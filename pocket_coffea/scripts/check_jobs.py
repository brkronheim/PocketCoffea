'''Simple script that checks the status of the jobs submitted by runner on condor.

The status of the jobs can be checked by looking at the file in the jobs folder.

- job_x.idle: The job is waiting to be executed
- job_x.running: The job is running
- job_x.done: The job has finished
- job_x.failed: The job has failed
    
    where x is the job id.
'''

import os
import click
import glob
import shutil
import subprocess as sp
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
import time
import re
import cloudpickle
from copy import deepcopy
from urllib.parse import urlparse
from pocket_coffea.utils.rucio import get_xrootd_sites_map, get_rucio_client
from pocket_coffea.utils.site_rewrite import _query_replicas
from pocket_coffea.utils.job_progress import (
    aggregate_by_group,
    load_job_to_group_map,
    render_progress_bar,
)
from pocket_coffea.executors.executors_cmsconnect import (
    clear_cmsconnect_job_status,
    restage_cmsconnect_job_configs,
)
from collections import Counter

queues = [
    "espresso",
    "microcentury",
    "longlunch",
    "workday",
    "tomorrow",
    "testmatch",
    "nextweek"
]
    


def get_tables(tot_jobs, idle_jobs, running_jobs, done_jobs, failed_jobs, details=False):
    # Summary table
    table1 = Table(title="Job Summary")
    table1.add_column("Total jobs", style="cyan", no_wrap=True)
    table1.add_column("Idle jobs", style="blue", no_wrap=True)
    table1.add_column("Running jobs", style="magenta", no_wrap=True)
    table1.add_column("Done jobs", style="green", no_wrap=True)
    table1.add_column("Failed jobs", style="red", no_wrap=True)
    table1.add_row(str(len(tot_jobs)),
                  str(len(idle_jobs)),
                  str(len(running_jobs)),
                  str(len(done_jobs)),
                  str(len(failed_jobs)))
    # Create a table to display the status
    if details:
        table2 = Table(title="Job Status")
        table2.add_column("Job ID", style="cyan", no_wrap=True)
        table2.add_column("Submitted", style="blue", no_wrap=True)
        table2.add_column("Running", style="magenta", no_wrap=True)
        table2.add_column("Done", style="green", no_wrap=True)
        table2.add_column("Failed", style="red", no_wrap=True)
        for job in tot_jobs:
            table2.add_row(job,
                          "X" if job in idle_jobs else "",
                          "X" if job in running_jobs else "",
                          "X" if job in done_jobs else "",
                          "X" if job in failed_jobs else "")
    else:
        table2 = None
    return table1, table2


# Layout setup
def create_layout(with_progress=False):
    """Two-column layout. The left column carries the summary table (and
    the per-group progress table when `with_progress` is True) and gets
    twice the width of the log panel on the right, since that's where the
    interesting content lives."""
    layout = Layout()
    layout.split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1),
    )
    if with_progress:
        # Fixed-height summary panel so it doesn't grow at the expense of the
        # per-group table; 9 rows covers the Panel border + Table title +
        # header row + data row + a bit of padding. Bumped from 7 to fit
        # everything without cropping the bottom of the table.
        layout["left"].split_column(
            Layout(name="summary", size=9),
            Layout(name="progress"),
        )
    return layout

def check_jobs_logs(jobs_folder):
    _sync_remote_status_files(jobs_folder)
     # Idle jobs
    idle_jobs = set([ a.split("/")[-1][:-5] for a in glob.glob(f"{jobs_folder}/job_*.idle")])
    # Running jobs
    running_jobs = set([a.split("/")[-1][:-8] for a in glob.glob(f"{jobs_folder}/job_*.running")])
    # Done jobs
    done_jobs = set([ a.split("/")[-1][:-5] for a in glob.glob(f"{jobs_folder}/job_*.done")])
    # Failed jobs
    failed_jobs = set([ a.split("/")[-1][:-7] for a in glob.glob(f"{jobs_folder}/job_*.failed")])
    for status_path in glob.glob(f"{jobs_folder}/job_*.status"):
        job = status_path.split("/")[-1][:-7]
        with open(status_path) as handle:
            status = handle.read().strip()
        if status == "done":
            done_jobs.add(job)
        elif status == "failed":
            failed_jobs.add(job)
        elif status == "running":
            running_jobs.add(job)

    jobs_config = Path(jobs_folder) / "jobs_config.yaml"
    if jobs_config.is_file():
        tot_jobs = sorted(path.stem for path in Path(jobs_folder).glob("job_*.sub"))
        condor_states = _condor_states_for_submission(
            jobs_folder, tot_jobs, _discover_cluster_ids(jobs_folder)
        )
        completed_outputs = _completed_remote_outputs(jobs_folder, tot_jobs)
        for job in tot_jobs:
            if job in done_jobs or job in failed_jobs:
                continue
            condor_state = condor_states.get(job, "unknown")
            if condor_state == "done":
                done_jobs.add(job)
            elif condor_state == "failed":
                failed_jobs.add(job)
            elif condor_state == "running":
                running_jobs.add(job)
            elif condor_state == "idle":
                idle_jobs.add(job)
            elif job in completed_outputs:
                done_jobs.add(job)

    failed_jobs = failed_jobs - done_jobs
    running_jobs = running_jobs - done_jobs - failed_jobs
    idle_jobs = idle_jobs - done_jobs - failed_jobs - running_jobs
    return list(idle_jobs), list(running_jobs), list(done_jobs), list(failed_jobs)


def _sync_remote_status_files(jobs_folder):
    jobs_config = Path(jobs_folder) / "jobs_config.yaml"
    if not jobs_config.is_file():
        return
    with open(jobs_config) as handle:
        payload = yaml.safe_load(handle) or {}
    status_destination = payload.get("status_destination")
    if not status_destination:
        return
    parsed = urlparse(status_destination)
    if parsed.scheme != "root" or not parsed.netloc:
        return
    remote_path = "/" + parsed.path.lstrip("/")
    try:
        listing = sp.run(
            ["xrdfs", parsed.netloc, "ls", remote_path],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, OSError, sp.TimeoutExpired):
        return
    if listing.returncode != 0:
        return

    known_jobs = {
        Path(sub_path).stem for sub_path in glob.glob(f"{jobs_folder}/job_*.sub")
    }
    legacy_statuses = []
    for entry in listing.stdout.splitlines():
        basename = Path(entry).name
        match = re.fullmatch(r"(job_\d+)\.(running|done|failed)", basename)
        if match and match.group(1) in known_jobs:
            Path(jobs_folder, basename).touch()
        elif basename.endswith(".status") and basename[:-7] in known_jobs:
            legacy_statuses.append(entry)

    for remote_status in legacy_statuses:
        local_status = str(Path(jobs_folder) / Path(remote_status).name)
        remote_url = f"root://{parsed.netloc}/{remote_status.lstrip('/')}"
        try:
            sp.run(
                ["xrdcp", "-f", remote_url, local_status],
                stdout=sp.DEVNULL,
                stderr=sp.DEVNULL,
                timeout=15,
                check=False,
            )
        except (FileNotFoundError, OSError, sp.TimeoutExpired):
            continue


def _discover_cluster_ids(jobs_folder):
    seen = set()
    for log_path in sorted(Path(jobs_folder, "logs").glob("job_*.log")):
        filename_match = re.fullmatch(r"job_(\d+)(?:\.\d+)?\.log", log_path.name)
        if filename_match:
            seen.add(filename_match.group(1))
        try:
            lines = log_path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            event_match = re.match(r"\d{3}\s+\((\d+)\.\d+\.\d+\)", line)
            if event_match:
                seen.add(event_match.group(1))
    return sorted(seen, key=int)


def _job_name_from_condor(proc_id, arguments, known_jobs):
    if arguments:
        job_id = arguments.split(None, 1)[0]
        if job_id.isdigit() and f"job_{job_id}" in known_jobs:
            return f"job_{job_id}"
    job_name = f"job_{proc_id}"
    return job_name if job_name in known_jobs else None


def _condor_states_for_submission(jobs_folder, tot_jobs, cluster_ids):
    del jobs_folder
    known_jobs = set(tot_jobs)
    states = {job: "unknown" for job in known_jobs}
    if not cluster_ids:
        return states

    attempts = {}

    def record(job_name, cluster_id, source_priority, state):
        rank = (int(cluster_id), source_priority)
        if rank >= attempts.get(job_name, ((-1), -1)):
            attempts[job_name] = rank
            states[job_name] = state

    constraint = " || ".join(
        f"ClusterId == {int(cluster_id)}" for cluster_id in cluster_ids
    )
    try:
        history = sp.run(
            [
                "condor_history",
                "-constraint",
                constraint,
                "-af",
                "ClusterId",
                "ProcId",
                "JobStatus",
                "ExitCode",
                "Args",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, OSError, sp.TimeoutExpired):
        history = None
    if history is not None and history.returncode == 0:
        for line in history.stdout.splitlines():
            fields = line.split(None, 4)
            if len(fields) < 4 or fields[0] not in cluster_ids:
                continue
            job_name = _job_name_from_condor(
                fields[1], fields[4] if len(fields) == 5 else "", known_jobs
            )
            if job_name is None:
                continue
            try:
                status = int(fields[2])
                exit_code = int(fields[3]) if fields[3] != "undefined" else None
            except ValueError:
                continue
            if status == 4:
                record(
                    job_name,
                    fields[0],
                    0,
                    "done" if exit_code == 0 else "failed",
                )
            elif status in {3, 5, 6}:
                record(job_name, fields[0], 0, "failed")

    try:
        current = sp.run(
            [
                "condor_q",
                "-constraint",
                constraint,
                "-af",
                "ClusterId",
                "ProcId",
                "JobStatus",
                "Args",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, OSError, sp.TimeoutExpired):
        current = None
    if current is not None and current.returncode == 0:
        for line in current.stdout.splitlines():
            fields = line.split(None, 3)
            if len(fields) < 3 or fields[0] not in cluster_ids:
                continue
            job_name = _job_name_from_condor(
                fields[1], fields[3] if len(fields) == 4 else "", known_jobs
            )
            if job_name is None:
                continue
            try:
                status = int(fields[2])
            except ValueError:
                continue
            if status == 1:
                record(job_name, fields[0], 1, "idle")
            elif status in {2, 7}:
                record(job_name, fields[0], 1, "running")
            elif status in {3, 5, 6}:
                record(job_name, fields[0], 1, "failed")
    return states


def _xrootd_directory_listing(remote_url):
    parsed = urlparse(remote_url)
    if parsed.scheme != "root" or not parsed.netloc:
        return set()
    remote_path = "/" + parsed.path.lstrip("/")
    try:
        listing = sp.run(
            ["xrdfs", parsed.netloc, "ls", remote_path],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (FileNotFoundError, OSError, sp.TimeoutExpired):
        return set()
    if listing.returncode != 0:
        return set()
    return {
        Path(entry).name
        for entry in listing.stdout.splitlines()
        if entry.strip()
    }


def _completed_remote_outputs(jobs_folder, tot_jobs):
    jobs_config = Path(jobs_folder) / "jobs_config.yaml"
    with open(jobs_config) as handle:
        jobs_payload = yaml.safe_load(handle) or {}
    jobs = jobs_payload.get("jobs_list") or {}
    expected_outputs = {}
    directory_urls = {}
    for job_name in tot_jobs:
        job_outputs = []
        for key in ("output_file", "root_output_file"):
            target = (jobs.get(job_name) or {}).get(key)
            if not target or not target.startswith("root://"):
                continue
            parsed = urlparse(target)
            remote_dir = parsed.path.rsplit("/", 1)[0]
            directory_key = (parsed.netloc, remote_dir)
            directory_urls[directory_key] = (
                f"root://{parsed.netloc}//{remote_dir.lstrip('/')}"
            )
            job_outputs.append((directory_key, Path(parsed.path).name))
        if job_outputs:
            expected_outputs[job_name] = job_outputs
    listings = {
        directory_key: _xrootd_directory_listing(remote_url)
        for directory_key, remote_url in directory_urls.items()
    }
    return {
        job_name
        for job_name, outputs in expected_outputs.items()
        if all(filename in listings[directory_key] for directory_key, filename in outputs)
    }


def get_progress_table(group_counts, label, multi_sample_overlap=False, bar_width=30):
    """Build a rich Table showing per-group progress, sorted by % done
    ascending so straggling groups surface at the top. Includes a stacked
    coloured progress bar column (done / running / idle / failed)."""
    title = f"Progress by {label}"
    if multi_sample_overlap:
        title += "  [dim](jobs touching multiple samples are counted under each)[/]"
    table = Table(title=title)
    table.add_column(label.capitalize(), style="cyan", no_wrap=True)
    table.add_column("Total", justify="right")
    table.add_column("Idle", justify="right", style="blue")
    table.add_column("Running", justify="right", style="magenta")
    table.add_column("Done", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Progress", justify="left", no_wrap=True)
    table.add_column("% Done", justify="right")

    rows = sorted(group_counts.items(), key=lambda kv: (kv[1]["pct_done"], kv[0]))
    for name, counts in rows:
        pct = f"{counts['pct_done']:.1f}%" if counts["total"] else "n/a"
        pct_style = "green" if counts["pct_done"] >= 99.5 else (
            "yellow" if counts["pct_done"] >= 50 else "red"
        )
        table.add_row(
            name,
            str(counts["total"]),
            str(counts["idle"]),
            str(counts["running"]),
            str(counts["done"]),
            str(counts["failed"]),
            render_progress_bar(counts, width=bar_width),
            f"[{pct_style}]{pct}[/]",
        )
    return table

def find_other_file(filepath, sitemap, xrootdfaillist=[], blacklist_sites=[], rucio_client=None):
    if filepath.startswith("root:/"):
        rootpref = filepath.split("/store/")[0]
        file = "/store/"+filepath.split("/store/")[1]
    else:
        rootpref = None
        file = filepath

    sites = _query_replicas(file, client=rucio_client)
    for site in sites:
        if site not in sitemap:
            continue
        sitepath = sitemap[site]
        if not isinstance(sitepath,str):
            continue
        if sitepath in blacklist_sites:
            continue
        if sitepath+file in xrootdfaillist:
            continue
        if rootpref:
            if rootpref in sitepath or sitepath in rootpref:
                continue
        return sitepath+file

    return filepath

def update_blacklist(xrootdfaillist,blacklist_threshold):
    sitepathlist = [i.split("/store/")[0] for i in xrootdfaillist]
    failedsitecounter = Counter(sitepathlist)
    blacklist_sites = []
    for site,fails in failedsitecounter.items():
        if fails > blacklist_threshold:
            blacklist_sites.append(site)
    return blacklist_sites


def _load_job_fileset(jobs_folder, failed_job):
    fileset_yaml = f"{jobs_folder}/fileset_{failed_job}.yaml"
    if os.path.isfile(fileset_yaml):
        with open(fileset_yaml) as handle:
            return yaml.safe_load(handle), fileset_yaml, None
    config_file = f"{jobs_folder}/config_{failed_job}.pkl"
    config = cloudpickle.load(open(config_file, "rb"))
    return config.filesets, config_file, config


def _save_job_fileset(fileset, target_path, config=None):
    if target_path.endswith(".yaml"):
        with open(target_path, "w") as handle:
            yaml.safe_dump(fileset, handle, sort_keys=False)
        return
    config.set_filesets_manually(fileset)
    cloudpickle.dump(config, open(target_path, "wb"))

def bump_jobqueue(sub_file, shift=1):
    with open(sub_file) as f:
        lines = f.readlines()
    next_jf = None
    updated_lines = []
    for line in lines:
        attribute = line.split("=", 1)[0].strip()
        if attribute not in {"+JobFlavour", "MY.JobFlavour"}:
            updated_lines.append(line)
            continue
        current_jf = line.split("=", 1)[1].strip().replace('"', '')
        if current_jf not in queues:
            updated_lines.append(line)
            continue
        next_jf = queues[min(queues.index(current_jf) + shift, len(queues) - 1)]
        updated_lines.append(f'{attribute} = "{next_jf}"\n')
    if next_jf is not None:
        with open(sub_file, "w") as f:
            f.writelines(updated_lines)
    return next_jf


def _extract_xrootd_failure(lines):
    for line_index, line in enumerate(lines):
        if "OSError: XRootD error" in line and line_index + 1 < len(lines):
            fields = lines[line_index + 1].strip().split()
            return fields[-1] if fields else None
        if "FileNotFoundError: file not found" in line and line_index + 3 < len(lines):
            return lines[line_index + 3].strip().strip("'")
        if "FileNotFoundError(" in line:
            match = re.search(r"filename=['\"](root://[^'\"]+)", line)
            if match:
                return match.group(1)
    return None


def _find_aborted_job_logs(logs_dir):
    aborted_logs = []
    for log_path in sorted(glob.glob(str(Path(logs_dir) / "job_*.log"))):
        try:
            with open(log_path, errors="replace") as handle:
                if any("job was aborted" in line.lower() for line in handle):
                    aborted_logs.append(log_path)
        except OSError:
            continue
    return aborted_logs


def _job_attempt_from_log_path(log_path):
    match = re.fullmatch(r"job_(\d+)\.(\d+)\.log", Path(log_path).name)
    if match is None:
        return None
    return match.group(1), str(int(match.group(2)))


def _submit_condor_job(jobs_folder, job_name):
    result = sp.run(
        ["condor_submit", f"{job_name}.sub"],
        cwd=jobs_folder,
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(
        stream.strip() for stream in (result.stdout, result.stderr) if stream.strip()
    )
    return result.returncode == 0, output


def _all_jobs_terminal(tot_jobs, done_jobs, failed_jobs, active_jobs=()):
    terminal_jobs = set(done_jobs) | (set(failed_jobs) - set(active_jobs))
    return set(tot_jobs) == terminal_jobs


def _resolve_jobs_folder(jobs_folder):
    jobs_folder = Path(jobs_folder)
    if any(jobs_folder.glob("job_*.sub")):
        return jobs_folder
    candidates = [
        child
        for child in jobs_folder.iterdir()
        if child.is_dir() and any(child.glob("job_*.sub"))
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        raise click.ClickException(
            f"Multiple job directories found under {jobs_folder}; pass one explicitly."
        )
    raise click.ClickException(f"No job_*.sub files found in {jobs_folder}.")

@click.command()
@click.option(
    "-j",
    "--jobs-folder",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Folder containing the jobs",
    required=True,
)
@click.option("-d","--details", is_flag=True, help="Show the details of the jobs")
@click.option("-r","--resubmit", is_flag=True, help="Resubmit the failed jobs")
@click.option("-m","--max-resubmit", type=int, help="Maximum number of resubmission", default=4)
@click.option("-b","--blacklist-threshold", type=int, help="Maximum number of allowed failed files at an xrootd site before it's blacklisted", default=3)
@click.option("-q","--queue-shift", type=int, help="How many queues to bump to if a job is removed due to time limit? E.g. 1 = bump to next queue, 2 = bump to next-to-next queue", default=1)
@click.option("--by", "group_by", type=click.Choice(["sample", "dataset", "none"]),
              default="sample",
              help="Show a per-group progress table below the summary. Requires "
                   "jobs_config.yaml in the jobs folder (created by manual-job "
                   "executors). Pass 'none' to disable. Default: sample.")
def check_jobs(jobs_folder, details, resubmit, max_resubmit, blacklist_threshold, queue_shift, group_by):
    jobs_folder = _resolve_jobs_folder(jobs_folder)
    # Get the list of files in the folder
    tot_jobs = sorted(path.stem for path in jobs_folder.glob("job_*.sub"))
    processed_logs_dir = jobs_folder / "logs" / "processedlogs"
    # Redo everything every 5 sec
    console = Console()

    resubmit_attempts = {}
    maxtimefile = f"{jobs_folder}/maxtime.txt"
    if os.path.isfile(maxtimefile):
            with open(maxtimefile,"r") as f:
                maxtimelist = [l.strip() for l in f.readlines()]
    else:
        maxtimelist = []

    # Load the per-job sample/dataset map if available and the user opted in.
    group_to_jobs = None
    group_label = None
    multi_sample_overlap = False
    if group_by != "none":
        sample_to_jobs, dataset_to_jobs = load_job_to_group_map(jobs_folder)
        if sample_to_jobs is None:
            rprint(f"[yellow]No jobs_config.yaml in {jobs_folder}; per-group progress "
                   f"table disabled.[/]")
        else:
            if group_by == "sample":
                group_to_jobs = sample_to_jobs
                group_label = "sample"
            else:
                group_to_jobs = dataset_to_jobs
                group_label = "dataset"
            # Detect uniform-split overlap: any job appearing under more than one group.
            all_jobs_listed = [j for jobs in group_to_jobs.values() for j in jobs]
            multi_sample_overlap = len(all_jobs_listed) != len(set(all_jobs_listed))

    if resubmit:
        sitemap = get_xrootd_sites_map()
        try:
            rucio_client = get_rucio_client()
        except Exception as e:
            print(f"WARNING: could not open a rucio client ({e}); replica lookups will fail.")
            rucio_client = None
        xrootdfailfile = f"{jobs_folder}/xrootdfaillist.txt"
        if os.path.isfile(xrootdfailfile):
            with open(xrootdfailfile,"r") as f:
                xrootdfaillist = [l.strip() for l in f.readlines()]
        else:
            xrootdfaillist = []
        processed_logs_dir.mkdir(parents=True, exist_ok=True)
        blacklist_sites = update_blacklist(xrootdfaillist,blacklist_threshold)
        if len(blacklist_sites) > 0:
            print("Blacklisted sites:",blacklist_sites)
    
    # Main loop
    show_progress = group_to_jobs is not None
    layout = create_layout(with_progress=show_progress)
    idle_jobs, running_jobs, done_jobs, failed_jobs = check_jobs_logs(jobs_folder)
    tables = get_tables(tot_jobs, idle_jobs, running_jobs, done_jobs, failed_jobs, details=details)
    if show_progress:
        layout["summary"].update(Panel(tables[0], title="Job Status"))
        gc = aggregate_by_group(group_to_jobs, idle_jobs, running_jobs, done_jobs, failed_jobs)
        layout["progress"].update(Panel(
            get_progress_table(gc, group_label, multi_sample_overlap=multi_sample_overlap)))
    else:
        layout["left"].update(Panel(tables[0], title="Job Status"))
    layout["right"].update(Panel("No logs yet", title="Log"))
    
    log_text = []
    definitive_failed = []
    resubmitted_and_failed = []
    step = 0
    
    with Live(layout, refresh_per_second=1/5, console=console):  # Refresh rate
        try:
            while True:
                step += 1
                idle_jobs, running_jobs, done_jobs, failed_jobs = check_jobs_logs(jobs_folder)
                tables = get_tables(tot_jobs, idle_jobs, running_jobs, done_jobs, failed_jobs, details=details)
                resubmitted_jobs = set()
                # Update the left panel(s) with fresh tables
                if show_progress:
                    layout["summary"].update(Panel(tables[0], title="Job Status"))
                    gc = aggregate_by_group(group_to_jobs, idle_jobs, running_jobs, done_jobs, failed_jobs)
                    layout["progress"].update(Panel(
                        get_progress_table(gc, group_label, multi_sample_overlap=multi_sample_overlap),
                        title=f"Progress by {group_label}"))
                else:
                    layout["left"].update(Panel(tables[0], title="Job Status"))

                # Checking failed jobs
                if len(failed_jobs) > 0:
                    if len(failed_jobs) > len(definitive_failed) and not resubmit:
                        log_text.append("[red]Failed jobs found. Check the details below. Use --resubmit to resubmit the failed jobs[/]")
                    resubmit_count = 0
                    for failed_job in failed_jobs:
                        attempts = resubmit_attempts.get(failed_job, 0)
                        failure_count = attempts + 1

                        failed_job_num = failed_job.split('_')[1]

                        if not failed_job in definitive_failed:
                            # Check the log files
                            glob_out = glob.glob(f"{jobs_folder}/logs/job_*.{failed_job_num}.out")
                            glob_err = glob.glob(f"{jobs_folder}/logs/job_*.{failed_job_num}.err")
                            xrootdfile = None
                            c = []
                            for log_path in glob_out[-1:] + glob_err[-1:]:
                                with open(log_path) as f:
                                    c.extend(f.readlines())
                            if c:
                                xrootdfile = _extract_xrootd_failure(c)
                                if xrootdfile:
                                    thisxrootdsite = xrootdfile.split('/store/')[0]
                                    log_text.append( f"[b]Job {failed_job} failed[/] {failure_count} times due to an XRootD error. Site: {thisxrootdsite}")
                                else:
                                    log_text.append( f"[b]Job {failed_job} failed[/] {failure_count} times. Last error:")
                                    log_text.append("\t"+ "".join(c[-3:]))
                            else:
                                log_text.append( f"Error in job {failed_job}: No .out/.err file found")

                            if resubmit and attempts < max_resubmit:
                                if xrootdfile:
                                    # Include the failed file in the global list so that it's not reused later
                                    if xrootdfile not in xrootdfaillist:
                                        with open(xrootdfailfile,"a") as f:
                                            f.write(xrootdfile+"\n")
                                        xrootdfaillist.append(xrootdfile)
                                        new_blacklist_sites = update_blacklist(xrootdfaillist,blacklist_threshold)
                                        if len(new_blacklist_sites) > len(blacklist_sites):
                                            diff = len(new_blacklist_sites) - len(blacklist_sites)
                                            log_text.append(f"[red][b]New blacklist sites[/]: {new_blacklist_sites[-diff:]}[/]")
                                            blacklist_sites = new_blacklist_sites

                                    # Move the logs so that this xrootdfile is not marked again as an XRootD failure
                                    for log_path in glob_out[-1:] + glob_err[-1:]:
                                        shutil.move(
                                            log_path,
                                            processed_logs_dir / Path(log_path).name,
                                        )

                                    # Update the filelist in the failed job's config to exclude this failed file
                                    current_fileset, job_state_path, config = _load_job_fileset(jobs_folder, failed_job)
                                    new_fileset = deepcopy(current_fileset)
                                    for sample, dct in new_fileset.items():
                                        fllist = dct['files']
                                        if xrootdfile in fllist:
                                            flidx = fllist.index(xrootdfile)
                                            newfl = find_other_file(xrootdfile,sitemap,xrootdfaillist,blacklist_sites,rucio_client=rucio_client)
                                            if newfl != xrootdfile:
                                                new_fileset[sample]['files'][flidx] = newfl
                                                _save_job_fileset(new_fileset, job_state_path, config)
                                                log_text.append(f"[b]Job {failed_job}[/]: Updated XRootD path of failed file to a new site.")
                                            else:
                                                log_text.append(f"[b]Job {failed_job}[/]: No alternative site found for {xrootdfile}. Resubmitting with the same file!")
                                        
                                    # Enforce blacklist              
                                    # Take this opportunity to replace all files in the config that are
                                    # at one of the blacklisted sites          
                                    if len(blacklist_sites) > 0:
                                        flcounter = 0
                                        samecounter = 0
                                        sitecounter = []
                                        for sample, dct in new_fileset.items():
                                            fllist = dct['files']
                                            newfllist = []
                                            for flname in fllist:
                                                thissite = flname.split("/store/")[0]
                                                if thissite in blacklist_sites:
                                                    newfl = find_other_file(flname,sitemap,xrootdfaillist,blacklist_sites,rucio_client=rucio_client)
                                                    newfllist.append(newfl)
                                                    if newfl != flname:
                                                        flcounter += 1
                                                        if thissite not in sitecounter:
                                                            sitecounter.append(thissite)
                                                    else:
                                                        samecounter += 1
                                                else:
                                                    newfllist.append(flname)
                                            
                                            new_fileset[sample]['files'] = newfllist

                                        _save_job_fileset(new_fileset, job_state_path, config)

                                        if flcounter > 0:
                                            log_text.append(f"[b]Job {failed_job}[/]: Replaced {flcounter} files in config because they were in {len(sitecounter)} blacklisted sites: {sitecounter}.")
                                        if samecounter > 0:
                                            log_text.append(f"[red][b]Job {failed_job}[/]: Could not replace {samecounter} files in config though they were in blacklisted sites, because no alternative site was found![/]")

                                cmsconnect_retry = False
                                try:
                                    if restage_cmsconnect_job_configs(jobs_folder, failed_job) is not None:
                                        clear_cmsconnect_job_status(jobs_folder, failed_job)
                                        cmsconnect_retry = True
                                except Exception as error:
                                    log_text.append(
                                        f"[red]Could not restage CMS Connect config for {failed_job}: {error}[/]"
                                    )
                                    continue

                                resubmit_attempts[failed_job] = attempts + 1
                                resubmit_succeeded, resubmit_log = _submit_condor_job(
                                    jobs_folder, failed_job
                                )

                                log_text.append(resubmit_log)
                                if resubmit_succeeded:
                                    Path(jobs_folder, f"{failed_job}.failed").unlink(missing_ok=True)
                                    Path(jobs_folder, f"{failed_job}.idle").touch()
                                    resubmitted_jobs.add(failed_job)
                                    resubmit_count += 1
                                elif cmsconnect_retry:
                                    Path(jobs_folder, f"{failed_job}.status").write_text("failed\n")

                                if resubmit_count > 0 and resubmit_count % 10 == 0:
                                    rprint(f"[green]Resubmitted {resubmit_count}/{len(failed_jobs)} jobs so far in step {step}[/]")   # Terminal output so that the user knows something's going on
                            else:
                                # Add it to the list of jobs that are definitely failed
                                definitive_failed.append(failed_job)
                    if resubmit_count > 0:
                        log_text.append(f"[red]Resubmitted {resubmit_count} failed jobs to condor[/]")

                active_jobs = set(resubmitted_jobs)
                if resubmit:
                    active_jobs.update(set(failed_jobs) - set(definitive_failed))
                if _all_jobs_terminal(
                    tot_jobs, done_jobs, failed_jobs, active_jobs
                ):
                    rprint("[green]All jobs are completed[/]")
                    rprint(f"Now merge outputs with [yellow]merge-outputs -jc {jobs_folder}[/].")
                    break

                # check in the logs for SYSTEM_PERIODIC_REMOVE
                # they are not failed but remain running/idle
                log_files = glob.glob(f"{jobs_folder}/logs/job_*.log")
                if not log_files:
                    time.sleep(5)
                    continue
                log_file = log_files[0]
                with open(log_file) as f:
                    c = f.readlines()
                
                for il, line in enumerate(c):
                    if line.startswith("009"):
                        # Match with a regex the job id from this
                        # line format "005 (5189350.010.000) 11/15 21:29:13 Job was aborted
                        pattern = re.compile(r"\((\d+)\.(\d+)\.\d+\)")
                        match = pattern.search(line)
                        if match:
                            cluster_id = match.group(1)
                            job_id = int(match.group(2))    # 010 -> 10
                            job_name = f"{cluster_id}_{job_id}"

                            # If this job was already resubmitted, skip
                            if job_name in maxtimelist:
                                continue

                            thisjob = f"job_{job_id}"
                            if thisjob in running_jobs or thisjob in idle_jobs:
                                if thisjob in running_jobs:
                                    running_jobs.remove(thisjob)
                                    Path(jobs_folder, f"{thisjob}.running").unlink(
                                        missing_ok=True
                                    )
                                
                                # Sometimes jobs which never run also get aborted; they have the idle tag
                                # but exist in the log file as and aborted job
                                if thisjob in idle_jobs:
                                    idle_jobs.remove(thisjob)
                                    Path(jobs_folder, f"{thisjob}.idle").unlink(
                                        missing_ok=True
                                    )

                                failed_jobs.append(thisjob)                                
                                Path(jobs_folder, f"{thisjob}.failed").touch()

                                maxtimelist.append(job_name)
                                with open(maxtimefile,'a') as f:
                                    f.write(job_name+"\n")

                                # Modify the sub file
                                # Check if next line has SYSTEM_PERIODIC_REMOVE
                                next_line = c[il + 1] if il + 1 < len(c) else ""
                                if "SYSTEM_PERIODIC_REMOVE" not in next_line:
                                    log_text.append(f"{thisjob} was aborted by condor. Check the log file for more details")
                                else:     
                                    sub_file = f"{jobs_folder}/{thisjob}.sub"
                                    next_jf = bump_jobqueue(sub_file, queue_shift)                                    

                                    log_text.append(f"{thisjob} was removed by the system due to max-time reached. Marked as failed and bumped to longer condor queue: {next_jf}.")
                
                # Now check jobs which were resubmitted by this script but then failed again
                # Look for "job was aborted"
                failedlogs = _find_aborted_job_logs(Path(jobs_folder) / "logs")
                for failedlog in failedlogs:
                    # Skip the OG log
                    if Path(log_file).name == Path(failedlog).name:
                        continue
                    job_attempt = _job_attempt_from_log_path(failedlog)
                    if job_attempt is None:
                        if failedlog not in resubmitted_and_failed:
                            # Report once, but don't keep reporting the same thing
                            log_text.append(f"[red]Detected a failed job log {failedlog} but could not determine the job id.[/]")
                            resubmitted_and_failed.append(failedlog)
                    else:
                        failedlogcluster, jobid = job_attempt
                        job_name = f"{failedlogcluster}_{jobid}"
                        if job_name in maxtimelist:
                            continue

                        thisjob = f"job_{jobid}"
                        if thisjob in running_jobs or thisjob in idle_jobs:
                            if thisjob in running_jobs:
                                running_jobs.remove(thisjob)
                                Path(jobs_folder, f"{thisjob}.running").unlink(
                                    missing_ok=True
                                )
                            
                            # Sometimes jobs which never run also get aborted; they have the idle tag
                            # but exist in the log file as and aborted job
                            if thisjob in idle_jobs:
                                idle_jobs.remove(thisjob)
                                Path(jobs_folder, f"{thisjob}.idle").unlink(
                                    missing_ok=True
                                )

                            failed_jobs.append(thisjob)                                
                            Path(jobs_folder, f"{thisjob}.failed").touch()

                            maxtimelist.append(job_name)
                            with open(maxtimefile,'a') as f:
                                f.write(job_name+"\n")

                            # Modify the sub file
                            # Check if log file has SYSTEM_PERIODIC_REMOVE
                            with open(failedlog,"r") as f:
                                lines = f.readlines()
                            
                            dobump = False
                            for line in lines:
                                if "SYSTEM_PERIODIC_REMOVE" in line:
                                    dobump = True
                                    break

                            if not dobump:
                                log_text.append(f"Resubmitted job, {thisjob}, was aborted [i]again[/] by condor. Check the log file for more details")
                            else:     
                                sub_file = f"{jobs_folder}/{thisjob}.sub"
                                next_jf = bump_jobqueue(sub_file, queue_shift)                                    

                                log_text.append(f"Resubmitted job, {thisjob}, was removed [i]again[/] by the system due to max-time reached. Marked as failed and bumped to longer condor queue: {next_jf}.")
                            
                            processed_logs_dir.mkdir(parents=True, exist_ok=True)
                            shutil.move(
                                failedlog,
                                processed_logs_dir / Path(failedlog).name,
                            )
                   
                if len(log_text):
                    if len(log_text) > 20:
                        log_text = log_text[-20:]
                    layout["right"].update(Panel("\n".join(log_text), title="Log"))

                time.sleep(5)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    check_jobs()

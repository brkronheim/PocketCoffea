"""Print a human-readable summary of a manual-executor job configuration.

Legacy manual executors produce a cloudpickled `Configurator` per job.
`condor@cmsconnect` produces a YAML descriptor that references one shared
configurator pickle and a compact per-job fileset YAML. This tool accepts
both formats and prints what the job will actually run on.

Typical use:
    pocket-coffea inspect-job /path/to/jobs_dir/job/config_job_42.pkl
    pocket-coffea inspect-job /path/to/cmsconnect_jobs/job/config_job_42.yaml
    pocket-coffea inspect-job .../config_job_42.pkl --files
"""
import os

import click
import cloudpickle
from rich import print as rprint
from rich.table import Table
from rich.console import Console

from pocket_coffea.utils.utils import load_job_config


@click.command(name="inspect-job")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-f", "--files", "show_files", is_flag=True,
              help="List every file URL per dataset.")
@click.option("-m", "--metadata", "show_metadata", is_flag=True,
              help="Print the full per-dataset metadata dict (verbose).")
@click.option("--workflow-options/--no-workflow-options", default=True,
              help="Print the workflow_options dict (default: on).")
def inspect_job(config_path, show_files, show_metadata, workflow_options):
    """Print a per-job .pkl or CMS Connect YAML descriptor."""
    console = Console()

    if config_path.endswith((".yaml", ".yml")):
        cfg = load_job_config(config_path)
        config_label = "Job descriptor"
    else:
        with open(config_path, "rb") as f:
            cfg = cloudpickle.load(f)
        config_label = "Job pickle"

    # --- header
    rprint(f"[bold]{config_label}:[/] {os.path.abspath(config_path)}")
    workflow = getattr(cfg, "workflow", None)
    wf_name = getattr(workflow, "__name__", str(workflow))
    wf_mod = getattr(workflow, "__module__", "")
    rprint(f"[bold]Workflow:[/] {wf_mod}.{wf_name}" if wf_mod else f"[bold]Workflow:[/] {wf_name}")

    if cfg.save_skimmed_files:
        rprint(f"[bold]save_skimmed_files:[/] {cfg.save_skimmed_files_folder}")
    else:
        rprint("[bold]save_skimmed_files:[/] off")

    if workflow_options and cfg.workflow_options:
        rprint("[bold]workflow_options:[/]")
        for k, v in cfg.workflow_options.items():
            rprint(f"  {k}: {v}")

    # --- datasets table
    filesets = getattr(cfg, "filesets", {}) or {}
    if not filesets:
        rprint("[yellow]No filesets in this pickle.[/]")
        return

    table = Table(title=f"Datasets in this job ({len(filesets)})")
    table.add_column("Dataset", style="cyan", no_wrap=True)
    table.add_column("Sample", style="magenta")
    table.add_column("Year", justify="right")
    table.add_column("isMC", justify="center")
    table.add_column("N files", justify="right", style="green")
    table.add_column("N events", justify="right", style="green")

    tot_files = 0
    tot_events = 0
    for ds_name, ds in filesets.items():
        meta = ds.get("metadata", {})
        files_list = ds.get("files", [])
        nev = meta.get("nevents", "?")
        try:
            nev_int = int(nev)
            tot_events += nev_int
            nev_str = f"{nev_int:_}"
        except (TypeError, ValueError):
            nev_str = str(nev)
        tot_files += len(files_list)

        table.add_row(
            ds_name,
            str(meta.get("sample", "?")),
            str(meta.get("year", "?")),
            str(meta.get("isMC", "?")),
            str(len(files_list)),
            nev_str,
        )
    table.add_section()
    table.add_row("[b]Total[/]", "", "", "", str(tot_files), f"{tot_events:_}")
    console.print(table)

    # --- per-dataset detail
    if show_metadata or show_files:
        for ds_name, ds in filesets.items():
            rprint(f"\n[bold cyan]── {ds_name}[/]")
            if show_metadata:
                rprint("  [bold]metadata:[/]")
                for k, v in (ds.get("metadata") or {}).items():
                    rprint(f"    {k}: {v}")
            if show_files:
                files_list = ds.get("files", [])
                rprint(f"  [bold]files ({len(files_list)}):[/]")
                for url in files_list:
                    rprint(f"    {url}")


if __name__ == "__main__":
    inspect_job()

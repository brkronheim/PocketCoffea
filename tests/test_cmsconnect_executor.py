import cloudpickle
import json
import os
import pytest
import subprocess
import sys
import tarfile
import yaml
from click.testing import CliRunner
from pathlib import Path
from unittest.mock import patch

from pocket_coffea.executors.executors_cmsconnect import (
    ExecutorFactoryCondorCMSConnect,
    _environment_cache_fingerprint,
    _find_python_site_packages,
    _get_editable_install_info,
    clear_cmsconnect_job_status,
    _mkdir_xrootd,
    restage_cmsconnect_job_configs,
)
from pocket_coffea.utils.utils import load_job_config
from pocket_coffea.utils.site_rewrite import GLOBAL_XROOTD_REDIRECTOR


class StubConfigurator:
    def __init__(self):
        self.filesets = {"original": {}}
        self.workflow_options = {"existing": True}
        self.do_postprocessing = True
        self.workflow = StubConfigurator
        self.save_skimmed_files = False

    def set_filesets_manually(self, filesets):
        self.filesets = filesets


def test_load_job_config_resolves_relative_yaml_references(tmp_path):
    configurator_path = tmp_path / "configurator.pkl"
    with configurator_path.open("wb") as handle:
        cloudpickle.dump(StubConfigurator(), handle)

    filesets = {
        "dataset": {
            "files": ["root://example.test//store/input.root"],
            "metadata": {"sample": "sample", "nevents": 10},
        }
    }
    (tmp_path / "fileset.yaml").write_text(yaml.safe_dump(filesets))
    descriptor = {
        "schema_version": 1,
        "configurator": "configurator.pkl",
        "fileset": "fileset.yaml",
        "workflow_options": {"dump_columns_as_arrays_per_chunk": "columns"},
    }
    descriptor_path = tmp_path / "config_job_0.yaml"
    descriptor_path.write_text(yaml.safe_dump(descriptor))

    config = load_job_config(descriptor_path)

    assert config.filesets == filesets
    assert config.workflow_options == {
        "existing": True,
        "dump_columns_as_arrays_per_chunk": "columns",
    }


def test_load_job_config_rejects_unknown_schema(tmp_path):
    descriptor_path = tmp_path / "config_job_0.yaml"
    descriptor_path.write_text(yaml.safe_dump({"schema_version": 2}))

    with pytest.raises(ValueError, match="schema_version"):
        load_job_config(descriptor_path)


def test_inspect_job_accepts_yaml_descriptor(tmp_path):
    from pocket_coffea.scripts.inspect_job import inspect_job

    with (tmp_path / "configurator.pkl").open("wb") as handle:
        cloudpickle.dump(StubConfigurator(), handle)
    (tmp_path / "fileset.yaml").write_text(yaml.safe_dump(sample_fileset()))
    descriptor_path = tmp_path / "config_job_0.yaml"
    descriptor_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "configurator": "configurator.pkl",
                "fileset": "fileset.yaml",
                "workflow_options": {},
            }
        )
    )

    result = CliRunner().invoke(inspect_job, [str(descriptor_path)])

    assert result.exit_code == 0, result.output
    assert "Job descriptor:" in result.output
    assert "dataset" in result.output


def make_factory(tmp_path, **run_options):
    factory = object.__new__(ExecutorFactoryCondorCMSConnect)
    factory.jobs_dir = str(tmp_path)
    factory.job_name = "test-job"
    factory.outputdir = "root://storage.example//store/user/test/output"
    factory.run_options = {
        "convert-parquet-to-root": False,
        "keep-coffea-output": True,
        "eos-prefix": "root://eosuser.cern.ch/",
        **run_options,
    }
    return factory


def sample_fileset():
    return {
        "dataset": {
            "files": ["root://input.example//store/input.root"],
            "metadata": {"sample": "sample", "nevents": 10},
        }
    }


def run_cmsconnect_wrapper(tmp_path, *arguments):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    command_log = tmp_path / "commands.log"
    command_stub = """#!/bin/bash
if [[ "$(basename "$0")" == "python" && "${1:-}" == "-" ]]; then
    printf '0\n'
    exit 0
fi
{
    printf '%s' "$(basename "$0")"
    printf '\t%s' "$@"
    printf '\n'
} >> "$CMSCONNECT_COMMAND_LOG"
"""
    for command in (
        "python",
        "runner",
        "xrdcp",
        "xrdfs",
        "condor_submit",
        "condor_q",
        "condor_history",
    ):
        command_path = fake_bin / command
        command_path.write_text(command_stub)
        command_path.chmod(0o755)
    environment = os.environ.copy()
    script = Path(__file__).parents[1] / "VHccPoCo" / "runCmsConnect.sh"
    jobs_dir = tmp_path / "jobs"
    environment.update(
        {
            "CMSCONNECT_COMMAND_LOG": str(command_log),
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "PYTHON_BIN": str(fake_bin / "python"),
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--skim-root",
            "off",
            "--output",
            "/eos/user/test/output",
            "--eos-base",
            "/eos/user/test/input",
            "--jobs-dir",
            str(jobs_dir),
            "--job-name",
            "analysis",
            *arguments,
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    commands = [line.split("\t") for line in command_log.read_text().splitlines()]
    return result, commands, jobs_dir


def test_run_cmsconnect_dry_run_builds_pinned_runner_command(tmp_path):
    result, commands, jobs_dir = run_cmsconnect_wrapper(
        tmp_path,
        "--dry-run",
        "--",
        "--jobs-dir",
        "wrong-jobs",
        "--job-name",
        "wrong-name",
        "--executor",
        "iterative",
        "--dry-run=false",
    )

    assert result.returncode == 0, result.stderr
    assert len(commands) == 1
    runner_command = commands[0]
    assert runner_command[:3] == [
        "python",
        "-m",
        "pocket_coffea.scripts.runner",
    ]
    assert runner_command[runner_command.index("-o") + 1] == str(
        jobs_dir / "analysis_submit_output"
    )
    assert runner_command[
        runner_command.index("--custom-run-options") + 1
    ].endswith("custom_run_options_vhqq_cmsconnect.yaml")
    assert runner_command[runner_command.index("--worker-image") + 1] == (
        "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/batch-team/containers/plusbatch/el9-full:latest"
    )
    assert runner_command[runner_command.index("--status-destination") + 1] == (
        "root://eosuser.cern.ch//eos/user/test/output/status/analysis"
    )
    dry_run_options = [
        argument for argument in runner_command if argument.startswith("--dry-run=")
    ]
    assert dry_run_options[-1] == "--dry-run=true"
    for option, expected in (
        ("--jobs-dir", str(jobs_dir)),
        ("--job-name", "analysis"),
        ("--executor", "condor@cmsconnect"),
    ):
        option_positions = [
            index for index, argument in enumerate(runner_command) if argument == option
        ]
        assert runner_command[option_positions[-1] + 1] == expected
    assert "--disable-column-output=true" in runner_command
    assert "--use-redirector" in runner_command


def test_run_cmsconnect_submit_monitors_exact_job_directory(tmp_path):
    result, commands, jobs_dir = run_cmsconnect_wrapper(tmp_path, "--submit")

    assert result.returncode == 0, result.stderr
    assert commands[0][:3] == ["python", "-m", "pocket_coffea.scripts.runner"]
    assert commands[1] == [
        "python",
        "-m",
        "pocket_coffea.scripts.check_jobs",
        "--jobs-folder",
        str(jobs_dir / "analysis"),
        "--resubmit",
        "--max-resubmit",
        "4",
        "--by",
        "sample",
    ]


def test_run_cmsconnect_no_use_redirector_overrides_custom_options(tmp_path):
    custom_options = tmp_path / "custom_options.yaml"
    custom_options.write_text("use-redirector: true\n")

    result, commands, _ = run_cmsconnect_wrapper(
        tmp_path,
        "--custom-run-options",
        str(custom_options),
        "--no-use-redirector",
    )

    assert result.returncode == 0, result.stderr
    runner_command = commands[0]
    assert "--no-use-redirector" in runner_command
    assert "--use-redirector" not in runner_command


def test_vhcc_local_monitor_delegates_to_package_entrypoint():
    script = Path(__file__).parents[1] / "VHccPoCo" / "check_jobs.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--jobs-folder" in result.stdout
    assert "--max-resubmit" in result.stdout


def test_monitor_keeps_successfully_resubmitted_job_active():
    from pocket_coffea.scripts.check_jobs import _all_jobs_terminal

    assert not _all_jobs_terminal(
        ["job_0", "job_1"],
        ["job_0"],
        ["job_1"],
        ["job_1"],
    )
    assert _all_jobs_terminal(
        ["job_0", "job_1"],
        ["job_0"],
        ["job_1"],
    )


def test_monitor_keeps_retryable_failure_active():
    from pocket_coffea.scripts.check_jobs import _all_jobs_terminal

    assert not _all_jobs_terminal(
        ["job_0"],
        [],
        ["job_0"],
        ["job_0"],
    )


def test_monitor_completes_without_condor_log(tmp_path):
    from pocket_coffea.scripts.check_jobs import check_jobs

    (tmp_path / "job_0.sub").touch()
    (tmp_path / "job_0.done").touch()

    result = CliRunner().invoke(
        check_jobs,
        ["--jobs-folder", str(tmp_path), "--by", "none"],
    )

    assert result.exit_code == 0, result.output
    assert "All jobs are completed" in result.output


def test_monitor_rejects_empty_jobs_folder(tmp_path):
    from pocket_coffea.scripts.check_jobs import check_jobs

    result = CliRunner().invoke(
        check_jobs,
        ["--jobs-folder", str(tmp_path), "--by", "none"],
    )

    assert result.exit_code != 0
    assert "No job_*.sub files found" in result.output


def test_monitor_restaging_failure_does_not_consume_retry(tmp_path):
    from pocket_coffea.scripts.check_jobs import check_jobs

    (tmp_path / "job_0.sub").touch()
    failed_state = ([], [], [], ["job_0"])
    states = iter([failed_state, failed_state, failed_state])

    def next_state(_jobs_folder):
        try:
            return next(states)
        except StopIteration:
            raise KeyboardInterrupt

    with (
        patch(
            "pocket_coffea.scripts.check_jobs.check_jobs_logs",
            side_effect=next_state,
        ),
        patch(
            "pocket_coffea.scripts.check_jobs.restage_cmsconnect_job_configs",
            side_effect=[RuntimeError("temporary staging error"), None],
        ) as restage,
        patch(
            "pocket_coffea.scripts.check_jobs._submit_condor_job",
            return_value=(True, "submitted"),
        ) as submit,
        patch(
            "pocket_coffea.scripts.check_jobs.get_xrootd_sites_map",
            return_value={},
        ),
        patch(
            "pocket_coffea.scripts.check_jobs.get_rucio_client",
            return_value=None,
        ),
        patch("pocket_coffea.scripts.check_jobs.time.sleep"),
    ):
        result = CliRunner().invoke(
            check_jobs,
            [
                "--jobs-folder",
                str(tmp_path),
                "--resubmit",
                "--max-resubmit",
                "1",
                "--by",
                "none",
            ],
        )

    assert result.exit_code == 0, result.output
    assert restage.call_count == 2
    submit.assert_called_once_with(tmp_path, "job_0")


def test_prepare_jobs_writes_yaml_job_descriptors(tmp_path):
    factory = make_factory(tmp_path)
    factory.config = StubConfigurator()

    job_configs = factory.prepare_jobs([sample_fileset()])

    assert job_configs == [str(tmp_path / "config_job_0.yaml")]
    descriptor = yaml.safe_load((tmp_path / "config_job_0.yaml").read_text())
    assert descriptor == {
        "schema_version": 1,
        "configurator": "configurator.pkl",
        "fileset": "fileset_job_0.yaml",
        "workflow_options": {},
    }
    jobs_config = yaml.safe_load((tmp_path / "jobs_config.yaml").read_text())
    assert jobs_config["jobs_list"]["job_0"]["config_file"] == "config_job_0.yaml"
    assert jobs_config["jobs_list"]["job_0"]["fileset_file"] == "fileset_job_0.yaml"


def test_prepare_jobs_can_disable_inherited_column_output(tmp_path):
    factory = make_factory(tmp_path, **{"disable-column-output": True})
    factory.config = StubConfigurator()
    factory.config.workflow_options["dump_columns_as_arrays_per_chunk"] = (
        "root://old.example//store/columns"
    )

    [job_config] = factory.prepare_jobs([sample_fileset()])

    descriptor = yaml.safe_load(Path(job_config).read_text())
    assert descriptor["workflow_options"] == {
        "dump_columns_as_arrays_per_chunk": None,
    }


def test_output_map_uses_remote_outputdir_as_default(tmp_path):
    factory = make_factory(tmp_path, **{"output-destination": None, "status-destination": None})
    path = factory._write_job_output_map(
        {"job_0": {"output_file": str(tmp_path / "output_job_0.coffea")}}
    )

    output_map = yaml.safe_load(open(path))
    assert output_map["0"]["coffea"] == (
        "root://storage.example//store/user/test/output/output_job_0.coffea"
    )
    assert output_map["0"]["status"] == (
        "root://storage.example//store/user/test/output/status/test-job/job_0.status"
    )


def test_output_map_rejects_worker_local_destination(tmp_path):
    factory = make_factory(tmp_path, **{"output-destination": "relative/output"})

    with pytest.raises(ValueError, match="output-destination"):
        factory._write_job_output_map(
            {"job_0": {"output_file": str(tmp_path / "output_job_0.coffea")}}
        )


def test_remote_output_uses_local_runner_directory(tmp_path, monkeypatch):
    from pocket_coffea.scripts.runner import _prepare_runner_outputdir

    monkeypatch.chdir(tmp_path)
    outputdir = "root://storage.example//store/user/test/analysis"
    with patch("pocket_coffea.scripts.runner.subprocess.run") as run:
        local_outputdir = _prepare_runner_outputdir(
            outputdir, "condor@cmsconnect"
        )

    assert local_outputdir.startswith(str(tmp_path))
    assert not local_outputdir.startswith("root://")
    assert os.path.isdir(local_outputdir)
    run.assert_called_once_with(
        ["xrdfs", "storage.example", "mkdir", "-p", "/store/user/test/analysis"],
        check=True,
    )


def test_remote_output_defaults_to_local_cmsconnect_jobs_dir(tmp_path, monkeypatch):
    from pocket_coffea.executors.executors_cmsconnect import (
        _local_cmsconnect_submission_dir,
    )

    monkeypatch.chdir(tmp_path)
    outputdir = "root://storage.example//store/user/test/analysis"
    factory = ExecutorFactoryCondorCMSConnect(
        run_options={"ignore-grid-certificate": True, "job-name": "my-job"},
        outputdir=outputdir,
    )

    expected_root = _local_cmsconnect_submission_dir(outputdir)
    assert factory.jobs_dir == os.path.join(expected_root, "my-job")
    assert os.path.isdir(factory.jobs_dir)


def test_mkdir_xrootd_uses_canonical_remote_path():
    with patch("pocket_coffea.executors.executors_cmsconnect.subprocess.run") as run:
        _mkdir_xrootd("root://eosuser.cern.ch//eos/user/u/user/output")

    run.assert_called_once_with(
        ["xrdfs", "eosuser.cern.ch", "mkdir", "-p", "/eos/user/u/user/output"],
        check=True,
    )


def test_analysis_archive_flattens_absolute_transfer_roots(tmp_path):
    analysis_dir = tmp_path / "external" / "analysis_pkg"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "module.py").write_text("VALUE = 1\n")
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    factory = make_factory(
        jobs_dir,
        **{"analysis-transfer-paths": [str(analysis_dir)]},
    )

    archive = factory._make_analysis_archive()

    with tarfile.open(archive) as handle:
        assert "analysis_pkg/module.py" in handle.getnames()


def test_submit_jobs_generates_bootstrap_safe_yaml_worker(tmp_path):
    factory = make_factory(
        tmp_path,
        **{
            "analysis-transfer-paths": [],
            "chunksize": 100,
            "cores-per-worker": 1,
            "custom-setup-commands": None,
            "disk-per-worker": "1GB",
            "dry-run": True,
            "ignore-grid-certificate": True,
            "max-retries": 3,
            "mem-per-worker": "1GB",
            "output-destination": "root://storage.example//store/user/test/output",
            "queue": None,
            "ship-python-env": False,
            "staging-area": "root://storage.example//store/user/test/staging",
            "status-destination": "/eos/user/t/test/status",
            "worker-image": "/cvmfs/example/image.sif",
        },
    )
    factory.config = StubConfigurator()
    factory.filesets = sample_fileset()
    factory._splits = [sample_fileset()]
    job_configs = factory.prepare_jobs(factory._splits)

    factory.submit_jobs(job_configs)

    script = (tmp_path / "job.sh").read_text()
    assert 'REMOTE_JOB_CONFIGS="$4"' in script
    assert 'STATUS_DIR="$6"' in script
    assert "write_status running" in script
    assert '"${STATUS_DIR%/}/$marker"' in script
    assert 'xrdcp -f "$REMOTE_JOB_CONFIGS" cmsconnect_job_configs.tar.gz' in script
    assert '--cfg "$JOB_CONFIG"' in script
    assert "config_job.pkl" not in script
    assert 'export POCKET_COFFEA_ANALYSIS_ROOT="$PWD/analysis_src"' in script
    assert "configurator.parameters.jets_calibration.jet_types.values()" in script
    assert "analysis_root.joinpath(*configured_path.parts[marker_index:])" in script
    assert 'remote_dir = "/" + parsed.path.rsplit("/", 1)[0].lstrip("/")' in script

    manifest = yaml.safe_load((tmp_path / "jobs_config.yaml").read_text())
    assert manifest["status_destination"] == "root://eosuser.cern.ch//eos/user/t/test/status"
    assert manifest["job_configs_archive"].endswith("/cmsconnect_job_configs.tar.gz")
    assert manifest["jobs_list"]["job_0"]["output_file"].startswith("root://")

    submit = (tmp_path / "jobs_all.sub").read_text()
    assert "cmsconnect_job_configs.tar.gz" in submit
    assert manifest["status_destination"] in submit
    assert "job_$(ProcId).status" not in submit
    assert "max_retries" not in submit


def test_prepare_jobs_rewrites_inputs_when_redirector_is_enabled(tmp_path):
    factory = make_factory(tmp_path, **{"use-redirector": True})
    factory.config = StubConfigurator()

    factory.prepare_jobs([sample_fileset()])

    manifest = yaml.safe_load((tmp_path / "jobs_config.yaml").read_text())
    fileset = yaml.safe_load((tmp_path / "fileset_job_0.yaml").read_text())
    expected_file = GLOBAL_XROOTD_REDIRECTOR + "store/input.root"
    assert manifest["use-redirector"] is True
    assert fileset["dataset"]["files"] == [expected_file]
    assert manifest["jobs_list"]["job_0"]["filesets"]["dataset"]["files"] == [
        expected_file
    ]


def test_restage_job_configs_publishes_updated_fileset(tmp_path):
    common_remote_archive = (
        "root://storage.example//store/staging/cmsconnect_job_configs.tar.gz"
    )
    factory = make_factory(
        tmp_path,
        **{
            "convert-parquet-to-root": False,
            "job_configs_archive": common_remote_archive,
        },
    )
    factory.config = StubConfigurator()
    job_configs = factory.prepare_jobs([sample_fileset(), sample_fileset()])
    manifest_path = tmp_path / "jobs_config.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["job_configs_archive"] = factory.run_options["job_configs_archive"]
    manifest_path.write_text(yaml.safe_dump(manifest))
    assert manifest["jobs_list"]["job_0"]["config_file"] == "config_job_0.yaml"
    assert manifest["jobs_list"]["job_0"]["fileset_file"] == "fileset_job_0.yaml"
    (tmp_path / "job_0.sub").write_text(
        "arguments = 0 100 remote_payload "
        f"{common_remote_archive} - remote_status\nqueue\n"
    )

    updated = sample_fileset()
    updated["dataset"]["files"] = ["root://replacement.example//store/replacement.root"]
    (tmp_path / "fileset_job_0.yaml").write_text(yaml.safe_dump(updated))

    with patch("pocket_coffea.executors.executors_cmsconnect.subprocess.run") as run:
        archive = restage_cmsconnect_job_configs(tmp_path, "job_0")

    with tarfile.open(archive) as handle:
        archived_fileset = yaml.safe_load(handle.extractfile("fileset_job_0.yaml"))
        assert set(handle.getnames()) == {"config_job_0.yaml", "fileset_job_0.yaml"}
    assert archived_fileset == updated
    remote_archive = (
        "root://storage.example//store/staging/"
        "cmsconnect_job_configs_job_0.tar.gz"
    )
    assert run.call_args_list[-1].args[0] == [
        "xrdcp",
        "-f",
        str(tmp_path / "cmsconnect_job_configs_job_0.tar.gz"),
        remote_archive,
    ]
    submit = (tmp_path / "job_0.sub").read_text()
    assert remote_archive in submit
    assert common_remote_archive not in submit
    assert job_configs == [
        str(tmp_path / "config_job_0.yaml"),
        str(tmp_path / "config_job_1.yaml"),
    ]


def test_restage_job_configs_reapplies_redirector_rewrite(tmp_path):
    factory = make_factory(
        tmp_path,
        **{
            "job_configs_archive": (
                "root://storage.example//store/staging/cmsconnect_job_configs.tar.gz"
            ),
            "use-redirector": True,
        },
    )
    factory.config = StubConfigurator()
    factory.prepare_jobs([sample_fileset()])
    manifest = yaml.safe_load((tmp_path / "jobs_config.yaml").read_text())
    manifest["job_configs_archive"] = factory.run_options["job_configs_archive"]
    (tmp_path / "jobs_config.yaml").write_text(yaml.safe_dump(manifest))
    (tmp_path / "job_0.sub").write_text(
        "arguments = 0 100 remote_payload "
        "root://storage.example//store/staging/cmsconnect_job_configs.tar.gz "
        "- remote_status\nqueue\n"
    )
    physical_fileset = {
        "dataset": {
            "files": ["root://site.example//store/replacement.root"],
            "metadata": {"sample": "sample", "nevents": 10},
        }
    }
    (tmp_path / "fileset_job_0.yaml").write_text(yaml.safe_dump(physical_fileset))

    with patch("pocket_coffea.executors.executors_cmsconnect.subprocess.run"):
        archive = restage_cmsconnect_job_configs(tmp_path, "job_0")

    with tarfile.open(archive) as handle:
        archived_fileset = yaml.safe_load(handle.extractfile("fileset_job_0.yaml"))
    assert archived_fileset["dataset"]["files"] == [
        GLOBAL_XROOTD_REDIRECTOR + "store/replacement.root"
    ]
    rewritten_fileset = yaml.safe_load((tmp_path / "fileset_job_0.yaml").read_text())
    assert rewritten_fileset == archived_fileset


def test_restage_legacy_manifest_restores_embedded_fileset(tmp_path):
    factory = make_factory(
        tmp_path,
        **{
            "job_configs_archive": (
                "root://storage.example//store/staging/cmsconnect_job_configs.tar.gz"
            ),
        },
    )
    factory.config = StubConfigurator()
    factory.prepare_jobs([sample_fileset()])
    manifest = yaml.safe_load((tmp_path / "jobs_config.yaml").read_text())
    manifest["job_configs_archive"] = factory.run_options["job_configs_archive"]
    manifest.pop("use-redirector", None)
    (tmp_path / "jobs_config.yaml").write_text(yaml.safe_dump(manifest))
    (tmp_path / "job_0.sub").write_text(
        "arguments = 0 100 remote_payload "
        "root://storage.example//store/staging/cmsconnect_job_configs.tar.gz "
        "- remote_status\nqueue\n"
    )
    embedded_fileset = {
        "dataset": {
            "files": ["root://xrootd-cms.infn.it//store/embedded.root"],
            "metadata": {"sample": "sample", "nevents": 10},
        }
    }
    manifest["jobs_list"]["job_0"]["filesets"] = embedded_fileset
    (tmp_path / "jobs_config.yaml").write_text(yaml.safe_dump(manifest))
    local_fileset = {
        "dataset": {
            "files": ["root://site.example//store/local.root"],
            "metadata": {"sample": "sample", "nevents": 10},
        }
    }
    (tmp_path / "fileset_job_0.yaml").write_text(yaml.safe_dump(local_fileset))

    with patch("pocket_coffea.executors.executors_cmsconnect.subprocess.run"):
        archive = restage_cmsconnect_job_configs(tmp_path, "job_0")

    with tarfile.open(archive) as handle:
        archived_fileset = yaml.safe_load(handle.extractfile("fileset_job_0.yaml"))
    assert archived_fileset == embedded_fileset
    assert yaml.safe_load((tmp_path / "fileset_job_0.yaml").read_text()) == embedded_fileset


def test_restage_job_configs_accepts_legacy_cwd_relative_paths(tmp_path):
    jobs_folder = tmp_path / "jobs_dir" / "job"
    jobs_folder.mkdir(parents=True)
    config_file = jobs_folder / "config_job_0.yaml"
    config_file.write_text("fileset: fileset_job_0.yaml\n")
    (jobs_folder / "fileset_job_0.yaml").write_text("dataset:\n  files: []\n")
    (jobs_folder / "job_0.sub").write_text(
        "arguments = 0 100 remote_payload "
        "root://storage.example//store/staging/cmsconnect_job_configs.tar.gz "
        "- remote_status\nqueue\n"
    )
    manifest = {
        "job_dir": str(jobs_folder),
        "job_configs_archive": (
            "root://storage.example//store/staging/cmsconnect_job_configs.tar.gz"
        ),
        "jobs_list": {
            "job_0": {"config_file": "jobs_dir/job/config_job_0.yaml"},
        },
    }
    (jobs_folder / "jobs_config.yaml").write_text(yaml.safe_dump(manifest))

    with patch("pocket_coffea.executors.executors_cmsconnect.subprocess.run"):
        archive = restage_cmsconnect_job_configs(jobs_folder, "job_0")

    with tarfile.open(archive) as handle:
        assert set(handle.getnames()) == {"config_job_0.yaml", "fileset_job_0.yaml"}


def test_clear_job_status_removes_local_and_remote_marker(tmp_path):
    manifest = {
        "status_destination": "root://eosuser.cern.ch//eos/user/t/test/status",
    }
    (tmp_path / "jobs_config.yaml").write_text(yaml.safe_dump(manifest))
    (tmp_path / "job_0.status").write_text("failed\n")
    (tmp_path / "job_0.running").touch()
    (tmp_path / "job_0.failed").touch()

    listing = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="".join(
            f"/eos/user/t/test/status/job_0.{state}\n"
            for state in ("running", "done", "failed", "status")
        ),
        stderr="",
    )
    with patch(
        "pocket_coffea.executors.executors_cmsconnect.subprocess.run",
        side_effect=[listing, None, None, None, None],
    ) as run:
        clear_cmsconnect_job_status(tmp_path, "job_0")

    assert not (tmp_path / "job_0.status").exists()
    assert not (tmp_path / "job_0.running").exists()
    assert not (tmp_path / "job_0.failed").exists()
    assert [call.args[0] for call in run.call_args_list] == [
        ["xrdfs", "eosuser.cern.ch", "ls", "/eos/user/t/test/status"],
        *[
        ["xrdfs", "eosuser.cern.ch", "rm", f"/eos/user/t/test/status/job_0.{state}"]
        for state in ("running", "done", "failed", "status")
        ],
    ]
    assert run.call_args_list[0].kwargs == {
        "capture_output": True,
        "text": True,
        "timeout": 30,
        "check": True,
    }
    for call in run.call_args_list[1:]:
        assert call.kwargs == {
            "check": True,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "timeout": 30,
        }


def test_clear_job_status_propagates_remote_cleanup_failure(tmp_path):
    manifest = {
        "status_destination": "root://eosuser.cern.ch//eos/user/t/test/status",
    }
    (tmp_path / "jobs_config.yaml").write_text(yaml.safe_dump(manifest))
    listing = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="/eos/user/t/test/status/job_0.failed\n",
        stderr="",
    )
    cleanup_error = subprocess.CalledProcessError(1, ["xrdfs", "rm"])

    with (
        patch(
            "pocket_coffea.executors.executors_cmsconnect.subprocess.run",
            side_effect=[listing, cleanup_error],
        ),
        pytest.raises(subprocess.CalledProcessError),
    ):
        clear_cmsconnect_job_status(tmp_path, "job_0")


def test_sync_remote_status_uses_one_directory_listing(tmp_path):
    from pocket_coffea.scripts.check_jobs import _sync_remote_status_files

    manifest = {
        "status_destination": "root://eosuser.cern.ch//eos/user/t/test/status",
    }
    (tmp_path / "jobs_config.yaml").write_text(yaml.safe_dump(manifest))
    for job_id in range(3):
        (tmp_path / f"job_{job_id}.sub").touch()

    listing = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            "/eos/user/t/test/status/job_0.running\n"
            "/eos/user/t/test/status/job_1.done\n"
            "/eos/user/t/test/status/job_2.failed\n"
            "/eos/user/t/test/status/unrelated.txt\n"
        ),
        stderr="",
    )
    with patch("pocket_coffea.scripts.check_jobs.sp.run", return_value=listing) as run:
        _sync_remote_status_files(tmp_path)

    run.assert_called_once_with(
        ["xrdfs", "eosuser.cern.ch", "ls", "/eos/user/t/test/status"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert (tmp_path / "job_0.running").is_file()
    assert (tmp_path / "job_1.done").is_file()
    assert (tmp_path / "job_2.failed").is_file()
    assert not (tmp_path / "unrelated.txt").exists()


def test_check_jobs_recognizes_remote_running_status(tmp_path):
    from pocket_coffea.scripts.check_jobs import check_jobs_logs

    (tmp_path / "job_0.idle").touch()
    (tmp_path / "job_0.status").write_text("running\n")

    idle, running, done, failed = check_jobs_logs(tmp_path)

    assert idle == []
    assert running == ["job_0"]
    assert done == []
    assert failed == []


def test_discover_cluster_ids_scans_all_logs_in_numeric_order(tmp_path):
    from pocket_coffea.scripts.check_jobs import _discover_cluster_ids

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "job_100.log").write_text("000 (100.000.000) submitted\n")
    (logs / "job_99.log").write_text("005 (101.000.000) terminated\n")

    assert _discover_cluster_ids(tmp_path) == ["99", "100", "101"]


def test_condor_state_recovery_maps_retry_arguments(tmp_path):
    from pocket_coffea.scripts.check_jobs import _condor_states_for_submission

    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            "123 0 4 0 0 400 payload\n"
            "123 1 4 1 1 400 payload\n"
        ),
        stderr="",
    )
    current = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="123 2 2 2 400 payload\n124 0 1 2 400 payload\n",
        stderr="",
    )
    with patch(
        "pocket_coffea.scripts.check_jobs.sp.run",
        side_effect=[completed, current],
    ) as run:
        states = _condor_states_for_submission(
            tmp_path,
            ["job_0", "job_1", "job_2"],
            ["123", "124"],
        )

    assert states == {
        "job_0": "done",
        "job_1": "failed",
        "job_2": "idle",
    }
    assert run.call_count == 2


def test_remote_output_recovery_requires_every_configured_output(tmp_path):
    from pocket_coffea.scripts.check_jobs import _completed_remote_outputs

    output_dir = "root://storage.example//store/user/test/output"
    manifest = {
        "jobs_list": {
            "job_0": {
                "output_file": f"{output_dir}/output_job_0.coffea",
                "root_output_file": f"{output_dir}/output_job_0_skim.root",
            },
            "job_1": {
                "root_output_file": f"{output_dir}/output_job_1_skim.root",
            },
        }
    }
    (tmp_path / "jobs_config.yaml").write_text(yaml.safe_dump(manifest))
    listing = {"output_job_0.coffea", "output_job_1_skim.root"}

    with patch(
        "pocket_coffea.scripts.check_jobs._xrootd_directory_listing",
        return_value=listing,
    ) as list_directory:
        completed = _completed_remote_outputs(tmp_path, ["job_0", "job_1"])

    assert completed == {"job_1"}
    list_directory.assert_called_once_with(output_dir)


def test_check_jobs_recovers_missing_cmsconnect_markers(tmp_path):
    from pocket_coffea.scripts.check_jobs import check_jobs_logs

    (tmp_path / "jobs_config.yaml").write_text("jobs_list: {}\n")
    for job_id in range(3):
        (tmp_path / f"job_{job_id}.sub").touch()
        (tmp_path / f"job_{job_id}.idle").touch()

    with (
        patch("pocket_coffea.scripts.check_jobs._sync_remote_status_files"),
        patch(
            "pocket_coffea.scripts.check_jobs._discover_cluster_ids",
            return_value=["123"],
        ),
        patch(
            "pocket_coffea.scripts.check_jobs._condor_states_for_submission",
            return_value={
                "job_0": "running",
                "job_1": "failed",
                "job_2": "unknown",
            },
        ),
        patch(
            "pocket_coffea.scripts.check_jobs._completed_remote_outputs",
            return_value={"job_2"},
        ),
    ):
        idle, running, done, failed = check_jobs_logs(tmp_path)

    assert idle == []
    assert running == ["job_0"]
    assert done == ["job_2"]
    assert failed == ["job_1"]


@pytest.mark.parametrize("attribute", ["+JobFlavour", "MY.JobFlavour"])
def test_bump_jobqueue_accepts_condor_attribute_styles(tmp_path, attribute):
    from pocket_coffea.scripts.check_jobs import bump_jobqueue

    submit_path = tmp_path / "job_0.sub"
    submit_path.write_text(f'{attribute} = "workday"\nqueue\n')

    assert bump_jobqueue(submit_path) == "tomorrow"
    assert '"tomorrow"' in submit_path.read_text()


def test_bump_jobqueue_without_flavour_is_a_noop(tmp_path):
    from pocket_coffea.scripts.check_jobs import bump_jobqueue

    submit_path = tmp_path / "job_0.sub"
    original = "executable = job.sh\nqueue\n"
    submit_path.write_text(original)

    assert bump_jobqueue(submit_path) is None
    assert submit_path.read_text() == original


def test_extract_xrootd_failure_handles_truncated_trace():
    from pocket_coffea.scripts.check_jobs import _extract_xrootd_failure

    assert _extract_xrootd_failure(["OSError: XRootD error\n"]) is None
    assert _extract_xrootd_failure(["FileNotFoundError: file not found\n"]) is None
    assert _extract_xrootd_failure(
        [
            "OSError: XRootD error\n",
            "attempted root://site.example//store/input.root\n",
        ]
    ) == "root://site.example//store/input.root"
    assert _extract_xrootd_failure(
        [
            "Skipping bad file. WorkItem(filename='root://site.example//store/input.root'). "
            "The error was: FileNotFoundError(2, 'No such file or directory').\n",
        ]
    ) == "root://site.example//store/input.root"


def test_find_aborted_job_logs_scans_log_contents(tmp_path):
    from pocket_coffea.scripts.check_jobs import (
        _find_aborted_job_logs,
        _job_attempt_from_log_path,
    )

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    aborted = logs_dir / "job_123.0.log"
    aborted.write_text("009 (...) Job was aborted\n")
    (logs_dir / "job_124.1.log").write_text("005 (...) Job terminated\n")

    assert _find_aborted_job_logs(logs_dir) == [str(aborted)]
    assert _job_attempt_from_log_path(aborted) == ("123", "0")
    assert _job_attempt_from_log_path(logs_dir / "job_123.log") is None


@pytest.mark.parametrize("returncode, expected", [(0, True), (1, False)])
def test_submit_condor_job_uses_process_returncode(tmp_path, returncode, expected):
    from pocket_coffea.scripts.check_jobs import _submit_condor_job

    result = subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout="1 job(s) submitted to cluster 123.\n" if returncode == 0 else "",
        stderr="submission failed\n" if returncode else "",
    )
    with patch("pocket_coffea.scripts.check_jobs.sp.run", return_value=result) as run:
        succeeded, output = _submit_condor_job(tmp_path, "job_0")

    assert succeeded is expected
    assert ("submitted" in output) is expected
    run.assert_called_once_with(
        ["condor_submit", "job_0.sub"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


def test_find_site_packages_uses_target_environment_version(tmp_path):
    site_packages = tmp_path / "lib" / "python3.11" / "site-packages"
    site_packages.mkdir(parents=True)
    (tmp_path / "pyvenv.cfg").write_text("version = 3.11.9\n")

    found_path, python_dir = _find_python_site_packages(tmp_path)

    assert found_path == str(site_packages)
    assert python_dir == "python3.11"


def test_editable_install_paths_are_decoded_and_import_normalized(tmp_path):
    site_packages = tmp_path / "site-packages"
    dist_info = site_packages / "demo_package-1.0.dist-info"
    dist_info.mkdir(parents=True)
    source_root = tmp_path / "source with space"
    package_source = source_root / "demo_package"
    package_source.mkdir(parents=True)
    (dist_info / "METADATA").write_text("Name: demo-package\n")
    (dist_info / "direct_url.json").write_text(
        json.dumps({"url": source_root.as_uri(), "dir_info": {"editable": True}})
    )

    _, resolve_map = _get_editable_install_info(str(site_packages))

    assert resolve_map == {"demo_package": str(package_source)}


def test_environment_fingerprint_changes_with_editable_source(tmp_path):
    env_path = tmp_path / "venv"
    env_path.mkdir()
    (env_path / "pyvenv.cfg").write_text("version = 3.12\n")
    source = tmp_path / "analysis" / "package"
    source.mkdir(parents=True)
    module = source / "module.py"
    module.write_text("VALUE = 1\n")

    first = _environment_cache_fingerprint(env_path, {"package": str(source)})
    module.write_text("VALUE = 200\n")
    second = _environment_cache_fingerprint(env_path, {"package": str(source)})

    assert first != second


def test_environment_fingerprint_changes_with_package_exclusions(tmp_path):
    env_path = tmp_path / "venv"
    env_path.mkdir()
    (env_path / "pyvenv.cfg").write_text("version = 3.12\n")

    first = _environment_cache_fingerprint(env_path, {}, {"cupy"})
    second = _environment_cache_fingerprint(env_path, {}, {"tensorflow"})

    assert first != second
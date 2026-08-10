import ast
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile

import cloudpickle
import yaml

from pocket_coffea.utils.site_rewrite import GLOBAL_XROOTD_REDIRECTOR, rewrite_fileset_to_redirector

from .executors_base import FuturesExecutorFactory, IterativeExecutorFactory
from .executors_manual_jobs import (
    ExecutorFactoryManualABC,
    INNER_RUN_OPTIONS_FILENAME,
    write_inner_run_options,
)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _xrootd_path(path, eos_prefix="root://eosuser.cern.ch/"):
    if path.startswith("root://"):
        return path
    if path.startswith("/eos/"):
        return eos_prefix.rstrip("/") + "/" + path
    return path


def _xrootd_join(base, *parts):
    return "/".join([base.rstrip("/")] + [part.strip("/") for part in parts])

def _xrootd_parent(path):
    return path.rsplit("/", 1)[0]


def _mkdir_xrootd(path):
    if not path.startswith("root://"):
        return
    remainder = path[len("root://"):]
    host, remote_path = remainder.split("/", 1)
    subprocess.run(["xrdfs", host, "mkdir", "-p", "/" + remote_path], check=True)


DEFAULT_ENV_EXCLUDE_PACKAGES = {
    "cupy",
    "cupy_backends",
    "jaxlib",
    "nvidia",
    "tensorflow",
    "tensorrt",
    "triton",
}


def _tar_filter(tarinfo):
    parts = set(tarinfo.name.split("/"))
    if parts & {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache"}:
        return None
    if tarinfo.name.endswith((".pyc", ".pyo")):
        return None
    return tarinfo


def _is_site_package_member(parts):
    return any(part in {"site-packages", "dist-packages"} for part in parts)


def _get_editable_install_info(site_packages_dir):
    """Detect pip editable installs in a site-packages directory.

    Modern pip (>=21.3) editable installs create:
      - ``__editable__.<pkg>.pth`` that imports a custom finder on startup
      - ``__editable__<pkg>_finder.py`` with a ``MAPPING`` dict that maps
        import names to absolute source paths on the *submitting* host.

    When the venv is shipped to a remote worker those absolute paths are
    invalid, so the .pth / finder files must be excluded from the env archive
    and the actual package source must be included directly under
    site-packages instead.

    Returns
    -------
    exclude_basenames : set[str]
        Filenames (basename only) that should be skipped when tarring the
        virtual environment.
    resolve_map : dict[str, str]
        Mapping from import name (e.g. ``pocket_coffea``) to the absolute
        path of the source directory on the submitting host.
    """
    exclude_basenames = set()
    resolve_map: dict[str, str] = {}

    # Map package-name-in-metadata -> import-name by reading METADATA / PKG-INFO
    def _import_name_from_dist_info(dist_info_dir):
        for meta in ("METADATA", "PKG-INFO"):
            path = os.path.join(dist_info_dir, meta)
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        if line.startswith("Name: "):
                            return line[len("Name: "):].strip()
        return None

    # Scan *.dist-info directories for editable installs
    dist_info_dirs = glob.glob(os.path.join(site_packages_dir, "*.dist-info"))
    for di_dir in dist_info_dirs:
        direct_url_path = os.path.join(di_dir, "direct_url.json")
        if not os.path.exists(direct_url_path):
            continue
        with open(direct_url_path) as f:
            info = json.load(f)
        dir_info = info.get("dir_info") or {}
        if not dir_info.get("editable"):
            continue

        url = info.get("url", "")
        if not url.startswith("file://"):
            continue
        source_path = url[len("file://"):]

        pkg_name = _import_name_from_dist_info(di_dir)
        if not pkg_name:
            # Fallback: derive from dist-info directory name
            pkg_name = os.path.basename(di_dir).rsplit("-", 1)[0]
        pkg_name_norm = pkg_name.replace("-", "_")

        # Collect all __editable__ files associated with this package
        for fname in os.listdir(site_packages_dir):
            if not fname.startswith("__editable__"):
                continue
            if pkg_name_norm not in fname:
                continue
            exclude_basenames.add(fname)

            # For .pth files, read the import to also catch the finder module
            if fname.endswith(".pth"):
                pth_path = os.path.join(site_packages_dir, fname)
                with open(pth_path) as f:
                    content = f.read().strip()
                m = re.match(r"import\s+(\S+)\s*;", content)
                if m:
                    finder_fname = m.group(1) + ".py"
                    if finder_fname in os.listdir(site_packages_dir):
                        exclude_basenames.add(finder_fname)
                        # Parse the MAPPING dict from the finder to learn
                        # the *import* name (which may differ from pkg_name)
                        finder_path = os.path.join(site_packages_dir, finder_fname)
                        try:
                            with open(finder_path) as f:
                                tree = ast.parse(f.read())
                            for node in ast.walk(tree):
                                # Handles both `MAPPING = {...}` (Assign) and
                                # `MAPPING: dict[str, str] = {...}` (AnnAssign)
                                targets = []
                                value = None
                                if isinstance(node, ast.Assign):
                                    targets = node.targets
                                    value = node.value
                                elif isinstance(node, ast.AnnAssign):
                                    targets = [node.target]
                                    value = node.value
                                if value is not None and isinstance(value, ast.Dict):
                                    for t in targets:
                                        if isinstance(t, ast.Name) and t.id == "MAPPING":
                                            for k, v in zip(value.keys, value.values):
                                                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                                                    resolve_map[k.value] = v.value
                        except Exception:
                            pass

        # If we haven't found a mapping via the finder, fall back to pkg_name.
        # The finder's MAPPING gives us the *import* name → actual source dir
        # (e.g. ``pocket_coffea`` → ``/path/to/pocket_coffea/pocket_coffea``).
        # When the fallback is used, we only have the project root (from
        # direct_url.json), so check for the common subdirectory name.
        if pkg_name not in resolve_map and pkg_name_norm not in resolve_map:
            candidate = os.path.join(source_path, pkg_name_norm)
            if os.path.isdir(candidate):
                resolve_map[pkg_name] = candidate
            else:
                resolve_map[pkg_name] = source_path

    return exclude_basenames, resolve_map


def _make_env_tar_filter(excluded_packages, editable_exclude_basenames=None):
    excluded_packages = {package.lower().replace("-", "_") for package in excluded_packages}
    editable_exclude = editable_exclude_basenames or set()

    def is_excluded_package(part):
        package_part = part.split(".", 1)[0]
        package_part = package_part.split("-", 1)[0]
        return any(package_part == package or package_part.startswith(f"{package}_") for package in excluded_packages)

    def filter_env_member(tarinfo):
        tarinfo = _tar_filter(tarinfo)
        if tarinfo is None:
            return None
        parts = tarinfo.name.split("/")
        basename = parts[-1]
        # Exclude pip editable-install .pth / finder files
        if basename in editable_exclude:
            return None
        normalized_parts = [part.lower().replace("-", "_") for part in parts]
        if set(normalized_parts) & {"test", "tests", "__pycache__", "share", "doc", "docs"}:
            return None
        if basename.endswith((".a", ".pyc", ".pyo")):
            return None
        if _is_site_package_member(parts):
            # Only check the top-level package name (the component right after
            # site-packages/ or dist-packages/).  We must NOT check nested
            # submodules, e.g. ``awkward/_nplikes/cupy.py`` — that ``cupy``
            # part is an awkward submodule, not the ``cupy`` package itself.
            for i, part in enumerate(normalized_parts):
                if part in {"site-packages", "dist-packages"} and i + 1 < len(normalized_parts):
                    top_level = normalized_parts[i + 1]
                    if is_excluded_package(top_level):
                        return None
                    break
        return tarinfo

    return filter_env_member


def _cache_key_for_path(path):
    resolved = os.path.realpath(path)
    return hashlib.sha256(resolved.encode()).hexdigest()[:12]


def _packing_code_hash():
    """Hash of this executor file, used to invalidate the env cache when the
    packing logic (e.g. editable-install handling) changes."""
    try:
        with open(__file__, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except Exception:
        return "000000000000"


class ExecutorFactoryCondorCMSConnect(ExecutorFactoryManualABC):
    def get(self):
        pass

    def prepare_jobs(self, splits):
        if _as_bool(self.run_options.get("use-redirector", False)):
            print(
                f"[cmsconnect] --use-redirector: rewriting every input file to "
                f"{GLOBAL_XROOTD_REDIRECTOR}."
            )
            splits = [rewrite_fileset_to_redirector(split) for split in splits]

        config_path = os.path.join(self.jobs_dir, "configurator.pkl")
        self.config.do_postprocessing = False
        cloudpickle.dump(self.config, open(config_path, "wb"))

        jobs_config = {
            "job_name": self.job_name,
            "job_dir": os.path.abspath(self.jobs_dir),
            "output_dir": os.path.abspath(self.outputdir),
            "config_pkl_total": config_path,
            "jobs_list": {},
        }

        fileset_files = []
        for index, split in enumerate(splits):
            fileset_file = os.path.join(self.jobs_dir, f"fileset_job_{index}.yaml")
            with open(fileset_file, "w") as handle:
                yaml.safe_dump(split, handle, sort_keys=False)
            fileset_files.append(fileset_file)
            output_file = os.path.join(os.path.abspath(self.outputdir), f"output_job_{index}.coffea")
            jobs_config["jobs_list"][f"job_{index}"] = {
                "filesets": split,
                "fileset_file": fileset_file,
                "output_file": output_file,
            }
            if _as_bool(self.run_options.get("convert-parquet-to-root", False)):
                jobs_config["jobs_list"][f"job_{index}"]["root_output_file"] = output_file.replace(".coffea", "_skim.root")

        with open(os.path.join(self.jobs_dir, "jobs_config.yaml"), "w") as handle:
            yaml.safe_dump(jobs_config, handle, sort_keys=False)
        return fileset_files

    def _make_python_env_archive(self):
        configured = self.run_options.get("python-env-archive")
        if configured:
            return os.path.abspath(configured)
        env_path = os.path.abspath(self.run_options.get("python-env-path") or sys.prefix)
        use_cache = _as_bool(self.run_options.get("python-env-cache", True))
        force_repack = _as_bool(self.run_options.get("python-env-force-repack", False))
        if use_cache:
            cache_dir = os.path.abspath(
                os.path.expanduser(
                    self.run_options.get("python-env-cache-dir")
                    or os.path.join("~", ".cache", "pocket_coffea", "cmsconnect_python_envs")
                )
            )
            os.makedirs(cache_dir, exist_ok=True)
            archive = os.path.join(
                cache_dir,
                f"python_env_{os.path.basename(env_path)}_{sys.version_info.major}{sys.version_info.minor}_{_cache_key_for_path(env_path)}_{_packing_code_hash()}.tar.gz",
            )
        else:
            archive = os.path.abspath(os.path.join(self.jobs_dir, "python_env.tar.gz"))
        if os.path.exists(archive) and not force_repack:
            print(f"Reusing cached Python environment archive {archive}")
            return archive
        excluded_packages = DEFAULT_ENV_EXCLUDE_PACKAGES | set(_as_list(self.run_options.get("python-env-exclude-packages")))
        print(f"Packing Python environment {env_path} -> {archive}")

        # Detect pip editable installs so we can replace the .pth/finder dance
        # with the actual source in the archive.
        py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        site_packages_dir = os.path.join(env_path, "lib", py_ver, "site-packages")
        editable_exclude, editable_resolve = _get_editable_install_info(site_packages_dir)

        # Determine the site-packages arcname prefix for adding resolved source
        sp_arcname = f"python_env/lib/{py_ver}/site-packages"

        with tarfile.open(archive, "w:gz") as tar:
            tar.add(
                env_path,
                arcname="python_env",
                filter=_make_env_tar_filter(excluded_packages, editable_exclude),
            )
            # Include the actual source for each editable package directly
            # under site-packages so imports work without the custom finder.
            for import_name, src_path in editable_resolve.items():
                resolved_path = os.path.normpath(src_path)
                if not os.path.isdir(resolved_path) and os.path.isfile(resolved_path):
                    # Single-module package (e.g. a .py file)
                    arc = f"{sp_arcname}/{os.path.basename(resolved_path)}"
                    tar.add(resolved_path, arcname=arc, filter=_tar_filter)
                elif os.path.isdir(resolved_path):
                    arc = f"{sp_arcname}/{import_name}"
                    tar.add(resolved_path, arcname=arc, filter=_tar_filter)
                else:
                    print(
                        f"Warning: editable package '{import_name}' source "
                        f"not found at {resolved_path}, skipping",
                        file=sys.stderr,
                    )
        return archive

    def _make_analysis_archive(self):
        archive = os.path.abspath(os.path.join(self.jobs_dir, "analysis_bundle.tar.gz"))
        paths = _as_list(self.run_options.get("analysis-transfer-paths"))
        selected = []
        for pattern in paths:
            matches = glob.glob(pattern)
            selected.extend(matches or ([pattern] if os.path.exists(pattern) else []))
        selected = sorted(dict.fromkeys(selected))
        if not selected:
            return None
        print(f"Packing analysis payload -> {archive}")
        with tarfile.open(archive, "w:gz") as tar:
            for path in selected:
                tar.add(path, filter=_tar_filter)
        return archive

    def _upload_with_xrdcp(self, local_path, remote_path):
        print(f"Staging {local_path} -> {remote_path}")
        _mkdir_xrootd(_xrootd_parent(remote_path))
        subprocess.run(["xrdcp", "-f", local_path, remote_path], check=True)

    def _make_payload_archive(self, fileset_files, analysis_archive, inner_yaml_path, output_map):
        archive = os.path.abspath(os.path.join(self.jobs_dir, "cmsconnect_payload.tar.gz"))
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(os.path.join(self.jobs_dir, "configurator.pkl"), arcname="configurator.pkl")
            tar.add(inner_yaml_path, arcname=INNER_RUN_OPTIONS_FILENAME)
            tar.add(output_map, arcname="job_output_map.yaml")
            for fileset_file in fileset_files:
                tar.add(fileset_file, arcname=os.path.basename(fileset_file))
            if analysis_archive:
                tar.add(analysis_archive, arcname="analysis_bundle.tar.gz")
            tar.add(os.path.join(self.jobs_dir, "jobs_config.yaml"), arcname="jobs_config.yaml")
        return archive

    def _write_job_output_map(self, jobs_config):
        path = os.path.join(self.jobs_dir, "job_output_map.yaml")
        payload = {}
        eos_prefix = self.run_options.get("eos-prefix", "root://eosuser.cern.ch/")
        output_destination = self.run_options.get("output-destination")
        if output_destination is None and os.path.abspath(self.outputdir).startswith("/eos/"):
            output_destination = os.path.abspath(self.outputdir)
        if output_destination is None:
            raise ValueError("condor@cmsconnect requires output-destination, or an -o path under /eos/.")
        output_destination = _xrootd_path(output_destination, eos_prefix)
        status_destination = self.run_options.get("status-destination") or _xrootd_join(output_destination, "status")
        status_destination = _xrootd_path(status_destination, eos_prefix)
        keep_coffea = _as_bool(self.run_options.get("keep-coffea-output", True))
        for job_name, job in jobs_config.items():
            job_id = job_name.split("_", 1)[1]
            payload[job_id] = {
                "status": _xrootd_join(status_destination, f"job_{job_id}.status"),
            }
            if keep_coffea:
                payload[job_id]["coffea"] = _xrootd_join(output_destination, os.path.basename(job["output_file"]))
            if "root_output_file" in job:
                payload[job_id]["root"] = _xrootd_join(output_destination, os.path.basename(job["root_output_file"]))
        with open(path, "w") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        return path

    def submit_jobs(self, fileset_files):
        abs_output_path = os.path.abspath(self.outputdir)
        abs_jobdir_path = os.path.abspath(self.jobs_dir)
        logs_dir = os.path.join(abs_jobdir_path, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        inner_yaml_path = write_inner_run_options(self.jobs_dir, self.run_options)
        analysis_archive = self._make_analysis_archive()
        env_archive = self._make_python_env_archive() if _as_bool(self.run_options.get("ship-python-env", True)) else None

        convert_parquet = _as_bool(self.run_options.get("convert-parquet-to-root", False))
        keep_coffea = _as_bool(self.run_options.get("keep-coffea-output", True))
        if not keep_coffea and not convert_parquet:
            raise ValueError("keep-coffea-output: false requires convert-parquet-to-root: true for condor@cmsconnect.")

        with open(os.path.join(self.jobs_dir, "jobs_config.yaml")) as handle:
            jobs_config = yaml.safe_load(handle)["jobs_list"]
        output_map = self._write_job_output_map(jobs_config)

        eos_prefix = self.run_options.get("eos-prefix", "root://eosuser.cern.ch/")
        staging_area = self.run_options.get("staging-area")
        if staging_area is None:
            output_destination = self.run_options.get("output-destination")
            if output_destination is None and abs_output_path.startswith("/eos/"):
                output_destination = abs_output_path
            if output_destination is None:
                raise ValueError("condor@cmsconnect requires staging-area, or output-destination/-o under /eos/.")
            staging_area = _xrootd_join(output_destination, "cmsconnect_staging", self.job_name)
        staging_area = _xrootd_path(staging_area, eos_prefix)

        with open(os.path.join(self.jobs_dir, "jobs_config.yaml")) as handle:
            jobs_payload = yaml.safe_load(handle)
        jobs_payload["staging_area"] = staging_area
        jobs_payload["status_destination"] = self.run_options.get("status-destination") or _xrootd_join(
            _xrootd_path(self.run_options.get("output-destination") or abs_output_path, eos_prefix), "status")
        with open(os.path.join(self.jobs_dir, "jobs_config.yaml"), "w") as handle:
            yaml.safe_dump(jobs_payload, handle, sort_keys=False)

        payload_archive = self._make_payload_archive(fileset_files, analysis_archive, inner_yaml_path, output_map)
        remote_payload = _xrootd_join(staging_area, os.path.basename(payload_archive))
        remote_env_archive = _xrootd_join(staging_area, "python_env.tar.gz") if env_archive else ""
        jobs_payload["payload_archive"] = remote_payload
        if remote_env_archive:
            jobs_payload["python_env_archive"] = remote_env_archive
        with open(os.path.join(self.jobs_dir, "jobs_config.yaml"), "w") as handle:
            yaml.safe_dump(jobs_payload, handle, sort_keys=False)
        if not _as_bool(self.run_options.get("dry-run", False)):
            if env_archive:
                self._upload_with_xrdcp(env_archive, remote_env_archive)
            self._upload_with_xrdcp(payload_archive, remote_payload)

        chunksize_cfg = self.run_options["chunksize"]
        self._validate_chunksize_keys(chunksize_cfg, self.filesets)
        per_job_chunksize = [self._resolve_chunksize_for_job(chunksize_cfg, split) for split in self._splits]

        columns_dir = self.run_options.get("parquet-output-dir", "columns")
        executor_args = "--executor iterative"
        if int(self.run_options["cores-per-worker"]) > 1:
            executor_args = f"--executor futures --scaleout {self.run_options['cores-per-worker']}"
        x509_export = ""
        if not self.run_options["ignore-grid-certificate"]:
            x509_export = "export X509_USER_PROXY=$PWD/" + os.path.basename(self.x509_path) + "\n"
        custom_setup = "\n".join(_as_list(self.run_options.get("custom-setup-commands")))
        if custom_setup:
            custom_setup += "\n"

        script = f"""#!/bin/bash
set -euo pipefail

JOB_ID="$1"
CHUNKSIZE="$2"
REMOTE_PAYLOAD="$3"
REMOTE_PYTHON_ENV="${{4:-}}"
CONFIG_PKL="configurator.pkl"
COLUMNS_DIR="{columns_dir}"
FILESET_YAML="fileset_job_${{JOB_ID}}.yaml"

# Ensure the per-job status file is always written — even when the script
# dies before reaching the explicit success/failure branches — so the
# submit host's job checker can pick the job up as 'failed' instead of
# leaving it stuck in 'idle'. Uses the venv's python (set later) via PATH
# so we re-resolve it at trap time; if no venv is shipped, the system
# python is used as a fallback.
write_status() {{
    local state="$1"
    if [ -x python_env/bin/python ]; then
        local py=python_env/bin/python
    else
        local py=python
    fi
    echo "$state" > "job_${{JOB_ID}}.status"
    if [ -f job_output_map.yaml ]; then
        "$py" - "$JOB_ID" "$state" <<'PYSTATUS' || true
import subprocess, sys
from urllib.parse import urlparse
try:
    import yaml
except Exception:
    sys.exit(0)
job_id, state = sys.argv[1], sys.argv[2]
with open("job_output_map.yaml") as handle:
    targets = yaml.safe_load(handle).get(job_id, {{}}) or {{}}
dest = targets.get("status")
if not dest:
    sys.exit(0)
if dest.startswith("root://"):
    parsed = urlparse(dest)
    remote_dir = parsed.path.rsplit("/", 1)[0]
    subprocess.run(["xrdfs", parsed.netloc, "mkdir", "-p", remote_dir],
                   check=False)
subprocess.run(["xrdcp", "-f", f"job_{{job_id}}.status", dest], check=False)
PYSTATUS
    fi
}}
trap 'write_status failed' ERR

{x509_export}
{custom_setup}
xrdcp -f "$REMOTE_PAYLOAD" cmsconnect_payload.tar.gz
tar xzf cmsconnect_payload.tar.gz

if [ -n "$REMOTE_PYTHON_ENV" ]; then
    xrdcp -f "$REMOTE_PYTHON_ENV" python_env.tar.gz
fi

if [ -f python_env.tar.gz ]; then
    mkdir -p python_env
    tar xzf python_env.tar.gz -C python_env --strip-components=1
    source python_env/bin/activate
fi

if [ -f analysis_bundle.tar.gz ]; then
    mkdir -p analysis_src
    tar xzf analysis_bundle.tar.gz -C analysis_src
    export PYTHONPATH="$PWD/analysis_src:$PWD:${{PYTHONPATH:-}}"
fi

export XRD_RUNFORKHANDLER=1
export MALLOC_TRIM_THRESHOLD_=0

# Pick a Python interpreter explicitly. The unpickle below may reference
# numpy/cloudpickle versions baked into the configurator.pkl on the submit
# host, so it must use the venv's interpreter (with the matching numpy)
# when one was shipped; otherwise fall back to whatever `python` resolves to
# in the singularity image.
if [ -x python_env/bin/python ]; then
    WORKER_PY=python_env/bin/python
else
    WORKER_PY=python
fi
"""
        script += f"""
"$WORKER_PY" - "$CONFIG_PKL" "$FILESET_YAML" "$COLUMNS_DIR" <<'PY'
import cloudpickle
import sys
import yaml

config_pkl, fileset_yaml, columns_dir = sys.argv[1:4]
with open(config_pkl, "rb") as handle:
    config = cloudpickle.load(handle)
with open(fileset_yaml) as handle:
    fileset = yaml.safe_load(handle)
config.set_filesets_manually(fileset)
workflow_options = getattr(config, "workflow_options", {{}}) or {{}}
if "dump_columns_as_arrays_per_chunk" in workflow_options:
    workflow_options["dump_columns_as_arrays_per_chunk"] = columns_dir
    config.workflow_options = workflow_options
with open("config_job.pkl", "wb") as handle:
    cloudpickle.dump(config, handle)
PY

set +e
"$WORKER_PY" -m pocket_coffea.scripts.runner --cfg config_job.pkl -o output {executor_args} --chunksize "$CHUNKSIZE" --custom-run-options {INNER_RUN_OPTIONS_FILENAME}
status=$?
set -e

if [ "$status" -eq 0 ]; then
    cp output/output_all.coffea output_job_${{JOB_ID}}.coffea
"""
        if convert_parquet:
            script += "    if [ -d \"$COLUMNS_DIR\" ]; then\n"
            script += "        \"$WORKER_PY\" -m pocket_coffea.scripts.parquet_to_root \"$COLUMNS_DIR\" output_job_${JOB_ID}_skim.root\n"
            script += "    fi\n"
        script += """    "$WORKER_PY" - "$JOB_ID" <<'PY'
import subprocess
import sys
from urllib.parse import urlparse
import yaml


def xrdcp_with_mkdir(src, dest):
    if dest.startswith("root://"):
        parsed = urlparse(dest)
        remote_dir = parsed.path.rsplit("/", 1)[0]
        subprocess.run(["xrdfs", parsed.netloc, "mkdir", "-p", remote_dir], check=True)
    subprocess.run(["xrdcp", "-f", src, dest], check=True)

job_id = sys.argv[1]
with open("job_output_map.yaml") as handle:
    targets = yaml.safe_load(handle)[job_id]
if "coffea" in targets:
    xrdcp_with_mkdir(f"output_job_{job_id}.coffea", targets["coffea"])
if "root" in targets:
    xrdcp_with_mkdir(f"output_job_{job_id}_skim.root", targets["root"])
PY
    echo done > "job_${JOB_ID}.status"
else
    echo failed > "job_${JOB_ID}.status"
    "$WORKER_PY" - "$JOB_ID" <<'PY'
import subprocess
import sys
from urllib.parse import urlparse
import yaml


def xrdcp_with_mkdir(src, dest):
    if dest.startswith("root://"):
        parsed = urlparse(dest)
        remote_dir = parsed.path.rsplit("/", 1)[0]
        subprocess.run(["xrdfs", parsed.netloc, "mkdir", "-p", remote_dir], check=True)
    subprocess.run(["xrdcp", "-f", src, dest], check=False)

job_id = sys.argv[1]
with open("job_output_map.yaml") as handle:
    targets = yaml.safe_load(handle)[job_id]
xrdcp_with_mkdir(f"job_{job_id}.status", targets["status"])
PY
    exit "$status"
fi
"$WORKER_PY" - "$JOB_ID" <<'PY'
import subprocess
import sys
from urllib.parse import urlparse
import yaml


def xrdcp_with_mkdir(src, dest):
    if dest.startswith("root://"):
        parsed = urlparse(dest)
        remote_dir = parsed.path.rsplit("/", 1)[0]
        subprocess.run(["xrdfs", parsed.netloc, "mkdir", "-p", remote_dir], check=True)
    subprocess.run(["xrdcp", "-f", src, dest], check=True)

job_id = sys.argv[1]
with open("job_output_map.yaml") as handle:
    targets = yaml.safe_load(handle)[job_id]
xrdcp_with_mkdir(f"job_{job_id}.status", targets["status"])
PY
"""

        job_script = os.path.join(self.jobs_dir, "job.sh")
        with open(job_script, "w") as handle:
            handle.write(script)
        os.chmod(job_script, 0o755)

        transfer_inputs = [
            os.path.join(abs_jobdir_path, "job.sh"),
        ]
        if not self.run_options["ignore-grid-certificate"]:
            transfer_inputs.append(self.x509_path)

        sub = {
            "executable": "job.sh",
            "error": f"{logs_dir}/job_$(ClusterId).$(ProcId).err",
            "output": f"{logs_dir}/job_$(ClusterId).$(ProcId).out",
            "log": f"{logs_dir}/job_$(ClusterId).log",
            "should_transfer_files": "YES",
            "when_to_transfer_output": "ON_EXIT",
            "transfer_input_files": ",".join(transfer_inputs),
            "transfer_output_files": "\"\"",
            "MY.XRDCP_CREATE_DIR": "True",
            "MY.SingularityImage": f'"{self.run_options["worker-image"]}"',
            "RequestCpus": self.run_options["cores-per-worker"],
            "RequestMemory": self.run_options["mem-per-worker"],
            "RequestDisk": self.run_options.get("disk-per-worker", "2GB"),
            "arguments": f"$(ProcId) $(chunksize) {remote_payload} {remote_env_archive}",
            "max_retries": self.run_options.get("max-retries", self.run_options.get("retries", 3)),
        }
        queue = self.run_options.get("queue")
        if queue:
            sub["MY.JobFlavour"] = f'"{queue}"'
        with open(os.path.join(self.jobs_dir, "jobs_all.sub"), "w") as handle:
            for key, value in sub.items():
                handle.write(f"{key} = {value}\n")
            handle.write("queue chunksize from (\n")
            for chunksize in per_job_chunksize:
                handle.write(f"  {chunksize}\n")
            handle.write(")\n")

        for index, chunksize in enumerate(per_job_chunksize):
            with open(os.path.join(self.jobs_dir, f"job_{index}.sub"), "w") as handle:
                for key, value in sub.items():
                    if isinstance(value, str):
                        value = value.replace("$(ProcId)", str(index)).replace("$(chunksize)", str(chunksize))
                    handle.write(f"{key} = {value}\n")
                handle.write("queue\n")
            open(os.path.join(self.jobs_dir, f"job_{index}.idle"), "w").close()

        if _as_bool(self.run_options.get("dry-run", False)):
            print(f"Dry run, not submitting jobs. You can find all files: {abs_jobdir_path}")
            return
        subprocess.run(["condor_submit", "jobs_all.sub"], cwd=abs_jobdir_path, check=True)

    def recreate_jobs(self, jobs_to_recreate):
        raise NotImplementedError("condor@cmsconnect resubmission is handled by check-jobs for compact fileset YAML jobs.")


def get_executor_factory(executor_name, **kwargs):
    if executor_name == "iterative":
        return IterativeExecutorFactory(**kwargs)
    elif executor_name == "futures":
        return FuturesExecutorFactory(**kwargs)
    elif executor_name == "condor":
        return ExecutorFactoryCondorCMSConnect(**kwargs)
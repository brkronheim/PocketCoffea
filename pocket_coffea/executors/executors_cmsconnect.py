import ast
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from urllib.parse import unquote, urlparse

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
    parsed = urlparse(path)
    remote_path = "/" + parsed.path.lstrip("/")
    subprocess.run(["xrdfs", parsed.netloc, "mkdir", "-p", remote_path], check=True)


def _normalize_xrootd_destination(path, eos_prefix, option_name):
    normalized = _xrootd_path(path, eos_prefix)
    if not isinstance(normalized, str) or not normalized.startswith("root://"):
        raise ValueError(
            f"condor@cmsconnect {option_name} must be a root:// URL or an /eos/ path, "
            f"got {path!r}."
        )
    parsed = urlparse(normalized)
    if not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"condor@cmsconnect {option_name} is not a complete XRootD destination: {path!r}.")
    return normalized


def _validate_condor_argument(value, option_name):
    if re.search(r'[\s"]', str(value)):
        raise ValueError(
            f"condor@cmsconnect {option_name} cannot contain whitespace or double quotes: {value!r}."
        )


def _write_job_configs_archive(job_config_files, archive):
    archive = os.path.abspath(archive)
    temporary_archive = archive + ".tmp"
    try:
        with tarfile.open(temporary_archive, "w:gz") as tar:
            for job_config_file in job_config_files:
                tar.add(job_config_file, arcname=os.path.basename(job_config_file))
                with open(job_config_file) as handle:
                    descriptor = yaml.safe_load(handle)
                fileset_file = os.path.join(os.path.dirname(job_config_file), descriptor["fileset"])
                tar.add(fileset_file, arcname=os.path.basename(fileset_file))
        os.replace(temporary_archive, archive)
    finally:
        if os.path.exists(temporary_archive):
            os.remove(temporary_archive)
    return archive


CMSCONNECT_STATUS_STATES = ("running", "done", "failed")


def _set_job_configs_submit_argument(submit_file, remote_archive):
    with open(submit_file) as handle:
        lines = handle.readlines()
    updated = False
    for index, line in enumerate(lines):
        key, separator, value = line.partition("=")
        if separator and key.strip().lower() == "arguments":
            arguments = value.split()
            if len(arguments) < 4:
                raise ValueError(f"Malformed arguments line in {submit_file}")
            arguments[3] = remote_archive
            lines[index] = f"arguments = {' '.join(arguments)}\n"
            updated = True
            break
    if not updated:
        raise ValueError(f"CMS Connect submit file has no arguments line: {submit_file}")
    with open(submit_file, "w") as handle:
        handle.writelines(lines)


def restage_cmsconnect_job_configs(jobs_folder, job_name=None):
    jobs_folder = os.path.abspath(os.fspath(jobs_folder))
    with open(os.path.join(jobs_folder, "jobs_config.yaml")) as handle:
        jobs_payload = yaml.safe_load(handle) or {}
    manifest_job_dir = jobs_payload.get("job_dir")
    if manifest_job_dir:
        manifest_job_dir = os.path.abspath(os.fspath(manifest_job_dir))
    remote_archive = jobs_payload.get("job_configs_archive")
    if not remote_archive:
        return None
    jobs = jobs_payload.get("jobs_list") or {}
    if job_name is not None:
        if job_name not in jobs:
            raise ValueError(f"CMS Connect jobs manifest has no entry for {job_name}")
        selected_jobs = {job_name: jobs[job_name]}
        archive_name = f"cmsconnect_job_configs_{job_name}.tar.gz"
        submit_file = os.path.join(jobs_folder, f"{job_name}.sub")
        if not os.path.isfile(submit_file):
            raise ValueError(f"CMS Connect submit file does not exist: {submit_file}")
        remote_archive = _xrootd_join(_xrootd_parent(remote_archive), archive_name)
    else:
        selected_jobs = jobs
        archive_name = "cmsconnect_job_configs.tar.gz"
    use_redirector = _as_bool(
        jobs_payload.get("use-redirector", jobs_payload.get("use_redirector", False))
    )
    job_config_files = []
    for job in selected_jobs.values():
        config_file = job.get("config_file")
        if not config_file:
            raise ValueError("CMS Connect jobs manifest is missing a per-job config_file")
        if not os.path.isabs(config_file):
            local_config = os.path.join(jobs_folder, config_file)
            if os.path.isfile(local_config):
                config_file = local_config
            elif manifest_job_dir:
                config_file = os.path.join(manifest_job_dir, os.path.basename(config_file))
            else:
                config_file = local_config
        if use_redirector:
            with open(config_file) as handle:
                descriptor = yaml.safe_load(handle) or {}
            fileset_file = os.path.join(os.path.dirname(config_file), descriptor["fileset"])
            with open(fileset_file) as handle:
                fileset = yaml.safe_load(handle) or {}
            rewritten_fileset = rewrite_fileset_to_redirector(fileset)
            if rewritten_fileset != fileset:
                with open(fileset_file, "w") as handle:
                    yaml.safe_dump(rewritten_fileset, handle, sort_keys=False)
        elif "use-redirector" not in jobs_payload and "use_redirector" not in jobs_payload:
            embedded_fileset = job.get("filesets")
            if embedded_fileset:
                with open(config_file) as handle:
                    descriptor = yaml.safe_load(handle) or {}
                fileset_file = os.path.join(os.path.dirname(config_file), descriptor["fileset"])
                with open(fileset_file) as handle:
                    fileset = yaml.safe_load(handle) or {}
                if fileset != embedded_fileset:
                    with open(fileset_file, "w") as handle:
                        yaml.safe_dump(embedded_fileset, handle, sort_keys=False)
        job_config_files.append(config_file)
    archive = _write_job_configs_archive(
        job_config_files, os.path.join(jobs_folder, archive_name)
    )
    _mkdir_xrootd(_xrootd_parent(remote_archive))
    subprocess.run(["xrdcp", "-f", archive, remote_archive], check=True)
    if job_name is not None:
        _set_job_configs_submit_argument(submit_file, remote_archive)
    return archive


def clear_cmsconnect_job_statuses(jobs_folder, job_names):
    jobs_folder = os.path.abspath(os.fspath(jobs_folder))
    job_names = set(job_names)
    for job_name in job_names:
        for state in (*CMSCONNECT_STATUS_STATES, "status", "idle"):
            local_status = os.path.join(jobs_folder, f"{job_name}.{state}")
            if os.path.exists(local_status):
                os.remove(local_status)
    with open(os.path.join(jobs_folder, "jobs_config.yaml")) as handle:
        jobs_payload = yaml.safe_load(handle) or {}
    status_destination = jobs_payload.get("status_destination")
    if not status_destination or not status_destination.startswith("root://"):
        return
    parsed = urlparse(status_destination)
    status_path = "/" + parsed.path.lstrip("/")
    listing = subprocess.run(
        ["xrdfs", parsed.netloc, "ls", status_path],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    expected_markers = {
        f"{job_name}.{state}"
        for job_name in job_names
        for state in (*CMSCONNECT_STATUS_STATES, "status", "idle")
    }
    remote_paths = [
        "/" + entry.lstrip("/")
        for entry in listing.stdout.splitlines()
        if os.path.basename(entry) in expected_markers
    ]
    for remote_path in remote_paths:
        subprocess.run(
            ["xrdfs", parsed.netloc, "rm", remote_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )


def clear_cmsconnect_job_status(jobs_folder, job_name):
    clear_cmsconnect_job_statuses(jobs_folder, [job_name])


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
    site_entries = os.listdir(site_packages_dir)
    for di_dir in dist_info_dirs:
        direct_url_path = os.path.join(di_dir, "direct_url.json")
        if not os.path.exists(direct_url_path):
            continue
        try:
            with open(direct_url_path) as f:
                info = json.load(f)
        except (OSError, ValueError) as error:
            print(f"Warning: cannot read editable install metadata {direct_url_path}: {error}", file=sys.stderr)
            continue
        dir_info = info.get("dir_info") or {}
        if not dir_info.get("editable"):
            continue

        parsed_url = urlparse(info.get("url", ""))
        if parsed_url.scheme != "file":
            continue
        source_path = unquote(parsed_url.path)

        pkg_name = _import_name_from_dist_info(di_dir)
        if not pkg_name:
            # Fallback: derive from dist-info directory name
            pkg_name = os.path.basename(di_dir).rsplit("-", 1)[0]
        pkg_name_norm = pkg_name.replace("-", "_")

        # Collect all __editable__ files associated with this package
        for fname in site_entries:
            if not fname.startswith("__editable__"):
                continue
            if pkg_name_norm.lower() not in fname.lower():
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
                    if finder_fname in site_entries:
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
        if pkg_name_norm not in resolve_map:
            candidate = os.path.join(source_path, pkg_name_norm)
            if os.path.isdir(candidate):
                resolve_map[pkg_name_norm] = candidate
            else:
                resolve_map[pkg_name_norm] = source_path

    return exclude_basenames, resolve_map


def _find_python_site_packages(env_path):
    env_path = os.path.abspath(os.fspath(env_path))
    candidates = sorted(
        set(
            glob.glob(os.path.join(env_path, "lib", "python*", "site-packages"))
            + glob.glob(os.path.join(env_path, "lib64", "python*", "site-packages"))
            + glob.glob(os.path.join(env_path, "Lib", "site-packages"))
        )
    )
    candidates = [path for path in candidates if os.path.isdir(path)]
    if not candidates:
        return None, None

    preferred_python_dir = None
    pyvenv_path = os.path.join(env_path, "pyvenv.cfg")
    if os.path.isfile(pyvenv_path):
        with open(pyvenv_path) as handle:
            for line in handle:
                if line.lower().startswith("version") and "=" in line:
                    version = line.split("=", 1)[1].strip()
                    match = re.match(r"(\d+)\.(\d+)", version)
                    if match:
                        preferred_python_dir = f"python{match.group(1)}.{match.group(2)}"
                    break

    if preferred_python_dir:
        for candidate in candidates:
            if preferred_python_dir in candidate.split(os.sep):
                return candidate, preferred_python_dir
    selected = candidates[-1]
    parent_name = os.path.basename(os.path.dirname(selected))
    python_dir = parent_name if parent_name.startswith("python") else None
    return selected, python_dir


def _environment_cache_fingerprint(env_path, editable_resolve, excluded_packages=()):
    env_path = os.path.abspath(os.fspath(env_path))
    digest = hashlib.sha256()

    for package in sorted(
        str(package).lower().replace("-", "_") for package in excluded_packages
    ):
        digest.update(package.encode())

    metadata_paths = []
    pyvenv_path = os.path.join(env_path, "pyvenv.cfg")
    if os.path.isfile(pyvenv_path):
        metadata_paths.append(pyvenv_path)
    metadata_paths.extend(glob.glob(os.path.join(env_path, "conda-meta", "*.json")))
    for library_root in ("lib", "lib64", "Lib"):
        metadata_paths.extend(
            glob.glob(os.path.join(env_path, library_root, "python*", "site-packages", "*.dist-info", "METADATA"))
        )
        metadata_paths.extend(
            glob.glob(os.path.join(env_path, library_root, "python*", "site-packages", "*.dist-info", "direct_url.json"))
        )

    for path in sorted(set(metadata_paths)):
        digest.update(os.path.relpath(path, env_path).encode())
        with open(path, "rb") as handle:
            digest.update(handle.read())

    for import_name, source_path in sorted(editable_resolve.items()):
        digest.update(import_name.encode())
        source_path = os.path.abspath(source_path)
        if os.path.isfile(source_path):
            source_files = [source_path]
        else:
            source_files = []
            for root, dirs, files in os.walk(source_path):
                dirs[:] = sorted(
                    directory
                    for directory in dirs
                    if directory not in {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
                )
                source_files.extend(os.path.join(root, filename) for filename in sorted(files))
        for path in source_files:
            stat = os.stat(path)
            digest.update(os.path.relpath(path, source_path).encode())
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:12]


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


def _local_cmsconnect_submission_dir(outputdir):
    parsed = urlparse(outputdir)
    output_name = os.path.basename(parsed.path.rstrip("/")) or "output"
    output_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", output_name)
    destination_hash = hashlib.sha256(outputdir.encode()).hexdigest()[:12]
    return os.path.abspath(
        os.path.join("cmsconnect_jobs", f"{output_name}_{destination_hash}")
    )


def _packing_code_hash():
    """Hash of this executor file, used to invalidate the env cache when the
    packing logic (e.g. editable-install handling) changes."""
    try:
        with open(__file__, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except Exception:
        return "000000000000"


class ExecutorFactoryCondorCMSConnect(ExecutorFactoryManualABC):
    def __init__(self, run_options, outputdir, **kwargs):
        if outputdir.startswith("root://") and not run_options.get("jobs-dir"):
            run_options = dict(run_options)
            run_options["jobs-dir"] = _local_cmsconnect_submission_dir(outputdir)
        super().__init__(run_options=run_options, outputdir=outputdir, **kwargs)

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

        output_dir = self.outputdir if self.outputdir.startswith("root://") else os.path.abspath(self.outputdir)
        jobs_config = {
            "job_name": self.job_name,
            "job_dir": os.path.abspath(self.jobs_dir),
            "output_dir": output_dir,
            "config_pkl_total": config_path,
            "use-redirector": _as_bool(self.run_options.get("use-redirector", False)),
            "jobs_list": {},
        }

        job_config_files = []
        for index, split in enumerate(splits):
            fileset_file = os.path.join(self.jobs_dir, f"fileset_job_{index}.yaml")
            with open(fileset_file, "w") as handle:
                yaml.safe_dump(split, handle, sort_keys=False)
            job_config_file = os.path.join(self.jobs_dir, f"config_job_{index}.yaml")
            workflow_options = {}
            if _as_bool(self.run_options.get("convert-parquet-to-root", False)):
                workflow_options["dump_columns_as_arrays_per_chunk"] = self.run_options.get(
                    "parquet-output-dir", "columns"
                )
            elif _as_bool(self.run_options.get("disable-column-output", False)):
                workflow_options["dump_columns_as_arrays_per_chunk"] = None
            with open(job_config_file, "w") as handle:
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "configurator": "configurator.pkl",
                        "fileset": os.path.basename(fileset_file),
                        "workflow_options": workflow_options,
                    },
                    handle,
                    sort_keys=False,
                )
            job_config_files.append(job_config_file)
            output_file = _xrootd_join(output_dir, f"output_job_{index}.coffea")
            jobs_config["jobs_list"][f"job_{index}"] = {
                "filesets": split,
                "fileset_file": os.path.basename(fileset_file),
                "config_file": os.path.basename(job_config_file),
                "output_file": output_file,
            }
            if _as_bool(self.run_options.get("convert-parquet-to-root", False)):
                jobs_config["jobs_list"][f"job_{index}"]["root_output_file"] = output_file.replace(".coffea", "_skim.root")

        with open(os.path.join(self.jobs_dir, "jobs_config.yaml"), "w") as handle:
            yaml.safe_dump(jobs_config, handle, sort_keys=False)
        return job_config_files

    def _make_python_env_archive(self):
        configured = self.run_options.get("python-env-archive")
        if configured:
            configured = os.path.abspath(os.path.expanduser(configured))
            if not os.path.isfile(configured) or not tarfile.is_tarfile(configured):
                raise ValueError(f"python-env-archive is not a readable tar archive: {configured}")
            return configured
        env_path = os.path.abspath(self.run_options.get("python-env-path") or sys.prefix)
        if not os.path.isdir(env_path):
            raise ValueError(f"python-env-path is not a directory: {env_path}")

        site_packages_dir, py_ver = _find_python_site_packages(env_path)
        if site_packages_dir is None:
            raise ValueError(f"Could not find site-packages under python-env-path {env_path}")
        editable_exclude, editable_resolve = _get_editable_install_info(site_packages_dir)
        excluded_packages = DEFAULT_ENV_EXCLUDE_PACKAGES | set(
            _as_list(self.run_options.get("python-env-exclude-packages"))
        )
        env_fingerprint = _environment_cache_fingerprint(
            env_path, editable_resolve, excluded_packages
        )

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
                f"python_env_{os.path.basename(env_path)}_{py_ver}_{_cache_key_for_path(env_path)}_"
                f"{env_fingerprint}_{_packing_code_hash()}.tar.gz",
            )
        else:
            archive = os.path.abspath(os.path.join(self.jobs_dir, "python_env.tar.gz"))
        if os.path.exists(archive) and not force_repack:
            if tarfile.is_tarfile(archive):
                print(f"Reusing cached Python environment archive {archive}")
                return archive
            print(f"Removing incomplete Python environment archive {archive}", file=sys.stderr)
            os.remove(archive)
        print(f"Packing Python environment {env_path} -> {archive}")

        site_packages_relative = os.path.relpath(site_packages_dir, env_path).replace(os.sep, "/")
        sp_arcname = f"python_env/{site_packages_relative}"
        temporary_archive = f"{archive}.tmp.{os.getpid()}"

        try:
            with tarfile.open(temporary_archive, "w:gz") as tar:
                tar.add(
                    env_path,
                    arcname="python_env",
                    filter=_make_env_tar_filter(excluded_packages, editable_exclude),
                )
                for import_name, src_path in editable_resolve.items():
                    resolved_path = os.path.normpath(src_path)
                    if not os.path.isdir(resolved_path) and os.path.isfile(resolved_path):
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
            os.replace(temporary_archive, archive)
        finally:
            if os.path.exists(temporary_archive):
                os.remove(temporary_archive)
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
        archive_roots = [os.path.basename(os.path.normpath(path)) for path in selected]
        if len(archive_roots) != len(set(archive_roots)):
            raise ValueError(
                "analysis-transfer-paths contains entries with duplicate basenames; "
                "each transferred root must have a unique name."
            )
        print(f"Packing analysis payload -> {archive}")
        with tarfile.open(archive, "w:gz") as tar:
            for path in selected:
                tar.add(path, arcname=os.path.basename(os.path.normpath(path)), filter=_tar_filter)
        return archive

    def _upload_with_xrdcp(self, local_path, remote_path):
        print(f"Staging {local_path} -> {remote_path}")
        _mkdir_xrootd(_xrootd_parent(remote_path))
        subprocess.run(["xrdcp", "-f", local_path, remote_path], check=True)

    def _make_payload_archive(self, job_config_files, analysis_archive, inner_yaml_path, output_map):
        archive = os.path.abspath(os.path.join(self.jobs_dir, "cmsconnect_payload.tar.gz"))
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(os.path.join(self.jobs_dir, "configurator.pkl"), arcname="configurator.pkl")
            tar.add(inner_yaml_path, arcname=INNER_RUN_OPTIONS_FILENAME)
            tar.add(output_map, arcname="job_output_map.yaml")
            if analysis_archive:
                tar.add(analysis_archive, arcname="analysis_bundle.tar.gz")
            tar.add(os.path.join(self.jobs_dir, "jobs_config.yaml"), arcname="jobs_config.yaml")
        return archive

    def _make_job_configs_archive(self, job_config_files):
        archive = os.path.abspath(os.path.join(self.jobs_dir, "cmsconnect_job_configs.tar.gz"))
        return _write_job_configs_archive(job_config_files, archive)

    def _write_job_output_map(self, jobs_config):
        path = os.path.join(self.jobs_dir, "job_output_map.yaml")
        payload = {}
        eos_prefix = self.run_options.get("eos-prefix", "root://eosuser.cern.ch/")
        output_destination = self.run_options.get("output-destination")
        if output_destination is None and (
            self.outputdir.startswith("root://") or os.path.abspath(self.outputdir).startswith("/eos/")
        ):
            output_destination = self.outputdir
        if output_destination is None:
            raise ValueError(
                "condor@cmsconnect requires output-destination, or an -o root:// or /eos/ path."
            )
        output_destination = _normalize_xrootd_destination(
            output_destination, eos_prefix, "output-destination"
        )
        status_destination = self.run_options.get("status-destination") or _xrootd_join(
            output_destination, "status", self.job_name
        )
        status_destination = _normalize_xrootd_destination(
            status_destination, eos_prefix, "status-destination"
        )
        self._output_destination = output_destination
        self._status_destination = status_destination
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

    def submit_jobs(self, job_config_files):
        output_path = self.outputdir if self.outputdir.startswith("root://") else os.path.abspath(self.outputdir)
        abs_jobdir_path = os.path.abspath(self.jobs_dir)
        logs_dir = os.path.join(abs_jobdir_path, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        convert_parquet = _as_bool(self.run_options.get("convert-parquet-to-root", False))
        keep_coffea = _as_bool(self.run_options.get("keep-coffea-output", True))
        if not keep_coffea and not convert_parquet:
            raise ValueError("keep-coffea-output: false requires convert-parquet-to-root: true for condor@cmsconnect.")

        with open(os.path.join(self.jobs_dir, "jobs_config.yaml")) as handle:
            jobs_payload = yaml.safe_load(handle)
        jobs_config = jobs_payload["jobs_list"]
        output_map = self._write_job_output_map(jobs_config)
        with open(output_map) as handle:
            output_targets = yaml.safe_load(handle)
        for job_name, job in jobs_config.items():
            targets = output_targets[job_name.split("_", 1)[1]]
            if "coffea" in targets:
                job["output_file"] = targets["coffea"]
            else:
                job.pop("output_file", None)
            if "root" in targets:
                job["root_output_file"] = targets["root"]

        eos_prefix = self.run_options.get("eos-prefix", "root://eosuser.cern.ch/")
        staging_area = self.run_options.get("staging-area")
        if staging_area is None:
            staging_area = _xrootd_join(self._output_destination, "cmsconnect_staging", self.job_name)
        staging_area = _normalize_xrootd_destination(staging_area, eos_prefix, "staging-area")

        inner_yaml_path = write_inner_run_options(self.jobs_dir, self.run_options)
        analysis_archive = self._make_analysis_archive()
        env_archive = self._make_python_env_archive() if _as_bool(self.run_options.get("ship-python-env", True)) else None
        job_configs_archive = self._make_job_configs_archive(job_config_files)

        payload_archive = os.path.abspath(os.path.join(self.jobs_dir, "cmsconnect_payload.tar.gz"))
        remote_payload = _xrootd_join(staging_area, os.path.basename(payload_archive))
        remote_job_configs = _xrootd_join(staging_area, os.path.basename(job_configs_archive))
        remote_env_archive = _xrootd_join(staging_area, "python_env.tar.gz") if env_archive else ""
        jobs_payload["output_dir"] = output_path
        jobs_payload["staging_area"] = staging_area
        jobs_payload["status_destination"] = self._status_destination
        jobs_payload["payload_archive"] = remote_payload
        jobs_payload["job_configs_archive"] = remote_job_configs
        if remote_env_archive:
            jobs_payload["python_env_archive"] = remote_env_archive
        with open(os.path.join(self.jobs_dir, "jobs_config.yaml"), "w") as handle:
            yaml.safe_dump(jobs_payload, handle, sort_keys=False)

        payload_archive = self._make_payload_archive(job_config_files, analysis_archive, inner_yaml_path, output_map)
        if not _as_bool(self.run_options.get("dry-run", False)):
            _mkdir_xrootd(self._status_destination)
            clear_cmsconnect_job_statuses(self.jobs_dir, jobs_config)
            if env_archive:
                self._upload_with_xrdcp(env_archive, remote_env_archive)
            self._upload_with_xrdcp(payload_archive, remote_payload)
            self._upload_with_xrdcp(job_configs_archive, remote_job_configs)

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
set -Eeuo pipefail

JOB_ID="$1"
CHUNKSIZE="$2"
REMOTE_PAYLOAD="$3"
REMOTE_JOB_CONFIGS="$4"
REMOTE_PYTHON_ENV="$5"
STATUS_DIR="$6"
COLUMNS_DIR="{columns_dir}"
JOB_CONFIG="config_job_${{JOB_ID}}.yaml"

write_status() {{
    local state="$1"
    local marker="job_${{JOB_ID}}.${{state}}"
    printf '%s\n' "$state" > "$marker"
    xrdcp -f "$marker" "${{STATUS_DIR%/}}/$marker" >/dev/null 2>&1 || true
}}
on_error() {{
    local exit_code=$?
    trap - ERR TERM INT
    write_status failed
    exit "$exit_code"
}}
trap on_error ERR
trap 'trap - ERR TERM INT; write_status failed; exit 1' TERM INT

{x509_export}
{custom_setup}
write_status running
xrdcp -f "$REMOTE_PAYLOAD" cmsconnect_payload.tar.gz
tar xzf cmsconnect_payload.tar.gz
xrdcp -f "$REMOTE_JOB_CONFIGS" cmsconnect_job_configs.tar.gz
tar xzf cmsconnect_job_configs.tar.gz

if [ "$REMOTE_PYTHON_ENV" != "-" ]; then
    xrdcp -f "$REMOTE_PYTHON_ENV" python_env.tar.gz
fi

if [ -f python_env.tar.gz ]; then
    mkdir -p python_env
    tar xzf python_env.tar.gz -C python_env --strip-components=1
    export VIRTUAL_ENV="$PWD/python_env"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    unset PYTHONHOME
fi

if [ -f analysis_bundle.tar.gz ]; then
    mkdir -p analysis_src
    tar xzf analysis_bundle.tar.gz -C analysis_src
    export POCKET_COFFEA_ANALYSIS_ROOT="$PWD/analysis_src"
    export PYTHONPATH="$PWD/analysis_src:$PWD:${{PYTHONPATH:-}}"
fi

export XRD_RUNFORKHANDLER=1
export MALLOC_TRIM_THRESHOLD_=0

# Pick a Python interpreter explicitly. The YAML job descriptor references a
# shared configurator pickle, which must be loaded with the matching runtime.
if [ -x python_env/bin/python ]; then
    WORKER_PY=python_env/bin/python
else
    WORKER_PY=python
fi

"$WORKER_PY" - "$PWD/analysis_src" <<'PY'
import os
from pathlib import Path
import sys

import cloudpickle


analysis_root = Path(sys.argv[1]).resolve()
config_path = Path("configurator.pkl")
with config_path.open("rb") as handle:
    configurator = cloudpickle.load(handle)

relocated_paths = []

def relocate_resources(node):
    # Configurations are nested DictConfig/ListConfig objects. Only relocate
    # absolute paths with a matching staged resource; URLs and CVMFS stay valid.
    from collections.abc import MutableMapping, MutableSequence
    from omegaconf.errors import MissingMandatoryValue, InterpolationResolutionError
    if isinstance(node, MutableMapping):
        keys = list(node)
    elif isinstance(node, MutableSequence):
        keys = range(len(node))
    else:
        return
    for key in keys:
        try:
            value = node[key]
        except (MissingMandatoryValue, InterpolationResolutionError):
            # Unused eras may have mandatory placeholders or interpolations
            # unavailable on workers. Relocation is not config validation:
            # preserve them so an actual consumer still raises if needed.
            continue
        if not isinstance(value, str) or not os.path.isabs(value):
            relocate_resources(value)
            continue
        configured_path = Path(os.path.normpath(value))
        for marker in ("params", "MVA"):
            if marker not in configured_path.parts:
                continue
            marker_index = configured_path.parts.index(marker)
            bundled_path = analysis_root.joinpath(*configured_path.parts[marker_index:])
            if bundled_path.exists() and str(bundled_path) != value:
                node[key] = str(bundled_path)
                relocated_paths.append((configured_path, bundled_path))
            break

relocate_resources(configurator.parameters)

if relocated_paths:
    temporary_path = config_path.with_suffix(".pkl.tmp")
    with temporary_path.open("wb") as handle:
        cloudpickle.dump(configurator, handle)
    os.replace(temporary_path, config_path)
    for configured_path, bundled_path in relocated_paths:
        print("Relocated analysis resource: %s -> %s" % (configured_path, bundled_path))
PY
"""
        script += f"""
set +e
"$WORKER_PY" -m pocket_coffea.scripts.runner --cfg "$JOB_CONFIG" -o output {executor_args} --chunksize "$CHUNKSIZE" --custom-run-options {INNER_RUN_OPTIONS_FILENAME}
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
        remote_dir = "/" + parsed.path.rsplit("/", 1)[0].lstrip("/")
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
    trap - ERR TERM INT
    write_status done
else
    trap - ERR TERM INT
    write_status failed
    exit "$status"
fi
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

        remote_env_argument = remote_env_archive or "-"
        remote_status_argument = self._status_destination
        for option_name, value in (
            ("payload archive", remote_payload),
            ("job configs archive", remote_job_configs),
            ("Python environment archive", remote_env_argument),
            ("status destination", remote_status_argument),
            ("worker-image", self.run_options["worker-image"]),
        ):
            _validate_condor_argument(value, option_name)

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
            "arguments": (
                f"$(ProcId) $(chunksize) {remote_payload} {remote_job_configs} "
                f"{remote_env_argument} {remote_status_argument}"
            ),
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

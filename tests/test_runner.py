import os
import pstats
from pathlib import Path
import pytest
import json
import numpy as np
from types import SimpleNamespace

from pocket_coffea.scripts import runner


@pytest.fixture
def base_path() -> Path:
    """Get the current folder of the test"""
    return Path(__file__).parent


def test_runner_local(base_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory):
    """Test the runner script"""
    monkeypatch.chdir(base_path / "test_full_configs/test_subsamples/")
    outputdir = tmp_path_factory.mktemp("test_runner")
    status = os.system(f"pocket-coffea run --cfg config_subsamples.py -o {outputdir} --test -lf 1 -lc 1 --chunksize 100")
    assert status == 0
    assert (outputdir / "output_all.coffea").exists()


def test_failed_jobs_tracking(base_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory):
    """Test that failed jobs are tracked when using --process-separately"""
    monkeypatch.chdir(base_path / "test_full_configs/test_subsamples/")
    outputdir = tmp_path_factory.mktemp("test_failed_jobs")
    
    # Run with process-separately to enable failed job tracking
    # Using --test and limiting files to ensure quick execution
    status = os.system(f"pocket-coffea run --cfg config_subsamples.py -o {outputdir} --test -lf 1 -lc 1 --chunksize 100 --process-separately")
    
    # The test should succeed (status 0 means no Python errors)
    assert status == 0
    
    # Check if failed_jobs.json exists (it might not exist if all jobs succeeded)
    failed_jobs_file = outputdir / "failed_jobs.json"
    # If the file doesn't exist, all jobs succeeded, which is also valid
    if failed_jobs_file.exists():
        with open(failed_jobs_file, 'r') as f:
            failed_jobs = json.load(f)
        # failed_jobs should be a list
        assert isinstance(failed_jobs, list)


def test_resubmit_failed_without_process_separately(base_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory):
    """Test that --resubmit-failed requires --process-separately"""
    monkeypatch.chdir(base_path / "test_full_configs/test_subsamples/")
    outputdir = tmp_path_factory.mktemp("test_resubmit_error")
    
    # Try to use --resubmit-failed without --process-separately
    status = os.system(f"pocket-coffea run --cfg config_subsamples.py -o {outputdir} --test --resubmit-failed 2>/dev/null")
    
    # Should fail (non-zero exit code)
    assert status != 0


def test_resubmit_failed_without_file(base_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory):
    """Test that --resubmit-failed fails when failed_jobs.json doesn't exist"""
    monkeypatch.chdir(base_path / "test_full_configs/test_subsamples/")
    outputdir = tmp_path_factory.mktemp("test_resubmit_no_file")
    
    # Try to use --resubmit-failed without having a failed_jobs.json file
    status = os.system(f"pocket-coffea run --cfg config_subsamples.py -o {outputdir} --test --process-separately --resubmit-failed 2>/dev/null")
    
    # Should fail (non-zero exit code)
    assert status != 0


def test_branch_tracing_collects_all_passes(monkeypatch: pytest.MonkeyPatch):
    class TraceArray:
        def __len__(self):
            return 1

    class Processor:
        def __init__(self):
            self.cfg = SimpleNamespace(save_skimmed_files=True)
            self.workflow_options = {
                "dump_columns_as_arrays_per_chunk": "output",
                "keep": True,
            }
            self.main_path_reached = False

        def skim_events(self):
            self.events = []
            self.has_events = False

        def apply_preselections(self, variation):
            self.events = []
            self.has_events = False

        def process(self, events):
            self.events = events
            self.skim_events()
            if not self.has_events:
                return
            self.apply_preselections("nominal")
            if not self.has_events:
                return
            self.main_path_reached = True

    events = SimpleNamespace(metadata={}, attrs={"@form": {}})
    processor = Processor()
    reports = {}

    def fake_make_length_tracer(events, length):
        tracer = TraceArray()
        report = set()
        reports[id(tracer)] = (length, report)
        return tracer, report

    def fake_attempt_tracing(callable_, tracer, throw):
        callable_(tracer)
        length, report = reports[id(tracer)]
        report.add(f"length-{length}")

    monkeypatch.setattr(runner, "coffea_trace", lambda fun, events: {"coffea"})
    monkeypatch.setattr(
        runner,
        "_pocket_coffea_form_keys_to_columns",
        lambda touched: frozenset(touched),
    )
    monkeypatch.setattr(
        "coffea.nanoevents.trace._make_length_zero_one_tracer",
        fake_make_length_tracer,
    )
    monkeypatch.setattr(
        "coffea.nanoevents.trace._attempt_tracing",
        fake_attempt_tracing,
    )

    branches = runner.traced_branch_printer(processor.process, events)

    assert branches == {"coffea", "length-0", "length-1"}
    assert processor.main_path_reached
    assert processor.cfg.save_skimmed_files
    assert processor.workflow_options == {
        "dump_columns_as_arrays_per_chunk": "output",
        "keep": True,
    }
    assert processor.skim_events.__func__ is Processor.skim_events


def test_shared_buffer_cache_compresses_and_round_trips():
    from pocket_coffea.utils.run import (
        _buffer_cache_capacity,
        _buffer_cache_storage,
        _shared_buffer_cache,
    )

    codec = _shared_buffer_cache.codec._codec
    cache_key = "test_shared_buffer_cache"
    values = np.arange(100_000, dtype=np.float32)

    assert _buffer_cache_capacity == 500 * 1024**2
    assert _buffer_cache_storage.n == _buffer_cache_capacity
    assert codec.cname == "zstd"
    assert codec.clevel == 1

    try:
        _shared_buffer_cache[cache_key] = values
        restored = _shared_buffer_cache[cache_key]
        assert np.array_equal(restored, values)
        assert len(_buffer_cache_storage[cache_key]) < values.nbytes
    finally:
        if cache_key in _shared_buffer_cache:
            del _shared_buffer_cache[cache_key]


def test_profiled_processor_captures_nested_process_calls(tmp_path, monkeypatch):
    class DummyProcessor:
        def process(self, events):
            return self.phase()

        def phase(self):
            return 7

    monkeypatch.setenv("POCKET_COFFEA_PROFILE_DIR", str(tmp_path))
    profiled = runner._ProfiledProcessor(DummyProcessor())

    assert profiled.process(None) == 7

    profile_paths = list(tmp_path.glob("*.prof"))
    assert len(profile_paths) == 1
    stats = pstats.Stats(str(profile_paths[0]))
    function_names = {function_key[2] for function_key in stats.stats}
    assert {"process", "phase"}.issubset(function_names)


def test_phase_profile_adds_named_root_to_nested_calls(tmp_path):
    from pocket_coffea.utils.profiling import profile_phase_call

    def nested_phase():
        return 7

    def phase():
        return nested_phase()

    profile_phase_call("demo", phase, output_dir=tmp_path)

    stats = pstats.Stats(str(next(tmp_path.glob("*.prof"))))
    phase_root = next(
        function_key
        for function_key in stats.stats
        if function_key[2] == "phase:demo"
    )
    phase_function = next(
        function_key for function_key in stats.stats if function_key[2] == "phase"
    )

    assert stats.stats[phase_function][4][phase_root][3] > 0


def test_phase_profile_skips_nested_cprofile(tmp_path):
    import cProfile
    from pocket_coffea.utils.profiling import profile_phase_call

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        assert profile_phase_call("nested", lambda: 7, output_dir=tmp_path) == 7
    finally:
        profiler.disable()

    assert not list(tmp_path.glob("*.prof"))


def test_phase_profile_labels_merge_across_variations(tmp_path):
    from pocket_coffea.utils.profiling import dump_phase_profile

    dump_phase_profile("nominal", 2.0, [("histograms", 1.0)], tmp_path)
    dump_phase_profile("up", 3.0, [("histograms", 2.0)], tmp_path)

    profile_paths = sorted(tmp_path.glob("phase-*.prof"))
    stats = pstats.Stats(str(profile_paths[0]))
    stats.add(str(profile_paths[1]))

    matches = [
        value for key, value in stats.stats.items()
        if key[2] == "phase:histograms"
    ]
    assert len(matches) == 1
    assert matches[0][0] == 2

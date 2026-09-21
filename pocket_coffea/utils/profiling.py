import cProfile
import itertools
import os
import pstats
import threading
from pathlib import Path


_PROFILE_SEQUENCE = itertools.count()


def _safe_file_component(value):
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def _add_profile_root(stats, root_name):
    stats.calc_callees()
    roots = {
        function_key
        for function_key, function_values in stats.stats.items()
        if not function_values[4] and stats.all_callees.get(function_key)
    }
    root_key = ("<pocket_coffea_phase>", 0, root_name)
    child_time = sum(stats.stats[function_key][3] for function_key in roots)
    root_time = stats.total_tt
    stats.stats[root_key] = (
        1,
        1,
        max(root_time - child_time, 0.0),
        root_time,
        {},
    )
    for function_key in roots:
        primitive_calls, total_calls, self_time, cumulative_time, callers = stats.stats[
            function_key
        ]
        updated_callers = dict(callers)
        updated_callers[root_key] = (1, 1, 0.0, cumulative_time)
        stats.stats[function_key] = (
            primitive_calls,
            total_calls,
            self_time,
            cumulative_time,
            updated_callers,
        )


def _dump_profile(profiler, output_dir, file_prefix, root_name=None):
    profile_directory = Path(output_dir)
    profile_directory.mkdir(parents=True, exist_ok=True)
    profile_path = profile_directory / (
        f"{file_prefix}-{os.getpid()}-{threading.get_ident()}-"
        f"{next(_PROFILE_SEQUENCE)}.prof"
    )
    if root_name is None:
        profiler.dump_stats(str(profile_path))
    else:
        stats = pstats.Stats(profiler)
        _add_profile_root(stats, root_name)
        stats.dump_stats(str(profile_path))


def is_profiler_active():
    probe = cProfile.Profile()
    try:
        probe.enable()
    except ValueError:
        return True
    probe.disable()
    return False


def profile_call(function, *args, output_dir, file_prefix, root_name=None, **kwargs):
    profiler = cProfile.Profile()
    try:
        profiler.enable()
    except ValueError:
        return function(*args, **kwargs)

    try:
        return function(*args, **kwargs)
    finally:
        profiler.disable()
        _dump_profile(
            profiler,
            output_dir,
            _safe_file_component(file_prefix),
            root_name=root_name,
        )


def profile_phase_call(label, function, *args, output_dir, **kwargs):
    return profile_call(
        function,
        *args,
        output_dir=output_dir,
        file_prefix=f"phase-{label}",
        root_name=f"phase:{label}",
        **kwargs,
    )


def dump_phase_profile(variation, elapsed, phases, output_dir):
    stats = pstats.Stats()
    variation_key = ("<pocket_coffea_phase>", 0, f"variation[{variation}]")
    stats.stats = {}
    phase_time = sum(duration for _, duration in phases)
    stats.stats[variation_key] = (
        1,
        1,
        max(elapsed - phase_time, 0.0),
        elapsed,
        {},
    )
    for label, duration in phases:
        phase_key = ("<pocket_coffea_phase>", 1, f"phase:{label}")
        stats.stats[phase_key] = (
            1,
            1,
            duration,
            duration,
            {variation_key: (1, 1, 0.0, duration)},
        )

    stats.total_tt = elapsed
    profile_directory = Path(output_dir)
    profile_directory.mkdir(parents=True, exist_ok=True)
    profile_path = profile_directory / (
        f"phase-{os.getpid()}-{threading.get_ident()}-"
        f"{next(_PROFILE_SEQUENCE)}.prof"
    )
    stats.dump_stats(str(profile_path))
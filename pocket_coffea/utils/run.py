import os
import logging
from functools import partial
from coffea.processor import Runner
from coffea.nanoevents.mapping import BufferCache

from pocket_coffea.utils.logging import try_and_log_error

# Shared in-memory buffer cache across all chunks.
# Avoids re-reading and re-decompressing branches from the ROOT file on every access.
# Without this, the lazy virtual arrays in Coffea 2026+ re-materialize from disk
# every time a branch is touched after a copy/deepcopy (ak.copy -> copy.deepcopy).
_shared_buffer_cache = BufferCache(cache=None, codec=None)


def get_runner(executor, chunksize, maxchunks, skipbadfiles, schema, format, error_log_file, exit_on_error=True):
    """
    Create and return a Coffea Runner wrapped with error logging,
    given the specified configuration parameters.
    Parameters
    ----------
    executor : str
        The executor type for the Coffea Runner (e.g., 'futures', 'iterative').
    chunksize : int
        The number of events per chunk to process.
    maxchunks : int
        The maximum number of chunks to process.
    skipbadfiles : bool
        Whether to skip bad files during processing.
    schema : coffea.nanoevents.schemas.BaseSchema
        The schema to use for NanoEvents.
    format : str
        The file format (e.g., 'root').
    error_log_file : str
        Path to the error log file for logging exceptions.
    exit_on_error : bool, optional
        If True, exits the program on error after logging. Default is False.
    Returns
    -------
    Runner
        A Coffea Runner instance configured with the specified parameters.
    """

    # Use a shared in-memory buffer cache so decompressed branch data survives
    # across chunk boundaries and repeated materializations (ak.copy / deepcopy).
    # The callable returns the *same* cache every time, so caching works across chunks.
    cachestrategy = lambda: _shared_buffer_cache

    # Create and return the Runner wrapped with error logging
    return try_and_log_error(
        error_log_file, exit_on_error=exit_on_error
    )(
        Runner(
            executor=executor,
            chunksize=chunksize,
            maxchunks=maxchunks,
            skipbadfiles=skipbadfiles,
            schema=schema,
            format=format,
            cachestrategy=cachestrategy,
        )
    )

import os
import logging
from functools import partial
from coffea.processor import Runner
from coffea.nanoevents.mapping import BufferCache
from numcodecs import Blosc
from zict import LRU

from pocket_coffea.utils.logging import try_and_log_error

# Shared compressed in-memory buffer cache across all chunks. The LRU weight is
# based on compressed bytes, so the limit bounds the cache's actual RAM payload.

_buffer_cache_capacity = int(
    os.environ.get("POCKET_COFFEA_BUFFER_CACHE_BYTES", 500 * 1024**2)
)
_buffer_cache_storage = LRU(
    n=_buffer_cache_capacity,
    d={},
    weight=lambda key, value: len(value),
)
_shared_buffer_cache = BufferCache(
    cache=_buffer_cache_storage,
    codec=Blosc("zstd", clevel=1, shuffle=Blosc.BITSHUFFLE),
)



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

    # Use the shared compressed buffer cache so branch data survives repeated
    # materializations (ak.copy / deepcopy) without retaining raw buffers.
    # The callable returns the same cache every time, so entries are shared
    # across chunks in the same process.
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

"""
Utilitiy functions for PyDoug io package.
"""

import numpy as np


def _auto_worker_count(samplearray: np.ndarray, numfiles: int) -> int:
    """
    Pick a conservative worker count based on CPU and available memory.

    Parameters
    ----------
    samplearray: np.ndarray
        A sample array to be imported.
    numfiles: int
        The total number of files being imported.

    Returns
    -------
    numworkers: int
        The number of workers estimated.
    """

    import os

    import psutil

    cpu_count: int = os.cpu_count() or 1
    available_bytes: int = psutil.virtual_memory().available
    bytes_per_file: int = samplearray.nbytes
    memory_capped_workers: int = max(1, int((available_bytes * 0.25) // bytes_per_file))
    return max(1, min(cpu_count, memory_capped_workers, numfiles))

"""
Tests for PyDoug's segment.label module.
"""

import numpy as np

from pydoug_core.segment import label


def test_connectivity(sample_3dgray_array) -> None:

    result: np.ndarray = label.connectivity(sample_3dgray_array >= 128)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.integer)
    assert len(np.unique(result)) > 1


def test_watershed(sample_arrays) -> None:

    pass

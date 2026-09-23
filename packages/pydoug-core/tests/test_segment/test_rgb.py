"""
Tests for PyDoug's segment.rgb module.
"""

import numpy as np
from skimage.data import coffee

from pydoug_core.segment import rgb


def test_rgb_functions() -> None:

    result1: np.ndarray = rgb.fz(coffee())
    result2: np.ndarray = rgb.qs(coffee())
    result3: np.ndarray = rgb.sl(np.stack([coffee()] * 5, axis=0))
    assert np.issubdtype(result1.dtype, np.integer)
    assert np.issubdtype(result2.dtype, np.integer)
    assert np.issubdtype(result3.dtype, np.integer)
    assert len(np.unique(result1)) > 1
    assert len(np.unique(result2)) > 1
    assert len(np.unique(result3)) > 1

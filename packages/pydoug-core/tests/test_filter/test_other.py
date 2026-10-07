"""
Tests for PyDoug's filter.other module.
"""

import numpy as np

from pydoug_core.filter import other


def test_entropy(sample_arrays):

    result1: np.ndarray = other.entropy(sample_arrays["2dgray"])
    result2: np.ndarray = other.entropy(sample_arrays["3dgray"])
    result3: np.ndarray = other.entropy(sample_arrays["2drgb"])
    result4: np.ndarray = other.entropy(sample_arrays["3drgb"])
    result5: np.ndarray = other.entropy(sample_arrays["2dgray"], Perform_Slicewise=True)
    result6: np.ndarray = other.entropy(sample_arrays["3dgray"], Perform_Slicewise=True)
    result7: np.ndarray = other.entropy(sample_arrays["2drgb"], Perform_Slicewise=True)
    result8: np.ndarray = other.entropy(sample_arrays["3drgb"], Perform_Slicewise=True)

    assert result1.dtype == np.float64
    assert result2.dtype == np.float64
    assert result3.dtype == np.float64
    assert result4.dtype == np.float64
    assert result5.dtype == np.float64
    assert result6.dtype == np.float64
    assert result7.dtype == np.float64
    assert result8.dtype == np.float64
    assert np.all(result1 == result5)
    assert np.all(result3 == result7)

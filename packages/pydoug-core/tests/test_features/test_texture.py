"""
Tests for PyDoug's features.texture module.
"""

import numpy as np
from skimage.data import brick

from pydoug_core.features import texture


def test_local_binary_pattern(sample_2drgb_array):

    test_array: np.ndarray = brick()[0:50, 0:50]
    test_array3d: np.ndarray = np.stack([test_array] * 5, axis=0)
    result0: np.ndarray = texture.local_binary_pattern(sample_2drgb_array)
    result1: np.ndarray = texture.local_binary_pattern(test_array3d)
    result2: np.ndarray = texture.local_binary_pattern(test_array, "NRI Uniform")
    result3: np.ndarray = texture.local_binary_pattern(test_array, "ROR")
    result4: np.ndarray = texture.local_binary_pattern(test_array, "Uniform")
    result5: np.ndarray = texture.local_binary_pattern(test_array, "Variance")

    assert np.all(result0 == sample_2drgb_array)
    assert result1.dtype == np.float64
    assert result2.dtype == np.float64
    assert result3.dtype == np.float64
    assert result4.dtype == np.float64
    assert result5.dtype == np.float64

"""
Tests for PyDoug's color.cspace module.
"""

import numpy as np

from pydoug_core.color import cspace


def test_convert_cspace(sample_2drgb_array) -> None:

    result0: np.ndarray = cspace.convert_cspace(sample_2drgb_array)
    result1: np.ndarray = cspace.convert_cspace(sample_2drgb_array, "RGB", "HSV")
    result2: np.ndarray = cspace.convert_cspace(sample_2drgb_array, "RGB", "RGB CIE")
    result3: np.ndarray = cspace.convert_cspace(sample_2drgb_array, "RGB", "XYZ")
    result4: np.ndarray = cspace.convert_cspace(sample_2drgb_array, "RGB", "YUV")
    result5: np.ndarray = cspace.convert_cspace(sample_2drgb_array, "RGB", "YIQ")
    result6: np.ndarray = cspace.convert_cspace(sample_2drgb_array, "RGB", "YPbPr")
    result7: np.ndarray = cspace.convert_cspace(sample_2drgb_array, "RGB", "YCbCr")
    result8: np.ndarray = cspace.convert_cspace(sample_2drgb_array, "RGB", "YDbDr")

    assert np.all(result0 == sample_2drgb_array)
    assert result1.shape == sample_2drgb_array.shape
    assert result2.shape == sample_2drgb_array.shape
    assert result3.shape == sample_2drgb_array.shape
    assert result4.shape == sample_2drgb_array.shape
    assert result5.shape == sample_2drgb_array.shape
    assert result6.shape == sample_2drgb_array.shape
    assert result7.shape == sample_2drgb_array.shape
    assert result8.shape == sample_2drgb_array.shape
    assert result1.dtype == np.dtype(np.float64)
    assert result2.dtype == np.dtype(np.float64)
    assert result3.dtype == np.dtype(np.float64)
    assert result4.dtype == np.dtype(np.float64)
    assert result5.dtype == np.dtype(np.float64)
    assert result6.dtype == np.dtype(np.float64)
    assert result7.dtype == np.dtype(np.float64)
    assert result8.dtype == np.dtype(np.float64)

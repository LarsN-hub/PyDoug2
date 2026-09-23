"""
Tests for PyDoug's segment.local module.
"""

import numpy as np

from pydoug_core.segment import local


def test_local_threshold(sample_2dgray_array, sample_3dgray_array) -> None:

    result1: np.ndarray = local.local_threshold(sample_2dgray_array, "Adaptive")
    result2: np.ndarray = local.local_threshold(sample_2dgray_array, "Mean")
    result3: np.ndarray = local.local_threshold(sample_2dgray_array, "Niblack")
    result4: np.ndarray = local.local_threshold(sample_2dgray_array, "Otsu")
    result5: np.ndarray = local.local_threshold(sample_2dgray_array, "Sauvola")
    result6: np.ndarray = local.local_threshold(sample_3dgray_array, "Mean")
    result7: np.ndarray = local.local_threshold(sample_3dgray_array, "Otsu")
    assert result1.dtype == np.uint8
    assert result2.dtype == np.uint8
    assert result3.dtype == np.uint8
    assert result4.dtype == np.uint8
    assert result5.dtype == np.uint8
    assert result6.dtype == np.uint8
    assert result7.dtype == np.uint8
    assert len(np.unique(result1)) == 2
    assert len(np.unique(result2)) == 2
    assert len(np.unique(result3)) == 2
    assert len(np.unique(result4)) == 2
    assert len(np.unique(result5)) == 2
    assert len(np.unique(result6)) == 2
    assert len(np.unique(result7)) == 2
    assert np.max(result1) == 255
    assert np.max(result2) == 255
    assert np.max(result3) == 255
    assert np.max(result4) == 255
    assert np.max(result5) == 255
    assert np.max(result6) == 255
    assert np.max(result7) == 255

"""
Tests for PyDoug's segment.thresh module.
"""

import numpy as np

from pydoug_core.segment import thresh


def test_threshold_array(sample_3dgray_array) -> None:

    result1: np.ndarray = thresh.threshold_array(sample_3dgray_array, (0, 128))
    test1: np.ndarray = np.zeros(sample_3dgray_array.shape, sample_3dgray_array.dtype)
    test1[(sample_3dgray_array >= 0) & (sample_3dgray_array <= 128)] = 255
    assert np.all(result1 == test1)


def test_preview_threshold(sample_3dgray_array) -> None:

    result1: np.ndarray = thresh.interactive_threshold(sample_3dgray_array, (0, 128))
    test1: np.ndarray = np.zeros(sample_3dgray_array.shape, sample_3dgray_array.dtype)
    test1[(sample_3dgray_array >= 0) & (sample_3dgray_array <= 128)] = 255
    assert np.all(result1 == test1)


def test_manual_threshold(sample_3dgray_array) -> None:

    result1: np.ndarray = thresh.manual_threshold(sample_3dgray_array, (0, 128))
    test1: np.ndarray = np.zeros(sample_3dgray_array.shape, sample_3dgray_array.dtype)
    test1[(sample_3dgray_array >= 0) & (sample_3dgray_array <= 128)] = 255
    assert np.all(result1 == test1)

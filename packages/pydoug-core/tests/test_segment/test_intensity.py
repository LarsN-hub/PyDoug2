"""
Tests for PyDoug's segment.intensity module.
"""

import numpy as np

from pydoug_core.segment import intensity


def test_threshold_array(sample_3dgray_array) -> None:

    result1: np.ndarray = intensity.threshold_array(sample_3dgray_array, (0, 128))
    test1: np.ndarray = np.zeros(sample_3dgray_array.shape, sample_3dgray_array.dtype)
    test1[(sample_3dgray_array >= 0) & (sample_3dgray_array <= 128)] = 255
    assert np.all(result1 == test1)


def test_preview_threshold(sample_3dgray_array) -> None:

    result1: np.ndarray = intensity.interactive_threshold(sample_3dgray_array, (0, 128))
    test1: np.ndarray = np.zeros(sample_3dgray_array.shape, sample_3dgray_array.dtype)
    test1[(sample_3dgray_array >= 0) & (sample_3dgray_array <= 128)] = 255
    assert np.all(result1 == test1)


def test_manual_threshold(sample_3dgray_array) -> None:

    result1: np.ndarray = intensity.manual_threshold(sample_3dgray_array, (0, 128))
    test1: np.ndarray = np.zeros(sample_3dgray_array.shape, sample_3dgray_array.dtype)
    test1[(sample_3dgray_array >= 0) & (sample_3dgray_array <= 128)] = 255
    assert np.all(result1 == test1)


def test_reassign(sample_3dgray_array) -> None:

    result1: np.ndarray = intensity.reassign(sample_3dgray_array, 128, 0)
    result2: np.ndarray = intensity.reassign(sample_3dgray_array, 128, 0, True)
    assert np.all(np.logical_not(result1 == 128))
    assert len(np.unique(result2)) <= 2


def test_hist_threshold(sample_3dgray_array, sample_masks) -> None:

    mod_array1: np.ndarray = np.random.binomial(20, 0.5, (5, 10, 10)).astype(np.uint8)
    mod_array2: np.ndarray = 255 - mod_array1
    min_array: np.ndarray = np.stack((mod_array1, mod_array2), 0)
    result1: np.ndarray = intensity.hist_threshold(sample_3dgray_array)
    result2: np.ndarray = intensity.hist_threshold(sample_3dgray_array, "Isodata")
    result3: np.ndarray = intensity.hist_threshold(sample_3dgray_array, "Li")
    result4: np.ndarray = intensity.hist_threshold(sample_3dgray_array, "Mean")
    result5: np.ndarray = intensity.hist_threshold(min_array, "Minimum")
    result6: np.ndarray = intensity.hist_threshold(sample_3dgray_array, "Triangle")
    result7: np.ndarray = intensity.hist_threshold(sample_3dgray_array, "Yen")
    result8: np.ndarray = intensity.hist_threshold(
        sample_3dgray_array, "Otsu", Otsu_Classes=3
    )
    result9: np.ndarray = intensity.hist_threshold(
        sample_3dgray_array, Apply_Mask=True, Mask=sample_masks["3dgray"]
    )
    assert result1.dtype == np.uint8
    assert result2.dtype == np.uint8
    assert result3.dtype == np.uint8
    assert result4.dtype == np.uint8
    assert result5.dtype == np.uint8
    assert result6.dtype == np.uint8
    assert result7.dtype == np.uint8
    assert result8.dtype == np.uint8
    assert result9.dtype == np.uint8
    assert len(np.unique(result1)) == 2
    assert len(np.unique(result2)) == 2
    assert len(np.unique(result3)) == 2
    assert len(np.unique(result4)) == 2
    assert len(np.unique(result5)) == 2
    assert len(np.unique(result6)) == 2
    assert len(np.unique(result7)) == 2
    assert len(np.unique(result8)) == 3
    assert len(np.unique(result9)) == 2

    result10: np.ndarray = intensity.hist_threshold(
        min_array, Apply_Mask=True, Mask=np.zeros(min_array.shape, np.bool)
    )
    assert result10.dtype == np.uint8
    for method in ("Isodata", "Li", "Mean", "Otsu", "Triangle", "Yen"):
        result10 = intensity.hist_threshold(
            min_array,
            method,
            Otsu_Classes=3,
            Apply_Mask=True,
            Mask=np.zeros(min_array.shape, np.bool),
        )
        assert result10.dtype == min_array.dtype


def test_hysteresis(sample_2dgray_array) -> None:

    result = intensity.hysteresis(sample_2dgray_array, (20, 200))
    assert result.dtype == np.uint8
    assert len(np.unique(result)) == 2
    assert np.max(result) == 255

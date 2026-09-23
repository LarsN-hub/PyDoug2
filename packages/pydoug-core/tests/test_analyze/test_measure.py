"""
Tests for PyDoug's analyze.measure module.
"""

from typing import Any

import numpy as np

from pydoug_core.analyze import measure


def test_stats(sample_3dgray_array) -> None:

    result1: dict[str, float | int] = measure.stats(sample_3dgray_array)
    maskarray: np.ndarray = np.zeros(sample_3dgray_array.shape, np.bool)
    maskarray[3] = 1
    result2: dict[str, float | int] = measure.stats(
        sample_3dgray_array, True, maskarray
    )
    result3: dict[str, float | int] = measure.stats(
        np.astype(sample_3dgray_array, np.float64)
    )
    result4: dict[str, float | int] = measure.stats(maskarray)
    result5: dict[str, float | int] = measure.stats(maskarray, True, maskarray)

    assert isinstance(result1["Mean"], float)
    assert isinstance(result1["Median"], float)
    assert not isinstance(result1["Mode"], float) and not isinstance(
        result1["Mode"], int
    )
    assert result1["Mode"].dtype == np.dtype(np.uint8)
    assert isinstance(result1["Std Dev"], float)
    assert isinstance(result1["Variance"], float)
    assert not isinstance(result1["Min"], float) and not isinstance(result1["Min"], int)
    assert not isinstance(result1["Max"], float) and not isinstance(result1["Max"], int)
    assert not isinstance(result1["Range"], float) and not isinstance(
        result1["Range"], int
    )
    assert result1["Dtype Min"] == 0
    assert result1["Dtype Max"] == 255

    assert isinstance(result2["Mean"], float)
    assert isinstance(result2["Median"], float)
    assert not isinstance(result2["Mode"], float) and not isinstance(
        result2["Mode"], int
    )
    assert result2["Mode"].dtype == np.dtype(np.uint8)
    assert isinstance(result2["Std Dev"], float)
    assert isinstance(result2["Variance"], float)
    assert not isinstance(result2["Min"], float) and not isinstance(result2["Min"], int)
    assert not isinstance(result2["Max"], float) and not isinstance(result2["Max"], int)
    assert not isinstance(result2["Range"], float) and not isinstance(
        result2["Range"], int
    )
    assert result2["Dtype Min"] == 0
    assert result2["Dtype Max"] == 255

    assert isinstance(result3["Mean"], float)
    assert isinstance(result3["Median"], float)
    assert isinstance(result3["Mode"], float)
    assert isinstance(result3["Std Dev"], float)
    assert isinstance(result3["Variance"], float)
    assert isinstance(result3["Min"], float)
    assert isinstance(result3["Max"], float)
    assert isinstance(result3["Range"], float)
    assert result3["Dtype Min"] == np.finfo(np.float64).min
    assert result3["Dtype Max"] == np.finfo(np.float64).max

    assert isinstance(result4["Mean"], float)
    assert isinstance(result4["Median"], float)
    assert isinstance(result4["Mode"], int)
    assert isinstance(result4["Std Dev"], float)
    assert isinstance(result4["Variance"], float)
    assert isinstance(result4["Min"], np.bool)
    assert isinstance(result4["Max"], np.bool)
    assert isinstance(result4["Range"], int)
    assert result4["Dtype Min"] == 0
    assert result4["Dtype Max"] == 1

    assert isinstance(result5["Mean"], float)
    assert isinstance(result5["Median"], float)
    assert isinstance(result5["Mode"], int)
    assert isinstance(result5["Std Dev"], float)
    assert isinstance(result5["Variance"], float)
    assert isinstance(result5["Min"], np.bool)
    assert isinstance(result5["Max"], np.bool)
    assert isinstance(result5["Range"], int)
    assert result5["Dtype Min"] == 0
    assert result5["Dtype Max"] == 1


def test_percentile(sample_2dgray_array) -> None:

    result1: dict = measure.percentiles(sample_2dgray_array)
    result2: dict = measure.percentiles(sample_2dgray_array, (30, 70))
    result3: dict = measure.percentiles(sample_2dgray_array, (-5, 105))
    result4: dict = measure.percentiles(sample_2dgray_array, (0, 100))
    assert list(result1.values())[0] < list(result2.values())[0]
    assert list(result1.values())[1] > list(result2.values())[1]
    assert result4 == result3


def test_thresholds(sample_3dgray_array, sample_masks) -> None:

    mod_array1: np.ndarray = np.random.binomial(20, 0.5, (5, 10, 10)).astype(np.uint8)
    mod_array2: np.ndarray = 255 - mod_array1
    min_array: np.ndarray = np.stack((mod_array1, mod_array2), 0)
    result1: Any = measure.thresholds(sample_3dgray_array)
    result2: Any = measure.thresholds(sample_3dgray_array, "Isodata")
    result3: Any = measure.thresholds(sample_3dgray_array, "Li")
    result4: Any = measure.thresholds(sample_3dgray_array, "Mean")
    result5: Any = measure.thresholds(min_array, "Minimum")
    result6: Any = measure.thresholds(sample_3dgray_array, "Triangle")
    result7: Any = measure.thresholds(sample_3dgray_array, "Yen")
    result8: Any = measure.thresholds(sample_3dgray_array, "Otsu", Otsu_Classes=3)
    result9: Any = measure.thresholds(
        sample_3dgray_array, Apply_Mask=True, Mask=sample_masks["3dgray"]
    )
    assert not isinstance(result1, np.ndarray)
    assert not isinstance(result2, np.ndarray)
    assert not isinstance(result3, np.ndarray)
    assert not isinstance(result4, np.ndarray)
    assert not isinstance(result5, np.ndarray)
    assert not isinstance(result6, np.ndarray)
    assert not isinstance(result7, np.ndarray)
    assert isinstance(result8, np.ndarray)
    assert not isinstance(result9, np.ndarray)

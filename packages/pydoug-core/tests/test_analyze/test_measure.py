"""
Tests for PyDoug's analyze.measure module.
"""

import numpy as np

from pydoug_core.analyze import measure


def test_stats(sample_3dgray_array) -> None:

    result1: dict[str, float | int] = measure.stats(sample_3dgray_array)
    maskarray: np.ndarray = np.zeros(sample_3dgray_array.shape, np.bool)
    maskarray[3] = 1
    result2: dict[str, float | int] = measure.stats(sample_3dgray_array, maskarray)
    result3: dict[str, float | int] = measure.stats(
        np.astype(sample_3dgray_array, np.float64)
    )
    result4: dict[str, float | int] = measure.stats(maskarray)
    result5: dict[str, float | int] = measure.stats(maskarray, maskarray)

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

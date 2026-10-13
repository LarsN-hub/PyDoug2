"""
Tests for PyDoug's hist.hist module.
"""

import numpy as np

from pydoug_core.hist import hist
from pydoug_core.results import PlotResult


def test_plot_histogram(sample_3dgray_array, sample_masks) -> None:

    result1: PlotResult = hist.plot_histogram(
        sample_3dgray_array,
        Normalize=True,
        X_Lims=(0, 10),
    )
    result2: PlotResult = hist.plot_histogram(
        sample_3dgray_array, Apply_Mask=True, Mask=sample_masks["3dgray"]
    )
    result3: PlotResult = hist.plot_histogram(
        sample_3dgray_array,
        Normalize=True,
        Y_Max=10,
    )
    assert isinstance(result1, PlotResult)
    assert isinstance(result2, PlotResult)
    assert isinstance(result3, PlotResult)


def test_invert(sample_3dgray_array) -> None:

    result1: np.ndarray = hist.invert(sample_3dgray_array)
    assert np.all(result1 == 255 - sample_3dgray_array)


def test_normalize(sample_2dgray_array) -> None:

    test_array: np.ndarray = np.copy(sample_2dgray_array)
    test_array[test_array > 250] = 250
    test_array[test_array < 5] = 5
    result1: np.ndarray = hist.normalize(test_array)
    result2: np.ndarray = hist.normalize(test_array, Output_Range="Image")
    result3: np.ndarray = hist.normalize(test_array, Input_Range="Dtype")
    result4: np.ndarray = hist.normalize(
        test_array, Input_Range="Dtype", Output_Range="Image"
    )
    result5: np.ndarray = hist.normalize(
        test_array, "Manual", (np.min(test_array), np.max(test_array))
    )
    result6: np.ndarray = hist.normalize(
        test_array,
        "Manual",
        (np.min(test_array), np.max(test_array)),
        "Manual",
        (np.iinfo(test_array.dtype).min, np.iinfo(test_array.dtype).max),
    )
    result7: np.ndarray = hist.normalize(
        test_array,
        "Image",
        (0, 0),
        "Manual",
        (np.iinfo(test_array.dtype).min, np.iinfo(test_array.dtype).max),
    )
    assert np.max(result1) == 255
    assert np.min(result1) == 0
    assert np.all(result2 == test_array)
    assert np.all(result3 == test_array)
    assert np.max(result4) < np.max(test_array)
    assert np.min(result4) > np.min(test_array)
    assert np.all(result5 == result1)
    assert np.all(result5 == result6)
    assert np.all(result5 == result7)


def test_saturate(sample_3dgray_array) -> None:

    test_bool = sample_3dgray_array.astype(np.bool)
    result0: np.ndarray = hist.saturate(test_bool)
    assert np.all(test_bool == result0)
    result1: np.ndarray = hist.saturate(sample_3dgray_array, (50, 200))
    result2: np.ndarray = hist.saturate(sample_3dgray_array, (50, 200), False)
    assert result1.dtype == sample_3dgray_array.dtype
    assert result2.dtype == sample_3dgray_array.dtype
    assert not np.all(result1 == result2)
    result3: np.ndarray = hist.saturate(sample_3dgray_array, (0, 255), False)
    assert result3.dtype == sample_3dgray_array.dtype
    assert np.all(result3 == sample_3dgray_array)


def test_equalize(sample_3dgray_array, sample_2dgray_array, sample_masks) -> None:

    result1: np.ndarray = hist.equalize(sample_3dgray_array, "Global")
    result2: np.ndarray = hist.equalize(
        sample_3dgray_array, "Global", Apply_Mask=True, Mask=sample_masks["3dgray"]
    )
    result3: np.ndarray = hist.equalize(sample_3dgray_array)
    result4: np.ndarray = hist.equalize(sample_3dgray_array, "Rank")
    result5: np.ndarray = hist.equalize(
        sample_3dgray_array, "Rank", Apply_Mask=True, Mask=sample_masks["3dgray"]
    )
    result6: np.ndarray = hist.equalize(sample_2dgray_array, "Rank")
    result7: np.ndarray = hist.equalize(sample_3dgray_array, Perform_Slicewise=True)
    result8: np.ndarray = hist.equalize(
        sample_3dgray_array,
        Perform_Slicewise=True,
        Apply_Mask=True,
        Mask=sample_masks["3dgray"],
    )
    assert result1.dtype == sample_3dgray_array.dtype
    assert result2.dtype == sample_3dgray_array.dtype
    assert result3.dtype == sample_3dgray_array.dtype
    assert result4.dtype == sample_3dgray_array.dtype
    assert result5.dtype == sample_3dgray_array.dtype
    assert result6.dtype == sample_2dgray_array.dtype
    assert result7.dtype == sample_3dgray_array.dtype
    assert result8.dtype == sample_3dgray_array.dtype


def test_adjust(sample_2dgray_array) -> None:

    result1: np.ndarray = hist.adjust(sample_2dgray_array, "Gamma")
    result2: np.ndarray = hist.adjust(sample_2dgray_array)
    result3: np.ndarray = hist.adjust(sample_2dgray_array, "Sigmoid")
    result4: np.ndarray = hist.adjust(sample_2dgray_array, Inverse=True)
    result5: np.ndarray = hist.adjust(sample_2dgray_array, "Sigmoid", Inverse=True)
    assert result1.dtype == sample_2dgray_array.dtype
    assert result2.dtype == sample_2dgray_array.dtype
    assert result3.dtype == sample_2dgray_array.dtype
    assert result4.dtype == sample_2dgray_array.dtype
    assert result5.dtype == sample_2dgray_array.dtype


def test_local_contrast(sample_3dgray_array, sample_2dgray_array, sample_masks) -> None:

    test_mask: np.ndarray = np.copy(sample_masks["2dgray"])
    result1: np.ndarray = hist.local_contrast(sample_2dgray_array)
    result2: np.ndarray = hist.local_contrast(
        sample_2dgray_array, Apply_Mask=True, Mask=test_mask
    )
    result3: np.ndarray = hist.local_contrast(
        sample_2dgray_array, Method="Local Extrema"
    )
    result4: np.ndarray = hist.local_contrast(
        sample_2dgray_array, Method="Local Extrema", Apply_Mask=True, Mask=test_mask
    )
    result5: np.ndarray = hist.local_contrast(sample_3dgray_array)
    assert result1.dtype == sample_2dgray_array.dtype
    assert result2.dtype == sample_2dgray_array.dtype
    assert result3.dtype == sample_2dgray_array.dtype
    assert result4.dtype == sample_2dgray_array.dtype
    assert result5.dtype == sample_3dgray_array.dtype

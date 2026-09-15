"""
Tests for PyDoug's hist.hist module.
"""

import numpy as np
from matplotlib.figure import Figure

from pydoug_core.hist import hist


def test_plot_histogram(sample_3dgray_array, sample_masks) -> None:

    fig1: Figure = hist.plot_histogram(sample_3dgray_array, Normalize=True)
    fig2: Figure = hist.plot_histogram(
        sample_3dgray_array, Apply_Mask=True, Mask=sample_masks["3dgray"]
    )
    assert isinstance(fig1, Figure)
    assert isinstance(fig2, Figure)


def test_convert_type(sample_3dgray_array) -> None:

    result1: np.ndarray = hist.convert_dtype(sample_3dgray_array, "Uint8")
    result2: np.ndarray = hist.convert_dtype(sample_3dgray_array, "Uint16")
    result3: np.ndarray = hist.convert_dtype(sample_3dgray_array, "Int16")
    result4: np.ndarray = hist.convert_dtype(sample_3dgray_array, "Float32")
    result5: np.ndarray = hist.convert_dtype(sample_3dgray_array, "Float64")
    result6: np.ndarray = hist.convert_dtype(sample_3dgray_array, "Bool")
    uint16_array: np.ndarray = np.astype(sample_3dgray_array, np.uint16)
    uint16_array[0, 0, 0] = 65535
    result7: np.ndarray = hist.convert_dtype(uint16_array, "Uint8")
    assert result1.dtype == np.uint8
    assert result2.dtype == np.uint16
    assert result3.dtype == np.int16
    assert result4.dtype == np.float32
    assert result5.dtype == np.float64
    assert result6.dtype == np.bool
    assert result7.dtype == np.uint8


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

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


def test_normalize(sample_arrays) -> None:

    test_array: np.ndarray = np.copy(sample_arrays["2dgray"])
    test_array[test_array > 250] = 250
    test_array[test_array < 5] = 5
    result1: np.ndarray = hist.normalize(test_array)
    result2: np.ndarray = hist.normalize(test_array, Output_Range="Image")
    result3: np.ndarray = hist.normalize(test_array, Input_Range="Dtype")
    result4: np.ndarray = hist.normalize(
        test_array, Input_Range="Dtype", Output_Range="Image"
    )
    assert np.max(result1) == 255
    assert np.min(result1) == 0
    assert np.all(result2 == test_array)
    assert np.all(result3 == test_array)
    assert np.max(result4) < np.max(test_array)
    assert np.min(result4) > np.min(test_array)

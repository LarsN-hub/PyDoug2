"""
Tests for PyDoug's analyze.measure module.
"""

import numpy as np
import pytest
from skimage.measure import label

from pydoug_core.analyze import measure
from pydoug_core.results import PlotResult


def test_heat_map(sample_2dgray_array, sample_masks) -> None:

    result01: PlotResult = measure.heat_map(sample_2dgray_array, Units="um")
    result02: PlotResult = measure.heat_map(
        sample_2dgray_array, Apply_Mask=True, Mask=sample_masks["2dgray"]
    )
    assert isinstance(result01, PlotResult)
    assert isinstance(result02, PlotResult)


def test_label_count(sample_masks) -> None:

    lab_array: np.ndarray = label(sample_masks["3dgray"])  # type: ignore
    result = measure.label_count(lab_array)
    assert result == label(sample_masks["3dgray"], return_num=True)[-1]  # type: ignore


def test_fractal_dim(sample_masks) -> None:

    with pytest.raises(ValueError):
        _ = measure.fractal_dim(sample_masks["2drgb"])

    result00: PlotResult = measure.fractal_dim(np.zeros((10, 10), np.bool))
    result01: PlotResult = measure.fractal_dim(
        np.random.randint(0, 2, (100, 100), np.bool)
    )
    result02: PlotResult = measure.fractal_dim(
        np.random.randint(0, 2, (100, 100, 100), np.bool)
    )
    assert isinstance(result00, PlotResult)
    assert isinstance(result01, PlotResult)
    assert isinstance(result02, PlotResult)

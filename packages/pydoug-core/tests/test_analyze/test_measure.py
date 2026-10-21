"""
Tests for PyDoug's analyze.measure module.
"""

import numpy as np
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

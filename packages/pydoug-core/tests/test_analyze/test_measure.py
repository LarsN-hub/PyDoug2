"""
Tests for PyDoug's analyze.measure module.
"""

import numpy as np
import pytest
from porespy.generators import overlapping_spheres
from skimage.measure import label

from pydoug_core.analyze import measure
from pydoug_core.results import PlotResult, ValueResult


def test_label_count(sample_masks) -> None:

    lab_array: np.ndarray = label(sample_masks["3dgray"])  # type: ignore
    result = measure.label_count(lab_array)
    assert result == label(sample_masks["3dgray"], return_num=True)[-1]  # type: ignore


def test_tortuosity() -> None:

    array2d: np.ndarray = overlapping_spheres([100, 100], porosity=0.65)
    array3d: np.ndarray = overlapping_spheres([100, 100, 100], porosity=0.65)

    with pytest.raises(ValueError):
        _ = measure.tortuosity(array2d)
    with pytest.raises(ValueError):
        _ = measure.tortuosity(array3d, Perform_Slicewise=True)

    result01: PlotResult | ValueResult = measure.tortuosity(array2d, "X")
    result02: PlotResult | ValueResult = measure.tortuosity(array2d, "Y")
    result03: PlotResult | ValueResult = measure.tortuosity(array3d)
    result04: PlotResult | ValueResult = measure.tortuosity(array3d, "X")
    result05: PlotResult | ValueResult = measure.tortuosity(array3d, "Y")
    result06: PlotResult | ValueResult = measure.tortuosity(
        array3d, "X", Perform_Slicewise=True, Units="um"
    )
    result07: PlotResult | ValueResult = measure.tortuosity(
        array3d, "Y", Perform_Slicewise=True
    )

    assert isinstance(result01, ValueResult)
    assert isinstance(result02, ValueResult)
    assert isinstance(result03, ValueResult)
    assert isinstance(result04, ValueResult)
    assert isinstance(result05, ValueResult)
    assert isinstance(result06, PlotResult)
    assert isinstance(result07, PlotResult)

    data01 = result01.values
    data02 = result02.values
    data03 = result03.values
    data04 = result04.values
    data05 = result05.values
    data06 = result06.data
    data07 = result07.data

    assert isinstance(data01, dict)
    assert isinstance(data02, dict)
    assert isinstance(data03, dict)
    assert isinstance(data04, dict)
    assert isinstance(data05, dict)
    assert isinstance(data06, dict)
    assert isinstance(data07, dict)

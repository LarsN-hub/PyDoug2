"""
Tests for PyDoug's _operations.specify_colors module in its napari plugin.
"""

import numpy as np
import pytest

from pydoug_core.results import VisualResult
from pydoug_napari._operations import colors


def test_get_cmap(sample_3dgray_array) -> None:

    with pytest.raises(ValueError):
        _ = colors.get_cmap(sample_3dgray_array.astype(np.float64))
    with pytest.raises(ValueError):
        _ = colors.get_cmap(sample_3dgray_array, "inferno", (1, 0))
    result01: dict = colors.get_cmap(sample_3dgray_array)
    result02: dict = colors.get_cmap(sample_3dgray_array, "viridis", (100, 150))
    result03: dict = colors.get_cmap(sample_3dgray_array, "cmr.amethyst", (-50, 300))
    for val in np.unique(sample_3dgray_array):
        assert val in result01.keys()
        assert val in result02.keys()
        assert val in result03.keys()


def test_value_gradient(sample_3dgray_array) -> None:

    result01: VisualResult = colors.value_gradient(sample_3dgray_array)
    assert np.all(result01.imarray == sample_3dgray_array)
    assert result01.layer_type == "labels"
    assert len(result01.vis_dict) == 5


def test_axial_gradient(sample_masks) -> None:

    with pytest.raises(ValueError):
        _ = colors.axial_gradient(sample_masks["2dgray"], "Z")
    result01: VisualResult = colors.axial_gradient(sample_masks["2dgray"], "X")
    result02: VisualResult = colors.axial_gradient(sample_masks["2dgray"], "Y")
    result03: VisualResult = colors.axial_gradient(sample_masks["3dgray"], "X")
    result04: VisualResult = colors.axial_gradient(sample_masks["3dgray"], "Y")
    result05: VisualResult = colors.axial_gradient(sample_masks["3dgray"], "Z")
    assert np.all((result01.imarray != 0) == (sample_masks["2dgray"] != 0))
    assert np.all((result02.imarray != 0) == (sample_masks["2dgray"] != 0))
    assert np.all((result03.imarray != 0) == (sample_masks["3dgray"] != 0))
    assert np.all((result04.imarray != 0) == (sample_masks["3dgray"] != 0))
    assert np.all((result05.imarray != 0) == (sample_masks["3dgray"] != 0))

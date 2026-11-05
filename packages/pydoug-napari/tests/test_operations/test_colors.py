"""
Tests for PyDoug's _operations.specify_colors module in its napari plugin.
"""

import numpy as np
import pytest
from matplotlib import colormaps

from pydoug_core.results import VisualResult
from pydoug_napari._operations import colors


def test_value_gradient(sample_3dgray_array) -> None:

    result01: VisualResult = colors.value_gradient(
        sample_3dgray_array, "inferno", (0, 0)
    )
    result02: VisualResult = colors.value_gradient(
        sample_3dgray_array, "inferno", (20, 100)
    )

    assert isinstance(result01.imarray, np.ndarray)
    assert isinstance(result02.imarray, np.ndarray)
    assert result01.layer_type, "labels"
    assert result02.layer_type, "labels"
    assert isinstance(result01.vis_dict, dict)
    assert isinstance(result02.vis_dict, dict)

    assert np.all(
        result01.vis_dict["colormap"][0]
        == colormaps["inferno"](np.linspace(0, 1, 256))[0]
    )
    assert np.all(
        result02.vis_dict["colormap"][20]
        == colormaps["inferno"](np.linspace(0, 1, 10))[0]
    )
    assert np.all(
        result02.vis_dict["colormap"][100]
        == colormaps["inferno"](np.linspace(0, 1, 10))[-1]
    )


def test_axial_gradient(sample_masks) -> None:

    with pytest.raises(ValueError):
        _ = colors.axial_gradient(sample_masks["2dgray"], "Z")
    result01: VisualResult = colors.axial_gradient(sample_masks["2dgray"])
    result02: VisualResult = colors.axial_gradient(sample_masks["3dgray"])

    assert result01.imarray.shape == sample_masks["2dgray"].shape
    assert result02.imarray.shape == sample_masks["3dgray"].shape
    assert result01.imarray.dtype != np.dtype(np.bool)
    assert result02.imarray.dtype != np.dtype(np.bool)

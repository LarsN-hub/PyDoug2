"""
Tests for PyDoug's _operations.specify_colors module in its napari plugin.
"""

import numpy as np

from pydoug_core.results import VisualResult
from pydoug_napari._operations import specify_colors


def test_value_gradient(sample_3dgray_array) -> None:

    result01: VisualResult = specify_colors.value_gradient(sample_3dgray_array)
    result02: VisualResult = specify_colors.value_gradient(
        sample_3dgray_array, "inferno", (20, 100)
    )

    assert isinstance(result01.imarray, np.ndarray)
    assert isinstance(result02.imarray, np.ndarray)
    assert result01.layer_type, "image"
    assert result02.layer_type, "image"
    assert isinstance(result01.vis_dict, dict)
    assert isinstance(result02.vis_dict, dict)
    assert result01.vis_dict["colormap"]["name"] == "inferno_custom"
    assert result02.vis_dict["colormap"]["name"] == "inferno_custom"
    assert result01.vis_dict["colormap"]["interpolation"] == "linear"
    assert result02.vis_dict["colormap"]["interpolation"] == "linear"
    assert isinstance(result01.vis_dict["colormap"]["colors"], np.ndarray)
    assert isinstance(result02.vis_dict["colormap"]["colors"], np.ndarray)

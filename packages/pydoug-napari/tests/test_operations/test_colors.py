"""
Tests for PyDoug's _operations.specify_colors module in its napari plugin.
"""

import numpy as np
import pytest
from matplotlib import colormaps

from pydoug_core.results import VisualResult
from pydoug_napari._operations import colors


def test_value_gradient_image(sample_3dgray_array, sample_masks) -> None:

    with pytest.raises(ValueError):
        _ = colors.value_gradient(sample_3dgray_array.astype(np.float64))

    result00: VisualResult = colors.value_gradient(sample_masks["2dgray"])
    result01: VisualResult = colors.value_gradient(sample_3dgray_array)
    result02: VisualResult = colors.value_gradient(
        sample_3dgray_array, "inferno", (20, 100)
    )
    result03: VisualResult = colors.value_gradient(sample_3dgray_array.astype(np.int8))
    result04: VisualResult = colors.value_gradient(
        sample_3dgray_array.astype(np.int8), "inferno", (-20, 50)
    )

    assert isinstance(result00.imarray, np.ndarray)
    assert isinstance(result01.imarray, np.ndarray)
    assert isinstance(result02.imarray, np.ndarray)
    assert isinstance(result03.imarray, np.ndarray)
    assert isinstance(result04.imarray, np.ndarray)
    assert result00.layer_type, "image"
    assert result01.layer_type, "image"
    assert result02.layer_type, "image"
    assert result03.layer_type, "image"
    assert result04.layer_type, "image"
    assert isinstance(result00.vis_dict, dict)
    assert isinstance(result01.vis_dict, dict)
    assert isinstance(result02.vis_dict, dict)
    assert isinstance(result03.vis_dict, dict)
    assert isinstance(result04.vis_dict, dict)
    assert result00.vis_dict["colormap"]["name"] == "inferno_custom"
    assert result01.vis_dict["colormap"]["name"] == "inferno_custom"
    assert result02.vis_dict["colormap"]["name"] == "inferno_custom"
    assert result03.vis_dict["colormap"]["name"] == "inferno_custom"
    assert result04.vis_dict["colormap"]["name"] == "inferno_custom"
    assert isinstance(result00.vis_dict["colormap"]["colors"], np.ndarray)
    assert isinstance(result01.vis_dict["colormap"]["colors"], np.ndarray)
    assert isinstance(result02.vis_dict["colormap"]["colors"], np.ndarray)
    assert isinstance(result03.vis_dict["colormap"]["colors"], np.ndarray)
    assert isinstance(result04.vis_dict["colormap"]["colors"], np.ndarray)

    assert result00.vis_dict["colormap"]["colors"].shape == (2, 4)
    assert np.all(
        result01.vis_dict["colormap"]["colors"]
        == colormaps["inferno"](np.linspace(0, 1, 256))
    )
    assert np.all(
        result02.vis_dict["colormap"]["colors"][0:20]
        == np.repeat(colormaps["inferno"](np.linspace(0, 1, 10))[0:1, :], 20, 0)
    )
    assert np.all(
        result02.vis_dict["colormap"]["colors"][100:]
        == np.repeat(colormaps["inferno"](np.linspace(0, 1, 10))[9:, :], 156, 0)
    )
    assert np.all(
        result03.vis_dict["colormap"]["colors"]
        == colormaps["inferno"](np.linspace(0, 1, 256))
    )
    assert np.all(
        result04.vis_dict["colormap"]["colors"][0 : (-20 + 128)]
        == np.repeat(colormaps["inferno"](np.linspace(0, 1, 10))[0:1, :], 108, 0)
    )
    assert np.all(
        result04.vis_dict["colormap"]["colors"][(50 + 128) :]
        == np.repeat(colormaps["inferno"](np.linspace(0, 1, 10))[9:, :], 78, 0)
    )


def test_value_gradient_labels(sample_3dgray_array) -> None:

    result01: VisualResult = colors.value_gradient(
        sample_3dgray_array, "inferno", (0, 0), "Labels"
    )
    result02: VisualResult = colors.value_gradient(
        sample_3dgray_array, "inferno", (20, 100), "Labels"
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
    result02: VisualResult = colors.axial_gradient(sample_masks["2dgray"], "X")
    result03: VisualResult = colors.axial_gradient(sample_masks["3dgray"])
    result04: VisualResult = colors.axial_gradient(sample_masks["3dgray"], "X")
    result05: VisualResult = colors.axial_gradient(sample_masks["3dgray"], "Z")
    result06: VisualResult = colors.axial_gradient(
        sample_masks["2dgray"], Return_Type="Labels"
    )
    result07: VisualResult = colors.axial_gradient(
        sample_masks["3dgray"], Return_Type="Labels"
    )
    result08: VisualResult = colors.axial_gradient(
        sample_masks["2dgray"], C_Lims=(2, 8)
    )
    result09: VisualResult = colors.axial_gradient(
        sample_masks["3dgray"], C_Lims=(2, 8)
    )

    assert result01.imarray.shape == sample_masks["2dgray"].shape
    assert result02.imarray.shape == sample_masks["2dgray"].shape
    assert result03.imarray.shape == sample_masks["3dgray"].shape
    assert result04.imarray.shape == sample_masks["3dgray"].shape
    assert result05.imarray.shape == sample_masks["3dgray"].shape
    assert result06.imarray.shape == sample_masks["2dgray"].shape
    assert result07.imarray.shape == sample_masks["3dgray"].shape
    assert result08.imarray.shape == sample_masks["2dgray"].shape
    assert result09.imarray.shape == sample_masks["3dgray"].shape
    assert result01.imarray.dtype != np.dtype(np.bool)
    assert result02.imarray.dtype != np.dtype(np.bool)
    assert result03.imarray.dtype != np.dtype(np.bool)
    assert result04.imarray.dtype != np.dtype(np.bool)
    assert result05.imarray.dtype != np.dtype(np.bool)
    assert result06.imarray.dtype != np.dtype(np.bool)
    assert result07.imarray.dtype != np.dtype(np.bool)
    assert result08.imarray.dtype != np.dtype(np.bool)
    assert result09.imarray.dtype != np.dtype(np.bool)

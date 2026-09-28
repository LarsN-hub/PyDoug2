"""
Tests for PyDoug's napari plugin operations.screenshot module.
"""

import os
from typing import Any

import numpy as np
import pytest
from napari.components.camera import (
    DepthAxisOrientation,
    HorizontalAxisOrientation,
    VerticalAxisOrientation,
)

from pydoug_napari._operations import screenshot

requires_gl = pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="needs a real OpenGL context; unavailable on the CI runner",
)


@requires_gl
def test_render_screenshots(sample_2dgray_array) -> None:

    base_settings: dict[str, Any] = {
        "colormap": None,
        "camera_angles": (0, 0, 0),
        "camera_zoom": 1,
        "camera_center": (0, 0, 0),
        "camera_perspective": 0,
        "camera_orientation": (
            DepthAxisOrientation.TOWARDS,
            VerticalAxisOrientation.DOWN,
            HorizontalAxisOrientation.RIGHT,
        ),
        "depiction": "volume",
        "dims_ndisplay": 2,
        "opacity": 1,
        "plane": None,
    }
    im_settings: dict[str, Any] = base_settings.copy()
    im_settings.update(
        {
            "blending": None,
            "layer_type": "image",
            "attenuation": 0.05,
            "contrast_limits": None,
            "gamma": 1,
            "interpolation2d": "nearest",
            "interpolation3d": "linear",
            "iso_threshold": None,
            "projection_mode": "mean",
            "rendering": "mip",
        }
    )
    lab_settings: dict[str, Any] = base_settings.copy()
    lab_settings.update(
        {
            "blending": "translucent",
            "layer_type": "labels",
            "iso_gradient_mode": "fast",
            "projection_mode": "none",
            "rendering": "iso_categorical",
        }
    )
    result1: np.ndarray = screenshot.render_image_screenshot(
        sample_2dgray_array, im_settings
    )
    result2: np.ndarray = screenshot.render_labels_screenshot(
        sample_2dgray_array.astype(np.int32), lab_settings
    )
    assert isinstance(result1, np.ndarray)
    assert isinstance(result2, np.ndarray)

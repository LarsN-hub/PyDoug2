"""
Take a screenshot and record its parameters in the workflow.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from pydoug_core.registry import register_operation


def _render_screenshot_impl(
    imarray: np.ndarray, Settings: dict[str, Any]
) -> np.ndarray:
    """
    Render `imarray` with the captured display settings in a fully
    isolated subprocess, and return the resulting screenshot array.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to capture a screenshot of in the napari viewer.
    Settings: dict
        The viewer settings.

    Returns
    -------
    screenshot_array: np.ndarray
        The screenshot as a 2D image.
    """

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp: Path = Path(tmp_str)
        array_path: Path = tmp / "input.npy"
        output_path: Path = tmp / "output.npy"
        config_path: Path = tmp / "config.json"
        np.save(array_path, imarray)
        config_path.write_text(
            json.dumps(
                {
                    "array_path": str(array_path),
                    "output_path": str(output_path),
                    "settings": Settings,
                }
            )
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pydoug_napari._operations._screenshot_subprocess",
                str(config_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(  # pragma: no cover
                f"Screenshot rendering failed:\n{result.stderr}"  # pragma: no cover
            )  # pragma: no cover
        if not output_path.exists():
            raise RuntimeError(  # pragma: no cover
                "Screenshot subprocess exited successfully but "  # pragma: no cover
                f"produced no output file.\nstderr: {result.stderr}"  # pragma: no cover
            )  # pragma: no cover
        return np.load(output_path)


@register_operation(
    category="visualization",
    display_name="Render Image Screenshot",
    captures_viewer_settings=True,
)
def render_image_screenshot(
    imarray: np.ndarray, Settings: dict[str, Any]
) -> np.ndarray:
    """
    Render `imarray` with the captured display settings in a fully
    isolated subprocess, and return the resulting screenshot array.
    Only accepts image layers.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to capture a screenshot of in the napari viewer.
    Settings: dict
        The viewer settings.

    Returns
    -------
    screenshot_array: np.ndarray
        The screenshot as a 2D image.
    """

    return _render_screenshot_impl(imarray, Settings)


@register_operation(
    category="visualization",
    display_name="Render Labels Screenshot",
    captures_viewer_settings=True,
    layer_type="labels",
)
def render_labels_screenshot(
    imarray: np.ndarray, Settings: dict[str, Any]
) -> np.ndarray:
    """
    Render `imarray` with the captured display settings in a fully
    isolated subprocess, and return the resulting screenshot array.
    Only accepts labels layers.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to capture a screenshot of in the napari viewer.
    Settings: dict
        The viewer settings.

    Returns
    -------
    screenshot_array: np.ndarray
        The screenshot as a 2D image.
    """

    return _render_screenshot_impl(imarray, Settings)

"""
Standalone entry point to render a screenshot off the main Qt thread.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from napari import Viewer
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication


def main(config_path: str) -> None:  # pragma: no cover
    """
    The main function. Takes a configuration path and creates a napari viewer with the
    desired layer added.

    Parameters
    ----------
    config_path: str
        The path to the configuration json file.
    """

    config: Any = json.loads(Path(config_path).read_text())
    imarray: np.ndarray = np.load(config["array_path"])
    settings: dict = config["settings"]

    app = QApplication.instance() or QApplication([])
    viewer: Viewer = Viewer(show=False)

    try:
        if settings["layer_type"] == "labels":
            viewer.add_labels(  # type: ignore
                imarray,
                blending=settings["blending"],
                colormap=settings["colormap"],
                depiction=settings["depiction"],
                iso_gradient_mode=settings["iso_gradient_mode"],
                opacity=settings["opacity"],
                plane=settings["plane"],
                projection_mode=settings["projection_mode"],
                rendering=settings["rendering"],
            )
        else:
            viewer.add_image(
                imarray,
                attenuation=settings["attenuation"],
                blending=settings["blending"],
                colormap=settings["colormap"],
                contrast_limits=settings["contrast_limits"],
                depiction=settings["depiction"],
                gamma=settings["gamma"],
                interpolation2d=settings["interpolation2d"],
                interpolation3d=settings["interpolation3d"],
                iso_threshold=settings["iso_threshold"],
                opacity=settings["opacity"],
                plane=settings["plane"],
                projection_mode=settings["projection_mode"],
                rendering=settings["rendering"],
            )
        viewer.scene.camera.angles = settings["camera_angles"]
        viewer.scene.camera.zoom = settings["camera_zoom"]
        viewer.scene.camera.center = settings["camera_center"]
        viewer.scene.camera.perspective = settings["camera_perspective"]
        try:  # pragma: no cover
            viewer.scene.camera.orientation = tuple(settings["camera_orientation"])
        except Exception as exc:  # pragma: no cover
            print(f"Could not set camera orientation: {exc}")
        viewer.dims.ndisplay = settings["dims_ndisplay"]

        qt_window = viewer.window._qt_window
        canvas_widget = viewer.window._qt_viewer.canvas.native
        target_w, target_h = settings.get("canvas_size", [1600, 1200])
        qt_window.resize(target_w + 400, target_h + 200)
        qt_window.setWindowFlag(Qt.WindowType.Tool, True)
        qt_window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        qt_window.setWindowOpacity(0.0)
        viewer.show()
        app.processEvents()
        qt_window.resize(
            qt_window.width() + (target_w - canvas_widget.width()),
            qt_window.height() + (target_h - canvas_widget.height()),
        )
        app.processEvents()
        np.save(config["output_path"], viewer.screenshot(canvas_only=True))

    finally:
        viewer.close()


if __name__ == "__main__":
    main(sys.argv[1])  # pragma: no cover

"""
Standalone entry point to render a screenshot off the main Qt thread.
"""

import sys

import numpy as np


def restore_cmap(corrupt_cmap: dict) -> dict:
    """
    Restore a colormap dictionary to napari-readable format.

    Parameters
    ----------
    corrupt_cmap: dict
        The jsonable colormap dictionary.

    Returns
    -------
    restore_cmap: dict
        The napari-readable colormap dictionary.
    """

    restored_cmap: dict = {}

    if "color_dict" in corrupt_cmap.keys():
        for key in corrupt_cmap["color_dict"].keys():
            rest_key = key
            if isinstance(key, str):
                try:
                    rest_key = int(key)
                except Exception:
                    try:
                        rest_key = float(key)
                    except Exception:
                        rest_key = None
            restored_cmap[rest_key] = np.array(corrupt_cmap["color_dict"][key])

    else:
        carray: np.ndarray = np.empty(
            (len(corrupt_cmap["colors"]), len(corrupt_cmap["colors"][0]))
        )
        for index, row in enumerate(corrupt_cmap["colors"]):
            carray[index, :] = np.array(row)
        restored_cmap["colors"] = carray
        restored_cmap["name"] = corrupt_cmap["name"]

    return restored_cmap


def main(config_path: str) -> None:  # pragma: no cover
    """
    The main function. Takes a configuration path and creates a napari viewer with the
    desired layer added.

    Parameters
    ----------
    config_path: str
        The path to the configuration json file.
    """

    import json
    from pathlib import Path
    from typing import Any

    import numpy as np
    from napari import Viewer
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication

    config: Any = json.loads(Path(config_path).read_text())
    imarray: np.ndarray = np.load(config["array_path"])
    settings: dict = config["settings"]
    if (
        "custom colormap" in settings.keys()
        and settings["custom colormap"]
        and isinstance(settings["colormap"], dict)
    ):
        settings["colormap"] = restore_cmap(settings["colormap"])

    app = QApplication.instance() or QApplication([])
    viewer: Viewer = Viewer(show=False)

    try:
        if settings["layer_type"] == "labels":
            if isinstance(settings["colormap"], dict):
                cmap: dict | None = settings["colormap"]
            else:
                cmap = None
            viewer.add_labels(  # type: ignore
                imarray,
                blending=settings["blending"],
                colormap=cmap,
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

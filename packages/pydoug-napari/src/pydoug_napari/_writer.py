"""
napari plugin for writing image files.
"""

from pathlib import Path
from typing import Any

from pydoug_core.io.imwriter import imwrite


def write_image(path: Path | str, data: Any, attributes: dict) -> list[str] | None:
    """
    Write an image layer to the specified path as a single image file. Called by napari
    when user chooses "Save Selected Layers" with this plugin's writer selected.

    Parameters
    ----------
    path: Path | str
        The file path to write the image data in data.
    data: Any
        The image data to be written.
    attributes: dict
        The layer attributes.
    """

    path = Path(path)
    imwrite(data, path.parent, path.stem, path.suffix.lstrip("."))
    return [str(path)] if path.exists() else None

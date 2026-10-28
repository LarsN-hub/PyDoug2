"""
napari plugin for writing image files.
"""

import os
from pathlib import Path
from typing import Any


def write_image(path: Path | str, data: Any, metadata: dict) -> list[str] | None:
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

    from pydoug_core.io.imwriter import imwrite

    path = Path(path)
    imwrite(data, path.parent, path.stem, path.suffix.lstrip("."))
    return [str(path)] if path.exists() else None


def write_image_sequence(
    path: Path | str, data: Any, metadata: dict
) -> list[str] | None:
    """
    Write an image layer to the specified path as an image sequence. Called by napari
    when user chooses "Save Selected Layers" with this plugin's writer selected.

    Parameters
    ----------
    path: Path | str
        The file path to write the image data in data.
    data: Any
        The image data to be written.
    metadata: dict
        The layer attributes.
    """

    from pydoug_core.io.imwriter import seqwrite

    path = Path(path)
    name: str = path.stem
    seqwrite(data, path.parent, name, "tif")
    written_paths: list[str] | None = (
        sorted(os.listdir(Path(path.parent) / name))
        if Path(Path(path.parent) / name).exists()
        else None
    )
    return written_paths

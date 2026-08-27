"""
napari plugin for reading image files.
"""

import os
from collections.abc import Callable
from pathlib import Path

from pydoug_core.io.imreader import imread, seqread

READABLE_EXTENSIONS: set[str] = {
    ".tif",
    ".tiff",
    ".png",
    ".h5",
    ".hdf",
    ".hdf5",
    ".he5",
}


def napari_get_reader(path: list[str | Path] | Path | str) -> Callable | None:
    """
    Return a reader function if this plugin can handle a path's extension.

    Parameters
    ----------
    path: list[str | Path] | Path | str
        The path or paths to the image file(s) to be read.

    Returns
    -------
    _reader_function if plugin can handle extensions in path, else None.
    """

    if isinstance(path, list):
        path = path[0]
    path = Path(path)
    if path.is_dir():
        dirlist: list[str] = sorted(os.listdir(path))
        for filepath in dirlist:
            if Path(filepath).suffix in READABLE_EXTENSIONS:
                return _reader_function
    elif path.suffix.lower() in READABLE_EXTENSIONS:
        return _reader_function
    return None


def _reader_function(path: list[str | Path] | Path | str) -> list[tuple]:
    """
    Read image(s) from a path or list of paths into image array(s).

    Parameters
    ----------
    path: list[str | Path] | Path | str
        The path or paths to the image file(s) to be read.

    Returns
    -------
    layer_data : list[tuple]
        A list of LayerData tuples where each tuple contains (data, metadata,
        layertype) where data is a numpy array, metadata is a dict of keyword arguments
        for the corresponding viewer.add_* method in napari, and layertype is a string
        naming the type of layer. Both "meta" and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided.
    """

    layer_data: list[tuple] = []
    if not isinstance(path, list):
        path = [path]
    for filepath in path:
        filepath = Path(filepath)
        data = seqread(filepath) if filepath.is_dir() else imread(filepath)
        metadata: dict[str, str] = {"name": filepath.stem}
        layertype: str = "image"
        layer_data.append((data, metadata, layertype))
    return layer_data

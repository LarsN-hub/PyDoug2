"""
Miscellaneous utility functions for PyDoug core package.
"""

from pathlib import Path
from typing import Literal

import numpy as np
from skimage.color import rgb2gray as skrgb2gray

from pydoug_core.registry import register_operation

__all__ = [
    "is3d",
    "isrgb",
    "gray2rgb",
    "rgb2gray",
    "chan_axis",
    "output_dtype",
    "axlengths",
    "tidypath",
]


def is3d(imarray: np.ndarray | None = None, *, imshape: None | tuple = None) -> bool:
    """
    Accept an image array or its shape and return if it is 3D or not.
    Assumes 3rd or 4th axis with size 3 indicates an RGB array.

    Parameters
    ----------
    imarray: np.ndarray | None
        Image array. Optional if shape is provided.
    imshape: None | tuple
        Shape of an image array. Optional if imarray is provided.

    Returns
    -------
    True if image array/shape is 3D, False if not.
    """

    if not imshape:
        if isinstance(imarray, np.ndarray):
            imshape = imarray.shape
        else:
            return False
    if isinstance(imshape, tuple):
        if len(imshape) < 3:
            return False
        elif len(imshape) == 3:
            if imshape[2] == 3:
                return False
            else:
                return True
        elif len(imshape) > 3:
            return True


def isrgb(imarray: np.ndarray | None = None, imshape: None | tuple = None) -> bool:
    """
    Accept an image array or its shape and return if it is RGB or not.
    Assumes 3rd or 4th axis with size 3 indicates an RGB array.

    Parameters
    ----------
    imarray: np.ndarray | None
        Image array. Optional if shape is provided.
    imshape: None | tuple
        Shape of an image array. Optional if imarray is provided.

    Returns
    -------
    True if image array/shape is RGB, False if not.
    """

    if not imshape:
        if isinstance(imarray, np.ndarray):
            imshape = imarray.shape
        else:
            return False
    if isinstance(imshape, tuple):
        check3d: bool = is3d(imshape=imshape)
        if not check3d and len(imshape) > 2:
            return True
        elif check3d and len(imshape) > 3:
            return True
        else:
            return False


@register_operation(category="utilities", display_name="Grayscale to RGB")
def gray2rgb(imarray: np.ndarray) -> np.ndarray:
    """
    Make a grayscale image array an RGB image array. Duplicates the grayscale array
    3x along a new axis to create the red, green, and blue layers with numpy's repeat:
    https://numpy.org/doc/stable/reference/generated/numpy.repeat.html

    Parameters
    ----------
    imarray: np.ndarray
        The grayscale image array.

    Returns
    -------
    rgbarray: np.ndarray
        The RGB image version of imarray.
        If isrgb(imarray) is True, returns the original array.
    """

    if isrgb(imarray):
        return imarray
    else:
        return np.repeat(np.expand_dims(imarray, imarray.ndim), 3, imarray.ndim)


@register_operation(category="utilities", display_name="RGB to Grayscale")
def rgb2gray(imarray: np.ndarray) -> np.ndarray:
    """
    Make an RGB image array a grayscale image array. Convenience wrapper around
    scikit-image's color.rgb2gray:
    https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.rgb2gray
    """

    if isrgb(imarray):
        return skrgb2gray(imarray)
    return imarray


def chan_axis(imarray: np.ndarray) -> int | None:
    """
    Return the channel axis of an RGB array. Returns None for a grayscale array.

    Parameteres
    -----------
    imarray: np.ndarray
        The image array to obtain the channel axis for.

    Returns
    -------
    ax: int | None
        The channel axis of an RGB image array or None for a grayscale array.
    """

    if isrgb(imarray):
        return imarray.ndim - 1
    else:
        return None


def output_dtype(
    imarray: np.ndarray,
    conv_dtype: np.dtype = np.dtype(np.float64),
    preserve_dtype: bool = True,
) -> np.dtype:
    """
    Return the required output dtype of a function where conversion is necessary.

    Parameters
    ----------
    imarray: np.ndarray
        The image array being converted.
    preserve_dtype: bool
        Whether or not to preserve the input dtype. Default is true.

    Returns
    -------
    outdtype: np.dtype
        The required output dtype for the function.
    """

    if preserve_dtype:
        return imarray.dtype
    return conv_dtype


def axstr2int(imarray: np.ndarray, axis: str = "x") -> int:
    """
    Convert an axis string label to integer label.

    Parameters
    ----------
    imarray: np.ndarray:
        The image array being evaluated.
    axis: str
        The axis string label (upper/lower "x", "y", "z", or "c").

    Returns
    -------
    axint: int
        The axis integer label.
    """

    if is3d(imarray):
        if axis.lower() == "x":
            return 2
        elif axis.lower() == "y":
            return 1
        elif axis.lower() == "z":
            return 0
        if isrgb(imarray):
            if axis.lower() == "c":
                return 3
            return 4
        return 3
    if axis.lower() == "x":
        return 1
    elif axis.lower() == "y":
        return 0
    if isrgb(imarray):
        if axis.lower() == "c":
            return 2
        return 3
    return 2


def axint2str(imarray: np.ndarray, axis: int = 0) -> Literal["X", "Y", "Z", "C"]:
    """
    Convert an axis string label to integer label.

    Parameters
    ----------
    imarray: np.ndarray:
        The image array being evaluated.
    axis: int
        The axis integer label (0, 1, 2, or 3).

    Returns
    -------
    axstr: Literal["X", "Y", "Z", "C"]
        The axis string label.
    """

    if is3d(imarray):
        if axis == 0:
            return "Z"
        elif axis == 1:
            return "Y"
        elif axis == 2:
            return "X"
        return "C"
    if axis == 0:
        return "Y"
    elif axis == 1:
        return "X"
    return "C"


def axlengths(imarray: np.ndarray) -> dict[str, int | None]:
    """
    Return the axial lengths of an image array in a dictionary.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to extract axial lengths from.

    Returns
    -------
    axlengths: dict[str, int | None]
        A dictionary containing the axial lengths in the format "x": 50. Always returns
        inegers for "x" and "y" but will return "None" for "z" in case of a 2D array.
    """

    if is3d(imarray):
        return {"x": imarray.shape[2], "y": imarray.shape[1], "z": imarray.shape[0]}
    else:
        return {"x": imarray.shape[1], "y": imarray.shape[0], "z": None}


def tidypath(filepath: Path, ext: str = "json") -> Path:
    """
    Accept a raw input path and tidy it up.

    Parameters
    ----------
    filepath: Path
        The raw input path
    ext: str
        The target extension (write "dir" for directory)

    Returns
    -------
    tidiedpath: Path
        The tidy path.
    """

    if ext[0] == ".":
        start_index: int = 1
        for char in ext[1:]:
            if char == ".":
                start_index += 1
            else:
                break
        ext = ext[start_index:]

    strpath: str = str(filepath)[: str(filepath).rfind(filepath.suffix)]
    if strpath[-1] == ".":
        end_index: int = 1
        for char in strpath[::-1][1:]:
            if char == ".":
                end_index += 1
            else:
                break
        strpath = strpath[:-end_index]

    if ext == "dir":
        return Path(strpath)
    return Path(strpath + f".{ext}")

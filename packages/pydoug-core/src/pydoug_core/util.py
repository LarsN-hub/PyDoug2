"""
Miscellaneous utility functions for PyDoug core package.
"""

from pathlib import Path
from typing import Literal

import numpy as np

from pydoug_core.registry import register_operation

__all__ = [
    "is3d",
    "isrgb",
    "gray2rgb",
    "rgb2gray",
    "chan_axis",
    "convert_dtype",
    "output_dtype",
    "axstr2int",
    "axint2str",
    "axlengths",
    "tidypath",
    "prep_mask",
    "dtype_2_str",
    "str_2_dtype",
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


def is_square(imarray: np.ndarray, along_z: bool = True) -> bool:
    """
    Determine if an image is square/cubic in bulk or along the z-axis.

    Parameters
    ----------
    imarray: np.ndarray
        The image to determine squareness for.
    along_z: bool
        If the image is 3D, whether or not to determine squareness along the z-axis
        (True) or determine cubicness in bulk (False). Default is True.

    Returns
    -------
    issquare: bool
        True if the image is square (2D) or square along its z-axis (3D). Else False.
    """

    if is3d(imarray):
        if along_z:
            if imarray.shape[1] == imarray.shape[2]:
                return True
            return False
        if imarray.shape[0] == imarray.shape[1] == imarray.shape[2]:
            return True
        return False
    if imarray.shape[0] == imarray.shape[1]:
        return True
    return False


@register_operation(category="color", display_name="Grayscale to RGB")
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


@register_operation(category="color", display_name="RGB to Grayscale")
def rgb2gray(imarray: np.ndarray, Preserve_Dtype: bool = True) -> np.ndarray:
    """
    Make an RGB image array a grayscale image array. Convenience wrapper around
    scikit-image's color.rgb2gray:
    https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.rgb2gray

    Parameters
    ----------
    imarray: np.ndarray
        The image to convert from RGB to grayscale.
    Preserve_Dtype: bool
        Whether or not to preserve the dtype of the original image. Default is True.
    """

    from skimage.color import rgb2gray as skrgb2gray

    if isrgb(imarray):
        if Preserve_Dtype:
            return convert_dtype(skrgb2gray(imarray), dtype_2_str(imarray.dtype))  # type: ignore
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


@register_operation(category="histogram", display_name="Convert Data Type")
def convert_dtype(
    imarray: np.ndarray,
    New_Type: Literal[
        "Uint8", "Uint16", "Int16", "Float32", "Float64", "Bool"
    ] = "Uint8",
) -> np.ndarray:
    """
    Convert the data type of an image. Convenience wrapper around the "img_as_XXXXX"
    functions in scikit-image's util package:
    https://scikit-image.org/docs/stable/api/skimage.util.html#

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be converted.
    New_Type: Literal["Bool", "Uint8", "Uint16", "Int16", "Float32", "Float64"]
        The data type to convert to.
        - Uint8: 8-bit unsigned integer format [0, 255]
        - Uint16: 16-bit unsigned integer format [0, 65,535]
        - Int16: 16-bit signed integer format [-32,768, 32,767]
        - Float32: 32-bit (single-precision) floating-point format [-3.4e+38, 3.4e+38]
        - Float64: 64-bit (double-precision) floating-point format [-1.8e+308, 1.8e+308]
        - Bool: Boolean True/False format

    Returns
    -------
    convarray: np.ndarray
        The converted image array.
    """

    from skimage.util import (
        img_as_bool,
        img_as_float32,
        img_as_float64,
        img_as_int,
        img_as_ubyte,
        img_as_uint,
    )

    if New_Type == "Bool" and imarray.dtype != np.bool:
        return img_as_bool(imarray)
    if New_Type == "Uint8" and imarray.dtype != np.uint8:
        return img_as_ubyte(imarray)
    if New_Type == "Uint16" and imarray.dtype != np.uint16:
        return img_as_uint(imarray)
    if New_Type == "Int16" and imarray.dtype != np.int16:
        return img_as_int(imarray)
    if New_Type == "Float32" and imarray.dtype != np.float32:
        return img_as_float32(imarray)
    if New_Type == "Float64" and imarray.dtype != np.float64:
        return img_as_float64(imarray)
    return imarray


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


def prep_mask(imarray: np.ndarray, maskarray: np.ndarray) -> np.ndarray:
    """
    Ensure that a mask array and an image array have the same dimensions.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be masked.
    maskarray: np.ndarray
        The mask array to be used.

    Returns
    -------
    maskarray_corr: np.ndarray
        The mask array with corrected dimensions.
    """

    if maskarray.ndim < imarray.ndim:
        if isrgb(imarray):
            if imarray.ndim == 3:
                return np.astype(np.stack([maskarray] * imarray.shape[2], 2), np.bool)
            if maskarray.ndim == 2:
                return np.astype(
                    np.stack(
                        [np.stack([maskarray] * imarray.shape[3], 2)]
                        * imarray.shape[0],
                        0,
                    ),
                    np.bool,
                )
            if maskarray.shape == imarray.shape[1:]:
                return np.astype(np.stack([maskarray] * imarray.shape[0], 0), np.bool)
            return np.astype(np.stack([maskarray] * imarray.shape[3], 3), np.bool)
        return np.astype(np.stack([maskarray] * imarray.shape[0], 0), np.bool)

    if maskarray.ndim > imarray.ndim:
        if isrgb(imarray):
            return np.astype(maskarray[0], np.bool)
        if maskarray.ndim == 4:
            if imarray.ndim == 3:
                return np.astype(maskarray[..., 0], np.bool)
            return np.astype(maskarray[0, :, :, 0], np.bool)
        if maskarray.shape[0:2] == imarray.shape:
            return np.astype(maskarray[..., 0], np.bool)
        return np.astype(maskarray[0, ...], np.bool)

    if maskarray.shape != imarray.shape:
        if maskarray.ndim == 3:
            if isrgb(imarray):
                return np.astype(
                    np.stack([maskarray[0]] * imarray.shape[2], 2), np.bool
                )
            return np.astype(
                np.stack([maskarray[..., 0]] * imarray.shape[0], 0), np.bool
            )

    return np.astype(maskarray, np.bool)


def dtype_2_str(dtype: np.dtype) -> str:
    """
    Convert a numpy dtype to a string usable by pydoug_core.util.convert_dtype.

    Parameters
    ----------
    dtype: np.dtype
        The dtype to convert.

    Returns
    -------
    dtype_str: str
        The converted str.
    """

    if dtype == np.uint8:
        return "Uint8"
    if dtype == np.uint16:
        return "Uint16"
    if dtype == np.int16:
        return "Int16"
    if dtype == np.dtype(np.bool):
        return "Bool"
    if dtype == np.float32:
        return "Float32"
    return "Float64"


def str_2_dtype(dtype_str: str) -> np.dtype:
    """
    Convert a string to a numpy dtype.

    Parameters
    ----------
    dtype_str: str
        The dtype string to convert.

    Returns
    -------
    dtype_str: np.dtype
        The dtype.
    """

    if dtype_str.lower() == "uint8":
        return np.dtype(np.uint8)
    if dtype_str.lower() == "uint16":
        return np.dtype(np.uint16)
    if dtype_str.lower() == "int16":
        return np.dtype(np.int16)
    if dtype_str.lower() == "bool":
        return np.dtype(np.bool)
    if dtype_str.lower() == "float32":
        return np.dtype(np.float32)
    return np.dtype(np.float64)

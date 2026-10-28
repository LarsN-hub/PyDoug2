"""
Logic operations.
"""

from typing import Annotated, Literal

import numpy as np

from pydoug_core.registry import register_operation

__all__ = ["apply_mask", "projection", "logic"]


@register_operation(
    category="math",
    display_name="Apply Mask",
    array_params=("Mask",),
    allow_in_place=True,
)
def apply_mask(
    imarray: np.ndarray,
    Mask: np.ndarray,
    Method: Literal["Out", "In"] = "Out",
    Masked_Color: Annotated[int, {"max": 65535, "min": -32768}] = 0,
    In_Place: bool = False,
) -> np.ndarray:
    """
    Apply a mask to an image array. Convenience wrapper around numpy's boolean indexing:
    https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be masked.
    Mask: np.ndarray
        The mask array to use. If "Mask" is not binary, it is first converted.
        If the image and "Mask" have different shapes, an attempt is made to modify
        "Mask" to fit imarray's dimensions:
        - If "Mask" is 2D and the image is 3D, "Mask" is projected along z axis.
        - If "Mask" is 3D and the image is 2D, first slice of "Mask" is used.
        - If "Mask" is grayscale and the image is RGB, "Mask" is made RGB.
        - If "Mask" is RGB and the image is grayscale, "Mask" is made grayscale.
    Method: Literal["Out", "In"]
        Whether to mask values outside (False) or inside (True) "Mask".
        Default is "Out".
    Masked_Color: int
        The color to assign masked values. Default is 0.
    In_Place: bool
        Whether or not to perform the masking on the input image (faster, saves RAM, but
        is non-reversible) or a copy (slower, consumes RAM, but is reversible).
        Default is False.

    Returns
    -------
    masked_array: np.ndarray
        The masked image array.
    """

    from pydoug_core.util import prep_mask

    if In_Place:
        if Method == "In":
            imarray[prep_mask(imarray, Mask)] = Masked_Color
        else:
            imarray[np.logical_not(prep_mask(imarray, Mask))] = Masked_Color
        return imarray
    masked_array: np.ndarray = np.copy(imarray)
    if Method == "In":
        masked_array[prep_mask(masked_array, Mask)] = Masked_Color
    else:
        masked_array[np.logical_not(prep_mask(masked_array, Mask))] = Masked_Color
    return masked_array


@register_operation(category="math", display_name="2D Projection")
def projection(
    imarray: np.ndarray,
    Method: Literal["First", "Last", "Thickness"] = "Thickness",
    Pixel_Size: Annotated[float, {"min": 0, "max": 1000000, "step": 0.0001}] = 1,
) -> np.ndarray:
    """
    Generate a 2D projection of binary 3D image. Can either count all pixels in a column
    to return thickness or find the height of the first/last pixel in the column to
    return height. Convenience wrapper around numpy's count_nonzero, argwhere, and flip:
    https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html
    https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html
    https://numpy.org/doc/stable/reference/generated/numpy.flip.html

    Parameters
    ----------
    imarray: np.ndarray
        The 3D binary image to create a projection from.
    Method: Literal
        The method for creating the projection. Default is "Thickness".
        - "First": Return the position of the first non-zero pixel along the z-axis.
        - "Last": Return the position of the last non-zero pixel along the z-axis.
        - "Thickness": Return the number of non-zero pixels along the z-axis by column.
    Pixel_Size: float
        The pixel edge length in real-world units. This value is multiplied by the
        output to return the thickness or position in real-world units.

    Returns
    -------
    proj_array: np.ndarray
        The projected image.
    """

    from pydoug_core.util import is3d, isrgb

    if not is3d(imarray) or isrgb(imarray):
        print("Image is either RGB or 2D, projection not created!")
        return imarray

    if Method == "First":
        return np.argmax(imarray, axis=0) * Pixel_Size

    elif Method == "Last":
        return np.argmax(np.flip(imarray, axis=0), axis=0) * Pixel_Size

    return np.count_nonzero(imarray, axis=0) * Pixel_Size


@register_operation(
    category="math",
    display_name="Logic",
    array_params=("Logic_Layer",),
)
def logic(
    imarray: np.ndarray,
    Logic_Layer: np.ndarray,
    Method: Literal["Intersect", "Subtract", "Union"] = "Union",
) -> np.ndarray:
    """
    Perform binary logic operations on two arrays. Think PowerPoint "Merge Shapes"
    operations. Both images must have the same dimensions. Images that are not binary
    will be converted to boolean as in "Convert Type".
    Convenience wrapper around numpy's boolean indexing:
    https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing

    Parameters
    ----------
    imarray: np.ndarray
        The base array to operate on.
    Logic_Image: np.ndarray
        The image used in the logic operation on the first image.
    Method: Literal["Intersect", "Subtract", "Union"]
        The logic method. Default is "Union".
        - "Intersect": Only pixels where both images are not background are kept.
        - "Subtract": Only pixels where the first image is not background are kept.
        - "Union": Pixels where either image are not background are kept.

    Returns
    -------
    log_array: np.ndarray
        The image after the logical operation.
    """

    from pydoug_core.util import convert_dtype

    log_array: np.ndarray = np.zeros(imarray.shape, np.bool)
    if Method == "Intersect":
        log_array[
            convert_dtype(imarray, "Bool") & convert_dtype(Logic_Layer, "Bool")
        ] = True
    elif Method == "Subtract":
        log_array[
            convert_dtype(imarray, "Bool")
            & np.logical_not(convert_dtype(Logic_Layer, "Bool"))
        ] = True
    else:
        log_array[
            convert_dtype(imarray, "Bool") | convert_dtype(Logic_Layer, "Bool")
        ] = True

    return log_array

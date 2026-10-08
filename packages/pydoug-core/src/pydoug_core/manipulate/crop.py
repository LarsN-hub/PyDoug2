"""
Crop images with masks.
"""

from typing import Annotated

import numpy as np

from pydoug_core.registry import register_operation

__all__ = ["crop"]


@register_operation(
    category="manipulate",
    display_name="Crop",
    array_params=("Mask",),
    allow_in_place=True,
)
def crop(
    imarray: np.ndarray,
    Mask: np.ndarray,
    Masked_Color: Annotated[int, {"max": 65535, "min": -32768}] = 0,
    In_Place: bool = False,
) -> np.ndarray:
    """
    Crop an image with a mask. Convenience wrapper around numpy's boolean indexing,
    argwhere, and squeeze:
    https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing
    https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html#numpy-argwhere
    https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html#numpy-squeeze

    Parameters
    ----------
    imarray: np.ndarray
        The image to crop.
    Mask: np.ndarray
        The mask array to use. If "Mask" is not binary, it is first converted.
        If the image and "Mask" have different shapes, an attempt is made to modify
        "Mask" to fit imarray's dimensions:
        - If "Mask" is 2D and the image is 3D, "Mask" is projected along z axis.
        - If "Mask" is 3D and the image is 2D, first slice of "Mask" is used.
        - If "Mask" is grayscale and the image is RGB, "Mask" is made RGB.
        - If "Mask" is RGB and the image is grayscale, "Mask" is made grayscale.
    Masked_Color: int
        In the case of a non-rectangular mask, the color to assign masked values.
        Default is 0.
    In_Place: bool
        Whether or not to perform the crop on the input image (faster, saves RAM, but
        is non-reversible) or a copy (slower, consumes RAM, but is reversible).
        Default is False.

    Returns
    -------
    croparray: np.ndarray
        The cropped image.
    """

    from pydoug_core.util import is3d, prep_mask

    if np.all(Mask):
        return imarray

    if is3d(Mask):
        z_start: np.int64 | None = np.min(np.argwhere(Mask)[:, 0])
        z_end: np.int64 | None = np.max(np.argwhere(Mask)[:, 0]) + 1
        r_start: np.int64 = np.min(np.argwhere(Mask)[:, 1])
        r_end: np.int64 = np.max(np.argwhere(Mask)[:, 1]) + 1
        c_start: np.int64 = np.min(np.argwhere(Mask)[:, 2])
        c_end: np.int64 = np.max(np.argwhere(Mask)[:, 2]) + 1
    else:
        z_start = None
        z_end = None
        r_start = np.min(np.argwhere(Mask)[:, 0])
        r_end = np.max(np.argwhere(Mask)[:, 0]) + 1
        c_start = np.min(np.argwhere(Mask)[:, 1])
        c_end = np.max(np.argwhere(Mask)[:, 1]) + 1

    if In_Place:
        imarray[np.logical_not(prep_mask(imarray, Mask))] = Masked_Color
        if is3d(imarray):
            if z_start is not None and z_end is not None:
                imarray = imarray[z_start:z_end, r_start:r_end, c_start:c_end]
            else:
                imarray = imarray[:, r_start:r_end, c_start:c_end]
        else:
            imarray = imarray[r_start:r_end, c_start:c_end]

        return np.squeeze(imarray)

    croparray: np.ndarray = np.copy(imarray)
    croparray[np.logical_not(prep_mask(imarray, Mask))] = Masked_Color
    if is3d(imarray):
        if z_start is not None and z_end is not None:
            croparray = croparray[z_start:z_end, r_start:r_end, c_start:c_end]
        else:
            croparray = croparray[:, r_start:r_end, c_start:c_end]
    else:
        croparray = croparray[r_start:r_end, c_start:c_end]

    return np.squeeze(croparray)

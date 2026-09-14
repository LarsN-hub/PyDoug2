"""
Mask logic operations.
"""

from typing import Annotated, Literal

import numpy as np

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["apply_mask"]


@register_operation(
    category="mask",
    display_name="Apply Mask",
    mask_params=("Mask",),
)
def apply_mask(
    imarray: np.ndarray,
    Mask: np.ndarray,
    Method: Literal["Out", "In"] = "Out",
    Masked_Intensity: Annotated[int, {"max": 65535, "min": -32768}] = 0,
    Conserve_RAM: bool = False,
) -> np.ndarray:
    """
    Apply a mask to an image array. Convenience wrapper around numpy's boolean indexing:
    https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be masked.
    Mask: np.ndarray
        The mask array to use. If Mask is not binary, it is first converted.
        If imarray and Mask have different shapes, an attempt is made to modify
        Mask to fit imarray's dimensions:
        - If Mask is 2D and imarray is 3D, Mask is projected along z axis.
        - If Mask is 3D and imarray is 2D, first slice of Mask is used.
        - If Mask is grayscale and imarray is RGB, Mask is made RGB.
        - If Mask is RGB and imarray is grayscale, Mask is made grayscale.
    Method: Literal["Out", "In"]
        Whether to mask values outside (False) or inside (True) Mask. Default is "Out".
    Masked_Intensity: int
        The intensity to assign masked values. Default is 0.
    Conserve_RAM: bool
        Whether or not to perform the masking on the input imarray or a copy.
        Default is False.

    Returns
    -------
    masked_array: np.ndarray
        The masked image array.
    """

    if Conserve_RAM:
        if Method == "In":
            imarray[util.prep_mask(imarray, Mask)] = Masked_Intensity
        else:
            imarray[np.logical_not(util.prep_mask(imarray, Mask))] = Masked_Intensity
        return imarray
    outarray: np.ndarray = np.copy(imarray)
    if Method == "In":
        outarray[util.prep_mask(outarray, Mask)] = Masked_Intensity
    else:
        outarray[np.logical_not(util.prep_mask(outarray, Mask))] = Masked_Intensity
    return outarray

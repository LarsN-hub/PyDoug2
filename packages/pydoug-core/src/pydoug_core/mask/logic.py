"""
Mask logic operations.
"""

from typing import Literal

import numpy as np

from pydoug_core import util
from pydoug_core.registry import register_operation


@register_operation(
    category="mask",
    display_name="Apply Mask",
    widget_options={"Masked_Intensity": {"max": 65535, "min": 0}},
)
def apply_mask(
    imarray: np.ndarray,
    maskarray: np.ndarray,
    Method: Literal["Out", "In"] = "Out",
    Masked_Intensity: int = 0,
    Conserve_RAM: bool = False,
) -> np.ndarray:
    """
    Apply a mask to an image array.
    """

    if Conserve_RAM:
        if Method == "In":
            imarray[util.prep_mask(imarray, maskarray)] = Masked_Intensity
        else:
            imarray[np.logical_not(util.prep_mask(imarray, maskarray))] = (
                Masked_Intensity
            )
        return imarray
    outarray: np.ndarray = np.copy(imarray)
    if Method == "In":
        outarray[util.prep_mask(outarray, maskarray)] = Masked_Intensity
    else:
        outarray[np.logical_not(util.prep_mask(outarray, maskarray))] = Masked_Intensity
    return outarray

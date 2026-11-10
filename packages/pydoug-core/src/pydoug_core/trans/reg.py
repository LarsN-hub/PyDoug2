"""
Image registration.
"""

import numpy as np

from pydoug_core.util import register_operation


@register_operation(
    category="transform",
    display_name="Register",
    array_params=("Fixed_Image",),
)
def register(
    imarray: np.ndarray,
    Fixed_Image: np.ndarray,
) -> np.ndarray:
    """
    Register an image to a fixed image.

    Parameters
    ----------
    imarray: np.ndarray
        The image to register.
    Fixed_Image: np.ndarray
        The fixed image to register the input image with.

    Returns
    -------
    regarray: np.ndarray
        The registered image.
    """

    return imarray

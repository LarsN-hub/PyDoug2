"""
Image arithmetic operations.
"""

from typing import Literal

import numpy as np

__all__ = ["im_arithmetic"]


def im_arithmetic(
    imarray: np.ndarray,
    Logic_Layer: np.ndarray,
    Method: Literal["Add", "Divide", "Multiply", "Subtract"] = "Add",
    Preserve_Dtype: bool = True,
) -> np.ndarray:
    """
    Perform arithmetic on two arrays. Both images must have the same dimensions.
    Convenience wrapper around numpy's arithmetic operations:
    https://numpy.org/doc/stable/reference/arrays.ndarray.html#arithmetic-matrix-multiplication-and-comparison-operations

    Parameters
    ----------
    imarray: np.ndarray
        The base array to operate on.
    Arithmetic_Image: np.ndarray
        The image used in the arithmetic operation on the first image.
    Method: Literal["Add", "Divide", "Multiply", "Subtract"]
        The logic method. Values outside the first image's dtype are saturated if
        "Preserve_Dtype" is True, otherwise the dtpye is converted. Default is "Add".
        - "Add": Add the images together.
        - "Divide": Divide the first image by "Arithmetic Image".
        - "Multiply": Multiply the images together.
        - "Subtract": Subtract "Arithmetic Image" from the first image.
    Preserve_Dtype: bool
        Whether or not to preserve the first image's dtype. Default is True.

    Returns
    -------
    math_array: np.ndarray
        The image after the arithmetic operation.
    """

    if Method == "Add":
        pass
    elif Method == "Divide":
        pass
    elif Method == "Multiply":
        pass
    elif Method == "Subtract":
        pass

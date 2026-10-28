"""
Image arithmetic operations.
"""

from typing import Annotated, Literal

import numpy as np

from pydoug_core.registry import register_operation

__all__ = ["arithmetic"]


@register_operation(
    category="math",
    display_name="Arithmetic",
    array_params=("Arithmetic_Layer",),
    array_param_flags={"Arithmetic_Layer": "Use_Arithmetic_Layer"},
)
def arithmetic(
    imarray: np.ndarray,
    Method: Literal["Add", "Divide", "Multiply", "Subtract"] = "Add",
    Arithmetic_Value: Annotated[
        float, {"step": 0.0001, "min": -32768, "max": 65535}
    ] = 1,
    Use_Arithmetic_Layer: bool = False,
    Arithmetic_Layer: np.ndarray | None = None,
    Preserve_Dtype: bool = True,
) -> np.ndarray:
    """
    Perform arithmetic on an image with a value or another image. Both images must have
    the same dimensions. Convenience wrapper around numpy's arithmetic operations:
    https://numpy.org/doc/stable/reference/arrays.ndarray.html#arithmetic-matrix-multiplication-and-comparison-operations

    Parameters
    ----------
    imarray: np.ndarray
        The base array to operate on.
    Arithmetic_Value: float
        The value to use for the arithmetic operation on the first image. Not used if
        "Use Arithmetic Layer" is True and "Arithmetic Layer" is provided.
    Use_Arithmetic_Layer: bool
        Whether or not to use the provided "Arithmetic Layer" for the arithmetic
        operation instead of the "Arithmetic Value". Default is False.
    Arithmetic_Layer: np.ndarray
        The image used in the arithmetic operation on the first image. Only used if
        "Use_Arithmetic_Layer" is True.
    Method: Literal["Add", "Divide", "Multiply", "Subtract"]
        The logic method. Values outside the first image's dtype are saturated if
        "Preserve_Dtype" is True, otherwise the dtpye is converted. Default is "Add".
        - "Add": Add the value (or other image) to the first image.
        - "Divide": Divide the first image by the value/image.
        - "Multiply": Multiply the first image by the value/image.
        - "Subtract": Subtract the other image/value from the first image.
    Preserve_Dtype: bool
        Whether or not to preserve the first image's dtype. Default is True.

    Returns
    -------
    math_array: np.ndarray
        The image after the arithmetic operation.
    """

    if np.issubdtype(imarray.dtype, np.integer):
        dtype_min = np.iinfo(imarray.dtype).min
        dtype_max = np.iinfo(imarray.dtype).max
    elif np.issubdtype(imarray.dtype, np.floating):
        dtype_min = np.finfo(imarray.dtype).min
        dtype_max = np.finfo(imarray.dtype).max
    else:
        dtype_min = 0
        dtype_max = 1

    if Method == "Add":
        if Use_Arithmetic_Layer and Arithmetic_Layer is not None:
            math_array: np.ndarray = (imarray + Arithmetic_Layer).astype(np.float64)
        else:
            math_array: np.ndarray = (imarray + Arithmetic_Value).astype(np.float64)

    elif Method == "Divide":
        if Use_Arithmetic_Layer and Arithmetic_Layer is not None:
            math_array: np.ndarray = (imarray / Arithmetic_Layer).astype(np.float64)
        else:
            math_array: np.ndarray = (imarray / Arithmetic_Value).astype(np.float64)

    elif Method == "Multiply":
        if Use_Arithmetic_Layer and Arithmetic_Layer is not None:
            math_array: np.ndarray = (imarray * Arithmetic_Layer).astype(np.float64)
        else:
            math_array: np.ndarray = (imarray * Arithmetic_Value).astype(np.float64)

    else:
        if Use_Arithmetic_Layer and Arithmetic_Layer is not None:
            math_array: np.ndarray = (imarray - Arithmetic_Layer).astype(np.float64)
        else:
            math_array: np.ndarray = (imarray - Arithmetic_Value).astype(np.float64)

    if Preserve_Dtype:
        math_array[math_array > dtype_max] = dtype_max
        math_array[math_array < dtype_min] = dtype_min

        return math_array.astype(imarray.dtype)

    return math_array

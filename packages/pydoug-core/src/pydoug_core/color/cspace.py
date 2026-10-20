"""
Convert color space.
"""

from typing import Literal

import numpy as np

from pydoug_core.registry import register_operation


@register_operation(category="visualize", display_name="Convert Color Space")
def convert_cspace(
    imarray: np.ndarray,
    Input_Color_Space: Literal[
        "RGB", "HSV", "RGB CIE", "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"
    ] = "RGB",
    Output_Color_Space: Literal[
        "RGB", "HSV", "RGB CIE", "XYZ", "YUV", "YIQ", "YPbPr", "YCbCr", "YDbDr"
    ] = "RGB",
    Preserve_Dtype: bool = True,
) -> np.ndarray:
    """
    Convert an image from one color space to another. Note, any PyDoug operations that
    work for color images are designed for RGB images. This functionality is solely for
    exporting images in your desired color space. Convenience wrapper around
    scikit-image's color.convert_colorspace:
    https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.convert_colorspace

    Parameters
    ----------
    imarray: np.ndarray
        The image for color space conversion.
    Input_Color_Space: Literal
        The input image's color space.
        - "RGB": red - green - blue
        - "HSV": hue - saturation - value
        - "RGB CIE": red-green-blue International Commission on Illumination (CIE)
        - "XYZ": x-y-z CIE
        - "YUV": luma - red projection - blue projection
        - "YIQ": luma - in-phase - quadrature
        - "YPbPr": luma - blue-difference - red-difference (analog)
        - "YCbCr": luma - blue-difference - red-difference (digital)
        - "YDbDr": luma - blue-difference - red-difference (SECAM)
    Output_Color_Space: Literal
        The output image's color space. Same options as "Input Color Space".
    Preserve_Dtype: bool
        Whether or not to return the same data type that was input. Default is True.

    Returns
    -------
    cspace_array: np.ndarray
        The image in the new color space.
    """

    from skimage.color import convert_colorspace
    from skimage.exposure import rescale_intensity

    from pydoug_core.util import convert_dtype, dtype_2_str

    if Input_Color_Space == Output_Color_Space:
        return imarray

    cspace_array: np.ndarray = convert_colorspace(
        imarray, Input_Color_Space.lower(), Output_Color_Space.lower()
    )

    if Preserve_Dtype:
        if np.max(cspace_array) > 1 or np.min(cspace_array) < -1:
            cspace_array = rescale_intensity(cspace_array, out_range=(-1, 1))  # type: ignore
        return convert_dtype(cspace_array, dtype_2_str(imarray.dtype))  # type: ignore
    return cspace_array

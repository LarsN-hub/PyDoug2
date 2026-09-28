"""
Filters that detect gradients and edges.
"""

from typing import Annotated, Literal

import numpy as np
from skimage.filters import (
    farid,
    farid_h,
    farid_v,
    laplace,
    prewitt,
    prewitt_h,
    prewitt_v,
    roberts,
    roberts_neg_diag,
    roberts_pos_diag,
    scharr,
    scharr_h,
    scharr_v,
    sobel,
    sobel_h,
    sobel_v,
)

from pydoug_core import util
from pydoug_core.registry import register_operation


@register_operation(category="filters", display_name="Gradient")
def gradient(
    imarray: np.ndarray,
    Method: Literal[
        "Farid", "Laplace", "Prewitt", "Roberts", "Scharr", "Sobel"
    ] = "Sobel",
    K_Size: Annotated[int, {"min": 1}] = 3,
    Horizontal: bool = False,
    Vertical: bool = False,
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"] = "Nearest",
    Constant_Value: Annotated[int, {"min": -32768, "max": 65535}] = 0,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Filter an image with a gradient response filter. Convenience wrapper around the
    gradient filters in scikit-image's filters package, including farid, laplace,
    prewitt, roberts, scharr, and sobel:
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.farid
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.farid_h
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.farid_v
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.laplace
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.prewitt
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.prewitt_h
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.prewitt_v
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.roberts
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.roberts_neg_diag
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.roberts_pos_diag
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr_h
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr_v
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel_h
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel_v

    Parameters
    ----------
    imarray: np.ndarray
        The image to be filtered.
    Method: Literal["Farid", "Laplace", "Prewitt", "Roberts", "Scharr", "Sobel"]
        The method to use to calculate the gradient response. "Roberts" will
        automatically be performed slicewise if the input image is 3D.
        Default is "Sobel".
    K_Size: int
        The k size if "Method" is "Laplace". Ignored for all other methods.
    Horizontal: bool
        Whether or not to apply the filter horizontally only. "Vertical" must be False
        or this will be ignored. Forces slicewise performance on 3D images and channel-
        wise performance on RGB images. Performs the negative diagonal if "Method" is
        "Roberts".
    Vertical: bool
        Whether or not to apply the filter vertically only. "Horizontal" must be False
        or this will be ignored. Forces slicewise performance on 3D images and channel-
        wise performance on RGB images. Performs the positive diagonal if "Method" is
        "Roberts".
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"]
        How the algorithm should handle image borders. Ignored if "Method" is "Laplace"
        or "Roberts". Ignored if "Horizontal" or "Vertical" are True.
        Default is "Nearest".
        - "Constant": The border is extended with a value defined in "Constant Value".
        - "Mirror": The border is extended by reflecting about the center of the pixel.
        - "Nearest": The border is extended by copying the edge pixel.
        - "Reflect": The border is extended by reflecting about the edge of the pixel.
        - "Wrap": The border is extended by wrapping around to the opposite border.
    Constant_Value: float
        If "Edge Mode" is "Constant", the constant value to use. Default is 0.
    Perform_Slicewise: bool
        Whether or not to perform the operation slice-by-slice. Default is False.

    Returns
    -------
    grad_array: np.ndarray
        The gradient filtered image.
    """

    def _grad(
        imarray, Method, K_Size, Horizontal, Vertical, Edge_Mode, Constant_Value
    ) -> np.ndarray:

        if Method == "Farid":
            if Horizontal and not Vertical:
                if util.isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[:, :, cindex] = farid_h(imarray[:, :, cindex])
                    return grad_array
                return farid_h(imarray)
            elif Vertical and not Horizontal:
                if util.isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[:, :, cindex] = farid_v(imarray[:, :, cindex])
                    return grad_array
                return farid_v(imarray)
            return farid(imarray, mode=Edge_Mode.lower(), cval=Constant_Value)

        elif Method == "Laplace":
            return laplace(imarray, K_Size)

        elif Method == "Prewitt":
            if Horizontal and not Vertical:
                if util.isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[:, :, cindex] = prewitt_h(imarray[:, :, cindex])
                    return grad_array
                return prewitt_h(imarray)
            elif Vertical and not Horizontal:
                if util.isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[:, :, cindex] = prewitt_v(imarray[:, :, cindex])
                    return grad_array
                return prewitt_v(imarray)
            return prewitt(imarray, mode=Edge_Mode.lower(), cval=Constant_Value)

        elif Method == "Roberts":
            if Horizontal and not Vertical:
                if util.isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[:, :, cindex] = roberts_neg_diag(
                            imarray[:, :, cindex]
                        )
                    return grad_array
                return roberts_neg_diag(imarray)
            elif Vertical and not Horizontal:
                if util.isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[:, :, cindex] = roberts_pos_diag(
                            imarray[:, :, cindex]
                        )
                    return grad_array
                return roberts_pos_diag(imarray)
            if util.isrgb(imarray):
                grad_array: np.ndarray = np.empty(imarray.shape)
                for cindex in range(0, imarray.shape[-1]):
                    grad_array[:, :, cindex] = roberts(imarray[:, :, cindex])
                return grad_array
            return roberts(imarray)

        elif Method == "Scharr":
            if Horizontal and not Vertical:
                if util.isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[:, :, cindex] = scharr_h(imarray[:, :, cindex])
                    return grad_array
                return scharr_h(imarray)
            elif Vertical and not Horizontal:
                if util.isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[:, :, cindex] = scharr_h(imarray[:, :, cindex])
                    return grad_array
                return scharr_v(imarray)
            return scharr(imarray, mode=Edge_Mode.lower(), cval=Constant_Value)

        else:
            if Horizontal and not Vertical:
                if util.isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[:, :, cindex] = sobel_h(imarray[:, :, cindex])
                    return grad_array
                return sobel_h(imarray)
            elif Vertical and not Horizontal:
                if util.isrgb(imarray):
                    grad_array: np.ndarray = np.empty(imarray.shape)
                    for cindex in range(0, imarray.shape[-1]):
                        grad_array[:, :, cindex] = sobel_v(imarray[:, :, cindex])
                    return grad_array
                return sobel_v(imarray)
            return sobel(imarray, mode=Edge_Mode.lower(), cval=Constant_Value)

    if Method == "Roberts":
        Perform_Slicewise = True
    if Horizontal and not Vertical:
        Perform_Slicewise = True
    if Vertical and not Horizontal:
        Perform_Slicewise = True

    if Perform_Slicewise and util.is3d(imarray):
        grad_array: np.ndarray = np.empty(imarray.shape)
        for index in range(0, imarray.shape[0]):
            grad_array[index] = _grad(
                imarray[index],
                Method,
                K_Size,
                Horizontal,
                Vertical,
                Edge_Mode,
                Constant_Value,
            )
        return grad_array
    else:
        return _grad(
            imarray,
            Method,
            K_Size,
            Horizontal,
            Vertical,
            Edge_Mode,
            Constant_Value,
        )

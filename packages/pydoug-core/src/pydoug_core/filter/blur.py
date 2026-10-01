"""
Filters that blur or smooth images.
"""

from typing import Annotated, Literal

import numpy as np
from skimage.filters import gaussian, median
from skimage.filters.rank import mean, modal
from skimage.morphology import ball, disk

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["smooth"]


@register_operation(category="filters", display_name="Smooth")
def smooth(
    imarray: np.ndarray,
    Method: Literal["Gaussian", "Mean", "Median", "Mode"] = "Gaussian",
    Sigma: Annotated[float, {"min": 0}] = 1,
    Truncate: Annotated[int, {"min": 1}] = 4,
    Radius: Annotated[int, {"min": 1}] = 3,
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"] = "Nearest",
    Constant_Value: Annotated[int, {"min": -32768, "max": 65535}] = 0,
    Perform_Slicewise: bool = False,
    Preserve_Dtype: bool = True,
) -> np.ndarray:
    """
    Filter an image with a smoothing filter. Convenience wrapper around median and
    gaussian from scikit-image's filters package and mean and modal from scikit-image's
    filters.rank package:
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.gaussian
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.median
    https://scikit-image.org/docs/stable/api/skimage.filters.rank.html#skimage.filters.rank.mean
    https://scikit-image.org/docs/stable/api/skimage.filters.rank.html#skimage.filters.rank.modal

    Parameters
    ----------
    imarray: np.ndarray
        The image to be smoothed.
    Method: Literal["Gaussian", "Geometric Mean", "Mean", "Median", "Mode"]
        The method for smoothing. All methods can handle RGB images but "Median",
        "Mean", and "Mode" operate separately along the RGB channel.
        Default is "Gaussian".
        - "Gaussian": Apply a Gaussian blur to the image.
        - "Mean": Replace each pixel with the local mean.
        - "Median": Replace each pixel with the local median.
        - "Mode": Replace each pixel with the local mode.
    Sigma: float
        The standard deviation to use if "Method" is "Gaussian". Ignored for others.
    Truncate: int
        The number of standard deviations used before the filter is truncated.
        Only used if "Method" is "Gaussian" and ignored for others.
    Radius: int
        The radius for the local kernel.
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"]
        How the algorithm should handle image borders. Only applies if "Method" is
        "Gaussian" or "Median". Ignored for others. Default is "Nearest".
        - "Constant": The border is extended with a value defined in "Constant Value".
        - "Mirror": The border is extended by reflecting about the center of the pixel.
        - "Nearest": The border is extended by copying the edge pixel.
        - "Reflect": The border is extended by reflecting about the edge of the pixel.
        - "Wrap": The border is extended by wrapping around to the opposite border.
    Constant_Value: float
        If "Edge Mode" is "Constant", the constant value to use. Default is 0.
    Perform_Slicewise: bool
        Whether or not to perform the operation slice-by-slice. Default is False.
    Preserve_Dtype: bool
        Whether or not to return the same data type that was input. Default is True.

    Returns
    -------
    smooth_array: np.ndarray
        The smoothed image.
    """

    def _smooth(
        imarray,
        Method,
        Sigma,
        Truncate,
        Radius,
        Edge_Mode,
        Constant_Value,
        Preserve_Dtype,
    ) -> np.ndarray:

        if util.is3d(imarray):
            fp = ball(Radius)
        else:
            fp = disk(Radius)
        chanax = util.chan_axis(imarray)

        if Method == "Median":
            if util.isrgb(imarray):
                smooth_array: np.ndarray = np.empty(imarray.shape, imarray.dtype)
                for cindex in range(0, imarray.shape[-1]):
                    smooth_array[..., cindex] = median(
                        imarray[..., cindex],
                        fp,
                        mode=Edge_Mode.lower(),
                        cval=Constant_Value,
                    )
                return smooth_array
            return median(imarray, fp, mode=Edge_Mode.lower(), cval=Constant_Value)

        elif Method == "Mean":
            if util.isrgb(imarray):
                smooth_array: np.ndarray = np.empty(imarray.shape, imarray.dtype)
                for cindex in range(0, imarray.shape[-1]):
                    smooth_array[..., cindex] = mean(imarray[..., cindex], fp)
                return smooth_array
            return mean(imarray, fp)

        elif Method == "Mode":
            if util.isrgb(imarray):
                smooth_array: np.ndarray = np.empty(imarray.shape, imarray.dtype)
                for cindex in range(0, imarray.shape[-1]):
                    smooth_array[..., cindex] = modal(imarray[..., cindex], fp)
                return smooth_array
            return modal(imarray, fp)

        else:
            smooth_array: np.ndarray = gaussian(
                imarray,
                Sigma,
                mode=Edge_Mode.lower(),
                cval=Constant_Value,
                truncate=Truncate,
                preserve_range=True,
                channel_axis=chanax,
            )
            if Preserve_Dtype:
                return smooth_array.astype(imarray.dtype)
            return smooth_array

    if Perform_Slicewise and util.is3d(imarray):
        if not Preserve_Dtype and Method == "Gaussian":
            smooth_array: np.ndarray = np.empty(imarray.shape, np.float64)
        else:
            smooth_array: np.ndarray = np.empty(imarray.shape, imarray.dtype)
        for index in range(0, imarray.shape[0]):
            smooth_array[index] = _smooth(
                imarray[index],
                Method,
                Sigma,
                Truncate,
                Radius,
                Edge_Mode,
                Constant_Value,
                Preserve_Dtype,
            )
        return smooth_array

    else:
        return _smooth(
            imarray,
            Method,
            Sigma,
            Truncate,
            Radius,
            Edge_Mode,
            Constant_Value,
            Preserve_Dtype,
        )

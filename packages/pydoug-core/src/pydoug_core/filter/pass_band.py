"""
Band-pass filters.
"""

from typing import Annotated, Literal

import numpy as np
from skimage.filters import butterworth, difference_of_gaussians, gabor

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["butter", "dog", "gab"]


@register_operation(category="filters", display_name="Butterworth")
def butter(
    imarray: np.ndarray,
    Cutoff_Frequency_Ratio: Annotated[
        float, {"min": 0, "max": 0.5, "step": 0.001}
    ] = 0.005,
    Order: Annotated[float, {"min": 0}] = 2,
    High_Pass: bool = True,
    Squared: bool = True,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Filter an image with a butterworth band-pass filter. Good for enhancing high or low
    frequency features. Can filter 2D/3D gray/RGB images. Dtype is not preserved.
    Convenience wrapper around scikit-image's filters.butterworth:
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.butterworth

    Parameters
    ----------
    imarray: np.ndarray
        The image to filter.
    Cutoff_Frequency_Ratio: float
        The cutoff ratio if "Method" is "Butterworth" in the range [0, 0.5]. Ignored
        for other methods.
    Order: float
        Affects the slope near the cutoff frequency in the frequency domain. Higher =
        steeper slope.
    High_Pass: bool
        Whether to perform a high-pass (True) filter or low-pass (False) filter.
        Default is True.
    Squared: bool
        Whether to perform a squared (True) or the square root (False) filter. Default
        is True.
    Perform_Slicewise: bool
        Whether or not to perform the operation slice-by-slice. Default is False.

    Returns
    -------
    bparray: np.ndarray
        The band-pass filtered image.
    """

    if Perform_Slicewise and util.is3d(imarray):
        bparray: np.ndarray = np.empty(imarray.shape)
        for index in range(0, imarray.shape[0]):
            bparray[index] = butterworth(
                imarray[index],
                Cutoff_Frequency_Ratio,
                High_Pass,
                Order,
                util.chan_axis(imarray[index]),
                squared_butterworth=Squared,
            )

        return bparray

    return butterworth(
        imarray,
        Cutoff_Frequency_Ratio,
        High_Pass,
        Order,
        util.chan_axis(imarray),
        squared_butterworth=Squared,
    )  # type: ignore


@register_operation(category="filters", display_name="Difference of Gaussians")
def dog(
    imarray: np.ndarray,
    Sigmas: tuple[Annotated[float, {"min": 0}], Annotated[float, {"min": 0}]] = (2, 5),
    Truncate: Annotated[int, {"min": 1}] = 4,
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"] = "Nearest",
    Constant_Value: Annotated[int, {"min": -32768, "max": 65535}] = 0,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Filter an image with a difference of Gaussians band-pass filter. Blurs the image
    with two Gaussian blurs then subtracts the more-blurred from the less-blurred to
    remove low and high frequency components. Dtype is not preserved.
    Convenience wrapper around scikit-image's filters.difference_of_gaussians:
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.difference_of_gaussians

    Parameters
    ----------
    imarray: np.ndarray
        The image to filter.
    Sigmas: tuple[float, float]
        The standard deviations to use for the Gaussian blurs.
    Truncate: int
        The number of standard deviations beyond which the filter is truncated.
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"]
        How the algorithm should handle image borders. Default is "Nearest".
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
    bparray: np.ndarray
        The band-pass filtered image.
    """

    if Perform_Slicewise and util.is3d(imarray):
        bparray: np.ndarray = np.empty(imarray.shape)
        for index in range(0, imarray.shape[0]):
            bparray[index] = difference_of_gaussians(
                imarray[index],
                min(Sigmas),
                max(Sigmas),
                mode=Edge_Mode.lower(),
                cval=Constant_Value,
                channel_axis=util.chan_axis(imarray[index]),
                truncate=Truncate,
            )

        return bparray

    return difference_of_gaussians(
        imarray,
        min(Sigmas),
        max(Sigmas),
        mode=Edge_Mode.lower(),
        cval=Constant_Value,
        channel_axis=util.chan_axis(imarray),
        truncate=Truncate,
    )


@register_operation(category="filters", display_name="Gabor")
def gab(
    imarray: np.ndarray,
    Frequency: Annotated[float, {"min": 0}] = 0.1,
    Theta: Annotated[float, {"min": 0, "max": 6.28}] = 0,
    Bandwidth: Annotated[int, {"min": 0}] = 1,
    Sigma_Count: Annotated[int, {"min": 1}] = 3,
    Offset: Annotated[float, {"min": 0, "max": 6.28}] = 0,
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"] = "Nearest",
    Constant_Value: Annotated[int, {"min": -32768, "max": 65535}] = 0,
) -> np.ndarray:
    """
    Filter an image with a 1D gabor band-pass filter. Good for classifying textures.
    Performed slicewise for 3D images and channelwise for RGB images. Dtype is
    preserved. Convenience wrapper around scikit-image's filters.gabor:
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.gabor

    Parameters
    ----------
    imarray: np.ndarray
        The image to filter.
    Frequency: float
        The spatial frequency (in pixels) of the function. Higher = more sensitive to
        finer details.
    Theta: float
        The orientation of the 1D filter in radians [0, 2*pi].
    Bandwidth: float
        Bandwidth captured by frequency.
    Sigma_Count: int
        The linear size of the kernel.
    Offset: float
        The phase offset of the function in radians [0, 2*pi].
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"]
        How the algorithm should handle image borders. Default is "Nearest".
        - "Constant": The border is extended with a value defined in "Constant Value".
        - "Mirror": The border is extended by reflecting about the center of the pixel.
        - "Nearest": The border is extended by copying the edge pixel.
        - "Reflect": The border is extended by reflecting about the edge of the pixel.
        - "Wrap": The border is extended by wrapping around to the opposite border.
    Constant_Value: float
        If "Edge Mode" is "Constant", the constant value to use. Default is 0.

    Returns
    -------
    bparray: np.ndarray
        The real result of the band-pass filtered image.
    """

    if util.isrgb(imarray):
        bparray: np.ndarray = np.empty(imarray.shape, imarray.dtype)
        for cindex in range(0, imarray.shape[-1]):
            if util.is3d(imarray):
                for index in range(0, imarray.shape[0]):
                    bparray[index, :, :, cindex], _ = gabor(
                        imarray[index, :, :, cindex],
                        Frequency,
                        Theta,  # type: ignore
                        Bandwidth,
                        n_stds=Sigma_Count,
                        offset=Offset,  # type: ignore
                        mode=Edge_Mode.lower(),
                        cval=Constant_Value,
                    )

            else:
                bparray[..., cindex], _ = gabor(
                    imarray[..., cindex],
                    Frequency,
                    Theta,  # type: ignore
                    Bandwidth,
                    n_stds=Sigma_Count,
                    offset=Offset,  # type: ignore
                    mode=Edge_Mode.lower(),
                    cval=Constant_Value,
                )

        return bparray

    if util.is3d(imarray):
        bparray: np.ndarray = np.empty(imarray.shape, imarray.dtype)
        for index in range(0, imarray.shape[0]):
            bparray[index], _ = gabor(
                imarray[index],
                Frequency,
                Theta,  # type: ignore
                Bandwidth,
                n_stds=Sigma_Count,
                offset=Offset,  # type: ignore
                mode=Edge_Mode.lower(),
                cval=Constant_Value,
            )

        return bparray

    bparray, _ = gabor(
        imarray,
        Frequency,
        Theta,  # type: ignore
        Bandwidth,
        n_stds=Sigma_Count,
        offset=Offset,  # type: ignore
        mode=Edge_Mode.lower(),
        cval=Constant_Value,
    )

    return bparray

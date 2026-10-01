"""
Denoising images.
"""

from math import ceil
from typing import Annotated, Literal

import numpy as np
import pywt
from skimage.filters.rank import mean_bilateral as rank_bilateral
from skimage.morphology import disk
from skimage.restoration import (
    denoise_bilateral,
    denoise_nl_means,
    denoise_tv_bregman,
    denoise_tv_chambolle,
    denoise_wavelet,
)

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["bilateral", "nl_means", "tv_bregman", "tv_chambolle", "wavelet"]


@register_operation(category="denoise", display_name="Bilateral Filter")
def bilateral(
    imarray: np.ndarray,
    Window_Size: Annotated[int, {"min": 0}] = 0,
    Sigma_Color: Annotated[float, {"min": 0}] = 0,
    Sigma_Spatial: Annotated[float, {"min": 0}] = 1,
    Bin_Count: Annotated[int, {"min": 1}] = 10000,
    Edge_Mode: Literal["Constant", "Edge", "Reflect", "Symmetric", "Wrap"] = "Edge",
    Constant_Value: Annotated[int, {"min": -32768, "max": 65535}] = 0,
    Rank: bool = False,
    Rank_Interval: tuple[
        Annotated[int, {"min": 0}],
        Annotated[int, {"min": 0}],
    ] = (10, 10),
    Preserve_Dtype: bool = True,
) -> np.ndarray:
    """
    Denoise an image with a bilateral filter. Applies slicewise along the Z axis to 3D
    images. Can denoise RGB images, but does so along the channel axis if "Rank" is
    checked. Convenience wrapper around scikit-image's restoration.denoise_bilateral and
    filters.rank.mean_bilateral:
    https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_bilateral
    https://scikit-image.org/docs/stable/api/skimage.filters.rank.html#skimage.filters.rank.mean_bilateral

    Parameters
    ----------
    imarray: np.ndarray
        The image to be denoised.
    Window_Size: int
        Window size for the filter. If 0, it is automatically calculated.
    Sigma_Color: float
        The intensity standard deviation. Higher = greater smoothing. If 0, the standard
        deviation of the image is used.
    Sigma_Spatial: float
        The distance standard deviation. Higher = greater smoothing.
    Bin_Count: int
        Number of values for Gaussian weights. Higher = improved accuracy.
    Edge_Mode: Literal["Constant", "Edge", "Reflect", "Symmetric", "Wrap"]
        How the algorithm should handle image borders. Default is "Edge".
        - "Constant": The border is extended with a value defined in "Constant Value".
        - "Edge": The border is extended by copying the edge pixel.
        - "Reflect": The border is extended by mirroring about the edge of the image.
        - "Symmetric": The border is extended by reflecting about the edge of the image.
        - "Wrap": The border is extended by wrapping around to the opposite border.
    Constant_Value: int
        If "Edge Mode" is "Constant", the constant value to use. Default is 0.
    Rank: bool
        Whether or not to appply a rank bilateral filter. Ignores all other parameters
        except "Window Size" and "Rank Thresholds" if True. Image must be grayscale.
        Default is False.
    Rank_Interval: tuple
        The lower and upper interval around the pixel's intensity that can be average.
        Only applies if "Rank" is checked.
    Preserve_Dtype: bool
        Whether or not to return the same data type that was input. "Rank" automatically
        preserves the dtype. Default is True.

    Returns
    -------
    dnarray: np.ndarray
        The denoised image.
    """

    output_dtype: np.dtype = util.output_dtype(imarray, preserve_dtype=Preserve_Dtype)
    output_dtype_str: str = util.dtype_2_str(output_dtype)
    if Rank:
        if Window_Size == 0:
            Window_Size = max(5, 2 * ceil(3 * Sigma_Spatial) + 1)

    if util.is3d(imarray):
        if Rank:
            dn_array: np.ndarray = np.empty(imarray.shape, dtype=imarray.dtype)
            fp = disk(Window_Size)
            if util.isrgb(imarray):
                for index in range(0, imarray.shape[0]):
                    for cindex in range(0, imarray.shape[-1]):
                        dn_array[index, :, :, cindex] = rank_bilateral(
                            imarray[index, :, :, cindex],
                            fp,
                            s0=Rank_Interval[0],
                            s1=Rank_Interval[1],
                        )
            else:
                for index in range(0, imarray.shape[0]):
                    dn_array[index] = rank_bilateral(
                        imarray[index], fp, s0=Rank_Interval[0], s1=Rank_Interval[1]
                    )

        else:
            dn_array: np.ndarray = np.empty(imarray.shape, dtype=output_dtype)
            chanax: int | None = util.chan_axis(imarray[0])
            for index in range(0, imarray.shape[0]):
                dn_array[index] = util.convert_dtype(
                    denoise_bilateral(
                        imarray[index],
                        Window_Size,
                        Sigma_Color,
                        Sigma_Spatial,
                        Bin_Count,
                        Edge_Mode.lower(),
                        Constant_Value,
                        channel_axis=chanax,
                    ),
                    output_dtype_str,  # type: ignore
                )

    else:
        if Rank:
            fp = disk(Window_Size)
            if util.isrgb(imarray):
                dn_array: np.ndarray = np.empty(imarray.shape, dtype=imarray.dtype)
                for cindex in range(0, imarray.shape[-1]):
                    dn_array[:, :, cindex] = rank_bilateral(
                        imarray[:, :, cindex],
                        fp,
                        s0=Rank_Interval[0],
                        s1=Rank_Interval[1],
                    )
            else:
                dn_array = rank_bilateral(
                    imarray, fp, s0=Rank_Interval[0], s1=Rank_Interval[1]
                )

        else:
            chanax: int | None = util.chan_axis(imarray)
            dn_array = util.convert_dtype(
                denoise_bilateral(
                    imarray,
                    Window_Size,
                    Sigma_Color,
                    Sigma_Spatial,
                    Bin_Count,
                    Edge_Mode.lower(),
                    Constant_Value,
                    channel_axis=chanax,
                ),
                output_dtype_str,  # type: ignore
            )

    return dn_array


@register_operation(category="denoise", display_name="Non-Local Means Filter")
def nl_means(
    imarray: np.ndarray,
    Patch_Size: Annotated[int, {"min": 1}] = 7,
    Patch_Distance: Annotated[int, {"min": 1}] = 11,
    Cut_Off_Distance: Annotated[float, {"min": 0}] = 0.1,
    Sigma: Annotated[float, {"min": 0}] = 0,
    Fast_Mode: bool = True,
    Slicewise: bool = False,
    Preserve_Dtype: bool = True,
) -> np.ndarray:
    """
    Denoise an image with a non-local means algorithm. Slowest of the denoising methods.
    Can denoise RGB images. Convenience wrapper around scikit-image's
    restoration.denoise_nl_means:
    https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_nl_means

    Parameters
    ----------
    imarray: np.ndarray
        The image to denoise.
    Patch_Size: int
        The size of patches used for denoising.
    Patch_Distance: int
        The max distance away from the pixel to search for patches.
    Cut_Off_Distance: float
        Higher = more smoothing.
    Sigma: float
        The intensity standard deviation. If 0, a less robust algorithm is used.
    Fast_Mode: bool
        Whether or not to use a faster approximation for the calculation. Less accurate.
        Deafult is True (recommended).
    Slicewise: bool
        Whether or not to apply the filter slicewise along the Z axis. No effect for 2D
        images.
    Preserve_Dtype: bool
        Whether or not to return the same data type that was input. Default is True.

    Returns
    -------
    dnarray: np.ndarray
        The denoised image.
    """

    output_dtype: np.dtype = util.output_dtype(imarray, preserve_dtype=Preserve_Dtype)

    if Slicewise and util.is3d(imarray):
        chanax: int | None = util.chan_axis(imarray[0])
        dnarray: np.ndarray = np.empty(imarray.shape, output_dtype)
        for index in range(0, imarray.shape[0]):
            dnarray[index] = denoise_nl_means(
                imarray[index],
                Patch_Size,
                Patch_Distance,
                Cut_Off_Distance,
                Fast_Mode,
                Sigma,
                preserve_range=True,
                channel_axis=chanax,
            ).astype(output_dtype, copy=False)

    else:
        chanax: int | None = util.chan_axis(imarray)
        dnarray = denoise_nl_means(
            imarray,
            Patch_Size,
            Patch_Distance,
            Cut_Off_Distance,
            Fast_Mode,
            Sigma,
            preserve_range=True,
            channel_axis=chanax,
        ).astype(output_dtype, copy=False)

    return dnarray


@register_operation(category="denoise", display_name="TV Bregman Filter")
def tv_bregman(
    imarray: np.ndarray,
    Weight: Annotated[float, {"min": 0}] = 5,
    Epsilon: Annotated[float, {"min": 0, "step": 0.001}] = 0.001,
    Maximum_Iterations: Annotated[int, {"min": 1}] = 100,
    Isotropic: bool = True,
    Slicewise: bool = False,
    Preserve_Dtype: bool = True,
) -> np.ndarray:
    """
    Denoise an image with the Bregman total variation filter. Can denoise RGB images,
    but 3D RGB images are processed slicewise along the Z axis. Convenience wrapper
    around scikit-image's restoration.denoise_tv_bregman:
    https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_tv_bregman

    Parameters
    ----------
    imarray: np.ndarray
        The image to denoise.
    Weight: float
        The strength of the denoising. Smaller = more smoothing.
    Epsilon: float
        The tolerance for the stop criterion. Smaller = tighter restriction.
    Maximum_Iterations: int
        The maximum number of iterations for optimization.
    Isotropic: bool
        Whether or not to perform isotropic denoising. Default is True.
    Slicewise: bool
        Whether or not to apply the filter slicewise along the Z axis. No effect for 2D
        images.
    Preserve_Dtype: bool
        Whether or not to return the same data type that was input. Default is True.

    Returns
    -------
    dnarray: np.ndarray
        The denoised image.
    """

    output_dtype: np.dtype = util.output_dtype(imarray, preserve_dtype=Preserve_Dtype)
    output_dtype_str: str = util.dtype_2_str(output_dtype)

    if util.is3d(imarray) and (Slicewise or util.isrgb(imarray)):
        chanax: int | None = util.chan_axis(imarray[0])
        dnarray: np.ndarray = np.empty(imarray.shape, output_dtype)
        for index in range(0, imarray.shape[0]):
            dnarray[index] = util.convert_dtype(
                denoise_tv_bregman(
                    imarray[index],
                    Weight,
                    Maximum_Iterations,
                    Epsilon,
                    Isotropic,
                    channel_axis=chanax,
                ),
                output_dtype_str,  # type: ignore
            )

    else:
        chanax: int | None = util.chan_axis(imarray)
        dnarray = util.convert_dtype(
            denoise_tv_bregman(
                imarray,
                Weight,
                Maximum_Iterations,
                Epsilon,
                Isotropic,
                channel_axis=chanax,
            ),
            output_dtype_str,  # type: ignore
        )

    return dnarray


@register_operation(category="denoise", display_name="TV Chambolle Filter")
def tv_chambolle(
    imarray: np.ndarray,
    Weight: Annotated[float, {"min": 0}] = 0.1,
    Epsilon: Annotated[float, {"min": 0, "step": 0.0001}] = 0.0002,
    Maximum_Iterations: Annotated[int, {"min": 1}] = 200,
    Slicewise: bool = False,
    Preserve_Dtype: bool = True,
) -> np.ndarray:
    """
    Denoise an image with the Chambolle total variation filter. Can denoise RGB images,
    but 3D RGB images are processed slicewise along the Z axis. Convenience wrapper
    around scikit-image's restoration.denoise_tv_chambolle:
    https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_tv_chambolle

    Parameters
    ----------
    imarray: np.ndarray
        The image to denoise.
    Weight: float
        The strength of the denoising. Smaller = more smoothing.
    Epsilon: float
        The tolerance for the stop criterion. Smaller = tighter restriction.
    Maximum_Iterations: int
        The maximum number of iterations for optimization.
    Isotropic: bool
        Whether or not to perform isotropic denoising. Default is True.
    Slicewise: bool
        Whether or not to apply the filter slicewise along the Z axis. No effect for 2D
        images. 3D RGB images are automatically processed slicewise.
    Preserve_Dtype: bool
        Whether or not to return the same data type that was input. Default is True.

    Returns
    -------
    dnarray: np.ndarray
        The denoised image.
    """

    output_dtype: np.dtype = util.output_dtype(imarray, preserve_dtype=Preserve_Dtype)
    output_dtype_str: str = util.dtype_2_str(output_dtype)

    if util.is3d(imarray) and (Slicewise or util.isrgb(imarray)):
        chanax: int | None = util.chan_axis(imarray[0])
        dnarray: np.ndarray = np.empty(imarray.shape, output_dtype)
        for index in range(0, imarray.shape[0]):
            dnarray[index] = util.convert_dtype(
                denoise_tv_chambolle(
                    imarray[index],
                    Weight,
                    Epsilon,
                    Maximum_Iterations,
                    channel_axis=chanax,
                ),
                output_dtype_str,  # type: ignore
            )

    else:
        chanax: int | None = util.chan_axis(imarray)
        dnarray = util.convert_dtype(
            denoise_tv_chambolle(
                imarray,
                Weight,
                Epsilon,
                Maximum_Iterations,
                channel_axis=chanax,
            ),
            output_dtype_str,  # type: ignore
        )

    return dnarray


@register_operation(
    category="denoise",
    display_name="Wavelet Filter",
    widget_options={"Wavelet": {"choices": pywt.wavelist()}},  # type: ignore
)
def wavelet(
    imarray: np.ndarray,
    Wavelet: str = "db1",
    Sigma: Annotated[float, {"min": 0}] = 0,
    Wavelet_Levels: Annotated[int, {"min": 0}] = 0,
    Mode: Literal["Soft", "Hard"] = "Soft",
    Method: Literal["BayesShrink", "VisuShrink"] = "BayesShrink",
    Rescale_Sigma: bool = True,
    Slicewise: bool = False,
    Preserve_Dtype: bool = True,
) -> np.ndarray:
    """
    Denoise an image with a Wavelet filter. Can denoise RGB images, but 3D RGB images
    are processed slicewise on along the Z axis. Convenience wrapper around
    scikit-image's restoration.denoise_wavelet:
    https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_wavelet

    Parameters
    ----------
    imarray: np.ndarray
        The image to denoise.
    Wavelet: str
        The wavelet to use for denoising. Default is "db1".
    Sigma: float
        The standard deviation to use when finding the wavelet detail coefficient
        thresholds. If 0 is provided, this is calculated automatically.
    Wavelet_Levels: int
        The number of wavelet decomposition levels to use. Automatically calcualted if
        0 is provided.
    Mode: Literal["Soft", "Hard"]
        The type of denoising performed. Default is "Soft".
    Method: Literal["BayesShrink" "VisuShrink"]
        The thresholding method to use. Default is "BayesShrink".
    Rescale_Sigma: bool
        Whether or not to rescale the user-provided sigma if the image is rescaled
        during the operation. Default is True (recommended).
    Slicewise: bool
        Whether or not to apply the filter slicewise along the Z axis. No effect for 2D
        images. 3D RGB images are automatically processed slicewise.
    Preserve_Dtype: bool
        Whether or not to return the same data type that was input. Default is True.

    Returns
    -------
    dnarray: np.ndarray
        The denoised image.
    """

    output_dtype: np.dtype = util.output_dtype(imarray, preserve_dtype=Preserve_Dtype)
    output_dtype_str: str = util.dtype_2_str(output_dtype)

    if util.is3d(imarray) and (Slicewise or util.isrgb(imarray)):
        chanax: int | None = util.chan_axis(imarray[0])
        dnarray: np.ndarray = np.empty(imarray.shape, output_dtype)
        for index in range(0, imarray.shape[0]):
            dnarray[index] = util.convert_dtype(
                denoise_wavelet(
                    imarray[index],
                    Sigma,
                    Wavelet,
                    Mode.lower(),
                    Wavelet_Levels,
                    False,
                    Method,
                    Rescale_Sigma,
                    channel_axis=chanax,
                ),
                output_dtype_str,  # type: ignore
            )

    else:
        chanax: int | None = util.chan_axis(imarray)
        dnarray = util.convert_dtype(
            denoise_wavelet(
                imarray,
                Sigma,
                Wavelet,
                Mode.lower(),
                Wavelet_Levels,
                False,
                Method,
                Rescale_Sigma,
                channel_axis=chanax,
            ),
            output_dtype_str,  # type: ignore
        )

    return dnarray

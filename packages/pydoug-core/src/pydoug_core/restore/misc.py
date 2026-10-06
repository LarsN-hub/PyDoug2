"""
Miscellaneous restoration filters.
"""

from typing import Annotated, Literal

import numpy as np
import pywt
from algotom.post.postprocessing import (
    remove_ring_based_fft,
    remove_ring_based_wavelet_fft,
)
from skimage.restoration import inpaint_biharmonic, rolling_ball

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["rolling", "inpaint"]


@register_operation(category="restore", display_name="Rolling Ball")
def rolling(
    imarray: np.ndarray,
    Radius: Annotated[int, {"min": 1}] = 5,
    Return_Background: bool = False,
) -> np.ndarray:
    """
    Use a rolling ball algorithm to attempt to subtract the background of an image.
    The background must be darker than the foreground.

    Parameters
    ----------
    imarray: np.ndarray
        The image to perform the rolling ball algorithm on.
    Radius: int
        The radius of the ball.
    Return_Background: bool
        Whether to return the background image (True) or the image with the background
        subtracted (False). Default is False.

    Returns
    -------
    sub_array: np.ndarray
        The image with the background subtracted.
    """

    bg: np.ndarray = rolling_ball(imarray, radius=Radius)
    if Return_Background:
        return bg
    return imarray - bg


@register_operation(
    category="restore",
    display_name="Inpaint",
    array_params=("Mask",),
)
def inpaint(imarray: np.ndarray, Mask: np.ndarray) -> np.ndarray:
    """
    Inpaint masked values of an image. Good for removing small specs from dust on a
    camera lens, etc. Convenience wrapper around scikit-image's
    restore.inpaint_biharmonic:
    https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.inpaint_biharmonic

    Parameters
    ----------
    imarray: np.ndarray
        The image to be deconvoluted.
    Mask: np.ndarray
        The mask defining values to be inpainted. If "Mask" is not binary it will be
        converted before use.

    Returns
    -------
    inpaint_array: np.ndarray
        The inpainted image.
    """

    return inpaint_biharmonic(imarray, util.prep_mask(imarray, Mask)).astype(
        imarray.dtype
    )


@register_operation(
    category="restore",
    display_name="Ring Removal",
    widget_options={"Wavelet": {"choices": pywt.wavelist()}},  # type: ignore
)
def ring_removal(
    imarray: np.ndarray,
    Method: Literal["FFT", "Wavelet"] = "FFT",
    FFT_Cutoff_Frequency: Annotated[int, {"min": 0}] = 20,
    FFT_Order: Annotated[int, {"min": 1}] = 8,
    FFT_Row_Count: Annotated[int, {"min": 1}] = 1,
    Wavelet: str = "db9",
    Wavelet_Levels: Annotated[int, {"min": 0}] = 5,
    Wavelet_Size: Annotated[int, {"min": 1}] = 1,
    Sort: bool = False,
) -> np.ndarray:
    """
    Remove rings from reconstructed images. Works for 2D/3D grayscale only. Requires
    images to be square (2D) or square along the z-axis (3D). Performed slicewise if 3D.
    Will apply a circular mask to the input image (non-reversible).
    Convenience wrapper around algotom's post.postprocessing ring removal functions,
    including remove_ring_based_fft and remove_ring_based_wavelet_fft:
    https://myalgotomo.readthedocs.io/en/latest/api/algotom.post.postprocessing.html#algotom.post.postprocessing.remove_ring_based_fft
    https://myalgotomo.readthedocs.io/en/latest/api/algotom.post.postprocessing.html#algotom.post.postprocessing.remove_ring_based_wavelet_fft

    Parameters
    ----------
    imarray: np.ndarray
        The image to remove rings.
    Method: Literal["FFT", "Wavelet"]
        The method to remove rings with. Default is "FFT".
        - "FFT": Combines the polar transform with the fft-based method.
        - "Wavelet": Combines the polar transform with the wavelet-fft-based method.
    FFT_Cutoff_Frequency: int
        If "Method" is "FFT", the cutoff frequency to use for the filter.
    FFT_Order: int
        If "Method" is "FFT", the order of the filter.
    FFT_Row_Count: int
        If "Method" is "FFT", the number of rows to apply to the filter.
    Wavelet: str
        If "Method" is "Wavelet", the wavelet to use. Default is "db9".
    Wavelet_Levels: int
        If "Method" is "Wavelet", the wavelet decompostion levels to use.
    Wavelet_Size: int
        If "Method" is "Wavelet", the damping parameter of the wavelet.
    Sort: bool
        For both methods, whether or not to sort the sinogram (True) or not (False).
        Default is False. See: https://doi.org/10.1364/OE.26.028396 for more detail.

    Returns
    -------
    rrarray: np.ndarray
        The ring removed image.
    """

    if util.isrgb(imarray):
        print("Image is RGB, ring removal not performed!")
        return imarray

    if not util.is_square(imarray):
        print("Image(s) not square, ring removal not performed!")
        return imarray

    if util.is3d(imarray):
        rrarray: np.ndarray = np.empty(imarray.shape, imarray.dtype)
        for index in range(0, imarray.shape[0]):
            if Method == "FFT":
                imarray[index] = remove_ring_based_fft(
                    imarray[index], FFT_Cutoff_Frequency, FFT_Order, FFT_Row_Count, Sort
                ).astype(imarray.dtype)
            else:
                imarray[index] = remove_ring_based_wavelet_fft(
                    imarray[index], Wavelet_Levels, Wavelet_Size, Wavelet, Sort
                ).astype(imarray.dtype)

        return rrarray

    if Method == "FFT":
        return remove_ring_based_fft(
            imarray, FFT_Cutoff_Frequency, FFT_Order, FFT_Row_Count, Sort
        ).astype(imarray.dtype)

    return remove_ring_based_wavelet_fft(
        imarray, Wavelet_Levels, Wavelet_Size, Wavelet, Sort
    ).astype(imarray.dtype)

"""
Image registration.
"""

import numpy as np
from scipy.fft import fft2, fftn, fftshift
from skimage.filters import difference_of_gaussians, window
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar

from pydoug_core import util

__all__ = ["register"]


def register(srcarray: np.ndarray, destarray: np.ndarray, filter: bool = True):
    """
    Register srcarray with destarray.

    Parameters
    ----------
    srcarray: np.ndarray
        The source array.
    destarray: np.ndarray
        The destingation array.
    filter: bool
        Whether or not to apply a bandpass filter before registration. Default is True.

    Returns
    --------
    regdict: dict[str, np.ndarray]
        A dictionary containing the registered array and the transformation parameters.
    """

    if util.is3d(srcarray) and not util.is3d(destarray):
        return {"reg": srcarray}

    lengths: list = []
    for index, length in enumerate(srcarray.shape):
        if destarray.shape[index] >= length:
            curlength: int = length
        else:
            curlength: int = destarray.shape[index]
        lengths.append(curlength)
    if util.is3d(srcarray):
        srcarray = srcarray[: lengths[0], : lengths[1], : lengths[2]]
        destarray = destarray[: lengths[0], : lengths[1], : lengths[2]]
    else:
        srcarray = srcarray[: lengths[0], : lengths[1]]
        destarray = destarray[: lengths[0], : lengths[1]]

    def _logpolfftwin(imarray: np.ndarray) -> np.ndarray:
        if util.is3d(imarray):
            fftwin: np.ndarray = np.abs(
                fftshift(fftn(imarray * window("hann", imarray.shape)))
            )
        else:
            fftwin: np.ndarray = np.abs(
                fftshift(fft2(imarray * window("hann", imarray.shape)))
            )
        shape: tuple = fftwin.shape
        radius: float | int = shape[0] // 8
        if util.is3d(imarray):
            return warp_polar(
                fftwin, radius=radius, output_shape=shape, scaling="log", order=0
            )[:, :, shape[1] // 2]
        else:
            return warp_polar(
                fftwin, radius=radius, output_shape=shape, scaling="log", order=0
            )[:, shape[0] // 2]

    if filter:
        logpolsrc: np.ndarray = _logpolfftwin(difference_of_gaussians(srcarray, 5, 20))
        logpoldest: np.ndarray = _logpolfftwin(
            difference_of_gaussians(destarray, 5, 20)
        )
    else:
        logpolsrc: np.ndarray = _logpolfftwin(srcarray)
        logpoldest: np.ndarray = _logpolfftwin(destarray)
    shifts, error, phasediff = phase_cross_correlation(
        logpolsrc, logpoldest, upsample_factor=10
    )

    return shifts, error, phasediff

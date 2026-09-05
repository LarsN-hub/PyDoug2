"""
Domain transformations of images.
"""

import numpy as np
from scipy import fft as ft
from skimage import transform

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["radon", "invradon", "fft", "invfft"]


@register_operation(category="transformations", display_name="Radon")
def radon(imarray: np.ndarray) -> np.ndarray:
    """
    Transform a spatial domain image into the projection domain.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be transformed.

    Returns
    -------
    radarray: np.ndarray
        The image array transformed into the projection domain.
    """

    imarray = util.rgb2gray(imarray)
    if util.is3d(imarray):
        firstrad: np.ndarray = transform.radon(imarray[0], circle=False)
        radarray: np.ndarray = np.empty(
            (imarray.shape[0], firstrad.shape[0], firstrad.shape[1]), firstrad.dtype
        )
        radarray[0] = firstrad
        for index in range(1, imarray.shape[0]):
            radarray[index] = transform.radon(imarray[index], circle=False)
        return radarray
    else:
        return transform.radon(imarray, circle=False)


@register_operation(category="transformations", display_name="Inverse Radon")
def invradon(imarray: np.ndarray) -> np.ndarray:
    """
    Transform a projection domain image into the spatial domain.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be transformed.

    Returns
    -------
    iradarray: np.ndarray
        The image array transformed into the spatial domain.
    """

    imarray = util.rgb2gray(imarray)
    if util.is3d(imarray):
        firstirad: np.ndarray = transform.iradon(imarray[0], circle=False)
        iradarray: np.ndarray = np.empty(
            (imarray.shape[0], firstirad.shape[0], firstirad.shape[1]), firstirad.dtype
        )
        iradarray[0] = firstirad
        for index in range(1, imarray.shape[0]):
            iradarray[index] = transform.iradon(imarray[index], circle=False)
        return iradarray
    else:
        return transform.iradon(imarray, circle=False)


@register_operation(category="transformations", display_name="FFT")
def fft(imarray: np.ndarray, Slicewise: bool = False) -> np.ndarray:
    """
    Transform a spatial domain image into the frequency domain.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be transformed.
    Slicewise: bool
        Whether to calcualte the fft of a 3D array as 2D slicewise or 3D total.
        Default is False (calculates 3D fft of 3D array).

    Returns
    -------
    fftarray: np.ndarray
        The image array transformed into the frequency domain.
    """

    imarray = util.rgb2gray(imarray)
    if util.is3d(imarray):
        if Slicewise:
            firstfft: np.ndarray = np.abs(ft.fftshift(ft.fft2(imarray[0])))
            fftarray: np.ndarray = np.empty(
                (imarray.shape[0], firstfft.shape[0], firstfft.shape[1]), firstfft.dtype
            )
            fftarray[0] = firstfft
            for index in range(1, imarray.shape[0]):
                fftarray[index] = np.abs(ft.fftshift(ft.fft2(imarray[index])))
            return fftarray
        return np.abs(ft.fftshift(ft.fftn(imarray)))
    return np.abs(ft.fftshift(ft.fft2(imarray)))


@register_operation(category="transformations", display_name="Inverse FFT")
def invfft(imarray: np.ndarray, Slicewise: bool = False) -> np.ndarray:
    """
    Transform a frequency domain image into the spatial domain.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be transformed.
    Slicewise: bool
        Whether to calcualte the inverse fft of a 3D array as 2D slicewise or 3D total.
        Default is False (calculates 3D inverse fft of 3D array).

    Returns
    -------
    ifftarray: np.ndarray
        The image array transformed into the spatial domain.
    """

    imarray = util.rgb2gray(imarray)
    if util.is3d(imarray):
        if Slicewise:
            firstifft: np.ndarray = np.abs(ft.ifftshift(ft.ifft2(imarray[0])))
            ifftarray: np.ndarray = np.empty(
                (imarray.shape[0], firstifft.shape[0], firstifft.shape[1]),
                firstifft.dtype,
            )
            ifftarray[0] = firstifft
            for index in range(1, imarray.shape[0]):
                ifftarray[index] = np.abs(ft.ifftshift(ft.ifft2(imarray[index])))
            return ifftarray
        return np.abs(ft.ifftshift(ft.ifftn(imarray)))
    return np.abs(ft.ifftshift(ft.ifft2(imarray)))


# @register_operation(category="transformations", display_name="Hough")
def hough(
    imarray: np.ndarray,
    Mode: str = "line",
    Radii: np.ndarray | None = None,
    Full_Output: bool = False,
) -> np.ndarray | tuple:
    """
    Transform an image from the spatial domain to the accumulator domain.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be transformed.
    Mode: str
        The mode to accumulate votes. Default is "line" but can also be "circle".
    Radii: np.ndarray | None
        The radii to be tested if mode is "circle". Auto-filled if None.
    Full_Output: bool
        Whether or not to provide the full output of the hough transform functions
        from skimage. Default is False (returns just the transformed array).

    Returns
    -------
    harray: np.ndarray
        The image array transformed into the accumulator domain if fullout is False or
        mode is "circle".
    (harray, angles, distances): tuple[
        np.ndarray,
        list[np.ndarray] | np.ndrarray,
        list[np.ndarray] | np.ndarray
    ]
        If fullout is True and mode is "line", returns a tuple containing the
        transformed image array, the measured angles (single array for 2D, list of
        arrays for 3D), and the measured distances (single array for 2D, list of arrays
        for 3D).
    """

    imarray = util.rgb2gray(imarray)
    if Mode == "circle":
        if not np.any(Radii):
            Radii = np.arange(min(5, min(imarray.shape) / 2), min(imarray.shape) / 2, 2)
        if util.is3d(imarray):
            firstarray = transform.hough_circle(imarray[0], Radii)
            harray: np.ndarray = np.empty(
                (
                    firstarray.shape[0],
                    firstarray.shape[1],
                    firstarray.shape[2],
                    imarray.shape[0],
                ),
                firstarray.dtype,
            )
            harray[:, :, :, 0] = firstarray
            for index in range(1, imarray.shape[0]):
                harray[:, :, :, index] = transform.hough_circle(imarray[index], Radii)
            return harray
        return transform.hough_circle(imarray, Radii)
    else:
        if util.is3d(imarray):
            firstarray, firstangles, firstdist = transform.hough_line(imarray[0])
            harray: np.ndarray = np.empty(
                (imarray.shape[0], firstarray.shape[0], firstarray.shape[1]),
                firstarray.dtype,
            )
            harray[0] = firstarray
            angles: list = [firstangles]
            dist: list = [firstdist]
            for index in range(1, imarray.shape[0]):
                harray[index], curangles, curdist = transform.hough_line(imarray[index])
                angles.append(curangles)
                dist.append(curdist)
            if not Full_Output:
                return harray
            return (harray, angles, dist)
        if not Full_Output:
            return transform.hough_line(imarray)[0]
        return transform.hough_line(imarray)

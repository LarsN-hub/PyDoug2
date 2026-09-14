"""
Domain transformations of images.
"""

from typing import Annotated

import numpy as np
from scipy import fft as ft
from scipy.ndimage import distance_transform_edt as edt
from skimage import transform

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["radon", "invradon", "fft", "invfft", "euclidian_dt"]


@register_operation(
    category="transformations",
    display_name="Distance",
)
def euclidian_dt(
    imarray: np.ndarray, Pixel_Size: Annotated[float, {"step": 0.0001, "min": 0}] = 1
) -> np.ndarray:
    """
    Transform a binary image into the distance domain by calculating the Euclidian
    distance from each foreground point to the nearest background point. Convenience
    wrapper around scipy's ndimage.distance_transform_edt:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be transformed.
    Pixel_Size: float
        The real-world length of a single pixel edge.

    Returns
    -------
    edtarray: np.ndarray
        The image array transformed into the distance domain.
    """

    if util.isrgb(imarray):
        return edt(util.rgb2gray(imarray), Pixel_Size)  # type: ignore
    return edt(imarray, Pixel_Size)  # type: ignore


@register_operation(category="transformations", display_name="FFT")
def fft(imarray: np.ndarray, Slicewise: bool = False) -> np.ndarray:
    """
    Transform a spatial domain image into the frequency domain. Convenience wrapper
    around several functions from scipy's fft package (fft2, fftn, and fftshift). Note
    Only returns the real component of the transform (using numpy's absolute):
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft2.html#scipy.fft.fft2
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fftn.html#scipy.fft.fftn
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fftshift.html#scipy.fft.fftshift
    https://numpy.org/doc/stable/reference/generated/numpy.absolute.html

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
    Transform a frequency domain image into the spatial domain. Convenience wrapper
    around several functions from scipy's fft package (ifft2, ifftn, and ifftshift).
    Only returns the real component of the transform (using numpy's absolute):
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ifft2.html#scipy.fft.ifft2
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ifftn.html#scipy.fft.ifftn
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ifftshift.html#scipy.fft.ifftshift
    https://numpy.org/doc/stable/reference/generated/numpy.absolute.html

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


@register_operation(category="transformations", display_name="Radon")
def radon(imarray: np.ndarray) -> np.ndarray:
    """
    Transform a spatial domain image into the projection domain (i.e., a sinogram) about
    a 180 degree rotation. Convenience wrapper around scikit-image's transform.radon:
    https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.radon

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
    Reconstruct a projection domain image (i.e., a sinogram) into the spatial domain.
    Convenience wrapper around scikit-image's transform.iradon:
    https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.iradon

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


# @register_operation(category="transformations", display_name="Hough")
def hough(
    imarray: np.ndarray,
    Mode: str = "line",
    Radii: np.ndarray | None = None,
    Full_Output: bool = False,
) -> np.ndarray | tuple:
    """
    Transform an image from the spatial domain to the accumulator domain. Convenience
    wrapper around hough_circle and hough_line from scikit-image's transform package:
    https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.hough_circle
    https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.hough_line

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

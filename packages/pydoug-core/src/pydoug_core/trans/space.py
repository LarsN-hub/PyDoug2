"""
Domain transformations of images.
"""

from typing import Annotated, Literal

import numpy as np

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["fft", "radon", "hough"]


@register_operation(category="transform", display_name="FFT")
def fft(
    imarray: np.ndarray, Inverse: bool = False, Perform_Slicewise: bool = False
) -> np.ndarray:
    """
    Transform a spatial domain image into the frequency domain and vice versa. Converts
    RGB to grayscale first. Only returns the real component of the transform (using
    numpy's absolute). Convenience wrapper around functions from scipy's fft package
    (fft2, ifft2, fftn, ifftn, fftshift, and ifftshift):
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft2.html#scipy.fft.fft2
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ifft2.html#scipy.fft.ifft2
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fftn.html#scipy.fft.fftn
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ifftn.html#scipy.fft.ifftn
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fftshift.html#scipy.fft.fftshift
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ifftshift.html#scipy.fft.ifftshift
    https://numpy.org/doc/stable/reference/generated/numpy.absolute.html

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be transformed.
    Inverse: bool
        Whether to perform an inverse FFT (True) or a forward FFT (False).
        Default is False.
    Perform_Slicewise: bool
        Whether to calcualte the fft of a 3D array as 2D slicewise or 3D total.
        Default is False (calculates 3D fft of 3D array).

    Returns
    -------
    fftarray: np.ndarray
        The transformed image array.
    """

    from scipy import fft as ft

    if Inverse:
        if util.is3d(imarray):
            if Perform_Slicewise:
                firstifft: np.ndarray = np.abs(
                    ft.ifftshift(ft.ifft2(util.rgb2gray(imarray[0])))
                )
                ifftarray: np.ndarray = np.empty(
                    (imarray.shape[0], firstifft.shape[0], firstifft.shape[1]),
                    firstifft.dtype,
                )
                ifftarray[0] = firstifft
                for index in range(1, imarray.shape[0]):
                    ifftarray[index] = np.abs(
                        ft.ifftshift(ft.ifft2(util.rgb2gray(imarray[index])))
                    )

                return ifftarray

            return np.abs(ft.ifftshift(ft.ifftn(util.rgb2gray(imarray))))

        return np.abs(ft.ifftshift(ft.ifft2(util.rgb2gray(imarray))))

    if util.is3d(imarray):
        if Perform_Slicewise:
            firstfft: np.ndarray = np.abs(
                ft.fftshift(ft.fft2(util.rgb2gray(imarray[0])))
            )
            fftarray: np.ndarray = np.empty(
                (imarray.shape[0], firstfft.shape[0], firstfft.shape[1]), firstfft.dtype
            )
            fftarray[0] = firstfft
            for index in range(1, imarray.shape[0]):
                fftarray[index] = np.abs(
                    ft.fftshift(ft.fft2(util.rgb2gray(imarray[index])))
                )

            return fftarray

        return np.abs(ft.fftshift(ft.fftn(util.rgb2gray(imarray))))

    return np.abs(ft.fftshift(ft.fft2(util.rgb2gray(imarray))))


@register_operation(category="transform", display_name="Radon")
def radon(imarray: np.ndarray, Inverse: bool = False) -> np.ndarray:
    """
    Transform a spatial domain image into the projection domain (i.e., a sinogram) about
    a 180 degree rotation. Converts RGB to grayscale first. Performed slicewise for 3D.
    Convenience wrapper around scikit-image's transform.radon:
    https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.radon

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be transformed.
    Inverse: bool
        Whether to perform an inverse (True) or forward (False) Radon transform.
        Default is False.

    Returns
    -------
    radarray: np.ndarray
        The transformed image array.
    """

    from skimage.transform import iradon
    from skimage.transform import radon as skiradon

    if Inverse:
        if util.is3d(imarray):
            firstirad: np.ndarray = iradon(util.rgb2gray(imarray[0]), circle=False)
            iradarray: np.ndarray = np.empty(
                (imarray.shape[0], firstirad.shape[0], firstirad.shape[1]),
                firstirad.dtype,
            )
            iradarray[0] = firstirad
            for index in range(1, imarray.shape[0]):
                iradarray[index] = iradon(imarray[index], circle=False)

            return iradarray

        return iradon(imarray, circle=False)

    if util.is3d(imarray):
        firstrad: np.ndarray = skiradon(util.rgb2gray(imarray[0]), circle=False)
        radarray: np.ndarray = np.empty(
            (imarray.shape[0], firstrad.shape[0], firstrad.shape[1]), firstrad.dtype
        )
        radarray[0] = firstrad
        for index in range(1, imarray.shape[0]):
            radarray[index] = skiradon(util.rgb2gray(imarray[index]), circle=False)

        return radarray

    return skiradon(util.rgb2gray(imarray), circle=False)


@register_operation(category="transform", display_name="Hough")
def hough(
    imarray: np.ndarray,
    Mode: Literal["Circle", "Line"] = "Line",
    Angle_Step: Annotated[float, {"min": 0.1, "step": 0.1}] = 1,
    Radius: Annotated[int, {"min": 1}] = 3,
) -> np.ndarray:
    """
    Transform an image from the spatial domain to the accumulator domain. Performed
    slicewise for 3D. Convenience wrapper around hough_circle and hough_line from
    scikit-image's transform package:
    https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.hough_circle
    https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.hough_line

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be transformed.
    Mode: Literal["Circle", "Line"]
        The mode to accumulate votes. Default is "Line".
        - "Line": accumulate line responses.
        - "Circle": accumulate circle responses.
    Angle_Step: float
        If "Mode" is "Line", the angle step to use in the range [-pi/2, pi/2]. For
        example, 1 will test 180 angles and 0.5 will test 360 angles, both in the range
        [-pi/2, pi/2].
    Radius: int
        If "Mode" is "Circle", the radius to use for accumulating circle responses.

    Returns
    -------
    harray: np.ndarray
        The image array transformed into the accumulator domain.

    """

    from skimage.transform import hough_circle, hough_line

    if Mode == "Circle":
        if util.is3d(imarray):
            firstarray = hough_circle(util.rgb2gray(imarray[0]), Radius)
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
                harray[:, :, :, index] = hough_circle(
                    util.rgb2gray(imarray[index]), Radius
                )

            return harray

        return hough_circle(util.rgb2gray(imarray), Radius)

    else:
        angles: np.ndarray = np.linspace(
            -np.pi / 2, np.pi / 2, round(180 / Angle_Step), endpoint=True
        )
        if util.is3d(imarray):
            firstarray = hough_line(util.rgb2gray(imarray[0]), angles)[0]
            harray: np.ndarray = np.empty(
                (imarray.shape[0], firstarray.shape[0], firstarray.shape[1]),
                firstarray.dtype,
            )
            harray[0] = firstarray
            for index in range(1, imarray.shape[0]):
                harray[index] = hough_line(util.rgb2gray(imarray[index]), angles)[0]

            return harray

        return hough_line(util.rgb2gray(imarray), angles)[0]

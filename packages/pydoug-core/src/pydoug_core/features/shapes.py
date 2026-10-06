"""
Detect shapes in images like blobs, lines, and circles.
"""

from math import sqrt
from typing import Annotated, Literal

import numpy as np
from skimage.draw import line
from skimage.feature import blob_dog, blob_doh, blob_log
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line

from pydoug_core.registry import register_operation
from pydoug_core.util import is3d, isrgb

__all__ = ["blobs", "lines"]


@register_operation(category="features", display_name="Blobs")
def blobs(
    imarray: np.ndarray,
    Method: Literal["DoG", "DoH", "LoG"] = "DoH",
    Sigmas: tuple[
        Annotated[float, {"min": 0}],
        Annotated[float, {"min": 0}],
    ] = (1, 30),
    Absolute_Threshold: Annotated[float, {"min": -32768, "max": 65535}] = 0.1,
    Relative_Threshold: Annotated[float, {"min": 0, "max": 1}] = 0,
    Overlap: Annotated[float, {"min": 0, "max": 1}] = 0.5,
    Sigma_Ratio: Annotated[float, {"min": 0}] = 1.6,
    Sigma_Count: Annotated[int, {"min": 0}] = 10,
    Log_Scale: bool = False,
    Populate_with_Sizes: bool = False,
    Return_Diameters: bool = False,
    Pixel_Size: Annotated[float, {"min": 0, "step": 0.0001}] = 1,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Detect blobs in a grayscale image. Works for 2D/3D, though "DoH" is performed
    slicewise automatically for 3D. Detects a single point at the center of the blob
    and the blob's approximate radius. Can return as binary or populate with radius /
    diameter value of each blob. Convenience wrapper around the blob detection functions
    in scikit-image's feature package, including blob_dog, blob_doh, and blob_log:
    https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_dog
    https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_doh
    https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_log

    Parameters
    ----------
    imarray: np.ndarray
        The image to detect blobs in.
    Method: Literal["DoG", "DoH", "LoG"]
        The method for blob detection.
        - "DoG": Difference of Gaussians. Medium speed. Bright blobs only.
        - "DoH": Difference of Hessians. Fast but least accurate. Dark or light blobs.
        - "LoG": Laplace of Gaussians. Slowest but most accurate. Bright blobs only.
    Sigmas: tuple[float, float]
        The minimum and maximum Gaussian kernel standard deviations to use. Keep min
        value low for small blobs and max value high for large blobs.
    Absolute_Threshold: float | None
        The absolute minimum intensity of response peaks. If 0, will be calculated
        automatically.
    Relative_Threshold: float | None
        The relative minimum intensity of response peaks (found as this value * log
        image max). If 0, will be calculated automatically. Overwrites
        "Absolute_Threshold" if both are provided.
    Overlap: float
        If two blobs overlap more than provided "Overlap" fraction, the smaller blob is
        eliminated.
    Sigma_Ratio: float
        If "Method" is "DoG", the ratio between sigmas to use for finding the difference
        of Gaussians.
    Sigma_Count: int
        If "Method" is "DoH" or "LoG", the number of standard deviations to use for the
        Gaussian kernel.
    Log_Scale: bool
        If "Method" is "DoH" or "LoG", whether or not to use a log scale.
        Default is False.
    Populate_with_Sizes: bool
        Whether to assign the size of the detected blob to the returned points (True)
        or just return a binary image with True values indicating blobs (False).
        Default is False.
    Return_Diameters: bool
        If "Populate_with_Sizes" is True, whether to return the diameter (True) or the
        radius (False). Default is False.
    Pixel_Size: float
        If "Populate_with_Sizes" is True, the pixel edge length in real-world units.
    Perform_Slicewise: bool
        Whether or not to perform the operation slicewise. Default is False.

    Returns
    -------
    blob_array: np.ndarray
        The blob response image.
    """

    if isrgb(imarray):
        print("Image is RGB, blob identification not performed!")
        return imarray

    abs_thresh: float | None = Absolute_Threshold
    if abs_thresh == 0:
        abs_thresh = None
    rel_thresh: float | None = Relative_Threshold
    if rel_thresh == 0:
        rel_thresh = None
    else:
        abs_thresh = None

    def _blobs(
        imarray,
        Method,
        Sigmas,
        abs_thresh,
        rel_thresh,
        Overlap,
        Sigma_Ratio,
        Sigma_Count,
        Log_Scale,
        Populate_with_Sizes,
        Pixel_Size,
        Return_Diameters,
    ) -> np.ndarray:

        if Method == "DoG":
            blob_response: np.ndarray = blob_dog(
                imarray,
                min(Sigmas),
                max(Sigmas),
                Sigma_Ratio,
                abs_thresh,
                Overlap,
                threshold_rel=rel_thresh,
            )
            blob_response[:, -1] *= sqrt(2)

        elif Method == "DoH":
            blob_response: np.ndarray = blob_doh(
                imarray,
                min(Sigmas),
                max(Sigmas),
                Sigma_Count,
                abs_thresh,
                Overlap,
                Log_Scale,
                threshold_rel=rel_thresh,
            )

        else:
            blob_response: np.ndarray = blob_log(
                imarray,
                min(Sigmas),
                max(Sigmas),
                Sigma_Count,
                abs_thresh,
                Overlap,
                Log_Scale,
                threshold_rel=rel_thresh,
            )
            blob_response[:, -1] *= sqrt(2)

        if not Populate_with_Sizes:
            blob_array: np.ndarray = np.zeros(imarray.shape, np.bool)
            blob_array[tuple(blob_response[:, :-1].T.astype(np.int32))] = True
        else:
            blob_array: np.ndarray = np.zeros(imarray.shape, np.float64)
            if Return_Diameters:
                blob_response[:, -1] *= 2
            blob_response[:, -1] *= Pixel_Size
            blob_array[tuple(blob_response[:, :-1].T.astype(np.int32))] = tuple(
                blob_response.T
            )[-1]

        return blob_array

    if (Perform_Slicewise or Method == "DoH") and is3d(imarray):
        if Populate_with_Sizes:
            blob_array = np.empty(imarray.shape)
        else:
            blob_array = np.empty(imarray.shape, np.bool)
        for index in range(0, imarray.shape[0]):
            blob_array[index] = _blobs(
                imarray[index],
                Method,
                Sigmas,
                abs_thresh,
                rel_thresh,
                Overlap,
                Sigma_Ratio,
                Sigma_Count,
                Log_Scale,
                Populate_with_Sizes,
                Pixel_Size,
                Return_Diameters,
            )

        return blob_array

    return _blobs(
        imarray,
        Method,
        Sigmas,
        abs_thresh,
        rel_thresh,
        Overlap,
        Sigma_Ratio,
        Sigma_Count,
        Log_Scale,
        Populate_with_Sizes,
        Pixel_Size,
        Return_Diameters,
    )


@register_operation(category="features", display_name="Lines")
def lines(
    imarray: np.ndarray,
    Angle_Step: Annotated[float, {"min": 0.1, "step": 0.1}] = 1,
    Minimum_Distance: Annotated[int, {"min": 1}] = 9,
    Minimum_Angle: Annotated[int, {"min": 0, "max": 180}] = 10,
    Threshold: Annotated[float, {"min": 0}] = 0,
    Probabilistic: bool = False,
    Line_Length: Annotated[int, {"min": 1}] = 20,
    Line_Gap: Annotated[int, {"min": 1}] = 10,
) -> np.ndarray:
    """
    Detect lines in a grayscale image. Performed slicewise for 3D images. Convenience
    wrapper around hough_line and hough_line_peaks in scikit-image's transform package:
    https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.hough_line
    https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.hough_line_peaks

    Parameters
    ----------
    imarray: np.ndarray
        The image to detect lines in.
    Angle_Step: float
        The angle step to use in the range [-pi/2, pi/2]. For example, 1 will test 180
        angles and 0.5 will test 360 angles, both in the range [-pi/2, pi/2].
    Minimum_Distance: int
        The minimum distance between corner response peaks.
    Minimum_Angle: float
        The minimum angle separating lines.
    Threshold: float
        The minimum intensity of corner response peaks. If 0, will be calculated
        automatically.
    Probabilistic: bool
        Whether or not to perform a probabilistic line transform (True) or not (False).
        Probabilistic transform requires additional parameters but can identify line
        endpoints. Default is False.
    Line_Length: int
        If "Probabilistic" is True, the minimum length of detected lines. Higher = more
        lines.
    Line_Gap: int
        If "Probabilistic" is True, the maximum gap between pixels to form a line.
        Higher = longer lines.

    Returns
    -------
    line_array: np.ndarray
        The line response image.
    """

    if isrgb(imarray):
        print("Image is RGB, lines detection not performed!")
        return imarray

    if Threshold == 0:
        thresh: float | None = None
    else:
        thresh = Threshold

    angles: np.ndarray = np.linspace(
        -np.pi / 2, np.pi / 2, round(180 / Angle_Step), endpoint=True
    )

    def _lines(
        imarray,
        angles,
        Minimum_Distance,
        Minimum_Angle,
        thresh,
        Probabilistic,
        Line_Length,
        Line_Gap,
    ) -> np.ndarray:

        lines_array: np.ndarray = np.zeros(imarray.shape, np.bool)

        if Probabilistic:
            if thresh:
                thresh = round(thresh)
            else:
                thresh = 10
            lines: list = probabilistic_hough_line(
                imarray, thresh, Line_Length, Line_Gap, angles
            )
            for cur_line in lines:
                lines_array[
                    line(cur_line[0][1], cur_line[0][0], cur_line[1][1], cur_line[1][0])
                ] = True

        else:
            hough_response, tested_angles, distances = hough_line(imarray, angles)
            for _, angle, d in zip(
                *hough_line_peaks(
                    hough_response,
                    tested_angles,
                    distances,
                    Minimum_Distance,
                    Minimum_Angle,
                    thresh,
                )
            ):
                x0: float = d * np.cos(angle)
                y0: float = d * np.sin(angle)
                m: float = np.tan(angle + np.pi / 2)
                b: float = y0 - x0 * m
                x_int: float = -b / m
                if b > 0 and b < imarray.shape[0]:
                    y1: int = round(b)
                    x1: int = 0
                else:
                    y1: int = 0
                    x1: int = round(x_int)
                y_opp: float = m * (imarray.shape[1] - 1) + b
                x_opp: float = ((imarray.shape[0] - 1) - b) / m
                if y_opp > 0 and y_opp < imarray.shape[0]:
                    y2: int = round(y_opp)
                    x2: int = round((y_opp - b) / m)
                else:
                    y2: int = round(m * x_opp + b)
                    x2: int = round(x_opp)
                lines_array[line(y1, x1, y2, x2)] = True

        return lines_array

    if is3d(imarray):
        harray: np.ndarray = np.empty(imarray.shape, np.bool)
        for index in range(0, imarray.shape[0]):
            harray[index] = _lines(
                imarray[index],
                angles,
                Minimum_Distance,
                Minimum_Angle,
                thresh,
                Probabilistic,
                Line_Length,
                Line_Gap,
            )

        return harray

    return _lines(
        imarray,
        angles,
        Minimum_Distance,
        Minimum_Angle,
        thresh,
        Probabilistic,
        Line_Length,
        Line_Gap,
    )

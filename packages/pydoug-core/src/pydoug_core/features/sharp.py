"""
Detect features in images like corners, blobs, and edges.
"""

from typing import Annotated, Literal

import numpy as np
from skimage.feature import (
    canny,
    corner_fast,
    corner_harris,
    corner_kitchen_rosenfeld,
    corner_moravec,
    corner_peaks,
    corner_shi_tomasi,
    peak_local_max,
)
from skimage.morphology import ball, disk

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["edges", "peaks", "corners"]


@register_operation(category="features", display_name="Edges")
def edges(
    imarray: np.ndarray,
    Sigma: Annotated[float, {"min": 0}] = 1,
    Thresholds: tuple[
        Annotated[float, {"min": -32768, "max": 65535}],
        Annotated[float, {"min": -32768, "max": 65535}],
    ] = (0, 0),
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"] = "Nearest",
    Constant_Value: Annotated[int, {"min": -32768, "max": 65535}] = 0,
) -> np.ndarray:
    """
    Identify edges in an image with the Canny algorithm. Only works on grayscale images.
    Performed slicewise for 3D images. Convenience wrapper around scikit-image's
    feature.canny:
    https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.canny

    Parameters
    ----------
    imarray: np.ndarray
        The image to detect edges in.
    Sigma: float
        The standard deviation of the Gaussian blur applied to the image.
    Thresholds: tuple[float, float]
        The low and high thresholds applied to the hysteresis thresholding to identify
        the edges. Automatically calculated if set to 0.
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"]
        If "Method" is "Kitchen-Rosenfeld", how the algorithm should handle image
        borders. Default is "Nearest". Ignored for other methods.
        - "Constant": The border is extended with a value defined in "Constant Value".
        - "Mirror": The border is extended by reflecting about the center of the pixel.
        - "Nearest": The border is extended by copying the edge pixel.
        - "Reflect": The border is extended by reflecting about the edge of the pixel.
        - "Wrap": The border is extended by wrapping around to the opposite border.
    Constant_Value: float
        If "Edge Mode" is "Constant", the constant value to use. Default is 0.

    Returns
    -------
    edge_array: np.ndarray
        The image after filtering with the Canny edge algorithm.
    """

    if util.isrgb(imarray):
        print("Image is RGB, edge detection not performed!")
        return imarray

    if Thresholds[0] > Thresholds[1]:
        thresholds: tuple = (Thresholds[1], Thresholds[0])
    else:
        thresholds: tuple = (Thresholds[0], Thresholds[1])
    if thresholds[0] == 0:
        thresholds = (None, thresholds[1])
    if thresholds[1] == 0:
        thresholds = (thresholds[0], None)

    if util.is3d(imarray):
        edge_array: np.ndarray = np.empty(imarray.shape, np.bool)
        for index in range(0, imarray.shape[0]):
            edge_array[index] = canny(
                imarray[index],
                Sigma,
                thresholds[0],
                thresholds[1],
                mode=Edge_Mode.lower(),
                cval=Constant_Value,
            )

        return edge_array

    return canny(
        imarray,
        Sigma,
        thresholds[0],
        thresholds[1],
        mode=Edge_Mode.lower(),
        cval=Constant_Value,
    )


@register_operation(category="features", display_name="Peaks")
def peaks(
    imarray: np.ndarray,
    Radius: Annotated[int, {"min": 0}] = 0,
    Minimum_Distance: Annotated[int, {"min": 1}] = 1,
    Absolute_Threshold: Annotated[float, {"min": -32768, "max": 65535}] = 0,
    Relative_Threshold: Annotated[float, {"min": -32768, "max": 65535}] = 0,
    Perform_Slicewise: bool = False,
) -> np.ndarray:
    """
    Identify local intensity maxima in an image. Can handle 2D/3D grayscale/RGB images.
    Convenience wrapper around scikit-image's feature.peak_local_max:
    https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max

    Parameters
    ----------
    imarray: np.ndarray
        The image to detect peaks in.
    Radius: int
        (Optional) The radius of the local footprint to find local maxima within. If 0,
        automatically generated.
    Minimum_Distance: int
        The minimum distance between peaks.
    Absolute_Threshold: float
        The abolsute minimum intensity of peaks. If 0, will be calculated automaticall.
    Relative_Threshold: float
        The relative minimum intensity of peaks (found as this value * image max).
        If 0, will be calculated automatically. Overwrites "Absolute_Threshold" if both
        are provided.
    Perform_Slicewise: bool
        Whether or not to perform the operation slicewise. Default is False.

    Returns
    -------
    peak_array: np.ndarray
        A binary image where True values are local peaks in the original image.
    """

    abs_thresh: float | None = Absolute_Threshold
    if abs_thresh == 0:
        abs_thresh = None
    rel_thresh: float | None = Relative_Threshold
    if rel_thresh == 0:
        rel_thresh = None
    else:
        abs_thresh = None

    if Radius == 0:
        footprint = None
    else:
        if Perform_Slicewise or not util.is3d(imarray):
            footprint = disk(Radius)
        else:
            footprint = ball(Radius)

    if util.isrgb(imarray):
        peak_array: np.ndarray = np.zeros(imarray.shape[:-1], np.bool)
    else:
        peak_array: np.ndarray = np.zeros(imarray.shape, np.bool)
    if Perform_Slicewise and util.is3d(imarray):
        for index in range(0, imarray.shape[0]):
            peak_coords = peak_local_max(
                imarray[index],
                Minimum_Distance,
                abs_thresh,
                rel_thresh,
                footprint=footprint,
            )
            if util.isrgb(imarray):
                peak_coords = peak_coords[:, :-1]
            peak_array[index][tuple(peak_coords.T)] = True

    peak_coords: np.ndarray = peak_local_max(
        imarray, Minimum_Distance, abs_thresh, rel_thresh, footprint=footprint
    )
    if util.isrgb(imarray):
        peak_coords = peak_coords[:, :-1]
    peak_array[tuple(peak_coords.T)] = True
    return peak_array


@register_operation(category="features", display_name="Corners")
def corners(
    imarray: np.ndarray,
    Method: Literal[
        "FAST", "Harris", "Kitchen-Rosenfeld", "Moravec", "Shi-Tomasi"
    ] = "Harris",
    Sigma: Annotated[float, {"min": 0}] = 1,
    FAST_N: Annotated[int, {"min": 1, "max": 16}] = 12,
    FAST_Threshold: Annotated[float, {"min": 0}] = 0.15,
    Harris_Method: Literal["Sensitivity", "Normalization"] = "Sensitivity",
    Harris_Sensitivity: Annotated[float, {"min": 0}] = 0.05,
    Harris_Normalization: Annotated[float, {"min": 0, "step": 0.000001}] = 0.000001,
    Moravec_Window_Size: Annotated[float, {"min": 1}] = 1,
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"] = "Nearest",
    Constant_Value: Annotated[int, {"min": -32768, "max": 65535}] = 0,
    Return_Peaks: bool = True,
    Radius: Annotated[int, {"min": 0}] = 0,
    Minimum_Distance: Annotated[int, {"min": 1}] = 1,
    Absolute_Threshold: Annotated[float, {"min": -32768, "max": 65535}] = 0,
    Relative_Threshold: Annotated[float, {"min": -32768, "max": 65535}] = 0,
) -> np.ndarray:
    """
    Identify corners in an image. Only works for grayscale. Performed slicewise for 3D.
    Convenience wrapper around the corner detection functions in scikit-image's feature
    package, including corner_fast, corner_harris, corner_kitchen_rosenfeld,
    corner_moravec, corner_peaks, and corner_shi_tomasi:
    https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.corner_fast
    https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.corner_harris
    https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.corner_kitchen_rosenfeld
    https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.corner_moravec
    https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.corner_peaks
    https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.corner_shi_tomasi

    Parameters
    ----------
    imarray: np.ndarray
        The image to identify corners in.
    Method: Literal[
        "Fast", "Foerstner", "Harris", "Kitchen-Rosenfeld", "Moravec", "Shi-Tomasi"
    ]
        The method for identifying corners. Default is Harris.
        - "FAST": Find corners with a Features from Accelerated Segment Test algorithm.
        - "Harris": Find corners with Harris' algorithm (improved on Moravec's).
        - "Kitchen-Rosenfeld": Find corners with Kitchen's and Rosenfeld's algorithm.
        - "Moravec": Find corners with Moravec's original algorithm. Fast & less robust.
        - "Shi-Tomasi": Find corners with Shi's and Tomasi's algorithm.
    Sigma: float
        The standard deviation for the Gaussian kernel used if "Method" is "Harris" or
        "Shi-Tomasi".
    FAST_N: int
        If "Method" is "FAST", the number of consecutive pixels (out of 16) that must be
        brighter/darker than the threshold for corner identification.
    FAST_Threshold: float
        If "Method" is "FAST", the threshold for identifying corner response pixels.
        Higher = fewer corners.
    Harris_Method: Literal["Sensitivity", "Normalization"]
        If "Method" is "Harris", the Harris method to use for corner detection.
        Default is "Sensitivity".
    Harris_Sensitivity: float
        If "Method" is "Harris", the sensitivity factor for the corner response.
    Harris_Normalization: float
        If "Method" is "Harris", the normalization factor for the corner response.
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"]
        If "Method" is "Kitchen-Rosenfeld", how the algorithm should handle image
        borders. Default is "Nearest". Ignored for other methods.
        - "Constant": The border is extended with a value defined in "Constant Value".
        - "Mirror": The border is extended by reflecting about the center of the pixel.
        - "Nearest": The border is extended by copying the edge pixel.
        - "Reflect": The border is extended by reflecting about the edge of the pixel.
        - "Wrap": The border is extended by wrapping around to the opposite border.
    Constant_Value: float
        If "Edge Mode" is "Constant", the constant value to use. Default is 0.
    Return_Peaks: bool
        Whether to return the peaks of the corner response image (True) found with
        scikit-image's feature.corern_peaks or just the corner reponse image (False).
        Default is True. Corner_peaks works similarly to the "Peaks" widget above with
        the parameters below this one.
    Radius: int
        (Optional) If "Return_Peaks" is True, the radius of the local footprint to find
        local maxima within. If 0, automatically generated.
    Minimum_Distance: int
        If "Return_Peaks" is True, the minimum distance between peaks.
    Absolute_Threshold: float
        If "Return_Peaks" is True, the abolsute minimum intensity of peaks.
        If 0, will be calculated automaticall.
    Relative_Threshold: float
        If "Return_Peaks" is True, the relative minimum intensity of peaks (found as
        this value * image max). If 0, will be calculated automatically. Overwrites
        "Absolute_Threshold" if both are provided.
    Perform_Slicewise: bool
        Whether or not to perform the operation slicewise. Default is False.

    Returns
    -------
    corner_array: np.ndarray
        The corner response image.
    """

    if util.isrgb(imarray):
        print("Image is RGB, corner identification not performed!")
        return imarray

    abs_thresh: float | None = Absolute_Threshold
    if abs_thresh == 0:
        abs_thresh = None
    rel_thresh: float | None = Relative_Threshold
    if rel_thresh == 0:
        rel_thresh = None
    else:
        abs_thresh = None

    if Radius == 0:
        footprint = None
    else:
        footprint = disk(Radius)

    def _corners(
        imarray,
        Method,
        Sigma,
        FAST_N,
        FAST_Threshold,
        Harris_Method,
        Harris_Sensitivity,
        Harris_Normalization,
        Moravec_Window_Size,
        Edge_Mode,
        Constant_Value,
        Return_Peaks,
        footprint,
        Minimum_Distance,
        abs_thresh,
        rel_thresh,
    ) -> np.ndarray:

        if Method == "FAST":
            corner_response: np.ndarray = corner_fast(imarray, FAST_N, FAST_Threshold)

        elif Method == "Kitchen-Rosenfeld":
            corner_response = corner_kitchen_rosenfeld(
                imarray, Edge_Mode.lower(), Constant_Value
            )

        elif Method == "Moravec":
            corner_response = corner_moravec(imarray, Moravec_Window_Size)

        elif Method == "Shi-Tomasi":
            corner_response = corner_shi_tomasi(imarray, Sigma)

        else:
            if Harris_Method == "Normalization":
                h_method: str = "k"
            else:
                h_method: str = "eps"
            corner_response = corner_harris(
                imarray, h_method, Harris_Sensitivity, Harris_Normalization, Sigma
            )

        if Return_Peaks:
            return corner_peaks(
                corner_response,
                Minimum_Distance,
                abs_thresh,
                rel_thresh,
                footprint=footprint,
                indices=False,
            )
        return corner_response

    if util.is3d(imarray):
        if Return_Peaks:
            corner_array: np.ndarray = np.empty(imarray.shape, np.bool)
        else:
            corner_array = np.empty(imarray.shape)
        for index in range(0, imarray.shape[0]):
            corner_array[index] = _corners(
                imarray[index],
                Method,
                Sigma,
                FAST_N,
                FAST_Threshold,
                Harris_Method,
                Harris_Sensitivity,
                Harris_Normalization,
                Moravec_Window_Size,
                Edge_Mode,
                Constant_Value,
                Return_Peaks,
                footprint,
                Minimum_Distance,
                abs_thresh,
                rel_thresh,
            )

        return corner_array

    return _corners(
        imarray,
        Method,
        Sigma,
        FAST_N,
        FAST_Threshold,
        Harris_Method,
        Harris_Sensitivity,
        Harris_Normalization,
        Moravec_Window_Size,
        Edge_Mode,
        Constant_Value,
        Return_Peaks,
        footprint,
        Minimum_Distance,
        abs_thresh,
        rel_thresh,
    )

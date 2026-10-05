"""
Local intensity-based thresholding operations to segment images.
"""

from typing import Annotated, Literal

import numpy as np
from skimage.filters import threshold_local, threshold_niblack, threshold_sauvola
from skimage.filters.rank import otsu as rank_otsu
from skimage.filters.rank import threshold as rank_thresh
from skimage.morphology import ball, disk
from skimage.segmentation import chan_vese, morphological_chan_vese, random_walker

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["local_threshold", "rand_walk", "chan_vese_algs"]


@register_operation(category="segmentation", display_name="Local Threshold")
def local_threshold(
    imarray: np.ndarray,
    Method: Literal["Adaptive", "Mean", "Niblack", "Sauvola", "Otsu"] = "Otsu",
    Radius: Annotated[int, {"min": 1}] = 5,
    k: Annotated[float, {"min": 0}] = 0.2,
    Adaptive_Method: Literal["Gaussian", "Generic", "Mean", "Median"] = "Gaussian",
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"] = "Nearest",
    Constant_Value: Annotated[int, {"min": -32768, "max": 65535}] = 0,
) -> np.ndarray:
    """
    Segment an image using local adaptive algorithms. Convenience wrapper around the
    local threshold functions in scikit-image's filters and filters.rank packages:
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_local
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_niblack
    https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_sauvola
    https://scikit-image.org/docs/stable/api/skimage.filters.rank.html#skimage.filters.rank.otsu
    https://scikit-image.org/docs/stable/api/skimage.filters.rank.html#skimage.filters.rank.threshold

    Parameters
    ----------
    imarray: np.ndarray
        The image array to segment.
    Method: Literal["Adaptive", "Mean", "Niblack", "Sauvola", "Otsu"]
        The local algorithm to use to segment the image. Default is Otsu.
    Radius: int
        The radius for the local footprint. Must be odd if "Method" is
        "Adaptive", "Niblack", or "Sauvola".
    k: float
        If "Method" is "Niblack" or "Sauvola", the k parameter for the calculation.
        Ignored for other methods.
    Adaptive_Method: Literal["Gaussian", "Generic", "Mean", "Median"]
        If "Method" is "Adaptive", the method used to calculate the local threshold.
        Default is "Gaussian".
    Edge_Mode: Literal["Constant", "Mirror", "Nearest", "Reflect", "Wrap"]
        If "Method" is "Adaptive", how the algorithm should handle image borders.
        Default is "Nearest". Ignored for other methods.
        - "Constant": The border is extended with a value defined in "Constant Value".
        - "Mirror": The border is extended by reflecting about the center of the pixel.
        - "Nearest": The border is extended by copying the edge pixel.
        - "Reflect": The border is extended by reflecting about the edge of the pixel.
        - "Wrap": The border is extended by wrapping around to the opposite border.
    Constant_Value: float
        If "Edge Mode" is "Constant", the constant value to use. Default is 0.

    Returns
    -------
    thresh_array: np.ndarray
        The segmented image array.
    """

    if Method == "Adaptive":
        if Radius % 2 == 0:
            Radius -= 1
        thresh_mask: np.ndarray = threshold_local(
            imarray,
            Radius,
            Adaptive_Method.lower(),
            mode=Edge_Mode.lower(),
            cval=Constant_Value,
        )

    if Method == "Mean":
        if util.is3d(imarray):
            footprint = ball(Radius)
        else:
            footprint = disk(Radius)
        return util.convert_dtype(
            rank_thresh(imarray, footprint).astype(np.bool), "Uint8"
        )

    if Method == "Niblack":
        thresh_mask: np.ndarray = threshold_niblack(imarray, Radius, k)

    if Method == "Otsu":
        if util.is3d(imarray):
            footprint = ball(Radius)
        else:
            footprint = disk(Radius)
        thresh_mask: np.ndarray = rank_otsu(imarray, footprint)

    if Method == "Sauvola":
        thresh_mask: np.ndarray = threshold_sauvola(imarray, Radius, k)

    return util.convert_dtype(imarray > thresh_mask, "Uint8")  # type: ignore


@register_operation(
    category="segmentation",
    display_name="Random Walker",
    array_params=("Markers",),
)
def rand_walk(
    imarray: np.ndarray,
    Markers: np.ndarray,
    Mode: Literal[
        "Brute Force",
        "Conjugate Gradient",
        "Conjugate Gradient - Jacobi",
    ] = "Conjugate Gradient - Jacobi",
    Beta: Annotated[int, {"min": 1}] = 100,
    Tolerance: Annotated[float, {"min": 0.001, "step": 0.001}] = 0.001,
) -> np.ndarray:
    """
    Segment an image with a random walker algorithm starting from pre-defined image
    markers. Good for segmenting visually distinct but noisy phases in an image.
    Accepts an image layer for "Markers" and returns an image layer.
    Convenience wrapper around scikit-image's segmentation.random_walker:
    https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.random_walker

    Parameters
    ----------
    imarray: np.ndarray
        The image to be segmented.
    Markers: np.ndarray
        The markers to use as seeds for the random walker algorithm. "Markers" must be
        the same size as imarray (but must not be RGB even if imarray is RGB).
    Mode: Literal["Brute Force", "Conjugate Gradient", "Conjugate Gradient - Jacobi"]
        The mode for solvin the linear system of the algorithm
        - "Brute Force": Fast for small images but very slow for large images.
        - "Conjugate Gradient": Slow, but less memory consuming for large images.
        - "Conjugate Gradient - Jacobi": Accelerated conjugate gradient.
    Beta: int
        The penalization coefficient for random walker motion. Higher = more difficult
        diffusion for walker.
    Tolerance: float
        The tolerance to achieve when solving the linear system if "Method" is one of
        the "Conjuage Gradient" options. Ignore for "Brute Force".

    Returns
    -------
    seg_array: np.ndarray
        The segmented image.
    """

    if Mode == "Brute Force":
        mode = "bf"
    elif Mode == "Conjugate Gradient":
        mode = "cg"
    else:
        mode = "cg_j"

    return random_walker(
        imarray, Markers, Beta, mode, Tolerance, channel_axis=util.chan_axis(imarray)
    )


@register_operation(category="segmentation", display_name="Chan-Vese")
def chan_vese_algs(
    imarray: np.ndarray,
    Initial_Level_Set: Literal["Checkerboard", "Disk"] = "Checkerboard",
    Lambda_1: Annotated[int, {"min": 1}] = 1,
    Lambda_2: Annotated[int, {"min": 1}] = 1,
    Smoothing: Annotated[int, {"min": 1}] = 1,
    Morphological_Snakes: bool = True,
    Edge_Length_Weight: Annotated[float, {"min": 0.01}] = 0.25,
    Tolerance: Annotated[float, {"min": 0.001, "step": 0.001}] = 0.001,
    Max_Iterations: Annotated[int, {"min": 1}] = 500,
    Speed_Factor: Annotated[float, {"min": 0}] = 0.5,
) -> np.ndarray:
    """
    Segment an image with the Chan-Vese algorithm. Can also be modified with the
    Morphological Active Contours Without Edges (MorphACWE) algorithm. Both are good for
    segmenting visually distinct phases without sharp borders. RGB image are segmented
    channelwise. MorphACWE can handle 3D while normal Chan-Vese is done slicewise.
    MorphACWE returns Int8 dtype while non-morph returns Boolean. Convenience wrapper
    around chan_vese and morphological_chan_vese in scikit-image's
    segmentation package:
    https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.chan_vese
    https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.morphological_chan_vese

    Parameters
    ----------
    imarray: np.ndarray
        The image to be segmented.
    Initial_Level_Set: Literal["Checkerboard", "Disk"]
        The initial level set to use for the algorithm. "Checkerboard" is fast but
        "Disk" tends to be more accurate.
    Lambda_1: float
        Difference from average weight parameter for the outer region. If "Lambda 1" >
        "Lambda 2" the outer area will have a greater range of values.
    Lambda_2: float
        Difference from average weight parameter for the inner region. If "Lambda 2" >
        "Lambda 1" the inner area will have a greater range of values.
    Max_Iterations: int
        The maximum number of iterations to apply the algorithm.
    Smoothing: int
        The number of times the smoothing operator is applied per iteration. Higher =
        smoother segmentation. Ignored if "Morphological Snakes" is False.
    Morphological_Snakes: bool
        Whether or not to use the morphological snakes version of the algorithm. Default
        is True, and does not require the use of parameters below this one.
    Edge_Length_Weight: float
        The weight applied to edge lengths. Higher = rounder edges, fewer small objects.
        Ignored for "Morphological Snakes".
    Tolerance: float
        The level set tolerance between iterations. Ignored for "Morphological Snakes".
    Speed_Factor: float
        Higher = speeds up each step but produces less accurate results.
        Ignored for "Morphological Snakes".

    Returns
    -------
    seg_array: np.ndarray
        The segmented image.
    """

    if Morphological_Snakes:
        if util.isrgb(imarray):
            seg_array: np.ndarray = np.empty(imarray.shape, np.int8)
            for cindex in range(0, imarray.shape[-1]):
                seg_array[..., cindex] = morphological_chan_vese(
                    imarray[..., cindex],
                    Max_Iterations,
                    Initial_Level_Set.lower(),
                    Smoothing,
                    Lambda_1,
                    Lambda_2,
                )

            return seg_array

        return morphological_chan_vese(
            imarray,
            Max_Iterations,
            Initial_Level_Set.lower(),
            Smoothing,
            Lambda_1,
            Lambda_2,
        )  # type: ignore

    if util.isrgb(imarray):
        seg_array: np.ndarray = np.empty(imarray.shape, np.bool)
        for cindex in range(0, imarray.shape[-1]):
            if util.is3d(imarray):
                for index in range(0, imarray.shape[0]):
                    seg_array[index, :, :, cindex] = chan_vese(
                        imarray[index, :, :, cindex],
                        Edge_Length_Weight,
                        Lambda_1,
                        Lambda_2,
                        Tolerance,
                        Max_Iterations,
                        Speed_Factor,
                        Initial_Level_Set.lower(),
                    )
            else:
                seg_array[..., cindex] = chan_vese(
                    imarray[..., cindex],
                    Edge_Length_Weight,
                    Lambda_1,
                    Lambda_2,
                    Tolerance,
                    Max_Iterations,
                    Speed_Factor,
                    Initial_Level_Set.lower(),
                )

        return seg_array

    if util.is3d(imarray):
        seg_array: np.ndarray = np.empty(imarray.shape, np.bool)
        for index in range(0, imarray.shape[0]):
            seg_array[index] = chan_vese(
                imarray[index],
                Edge_Length_Weight,
                Lambda_1,
                Lambda_2,
                Tolerance,
                Max_Iterations,
                Speed_Factor,
                Initial_Level_Set.lower(),
            )

        return seg_array

    return chan_vese(
        imarray,
        Edge_Length_Weight,
        Lambda_1,
        Lambda_2,
        Tolerance,
        Max_Iterations,
        Speed_Factor,
        Initial_Level_Set.lower(),
    )  # type: ignore

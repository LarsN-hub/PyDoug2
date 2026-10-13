"""
Image information.
"""

from typing import Annotated, Any, Literal

import numpy as np

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = [
    "stats",
    "percentiles",
    "thresholds",
    "euler_num",
    "blur_effect",
    "shannon_entropy",
    "inertia_tensor_eigs",
]


@register_operation(
    category="info",
    display_name="Stats",
    array_params=("Mask",),
    array_param_flags={"Mask": "Apply_Mask"},
    output_kind="value",
    recordable_toggle=True,
)
def stats(
    imarray: np.ndarray, Apply_Mask: bool = False, Mask: np.ndarray | None = None
) -> dict[str, float | int]:
    """
    Calculate the grayscale statistics of an image array. Convenience wrapper around
    functions in numpy's statistics section, except for mode which is calculated using
    scipy's stats.mode function:
    https://numpy.org/doc/stable/reference/routines.statistics.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html#scipy.stats.mode

    Parameters
    ----------
    imarray: np.ndarray
        The image array to calculate grayscale statistics for.
    Apply_Mask: bool = False
        Whether or not to apply a mask to the operation. Default is False.
    Mask: np.ndarray | None
        (Optional) the mask to be applied before statistics calculation if "Apply_Mask"
        is True.

    Returns
    -------
    statdict: dict[str, float]
        A dictionary containing the stats "Mean", "Median", "Mode", "Std Dev",
        "Variance", "Min", "Max", "Range", "Dtype Min", and "Dtype Max".
    """

    from scipy.stats import mode

    if np.issubdtype(imarray.dtype, np.integer):
        imdmin = np.iinfo(imarray.dtype).min
        imdmax = np.iinfo(imarray.dtype).max
        boolarray: bool = False
    elif np.issubdtype(imarray.dtype, np.floating):
        imdmin = np.finfo(imarray.dtype).min
        imdmax = np.finfo(imarray.dtype).max
        boolarray: bool = False
    else:
        imdmin = 0
        imdmax = 1
        boolarray: bool = True

    if Mask is not None and Apply_Mask:
        Mask = util.prep_mask(imarray, Mask)
        immean: float = np.mean(imarray[Mask])
        immedian: float | int = np.median(imarray[Mask])
        imstd: float = np.std(imarray[Mask])
        imvar: float = np.var(imarray[Mask])
        immin: float | int = np.min(imarray[Mask])
        immax: float | int = np.max(imarray[Mask])
        if not boolarray:
            immode: float | int = mode(imarray[Mask], axis=None)[0]  # type: ignore
            imrange: float | int = immax - immin
        else:
            onecount: np.int64 = np.count_nonzero(imarray[Mask])
            zercount: np.int64 = len(imarray[Mask]) - onecount
            immode: int = 1 if onecount >= zercount else 0
            imrange = int(immax) - int(immin)
    else:
        immean: float = np.mean(imarray)
        immedian: float | int = np.median(imarray)
        imstd: float = np.std(imarray)
        imvar: float = np.var(imarray)
        immin: float | int = np.min(imarray)
        immax: float | int = np.max(imarray)
        if not boolarray:
            immode: float | int = mode(imarray, axis=None)[0]  # type: ignore
            imrange: float | int = immax - immin
        else:
            onecount: np.int64 = np.count_nonzero(imarray)
            zercount: np.int64 = len(imarray) - onecount
            immode: int = 1 if onecount >= zercount else 0
            imrange = int(immax) - int(immin)

    return {
        "Mean": immean,
        "Median": immedian,
        "Mode": immode,
        "Std Dev": imstd,
        "Variance": imvar,
        "Min": immin,
        "Max": immax,
        "Range": imrange,
        "Dtype Min": imdmin,
        "Dtype Max": imdmax,
    }


@register_operation(
    category="info",
    display_name="Percentiles",
    output_kind="value",
    recordable_toggle=True,
)
def percentiles(
    imarray: np.ndarray,
    percentiles: tuple[
        Annotated[float, {"step": 0.0001, "max": 100, "min": 0}],
        Annotated[float, {"step": 0.0001, "max": 100, "min": 0}],
    ] = (1, 99),
) -> dict[float, float]:
    """
    Calculate the lower and upper intensity values for the given percentiles.
    Convenience wrapper around numpy's percentile:
    https://numpy.org/doc/stable/reference/generated/numpy.percentile.html#numpy-percentile

    Parameters
    ----------
    imarray: np.ndarray
        The image array to calculate percentile intensities for.
    percentiles: tuple[float, float]
        A tuple containing the lower and upper percentiles in the range [0, 100]. Values
        outside [0, 100] will be clipped to 0 (first value) or 100 (second value).

    Returns
    -------
    intensities: dict[float, float]
        The percentile values in a dictionary containing keys for each given percentile.
    """

    if percentiles[0] > 100 or percentiles[0] < 0:
        percentiles = (0, percentiles[1])
    if percentiles[1] > 100 or percentiles[1] < 0:
        percentiles = (percentiles[0], 100)

    if percentiles == (0, 100):
        return {0: np.min(imarray), 100: np.max(imarray)}
    return {
        min(percentiles): np.percentile(imarray, min(percentiles)),
        max(percentiles): np.percentile(imarray, max(percentiles)),
    }


@register_operation(
    category="info",
    display_name="Thresholds",
    array_params=("Mask",),
    array_param_flags={"Mask": "Apply_Mask"},
    output_kind="value",
    recordable_toggle=True,
)
def thresholds(
    imarray: np.ndarray,
    Method: Literal[
        "Isodata", "Li", "Mean", "Minimum", "Otsu", "Triangle", "Yen"
    ] = "Otsu",
    Otsu_Classes: Annotated[int, {"min": 2}] = 2,
    Bin_Count: Annotated[int, {"min": 1}] = 256,
    Max_Iteration_Count: Annotated[int, {"min": 1, "max": 100000}] = 10000,
    Apply_Mask: bool = False,
    Mask: np.ndarray | None = None,
) -> Any:
    """
    Calculate segmentation thresholds for an image using histogram-based algorithms.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to segment.
    Method: Literal["Isodata", "Li", "Mean", "Minimum", "Otsu", "Triangle", "Yen"]
        The algorithm to use for identifying the threshold value(s). Default is Otsu.
    Otsu_Classes: int
        The number of classes to segmented if "Method" is "Otsu".
        Ignored for other methods.
    Bin_Count: int
        The number of bins to use for calculating the histogram.
        Ignored for integer dtypes.
    Max_Iteration_Count: int
        The maximum number of iterations to use for smoothing the histogram.
        Only applied if "Method" is "Minimum".
    Apply_Mask: bool
        Whether or not to apply a mask. Default is False.
    Mask: np.ndarray | None
        (Optional) If Apply Mask is True, only values within the provided mask array
        will be included in the histogram.

    Returns
    -------
    thresh_vals: float | int | np.ndarray
        The threshold value(s).
    """

    from pydoug_core.segment.segment_util import calc_thresholds

    if Apply_Mask and Mask is not None:
        eval_array: np.ndarray = imarray[util.prep_mask(imarray, Mask)]
    else:
        eval_array = imarray

    return calc_thresholds(
        eval_array, Method, Otsu_Classes, Bin_Count, Max_Iteration_Count
    )


@register_operation(
    category="info",
    display_name="Euler Number",
    output_kind="value",
    recordable_toggle=True,
)
def euler_num(
    imarray: np.ndarray,
    Connectivity: Literal[1, 2, 3] = 2,
) -> int:
    """
    Calculate the Euler number of a binary image (the number of objects minus the number
    of holes). Connectivity is for the objects (note that the connectivity of holes is
    dependent on the object connectivity, e.g., object connectivity of 2 forces a holes
    connectivity of 1 in 2D). Image can be 2D/3D grayscale/RGB but non-binary images are
    binarized before labeling and RGB images are converted to grayscale. Convenience
    wrapper around scikit-image's measure.euler_number:
    https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.euler_number

    Parameters
    ----------
    imarray: np.ndarray
        The binary image.
    Connectivity: Literal
        The method to use for determing label connectivity for objects. Default is 2.
        - 1: Only pixels with faces in contact are considered connected.
            - In 2D, this means only four most adjacent pixels.
            - In 3D, this means only the 6 most adjacent voxels.
        - 2: Includes 1, and adds pixels with edges in contact.
            - In 2D, this means the surrounding 8 pixels.
            - Not supported for 3D in this operation (defaults to 3).
        - 3: (3D only) Includes 1-2, and adds pixels with corners in contact.
            - In 3D, this means the surrounding 26 voxels.

    Returns
    -------
    euler_num: int
        The Euler number of the image.
    """

    from skimage.measure import euler_number

    from pydoug_core.util import is3d, rgb2gray

    if Connectivity == 2 and is3d(imarray):
        Connectivity = 3

    return int(euler_number(np.round(rgb2gray(imarray)).astype(np.bool), Connectivity))  # type: ignore


@register_operation(
    category="info",
    display_name="Blur Effect",
    output_kind="value",
    recordable_toggle=True,
)
def blur_effect(
    imarray: np.ndarray,
    H_Size: Annotated[int, {"min": 1}] = 11,
) -> float:
    """
    Estimate the blur effect applied to an image. Returns a value between 0 (no blur)
    and 1 (max blur). Image can be 2D/3D grayscale/RGB. Convenience wrapper around
    scikit-image's measure.blur_effect:
    https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.blur_effect

    Parameters
    ----------
    imarray: np.ndarray
        The image to test the blur effect for.
    H_Size: int
        The size of the re-blurring filter (higher detects larger blur).

    Returns
    -------
    blur_val: float
        The blur effect.
    """

    from skimage.measure import blur_effect as skiblur_effect

    from pydoug_core.util import chan_axis

    return skiblur_effect(imarray, H_Size, channel_axis=chan_axis(imarray))  # type: ignore


@register_operation(
    category="info",
    display_name="Shannon Entropy",
    output_kind="value",
    recordable_toggle=True,
)
def shannon_entropy(
    imarray: np.ndarray,
    Base: Annotated[int, {"min": 2}] = 2,
) -> float:
    """
    Calculate the Shannon entropy for the image. Higher entropy = less order. Image must
    be 2D grayscale (RGB image will be converted). Convenience wrapper around
    scikit-image's measure.shannon_entropy:
    https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.shannon_entropy

    Parameters
    ----------
    imarray: np.ndarray
        The 2D iamge to.
    Base: int
        The logarithmic base to use for the calculation.

    Returns
    -------
    entropy_val: float
        The Shannon entropy.
    """

    from skimage.measure import shannon_entropy as ski_entropy

    from pydoug_core.util import is3d, rgb2gray

    if is3d(imarray):
        print("Image is 3D, entropy not calculated!")
        return 0

    return ski_entropy(rgb2gray(imarray), Base)  # type: ignore


@register_operation(
    category="info",
    display_name="Inertia Tensor Eigenvalues",
    output_kind="value",
    recordable_toggle=True,
)
def inertia_tensor_eigs(
    imarray: np.ndarray,
    Maximum_Moments_Order: Annotated[int, {"min": 1}] = 3,
) -> dict[str, float]:
    """
    Calculate the inertia tensor eigenvalues of the image. Iimage can be 2D/3D grayscale
    or RGB, but RGB images are converted first. Convenience wrapper around
    scikit-image's measure.inertia_tensor_eigvals and measure.moments:
    https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.inertia_tensor_eigvals
    https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.moments

    Parameters
    ----------
    imarray: np.ndarray
        The image to calculate the inertia tensor eigenvalues for.
    Maximum_Moments_Order: int
        The maximum order of moments to calculate the inertia tensor. Default is 3.

    Returns
    -------
    inertia_eigs: dict[str, float]
        Dictionary containing the inertia tensor eigenvalues for each axis.
    """

    from skimage.measure import inertia_tensor_eigvals, moments

    from pydoug_core.util import is3d, rgb2gray

    results: list = inertia_tensor_eigvals(
        rgb2gray(imarray), moments(rgb2gray(imarray), Maximum_Moments_Order)
    )
    if is3d(imarray):
        return {"X": results[2], "Y": results[1], "Z": results[0]}

    return {"X": results[1], "Y": results[0]}

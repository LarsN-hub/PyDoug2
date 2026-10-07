"""
Measured and calculated values of images.
"""

from typing import Annotated, Any, Literal

import numpy as np
from scipy.stats import mode

from pydoug_core import util
from pydoug_core.registry import register_operation
from pydoug_core.segment.segment_util import calc_thresholds

__all__ = ["stats", "percentiles", "thresholds"]


@register_operation(
    category="measurements",
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
    category="measurements",
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
    category="measurements",
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

    if Apply_Mask and Mask is not None:
        eval_array: np.ndarray = imarray[util.prep_mask(imarray, Mask)]
    else:
        eval_array = imarray

    return calc_thresholds(
        eval_array, Method, Otsu_Classes, Bin_Count, Max_Iteration_Count
    )


@register_operation(
    category="measurements",
    display_name="Label Count",
    layer_type="labels",
    output_kind="value",
)
def label_count(imarray: np.ndarray) -> int:
    """
    Count the number of unique labels in a labeled segmented image. Convenience wrapper
    around numpy's max function (labels are generated sequentially, so the max value
    in a labeled image should be the number of unique labels):
    https://numpy.org/doc/stable/reference/generated/numpy.max.html

    Parameters
    ----------
    imarray: np.ndarray
        The labeled segmented image to count unique labels.

    Returns
    -------
    lab_count: int
        The unique label count.
    """

    return np.max(imarray)

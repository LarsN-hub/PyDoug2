"""
Measured and calculated values of images.
"""

from typing import Annotated

import numpy as np
from scipy.stats import mode

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["stats"]


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
    Mask: np.ndarray | None:
        (Optional) the mask to be applied before statistics calculation.

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

    if Mask is not None:
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
    category="measurements", display_name="Percentiles", output_kind="value"
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

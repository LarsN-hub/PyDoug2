"""
Measured and calculated values of images.
"""

import numpy as np
from scipy.stats import mode

from pydoug_core import util
from pydoug_core.registry import register_operation

__all__ = ["stats"]


@register_operation(
    category="measurements",
    display_name="Stats",
    mask_params=("Mask",),
    output_kind="value",
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

"""
Plot and modify an image array's histogram.
"""

from typing import Annotated, Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from skimage import util as skutil
from skimage.exposure import rescale_intensity

from pydoug_core import util
from pydoug_core.analyze import plot_util
from pydoug_core.registry import register_operation

__all__ = ["plot_histogram", "normalize"]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"


@register_operation(
    category="histogram",
    display_name="Plot Histogram",
    mask_params=("Mask",),
    output_kind="plot",
    recordable_toggle=True,
)
def plot_histogram(
    imarray: np.ndarray,
    Bin_Count: int = 256,
    Normalize: bool = False,
    Apply_Mask: bool = False,
    Mask: np.ndarray | None = None,
) -> Figure:
    """
    Plot an image array's histogram. Convenience wrapper around scikit-image's
    exposure.histogram:
    https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.histogram

    Parameters
    ----------
    imarray: np.ndarray
        The image array to plot the histogram for.
    Bin_Count: int
        The number of bins to use in the histogram. Ignored for integer arrays.
    Normalize: bool
        Whether or not to normalize the counts by the total sum. Default is False.
    Apply_Mask: bool
        Whether or not to apply a mask. Default is False.
    Mask: np.ndarray | None
        (Optional) If Apply Mask is True, only values within the provided mask array
        will be included in the histogram.

    Returns
    -------
    hist_fig: plt.Figure
        The figure containing the histogram plot.
    """

    if Apply_Mask and Mask is not None:
        values: np.ndarray = imarray[util.prep_mask(imarray, Mask)]
    else:
        values = np.copy(imarray)
    fig: Figure = Figure()
    ax: Axes = fig.add_subplot()
    ax = plot_util.hist_ax(values, ax, Bin_Count, normalize=Normalize)
    ax.set_xlabel("Gray Value", fontsize=15)
    if Normalize:
        ax.set_ylabel("Probability", fontsize=15)
    else:
        ax.set_ylabel("Counts", fontsize=15)
    ax.tick_params("both", labelsize=13)
    fig.set_layout_engine("constrained")
    return fig


@register_operation(category="histogram", display_name="Convert Data Type")
def convert_dtype(
    imarray: np.ndarray,
    New_Type: Literal[
        "Uint8", "Uint16", "Int16", "Float32", "Float64", "Bool"
    ] = "Uint8",
) -> np.ndarray:
    """
    Convert the data type of an image. Convenience wrapper around the "img_as_XXXXX"
    functions in scikit-image's util package:
    https://scikit-image.org/docs/stable/api/skimage.util.html#

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be converted.
    New_Type: Literal["Bool", "Uint8", "Uint16", "Int16", "Float32", "Float64"]
        The data type to convert to.
        - Uint8: 8-bit unsigned integer format [0, 255]
        - Uint16: 16-bit unsigned integer format [0, 65,535]
        - Int16: 16-bit signed integer format [-32,768, 32,767]
        - Float32: 32-bit (single-precision) floating-point format [-3.4e+38, 3.4e+38]
        - Float64: 64-bit (double-precision) floating-point format [-1.8e+308, 1.8e+308]
        - Bool: Boolean True/False format

    Returns
    -------
    convarray: np.ndarray
        The converted image array.
    """

    if New_Type == "Bool" and imarray.dtype != np.bool:
        return skutil.img_as_bool(imarray)
    if New_Type == "Uint8" and imarray.dtype != np.uint8:
        return skutil.img_as_ubyte(imarray)
    if New_Type == "Uint16" and imarray.dtype != np.uint16:
        return skutil.img_as_uint(imarray)
    if New_Type == "Int16" and imarray.dtype != np.int16:
        return skutil.img_as_int(imarray)
    if New_Type == "Float32" and imarray.dtype != np.float32:
        return skutil.img_as_float32(imarray)
    if New_Type == "Float64" and imarray.dtype != np.float64:
        return skutil.img_as_float64(imarray)
    return imarray


@register_operation(category="histogram", display_name="Normalize")
def normalize(
    imarray: np.ndarray,
    Input_Range: Literal["Image", "Dtype", "Manual"] = "Image",
    Manual_Input: tuple[
        Annotated[float, {"step": 0.0001, "min": -32768, "max": 65535}],
        Annotated[float, {"step": 0.0001, "min": -32768, "max": 65535}],
    ] = (0, 0),
    Output_Range: Literal["Image", "Dtype", "Manual"] = "Dtype",
    Manual_Output: tuple[
        Annotated[float, {"step": 0.0001, "min": -32768, "max": 65535}],
        Annotated[float, {"step": 0.0001, "min": -32768, "max": 65535}],
    ] = (0, 0),
) -> np.ndarray:
    """
    Shrink or expand an image's histogram to a new range. Convenience wrapper around
    scikit-image's exposure.rescale_intensity:
    https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.rescale_intensity

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be normalized.
    Input_Range: Literal["Image", "Dtype", "Manual"]
        Method for determining the input range to be normalized.
        - "Image": (default) uses the current range of the input image.
        - "Dtype": uses the max range of the image's data type.
        - "Manual": uses the range specified in "Manual Input".
    Manual_Input: tuple[float, float]
        If Input Range is "Manual", the max and min values of the input range.
    Output_Range: Literal["Image", "Dtype", "Manual"]
        Method for determining the output range after normalization.
        - "Image": uses the current range of the input image.
        - "Dtype": (default) uses the max range of the image's data type.
        - "Manual": uses the range specified in "Manual Input".
    Manual_Output: tuple[float, float]
        If Output Range is "Manual", the max and min values of the output range.

    Returns
    -------
    normarray: np.ndarray
        The normalized image array.
    """

    if Input_Range == "Image" and Output_Range == "Image":
        return imarray

    if Input_Range == "Manual":
        input: str | tuple[float, float] = Manual_Input
    else:
        input = Input_Range.lower()
    if Output_Range == "Manual":
        output: str | tuple[float, float] = Manual_Output
    else:
        output = Output_Range.lower()

    return rescale_intensity(imarray, input, output).astype(imarray.dtype)  # type: ignore


@register_operation(category="histogram", display_name="Saturate")
def saturate(
    imarray: np.ndarray,
    Bounds: tuple[
        Annotated[float, {"step": 0.0001}], Annotated[float, {"step": 0.0001}]
    ] = (0, 0),
    Normalize_to_Dtype: bool = True,
) -> np.ndarray:
    """
    Saturate an image by setting values outside the given range to the range's bounds.

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be saturated.
    Bounds: tuple[float, float]
        The intensity bounds to use for saturation. Percentiles are not supported, but
        can be calculated in "Measurements" to use here.
    Normalize_to_Dtype: bool
        Whether or not to stretch the final histgoram to the dtype's max range after
        saturation. Default is True.

    Returns
    -------
    satarray: np.ndarray
        The saturated image array.
    """

    if imarray.dtype == np.bool:
        return imarray

    satarray: np.ndarray = np.copy(imarray)
    satarray[satarray >= max(Bounds)] = max(Bounds)
    satarray[satarray <= min(Bounds)] = min(Bounds)

    if Normalize_to_Dtype:
        satarray = normalize(satarray)
    return satarray

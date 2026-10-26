"""
Generate distributions.
"""

from typing import Annotated, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pydoug_core.registry import register_operation
from pydoug_core.results import PlotResult, ValueResult
from pydoug_core.util import is3d, prep_mask

__all__ = ["hist_distrib", "size_distrib"]


def hist_distrib(
    values: np.ndarray,
    nbins: int = 256,
    source_range: Literal["values", "dtype"] = "dtype",
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """
    Generate an histogram for the given values. Convenience wrapper around
    scikit-image's exposure.histogram:
    https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.histogram

    Parameters
    ----------
    values: np.ndarray
        The values to generate a histogram for.
    nbins: int
        The number of bins to use. Default is 256.
    source_range: Literal["values", "dtype"]
        Whether to use the values or their dtype as the source for the x-axis range.
    normalize: bool
        Whether or not to normalize the histogram. Default is False.

    Returns
    -------
    hist_vals: np.ndarray
        The counts corresponding to each bin center in "bin_centers".
    bin_centers: np.ndarray
        The bin centers corresponding to each histogram value in "hist_vals".
    stats_dict: dict[str, int | float]
        A dictionary containing statistics about the histogram.
    """

    from skimage.exposure import histogram

    from pydoug_core.analyze.image_info import stats

    input_range: str = source_range
    if source_range == "values":
        input_range = "image"
    if values.size == 0:
        values = np.array([0], values.dtype)
    hist_vals, bin_centers = histogram(
        values, nbins if nbins != 0 else np.unique(values).size, input_range, normalize
    )
    stats_dict: dict[str, float | int] = stats(values)  # type: ignore

    return hist_vals, bin_centers, stats_dict


@register_operation(
    category="measure",
    display_name="Size Distribution",
    layer_type="labels",
    output_kind="plot",
    array_params=("Mask",),
    array_param_flags={"Mask": "Apply_Mask"},
)
def size_distrib(
    labarray: np.ndarray,
    Measurement: Literal[
        "Equivalent Diameter", "Equivalent Radius", "Area", "Volume"
    ] = "Equivalent Diameter",
    Pixel_Size: Annotated[float, {"min": 0, "step": 0.0001}] = 1,
    Units: str = "pixels",
    Bin_Count: Annotated[int, {"min": 0}] = 0,
    Normalize: bool = False,
    Y_Max: Annotated[float, {"min": 0, "max": 1000000000000000, "step": 0.0001}] = 0,
    X_Lims: tuple[
        Annotated[float, {"min": 0, "max": 1000000000000000, "step": 0.0001}],
        Annotated[float, {"min": 0, "max": 1000000000000000, "step": 0.0001}],
    ] = (0, 0),
    Apply_Mask: bool = False,
    Mask: np.ndarray | None = None,
) -> PlotResult | ValueResult:
    """
    Plot a frequency-weighted histogram of labeled object sizes. Convenience wrapper
    around scikit-image's exposure.histogram and matplotlib's hist:
    https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.histogram
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html

    Parameters
    ----------
    labarray: np.ndarray
        The labeled segmented image.
    Measurement: Literal
        The measurement to perform.
        - "Equivalent Diameter": Finds the diameter of a circle/sphere with the same
        area/volume as the labeled object. If image is 2D assumes circles else assumes
        spheres.
        - "Equivalent Radius": Finds the radius of a circle/sphere with the same
        area/volume as the labeled object. If image is 2D assumes circles else assumes
        spheres.
        - "Area": Multiplies number of pixels in each label by "Pixel Size" ^ 2.
        - "Volume": Multiplies number of pixels in each label by "Pixel Size" ^ 3.
    Pixel_Size: float
        The edge-length of pixels in real-world units.
    Units: str
        The real-world units for "Pixel Size".
    Bin_Count: int
        The number of bins to use in the histogram. Ignored if set to 0.
    Normalize: bool
        Whether or not to normalize the counts by the total sum. Default is False.
    Y_Max: float
        The maximum y-axis value to plot. Automatically determined if 0.
    X_Lims: tuple[float, float]
        The limits of the x-axis in the form [min, max]. Ignored if both are 0.
    Apply_Mask: bool
        Whether to mask the measurement region (True) or not (False).
    Mask: np.ndarray | None
        If "Apply Mask" is "True", the mask to use.

    Returns
    -------
    size_pr: PlotResult
        The size distribution PlotResult object.
    """

    if Apply_Mask and Mask is not None:
        values: np.ndarray = labarray[prep_mask(labarray, Mask)]
    else:
        values = labarray
    if np.all(np.unique(values) == np.array([0])):
        return ValueResult(
            {
                "Bin Centers": np.array([0]),
                "Probability Density" if Normalize else "Counts": np.array([0]),
            }
        )

    hist_vals, _, _ = hist_distrib(values, source_range="values")
    hist_vals = hist_vals.astype(np.float64)
    if Measurement in "Area":
        hist_vals *= Pixel_Size**2
    elif Measurement == "Volume":
        hist_vals *= Pixel_Size**3
    elif Measurement in ("Equivalent Diameter", "Equivalent Radius"):
        if is3d(labarray):
            hist_vals *= Pixel_Size**3
            hist_vals = np.cbrt((hist_vals * 3) / (4 * np.pi))
        else:
            hist_vals *= Pixel_Size**2
            hist_vals = np.sqrt(hist_vals / np.pi)
        if Measurement == "Equivalent Diameter":
            hist_vals *= 2
    size_vals, size_bin_centers, size_stats = hist_distrib(
        hist_vals, Bin_Count, "values", Normalize
    )
    size_stats.pop("Dtype Min")
    size_stats.pop("Dtype Max")

    from pydoug_core.analyze.plot_util import hist_ax

    display_units: str = r"\mu m" if Units == "um" else Units
    if Measurement in ("Area", "Volume"):
        display_units = (
            rf"{display_units}^{{2}}"
            if Measurement == "Area"
            else rf"{display_units}^{{3}}"
        )
    fig: Figure = Figure()
    ax: Axes = fig.add_subplot()
    ax = hist_ax(
        size_vals,
        size_bin_centers,
        ax,
        rf"{Measurement} ($\mathrm{{{display_units}}}$)",
        "Probability Density" if Normalize else "Counts",
        Y_Max,
        None if (X_Lims[0] == X_Lims[1] == 0) else X_Lims,
    )
    fig.set_layout_engine("constrained")

    return PlotResult(
        fig,
        {
            "Bin Centers": size_bin_centers,
            "Probability Density" if Normalize else "Counts": size_vals,
        },  # type: ignore
        x_columns=("Bin Centers",),
        tables={"statistics": size_stats},
    )


@register_operation(
    category="measure",
    display_name="Aspect Ratio Distribution",
    layer_type="labels",
    output_kind="plot",
    array_params=("Mask",),
    array_param_flags={"Mask": "Apply_Mask"},
)
def aspect_ratio_distrib(
    labarray: np.ndarray,
    Bin_Count: Annotated[int, {"min": 0}] = 0,
    Normalize: bool = False,
    Y_Max: Annotated[float, {"min": 0, "max": 1000000000000000, "step": 0.0001}] = 0,
    X_Lims: tuple[
        Annotated[float, {"min": 0, "max": 1000000000000000, "step": 0.0001}],
        Annotated[float, {"min": 0, "max": 1000000000000000, "step": 0.0001}],
    ] = (0, 0),
    Apply_Mask: bool = False,
    Mask: np.ndarray | None = None,
) -> PlotResult | ValueResult:
    """
    Plot a frequency-weighted histogram of labeled object aspect ratios. Convenience
    wrapper around scikit-image's exposure.histogram and measure.regionprops as well as
    matplotlib's hist:
    https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.histogram
    https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html

    Parameters
    ----------
    labarray: np.ndarray
        The labeled segmented image.
    Pixel_Size: float
        The edge-length of pixels in real-world units.
    Units: str
        The real-world units for "Pixel Size".
    Bin_Count: int
        The number of bins to use in the histogram. Ignored if set to 0.
    Normalize: bool
        Whether or not to normalize the counts by the total sum. Default is False.
    Y_Max: float
        The maximum y-axis value to plot. Automatically determined if 0.
    X_Lims: tuple[float, float]
        The limits of the x-axis in the form [min, max]. Ignored if both are 0.
    Apply_Mask: bool
        Whether to mask the measurement region (True) or not (False).
    Mask: np.ndarray | None
        If "Apply Mask" is "True", the mask to use.

    Returns
    -------
    arpr: PlotResult
        The aspect ratio PlotResult object.
    """

    if Apply_Mask and Mask is not None:
        values: np.ndarray = np.copy(labarray)
        values[np.logical_not(prep_mask(labarray, Mask))] = 0
    else:
        values = labarray
    if np.all(np.unique(values) == np.array([0])):
        return ValueResult(
            {
                "Bin Centers": np.array([0]),
                "Probability Density" if Normalize else "Counts": np.array([0]),
            }
        )

    from skimage.measure import regionprops

    labels: np.ndarray = np.unique(values)
    labels = labels[1:] if labels[0] == 0 else labels
    aspect_ratios: np.ndarray = np.empty(labels.size)
    props = regionprops(values)
    for index, region in enumerate(props):
        aspect_ratios[index] = region.axis_major_length / max(
            region.axis_minor_length, 0.01
        )
    ar_values, ar_bin_centers, ar_stats = hist_distrib(
        aspect_ratios, Bin_Count, "values", Normalize
    )
    ar_stats.pop("Dtype Min")
    ar_stats.pop("Dtype Max")

    from pydoug_core.analyze.plot_util import hist_ax

    x_lims: tuple[float, float] | None = (
        None if (X_Lims[0] == X_Lims[1] == 0) else X_Lims
    )
    fig: Figure = Figure()
    ax: Axes = fig.add_subplot()
    ax = hist_ax(
        ar_values,
        ar_bin_centers,
        ax,
        "Aspect Ratio",
        "Probability Density" if Normalize else "Counts",
        Y_Max,
        x_lims,
    )
    fig.set_layout_engine("constrained")

    return PlotResult(
        fig,
        {
            "Bin Centers": ar_bin_centers,
            "Probability Density" if Normalize else "Counts": ar_values,
        },  # type: ignore
        x_columns=("Bin Centers",),
        tables={"statistics": ar_stats},
    )

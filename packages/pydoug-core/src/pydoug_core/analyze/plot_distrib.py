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

__all__ = ["hist_distrib", "size_distrib", "dimless_morph_distrib"]


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
                Measurement: np.array([0]),
                "Probability Density" if Normalize else "Counts": np.array([0]),
            }
        )

    hist_vals, _, _ = hist_distrib(values, source_range="values")
    hist_vals = hist_vals.astype(np.float64)
    display_units: str = r"\mu m" if Units == "um" else Units
    units: dict = {"Probability Density": None} if Normalize else {"Counts": None}
    if Measurement in "Area":
        hist_vals *= Pixel_Size**2
        display_units = rf"{display_units}^{{2}}"
        units[Measurement] = f"{Units}^2"
    elif Measurement == "Volume":
        hist_vals *= Pixel_Size**3
        display_units = rf"{display_units}^{{3}}"
        units[Measurement] = f"{Units}^3"
    elif Measurement in ("Equivalent Diameter", "Equivalent Radius"):
        if is3d(labarray):
            hist_vals *= Pixel_Size**3
            hist_vals = np.cbrt((hist_vals * 3) / (4 * np.pi))
        else:
            hist_vals *= Pixel_Size**2
            hist_vals = np.sqrt(hist_vals / np.pi)
        if Measurement == "Equivalent Diameter":
            hist_vals *= 2
        units[Measurement] = Units
    size_vals, size_bin_centers, size_stats = hist_distrib(
        hist_vals, Bin_Count, "values", Normalize
    )
    size_stats.pop("Dtype Min")
    size_stats.pop("Dtype Max")

    from pydoug_core.analyze.plot_util import hist_ax

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
            Measurement: size_bin_centers,
            "Probability Density" if Normalize else "Counts": size_vals,
        },  # type: ignore
        x_columns=(Measurement,),
        tables={"statistics": size_stats},
        units=units,
    )


@register_operation(
    category="measure",
    display_name="Dimensionless Morphology Distribution",
    layer_type="labels",
    output_kind="plot",
    array_params=("Mask",),
    array_param_flags={"Mask": "Apply_Mask"},
)
def dimless_morph_distrib(
    labarray: np.ndarray,
    Measurement: Literal["Aspect Ratio", "Sphericity"] = "Aspect Ratio",
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
    Plot a frequency-weighted histogram of dimensionless morphological descriptors of
    labeled objects in a segmented image. Convenience wrapper around scikit-image's
    exposure.histogram and measure.regionprops, PyDoug's measure tab's bulk and surface
    value calculation widgets, and matplotlib's hist:
    https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.histogram
    https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html

    Parameters
    ----------
    labarray: np.ndarray
        The labeled segmented image.
    Measurement: Literal
        The measurement to perform on each labeled region.
        - "Aspect Ratio": Find the ratio of the longest diameter to the shortest
        diameter for each labeled object.
        - "Sphericity": Find the ratio of the spherical/circular equivalent surface
        area/perimeter to the measured surface area/perimeter (slashes denote 3D/2D).
        1 = perfectly spherical/circular while lower indicates further deviation. In
        practice, some values will be nonsensically greater than 1 due to
        inconsistencies between surface and bulk calculations with pixelated data.
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
                Measurement: np.array([0]),
                "Probability Density" if Normalize else "Counts": np.array([0]),
            }
        )

    labels: np.ndarray = np.unique(values)
    labels = labels[1:] if labels[0] == 0 else labels
    dimless_values: np.ndarray = np.empty(labels.size)

    if Measurement == "Aspect Ratio":
        from skimage.measure import regionprops

        props = regionprops(values)
        for index, region in enumerate(props):
            dimless_values[index] = region.axis_major_length / max(
                region.axis_minor_length, 0.01
            )

    else:
        from pydoug_core.analyze.measure import bulk, surf

        surf_measurement: str = "Area" if is3d(labarray) else "Perimeter"
        bulk_measurement: str = "Volume" if is3d(labarray) else "Area"
        surf_result: PlotResult | ValueResult = surf(values, surf_measurement)
        bulk_result: PlotResult | ValueResult = bulk(values, bulk_measurement)
        assert isinstance(surf_result, ValueResult)
        assert isinstance(bulk_result, ValueResult)
        surf_dict: dict = surf_result.values  # type: ignore
        bulk_dict: dict = bulk_result.values  # type: ignore
        for index, label in enumerate(labels):
            if is3d(labarray):
                dimless_values[index] = (
                    (np.cbrt((bulk_dict[label] * 3) / (4 * np.pi)) ** 2) * 4 * np.pi
                ) / surf_dict[label]
            else:
                dimless_values[index] = (
                    np.sqrt(bulk_dict[label] / np.pi) * 2 * np.pi
                ) / surf_dict[label]

    hist_values, hist_bin_centers, hist_stats = hist_distrib(
        dimless_values, Bin_Count, "values", Normalize
    )
    hist_stats.pop("Dtype Min")
    hist_stats.pop("Dtype Max")

    from pydoug_core.analyze.plot_util import hist_ax

    fig: Figure = Figure()
    ax: Axes = fig.add_subplot()
    ax = hist_ax(
        hist_values,
        hist_bin_centers,
        ax,
        Measurement,
        "Probability Density" if Normalize else "Counts",
        Y_Max,
        None if (X_Lims[0] == X_Lims[1] == 0) else X_Lims,
    )
    fig.set_layout_engine("constrained")

    return PlotResult(
        fig,
        {
            Measurement: hist_bin_centers,
            "Probability Density" if Normalize else "Counts": hist_values,
        },  # type: ignore
        x_columns=(Measurement,),
        tables={"statistics": hist_stats},
    )


@register_operation(
    category="measure",
    display_name="Chord Length Distribution",
    output_kind="plot",
    array_params=("Mask",),
    array_param_flags={"Mask": "Apply_Mask"},
)
def chord_length_distrib(
    imarray: np.ndarray,
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
    Create a ditribution of chord lengths for an image with chords generated using the
    "Chords" widget in the "Features" tab. Convenience wrapper around porespy's
    metrics.chord_counts, scikit-image's exposure.histogram, and matplotlib's hist:
    https://porespy.org/autoapi/porespy/metrics/chord_counts.html
    https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.histogram
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html

    Parameters
    ----------
    imarray: np.ndarray
        The image array with chords already added.
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
    chord_result: PlotResult | ValueResult
        The chord length distribution result object.
    """

    if Apply_Mask and Mask is not None:
        values: np.ndarray = np.copy(imarray)
        values[np.logical_not(prep_mask(imarray, Mask))] = 0
    else:
        values = imarray
    if np.all(np.unique(values) == np.array([0])):
        return ValueResult(
            {
                "Chord Length": np.array([0]),
                "Probability Density" if Normalize else "Counts": np.array([0]),
            }
        )

    from porespy.metrics import chord_counts

    lengths: np.ndarray = chord_counts(imarray) * Pixel_Size
    hist_values, hist_bin_centers, hist_stats = hist_distrib(
        lengths, Bin_Count, "values", Normalize
    )
    hist_stats.pop("Dtype Min")
    hist_stats.pop("Dtype Max")

    from pydoug_core.analyze.plot_util import hist_ax

    display_units: str = r"\mu m" if Units == "um" else Units
    units: dict = {"Probability Density": None} if Normalize else {"Counts": None}
    units.update({"Chord Length": Units})
    fig: Figure = Figure()
    ax: Axes = fig.add_subplot()
    ax = hist_ax(
        hist_values,
        hist_bin_centers,
        ax,
        rf"Chord Length ($\mathrm{{{display_units}}}$)",
        "Probability Density" if Normalize else "Counts",
        Y_Max,
        None if (X_Lims[0] == X_Lims[1] == 0) else X_Lims,
    )
    fig.set_layout_engine("constrained")

    return PlotResult(
        fig,
        {
            "Chord Length": hist_bin_centers,
            "Probability Density" if Normalize else "Counts": hist_values,
        },  # type: ignore
        ("Chord Length",),
        tables={"statistics": hist_stats},
        units=units,
    )

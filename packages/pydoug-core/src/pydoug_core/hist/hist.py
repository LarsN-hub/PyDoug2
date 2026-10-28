"""
Plot and modify an image array's histogram.
"""

from typing import Annotated, Literal

import numpy as np
from skimage.morphology import ball, disk

from pydoug_core import util
from pydoug_core.registry import register_operation
from pydoug_core.results import PlotResult

__all__ = [
    "plot_histogram",
    "invert",
    "normalize",
    "saturate",
    "equalize",
    "adjust",
    "local_contrast",
]


@register_operation(
    category="histogram",
    display_name="Plot Histogram",
    array_params=("Mask",),
    array_param_flags={"Mask": "Apply_Mask"},
    output_kind="plot",
    recordable_toggle=True,
)
def plot_histogram(
    imarray: np.ndarray,
    Bin_Count: Annotated[int, {"min": 1}] = 256,
    Normalize: bool = False,
    Units: str = "",
    X_Label: str = "Gray Value",
    Y_Max: Annotated[float, {"min": 0, "max": 1000000000000000, "step": 0.0001}] = 0,
    X_Lims: tuple[
        Annotated[
            float, {"min": -1000000000000000, "max": 1000000000000000, "step": 0.0001}
        ],
        Annotated[
            float, {"min": -1000000000000000, "max": 1000000000000000, "step": 0.0001}
        ],
    ] = (0, 0),
    Apply_Mask: bool = False,
    Mask: np.ndarray | None = None,
) -> PlotResult:
    """
    Plot an image array's histogram. Convenience wrapper around scikit-image's
    exposure.histogram and matplotlib's hist:
    https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.histogram
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html

    Parameters
    ----------
    imarray: np.ndarray
        The image array to plot the histogram for.
    Bin_Count: int
        The number of bins to use in the histogram. Ignored for integer arrays.
    Normalize: bool
        Whether or not to normalize the counts by the total sum. Default is False.
    Units: str
        (Optional) units to add to the x axis. If not provided, no units are added.
    X_Label: str
        The label for the x-axis. By changing this, this widget can be used to plot
        distributions other than simply grayscale values (e.g., a path length
        distribution after finding path length at each pixel in the "Path Length" widget
        of the "Features" tab).
    Y_Max: float
        The maximum y-axis value to plot. Automatically determined if 0.
    X_Lims: tuple[float, float]
        The limits of the x-axis in the form [min, max]. Ignored if both are 0.
    Apply_Mask: bool
        Whether or not to apply a mask. Default is False.
    Mask: np.ndarray | None
        (Optional) If Apply Mask is True, only values within the provided mask array
        will be included in the histogram.

    Returns
    -------
    plot_result: PlotResult
        The PlotResult object containing the plot figure and the data.
    """

    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from pydoug_core.analyze.plot_distrib import hist_distrib
    from pydoug_core.analyze.plot_util import hist_ax

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Arial"

    if Apply_Mask and Mask is not None:
        values: np.ndarray = imarray[util.prep_mask(imarray, Mask)]
    else:
        values = np.copy(imarray)
    if values.size == 0:
        values = np.array([0], imarray.dtype)
    hist_vals, bin_centers, hist_stats = hist_distrib(
        values, Bin_Count, "values", normalize=Normalize
    )
    display_units: str = r"\mu m" if Units == "um" else Units
    x_label: str = (
        rf"{X_Label} ($\mathrm{{{display_units}}}$)" if Units != "" else X_Label
    )
    units: dict = {"Probability Density" if Normalize else "Counts": None}
    units.update({X_Label: Units if Units != "" else None})

    fig: Figure = Figure()
    ax: Axes = fig.add_subplot()
    ax = hist_ax(
        hist_vals,
        bin_centers,
        ax,
        x_label,
        "Probability Density" if Normalize else "Counts",
        Y_Max,
        None if (X_Lims[0] == X_Lims[1] == 0) else X_Lims,
    )
    fig.set_layout_engine("constrained")

    return PlotResult(
        figure=fig,
        data={
            X_Label: bin_centers,
            "Probability Density" if Normalize else "Counts": hist_vals,
        },  # type: ignore
        x_columns=(X_Label,),
        tables={"statistics": hist_stats},
        units=units,
    )


@register_operation(category="histogram", display_name="Invert")
def invert(imarray: np.ndarray) -> np.ndarray:
    """
    Invert an image's histogram. Convenience wrapper around scikit-image's util.invert:
    https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.invert

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be inverted.

    Returns
    -------
    invarray: np.ndarray
        The ivnerted image array.
    """

    from skimage.util import invert as skinvert

    return skinvert(imarray)


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

    from skimage.exposure import rescale_intensity

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
    Normalizing after saturation enhances the contrast of an image.
    Convenience wrapper around numpy's clip:
    https://numpy.org/doc/stable/reference/generated/numpy.clip.html#numpy.clip

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

    satarray: np.ndarray = np.clip(imarray, min(Bounds), max(Bounds))
    if Normalize_to_Dtype:
        satarray = normalize(satarray)

    return satarray


@register_operation(
    category="histogram",
    display_name="Equalize",
    array_params=("Mask",),
    array_param_flags={"Mask": "Apply_Mask"},
)
def equalize(
    imarray: np.ndarray,
    Method: Literal["Global", "CLAHE", "Rank"] = "CLAHE",
    Radius: Annotated[int, {"min": 1}] = 5,
    CLAHE_Clip_Limit: Annotated[float, {"max": 1, "min": 0}] = 0.5,
    Perform_Slicewise: bool = False,
    Apply_Mask: bool = False,
    Mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Equalize the histogram of an image array. Equalization spreads out the most common
    intensity values along the length of the histogram to improve contrast. Convenience
    wrapper around scikit-image's exposure.equalize_hist (Global),
    exposure.equalize_adapthist (CLAHE), and filters.rank.equalize (Rank):
    https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_hist
    https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
    https://scikit-image.org/docs/stable/api/skimage.filters.rank.html#skimage.filters.rank.equalize

    Parameters
    ----------
    imarray: np.ndarray
        - The image array to be equalized.
    Method: Literal["Global", "CLAHE", "Rank"]
        The method to use for equalization.
        - Global: The entire histogram is used for equalization.
        - CLAHE: (C)ontrast (L)imited (A)daptive (H)istogram (E)qualization uses
        multiple histograms from the image (determined by Radius) for equalization.
        - Rank: Uses the local histogram around each pixel (determined by Radius) for
        equalization.
    Radius: int
        The local radius for the kernel (CLAHE) or rank filter (Rank). Ignored if
        "Global" is selected for "Method".
    CLAHE_Clip_Limit: float
        The clip limit in the range [0, 1] to use for CLAHE. Higher = more contrast.
    Perform_Slicewise: bool
        Whether or not to perform the operation slice-by-slice. Default is False.
    Apply_Mask: bool
        Whether or not to apply a mask to the histogram sampling region before
        equalization. Ignored if "CLAHE" is selected for "Method".
    Mask: np.ndarray
        The mask array to apply if "Apply Mask" is True.

    Returns
    -------
    eqarray: np.ndarray
        The histogram-equalized image array.
    """

    from skimage.exposure import equalize_adapthist, equalize_hist
    from skimage.filters.rank import equalize as equalize_rank

    def _equalize(
        imarray: np.ndarray,
        Method: Literal["Global", "CLAHE", "Rank"],
        Radius: int,
        CLAHE_Clip_Limit: float,
        Apply_Mask: bool,
        Mask: np.ndarray | None,
    ) -> np.ndarray:

        input_dtype: str = util.dtype_2_str(imarray.dtype)

        if Method == "Global":
            if Apply_Mask and Mask is not None:
                if not np.any(Mask):
                    return imarray
                return util.convert_dtype(
                    equalize_hist(imarray, mask=Mask),
                    input_dtype,  # type: ignore
                )
            return util.convert_dtype(equalize_hist(imarray), input_dtype)  # type: ignore

        if Method == "CLAHE":
            return util.convert_dtype(
                equalize_adapthist(imarray, Radius * 2, CLAHE_Clip_Limit),
                input_dtype,  # type: ignore
            )

        if Method == "Rank":
            if util.is3d(imarray):
                footprint = ball(Radius)
            else:
                footprint = disk(Radius)
            if Apply_Mask and Mask is not None:
                return util.convert_dtype(
                    equalize_rank(imarray, footprint, mask=Mask),
                    input_dtype,  # type: ignore
                )
            return util.convert_dtype(equalize_rank(imarray, footprint), input_dtype)  # type: ignore

    if Apply_Mask and Mask is not None:
        Mask = util.prep_mask(imarray, Mask)

    if Perform_Slicewise and util.is3d(imarray):
        eqarray: np.ndarray = np.empty(imarray.shape, imarray.dtype)
        for slice_index in range(0, eqarray.shape[0]):
            if Apply_Mask and Mask is not None:
                eqarray[slice_index] = _equalize(
                    imarray[slice_index],
                    Method,
                    Radius,
                    CLAHE_Clip_Limit,
                    Apply_Mask,
                    Mask[slice_index],
                )

            else:
                eqarray[slice_index] = _equalize(
                    imarray[slice_index],
                    Method,
                    Radius,
                    CLAHE_Clip_Limit,
                    Apply_Mask,
                    None,
                )

        return eqarray

    return _equalize(imarray, Method, Radius, CLAHE_Clip_Limit, Apply_Mask, Mask)


@register_operation(category="histogram", display_name="Adjust")
def adjust(
    imarray: np.ndarray,
    Method: Literal["Gamma", "Log", "Sigmoid"] = "Log",
    Gamma: Annotated[int, {"min": 0}] = 1,
    Gain: Annotated[int, {"min": 1}] = 1,
    Sigmoid_Cutoff: float = 0.5,
    Inverse: bool = False,
) -> np.ndarray:
    """
    Adjust the contrast of an image using one of several methods. Convenience wrapper
    around adjust_gamma, adjust_log, and adjust_sigmoid in scikit-image's exposure
    package:
    https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_gamma
    https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_log
    https://scikit-image.org/docs/stable/api/skimage.exposure.html#skimage.exposure.adjust_sigmoid

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be adjusted.
    Method: Literal["Gamma", "Log", "Sigmoid"]
        The method to use for contrast adjustment.
        - Gamma: perform gamma correction on the image.
        - Log: perform logarithmic correction on the image.
        - Sigmoid: perform sigmoid correction on the image.
    Gamma: float
        The gamma value used if "Gamma" is selected for "Method". Ignored for others.
    Gain: float
        The gain value used by all methods.
    Sigmoid_Cutoff: float
        The cutoff value used if "Sigmoid" is selected for "Method". Ignored for others.
    Inverse: bool
        Whether or not to use an inverse transform if "Log" or "Sigmoid" are selected
        for "Method". Ignored for "Gamma".

    Returns
    -------
    adjarray: np.ndarray
        The contrast-adjusted image array.
    """

    from skimage.exposure import adjust_gamma, adjust_log, adjust_sigmoid

    if Method == "Gamma":
        return adjust_gamma(imarray, Gamma, Gain)

    if Method == "Log":
        return adjust_log(imarray, Gain, Inverse)

    if Method == "Sigmoid":
        return adjust_sigmoid(imarray, Sigmoid_Cutoff, Gain, Inverse)


@register_operation(
    category="histogram",
    display_name="Local Contrast",
    array_params=("Mask",),
    array_param_flags={"Mask": "Apply_Mask"},
)
def local_contrast(
    imarray: np.ndarray,
    Method: Literal["Autolevel", "Local Extrema"] = "Autolevel",
    Radius: Annotated[int, {"min": 1}] = 5,
    Apply_Mask: bool = False,
    Mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Perform local contrast enhancement on an image array. Convenience wrapper around
    autolevel, and enhance_contrast from scikit-image's filters.rank package:
    https://scikit-image.org/docs/stable/api/skimage.filters.rank.html#skimage.filters.rank.autolevel
    https://scikit-image.org/docs/stable/api/skimage.filters.rank.html#skimage.filters.rank.enhance_contrast

    Parameters
    ----------
    imarray: np.ndarray
        The image array to be contrast-enhanced.
    Method: Literal["Autolevel", "Extrema"]
        The method to use for local contrast enhancement. Default is "Autolevel".
    Radius: int
        The radius for the local footprint.
    Apply_Mask: bool
        Whether or not to apply a mask defining the area used for contrast enhancement.
        Default is False.
    Mask: np.ndarray
        If "Apply Mask" is True, the mask array to use.

    Returns
    -------
    enhanced_array: np.ndarray
        The contrast-enhanced image array.
    """

    from skimage.filters.rank import autolevel, enhance_contrast

    if util.is3d(imarray):
        footprint = ball(Radius)
    else:
        footprint = disk(Radius)

    if Method == "Autolevel":
        if Apply_Mask and Mask is not None:
            return autolevel(imarray, footprint, mask=util.prep_mask(imarray, Mask))
        return autolevel(imarray, footprint)

    if Method == "Local Extrema":
        if Apply_Mask and Mask is not None:
            return enhance_contrast(
                imarray, footprint, mask=util.prep_mask(imarray, Mask)
            )
        return enhance_contrast(imarray, footprint)

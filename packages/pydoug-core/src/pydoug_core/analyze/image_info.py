"""
Image information.
"""

from typing import Annotated, Any, Literal

import numpy as np

from pydoug_core import util
from pydoug_core.registry import register_operation
from pydoug_core.results import PlotResult

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
    imarray: np.ndarray,
    Apply_Mask: bool = False,
    Mask: np.ndarray | None = None,
    Perform_Slicewise: bool = False,
    Y_Lims: tuple[
        Annotated[
            float, {"min": -1000000000000000, "max": 1000000000000000, "step": 0.0001}
        ],
        Annotated[
            float, {"min": -1000000000000000, "max": 1000000000000000, "step": 0.0001}
        ],
    ] = (0, 0),
) -> dict[str, float | int] | PlotResult:
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
    Perform_Slicewise: bool
        Whether to perform the operation slicewise and return a plot of the stats for
        each slice (True) or not (False). Default is False (returns single values).
        Note that only the mean, min, max, and stdv are plotted but all values are still
        output upon batch running.
    Y_Lims: tuple[float, float]
        If "Perform Slicewise" is "True", the limits of the plot's y-axis in the form
        [min, max]. Ignored if both are 0.

    Returns
    -------
    statdict: dict[str, float]
        A dictionary containing the stats "Mean", "Median", "Mode", "Std Dev",
        "Variance", "Count", "Min", "Max", "Range", "Dtype Min", and "Dtype Max".
    statplot: PlotResult
        If "Perform Slicewise" is "True", the PlotResult object.
    """

    from scipy.stats import mode

    def _stats(values: np.ndarray) -> dict[str, float | int]:

        if np.issubdtype(values.dtype, np.integer):
            imdmin = np.iinfo(values.dtype).min
            imdmax = np.iinfo(values.dtype).max
            boolarray: bool = False
        elif np.issubdtype(values.dtype, np.floating):
            imdmin = np.finfo(values.dtype).min
            imdmax = np.finfo(values.dtype).max
            boolarray: bool = False
        else:
            imdmin = 0
            imdmax = 1
            boolarray: bool = True

        if values.size == 0:
            return {
                "Mean": np.nan,
                "Median": np.nan,
                "Mode": np.nan,
                "Std Dev": np.nan,
                "Variance": np.nan,
                "Count": 0,
                "Min": np.nan,
                "Max": np.nan,
                "Range": np.nan,
                "Dtype Min": imdmin,
                "Dtype Max": imdmax,
            }

        immean: float = np.mean(values)
        immedian: float | int = np.median(values)
        imstd: float = np.std(values)
        imvar: float = np.var(values)
        imcount: int = np.unique(values).size
        immin: float | int = np.min(values)
        immax: float | int = np.max(values)
        if not boolarray:
            immode: float | int = mode(values, axis=None)[0]  # type: ignore
            imrange: float | int = immax - immin
        else:
            onecount: np.int64 = np.count_nonzero(values)
            zercount: np.int64 = values.size - onecount
            immode: int = 1 if onecount >= zercount else 0
            imrange = int(immax) - int(immin)

        return {
            "Mean": immean,
            "Median": immedian,
            "Mode": immode,
            "Std Dev": imstd,
            "Variance": imvar,
            "Count": imcount,
            "Min": immin,
            "Max": immax,
            "Range": imrange,
            "Dtype Min": imdmin,
            "Dtype Max": imdmax,
        }

    if Perform_Slicewise and util.is3d(imarray):
        stats_dict: dict[str, np.ndarray] = {
            "Slice": np.arange(0, imarray.shape[0]),
            "Mean": np.empty(imarray.shape[0]),
            "Median": np.empty(imarray.shape[0]),
            "Mode": np.empty(imarray.shape[0]),
            "Std Dev": np.empty(imarray.shape[0]),
            "Variance": np.empty(imarray.shape[0]),
            "Count": np.empty(imarray.shape[0], np.int32),
            "Min": np.empty(imarray.shape[0]),
            "Max": np.empty(imarray.shape[0]),
            "Range": np.empty(imarray.shape[0]),
            "Dtype Min": np.empty(imarray.shape[0], imarray.dtype),
            "Dtype Max": np.empty(imarray.shape[0], imarray.dtype),
        }

        for index in range(0, imarray.shape[0]):
            if Apply_Mask and Mask is not None:
                cur_dict: dict[str, float | int] = _stats(
                    imarray[index][util.prep_mask(imarray[index], Mask[index])]
                )
            else:
                cur_dict = _stats(imarray[index])
            stats_dict["Mean"][index] = cur_dict["Mean"]
            stats_dict["Median"][index] = cur_dict["Median"]
            stats_dict["Mode"][index] = cur_dict["Mode"]
            stats_dict["Std Dev"][index] = cur_dict["Std Dev"]
            stats_dict["Variance"][index] = cur_dict["Variance"]
            stats_dict["Count"][index] = cur_dict["Count"]
            stats_dict["Min"][index] = cur_dict["Min"]
            stats_dict["Max"][index] = cur_dict["Max"]
            stats_dict["Range"][index] = cur_dict["Range"]
            stats_dict["Dtype Min"][index] = cur_dict["Dtype Min"]
            stats_dict["Dtype Max"][index] = cur_dict["Dtype Max"]

        from matplotlib.axes import Axes
        from matplotlib.figure import Figure

        from pydoug_core.analyze.plot_util import stats_ax

        y_lims: tuple[float, float] | None = (
            None if (Y_Lims[0] == Y_Lims[1] == 0) else Y_Lims
        )
        fig: Figure = Figure()
        ax: Axes = fig.add_subplot()
        ax = stats_ax(stats_dict, ax, y_lims)
        fig.set_layout_engine("constrained")

        return PlotResult(figure=fig, data=stats_dict, x_columns=("Slice",))  # type: ignore

    if Apply_Mask and Mask is not None:
        values: np.ndarray = imarray[util.prep_mask(imarray, Mask)]
    else:
        values = imarray

    return _stats(values)


@register_operation(
    category="info",
    display_name="Line Scan",
    layer_type="shapes",
    output_kind="value",
    array_params=("Image_Layer",),
    primary_params=("Image_Layer",),
    recordable_toggle=True,
)
def linescan(
    shapes_layer,
    Image_Layer: np.ndarray,
    Y_Lims: tuple[
        Annotated[
            float, {"min": -1000000000000000, "max": 1000000000000000, "step": 0.0001}
        ],
        Annotated[
            float, {"min": -1000000000000000, "max": 1000000000000000, "step": 0.0001}
        ],
    ] = (0, 0),
) -> PlotResult:
    """
    Plot a line scan from a line drawn in a shapes layer. If multiple shapes are drawn
    in the shapes layer, the first line encountered will be used. Fails if no line is
    found.

    Parameters
    ----------
    shapes_layer: Shapes
        The napari shapes layer.
    Image_Layer: np.ndarray
        The image to apply the line to for the line scan.
    Y_Lims: tuple[float, float]
        The limits of the y-axis in the form [min, max]. Ignored if both are 0.

    Returns
    -------
    linescan_plot: PlotResult
        The line scan PlotResult object
    """

    found_coords: bool = False
    coords: np.ndarray = np.array([])
    for index, shape_type in enumerate(shapes_layer.shape_type):
        if shape_type == "line":
            coords = np.copy(shapes_layer.data[index])
            found_coords = True
            break

    if not found_coords:
        raise ValueError("No line found in the shapes layer, line scan not performed.")

    if coords.shape[1] == 3:
        if util.is3d(Image_Layer):
            eval_array: np.ndarray = np.copy(Image_Layer[int(round(coords[0, 0]))])
        else:
            eval_array = Image_Layer
        coords = coords[:, 1:]
    else:
        if util.is3d(Image_Layer):
            eval_array = Image_Layer[0]
        else:
            eval_array = Image_Layer

    y_lims: tuple[float, float] | None = (
        None if (Y_Lims[0] == Y_Lims[1] == 0) else Y_Lims
    )

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from skimage.measure import profile_line

    from pydoug_core.analyze.plot_util import line_ax

    line_vals: np.ndarray = profile_line(
        eval_array, coords[0, :], coords[1, :], order=0, mode="constant", cval=0
    )
    pos_vals: np.ndarray = np.linspace(0, len(line_vals), num=len(line_vals))
    fig: Figure = Figure()
    ax: Axes = fig.add_subplot()
    ax = line_ax(pos_vals, line_vals, ax, y_lims, "Distance (pixels)", "Gray Value")
    fig.set_layout_engine("constrained")

    return PlotResult(
        figure=fig,
        data={"Distance (pixels)": pos_vals, "Gray Value": line_vals},  # type: ignore
        x_columns=("Distance (pixels)",),
    )


@register_operation(
    category="info",
    display_name="Percentiles",
    output_kind="value",
    recordable_toggle=True,
    array_params=("Mask",),
    array_param_flags={"Mask": "Apply_Mask"},
)
def percentiles(
    imarray: np.ndarray,
    Percentiles: tuple[
        Annotated[float, {"step": 0.0001, "max": 100, "min": 0}],
        Annotated[float, {"step": 0.0001, "max": 100, "min": 0}],
    ] = (1, 99),
    Apply_Mask: bool = False,
    Mask: np.ndarray | None = None,
    Perform_Slicewise: bool = False,
) -> dict[float, float] | PlotResult:
    """
    Calculate the lower and upper intensity values for the given percentiles.
    Convenience wrapper around numpy's percentile:
    https://numpy.org/doc/stable/reference/generated/numpy.percentile.html#numpy-percentile

    Parameters
    ----------
    imarray: np.ndarray
        The image array to calculate percentile intensities for.
    Percentiles: tuple[float, float]
        A tuple containing the lower and upper percentiles in the range [0, 100]. Values
        outside [0, 100] will be clipped to 0 (first value) or 100 (second value).
    Apply_Mask: bool = False
        Whether or not to apply a mask to the operation. Default is False.
    Mask: np.ndarray | None
        (Optional) the mask to be applied before statistics calculation if "Apply_Mask"
        is True.
    Perform_Slicewise: bool
        Whether to perform the operation slicewise and return a plot of the percentiles
        for each slice (True) or not (False). Default is False (returns single values).

    Returns
    -------
    intensities: dict[float, float]
        The percentile values in a dictionary containing keys for each given percentile.
    intensities_plot: PlotResult
        A PlotResult object containing the plot and data for the percentile intensities
        for each slice.
    """

    def _percentiles(
        values: np.ndarray,
        Percentiles: tuple[float, float],
    ) -> dict[float, float]:

        if values.size == 0:
            return {min(Percentiles): np.nan, max(Percentiles): np.nan}

        if Percentiles == (0, 100):
            return {0: np.min(values), 100: np.max(values)}

        return {
            min(Percentiles): np.percentile(values, min(Percentiles)),
            max(Percentiles): np.percentile(values, max(Percentiles)),
        }

    if Percentiles[0] > 100 or Percentiles[0] < 0:
        Percentiles = (0, Percentiles[1])
    if Percentiles[1] > 100 or Percentiles[1] < 0:
        Percentiles = (Percentiles[0], 100)

    if Perform_Slicewise and util.is3d(imarray):
        min_array: np.ndarray = np.empty(imarray.shape[0])
        max_array: np.ndarray = np.copy(min_array)
        slice_array: np.ndarray = np.arange(0, imarray.shape[0])
        if Apply_Mask and Mask is not None:
            Mask = util.prep_mask(imarray, Mask)
        for index in range(0, imarray.shape[0]):
            if Apply_Mask and Mask is not None:
                cur_percentiles: dict = _percentiles(
                    imarray[index][Mask[index]], Percentiles
                )
            else:
                cur_percentiles: dict = _percentiles(imarray[index], Percentiles)
            min_array[index] = cur_percentiles[min(Percentiles)]
            max_array[index] = cur_percentiles[max(Percentiles)]

        from matplotlib.axes import Axes
        from matplotlib.figure import Figure

        from pydoug_core.analyze.plot_util import multi_line

        fig: Figure = Figure()
        ax: Axes = fig.add_subplot()
        ax = multi_line(
            slice_array,
            [min_array, max_array],
            ax,
            "Slice",
            "Gray Value",
            data_labels=[f"{min(Percentiles)}", f"{max(Percentiles)}"],
            colors=["blue", "red"],
        )
        fig.set_layout_engine("constrained")

        return PlotResult(
            figure=fig,
            data={
                "Slice": slice_array,
                f"{min(Percentiles)}": min_array,
                f"{max(Percentiles)}": max_array,
            },  # type: ignore
            x_columns=("Slice",),
        )

    if Apply_Mask and Mask is not None:
        return _percentiles(imarray[util.prep_mask(imarray, Mask)], Percentiles)

    return _percentiles(imarray, Percentiles)


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

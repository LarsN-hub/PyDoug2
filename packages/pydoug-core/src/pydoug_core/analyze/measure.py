"""
Measured and calculated values of images.
"""

from typing import Annotated, Literal

import numpy as np

from pydoug_core import util
from pydoug_core.registry import register_operation
from pydoug_core.results import PlotResult

__all__ = ["bulk", "mesh_surface_area", "label_count"]


@register_operation(
    category="measure",
    display_name="Bulk Value",
    array_params=("Mask",),
    array_param_flags={"Mask": "Apply_Mask"},
    output_kind="value",
)
def bulk(
    imarray: np.ndarray,
    Measurement: Literal["Area", "Length", "Volume"] = "Volume",
    Pixel_Size: Annotated[float, {"min": 0, "step": 0.0001}] = 1,
    Units: str = "pix",
    Normalize: bool = False,
    Normalization_Method: Literal["Dataset", "Segmented", "Phase"] = "Dataset",
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
    Plot_Background: bool = False,
    Normalize_by_Slice: bool = False,
) -> dict | PlotResult:
    """
    Measure bulk values (area, length, volume) of segmented data. If there are multiple
    segmented phases, the measurement for each unique intensity is returned.
    Convenience wrapper around numpy's count_nonzero:
    https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html

    Parameters
    ----------
    imarray: np.ndarray
        The image to measure.
    Measurement: Literal
        The measurement to perform.
        - "Area": Multiplies number of segmented pixels by "Pixel Size" ^ 2.
        - "Length": Multiplies number of segmented pixels by "Pixel Size".
        - "Volume": Multiplies number of segmented pixels by "Pixel Size" ^ 3.
    Pixel_Size: float
        The edge-length of pixels in real-world units.
    Units: str
        The real-world units for "Pixel Size". If "um" is provided, the u will be
        converted to a micro symbol automatically.
    Normalize: bool
        Whether to normalize the output by dividing it by a normalization factor
        specified in "Normalization Method" (True) or to leave the value absolute
        (False). Default is False.
    Normalization_Method: Literal
        If "Normalize" is "True", the normalization method to use. Default is "Dataset".
        - "Dataset": Divides each output by the total measurement for the dataset (e.g.,
        divide each volume by the dataset volume). Affected by "Mask" if used.
        - "Segmented": Same as "Dataset" but excludes background values from the value
        used for normalization. Affected by "Mask" if used.
        - "Phase": Divides each output by its total measurement. Will return 1 unless
        "Perform Slicewise" is "True" and "Normalize by Slice" is "False". Affected by
        "Mask" if used.
    Apply_Mask: bool
        Whether to mask the region where the value is measured (True) or not (False).
    Mask: np.ndarray | None
        If "Apply Mask" is "True", the mask to use.
    Perform_Slicewise: bool
        Whether to perform the operation slicewise and produce a plot of the measured
        value along the z-axis (True) or to return a single value for the entire dataset
        (False). Default is False. Ignored for 2D images.
    Y_Lims: tuple[float, float]
        If "Perform Slicewise" is "True", the limits of the plot's y-axis in the form
        [min, max]. Ignored if both are 0.
    Plot_Background: bool
        If "Perform Slicewise" is "True", whether to plot the background phase (True)
        or to only plot non-zero segmented phases (False). Default is False.
    Normalize_by_Slice: bool
        If "Perform Slicewise" and "Normalize" are both "True", whether to normalize
        each output by the values measured for each slice (True) or by the bulk value
        for the entire dataset (False). Default is False.

    Returns
    -------
    bulk_dict: dict
        A dictionary containing the bulk measurement for each unique intensity in the
        dataset.
    bulk_plot: PlotResult
        The PlotResult object if "Perform Slicewise" is "True".
    """

    if Units == "um":
        Units = "\u00b5m"
    slice_key: str = "Slice"

    def _bulk(values: np.ndarray) -> dict:
        unique_vals: np.ndarray = np.unique(values)
        bulk_dict: dict = {}
        for val in unique_vals:
            bulk_dict[val] = np.count_nonzero(values == val)

        return bulk_dict

    def _norm(bulk_dict: dict) -> dict:
        if Normalization_Method == "Dataset":
            dset_tot: int = (
                imarray[0].size
                if Normalize_by_Slice and util.is3d(imarray)
                else imarray.size
            )
            for key in list(bulk_dict.keys()):
                if key != slice_key:
                    bulk_dict[key] /= dset_tot
        elif Normalization_Method == "Segmented":
            seg_tot: np.integer | np.ndarray = (
                np.count_nonzero(imarray, axis=(1, 2))
                if Normalize_by_Slice and util.is3d(imarray)
                else np.count_nonzero(imarray)
            )
            for key in list(bulk_dict.keys()):
                if key != slice_key:
                    bulk_dict[key] /= seg_tot
        else:
            for key in list(bulk_dict.keys()):
                if key != slice_key:
                    eval_array: np.ndarray = imarray == key
                    phs_tot: np.integer | np.ndarray = (
                        np.count_nonzero(eval_array, axis=(1, 2))
                        if Normalize_by_Slice and util.is3d(imarray)
                        else np.count_nonzero(eval_array)
                    )
                    bulk_dict[key] /= phs_tot

        return bulk_dict

    if Perform_Slicewise and util.is3d(imarray):
        unique_vals: np.ndarray = (
            np.unique(imarray[util.prep_mask(imarray, Mask)])
            if Apply_Mask and Mask is not None
            else np.unique(imarray)
        )
        bulk_dict = {
            slice_key: np.arange(0, imarray.shape[0]),
        }
        for val in unique_vals:
            bulk_dict[val] = np.empty(imarray.shape[0])
        for index in range(0, imarray.shape[0]):
            cur_bulk_dict: dict = (
                _bulk(imarray[index][util.prep_mask(imarray, Mask)[index]])
                if Apply_Mask and Mask is not None
                else _bulk(imarray[index])
            )
            for val in unique_vals:
                bulk_dict[val][index] = (
                    cur_bulk_dict[val] if val in cur_bulk_dict.keys() else 0
                )
        if Normalize:
            bulk_dict = _norm(bulk_dict)
            if Normalization_Method == "Dataset":
                y_label: str = (
                    f"Fraction of Total {Measurement}"
                    if not Normalize_by_Slice
                    else f"Fraction of Positional {Measurement}"
                )
            elif Normalization_Method == "Segmented":
                y_label = (
                    f"Fraction of Segmented {Measurement}"
                    if not Normalize_by_Slice
                    else f"Fraction of Positional Segmented {Measurement}"
                )
            else:
                y_label = (
                    f"Fraction of Phase {Measurement}"
                    if not Normalize_by_Slice
                    else f"Fraction of Positional Phase {Measurement}"
                )
        else:
            if Measurement == "Area":
                y_label = f"Area ({Units}\u00b2)"
                for key in list(bulk_dict.keys()):
                    if key != slice_key:
                        bulk_dict[key] *= Pixel_Size**2
            elif Measurement == "Volume":
                y_label = f"Volume ({Units}\u00b3)"
                for key in list(bulk_dict.keys()):
                    if key != slice_key:
                        bulk_dict[key] *= Pixel_Size**3
            else:
                y_label = f"Length ({Units})"
                for key in list(bulk_dict.keys()):
                    if key != slice_key:
                        bulk_dict[key] *= Pixel_Size

        from matplotlib.axes import Axes
        from matplotlib.figure import Figure

        from pydoug_core.analyze.plot_util import multi_line

        fig: Figure = Figure()
        ax: Axes = fig.add_subplot()
        y_vals: list = []
        data_labels: list = []
        for key in list(bulk_dict.keys()):
            if key != slice_key:
                if key == 0 and not Plot_Background:
                    continue
                y_vals.append(bulk_dict[key])
                data_labels.append(key)
        y_lims: tuple[float, float] | None = (
            None if (Y_Lims[0] == Y_Lims[1] == 0) else Y_Lims
        )
        ax = multi_line(
            bulk_dict[slice_key],
            y_vals,
            ax,
            f"Position ({Units})",
            y_label,
            y_lims,
            data_labels,
            None,
            True,
        )
        fig.set_layout_engine("constrained")

        return PlotResult(fig, bulk_dict, (slice_key,))

    if Apply_Mask and Mask is not None:
        bulk_dict: dict = _bulk(imarray[util.prep_mask(imarray, Mask)])
    else:
        bulk_dict = _bulk(imarray)
    if Normalize:
        bulk_dict = _norm(bulk_dict)
    else:
        for key in list(bulk_dict.keys()):
            if Measurement == "Area":
                bulk_dict[key] *= Pixel_Size**2
            elif Measurement == "Volume":
                bulk_dict[key] *= Pixel_Size**3
            else:
                bulk_dict[key] *= Pixel_Size

    return bulk_dict


@register_operation(
    category="measure",
    display_name="Mesh Surface Area",
    layer_type="surface",
    output_kind="value",
    recordable_toggle=True,
)
def mesh_surface_area(
    surface: tuple,
    Pixel_Size: Annotated[float, {"min": 0, "max": 1000000, "step": 0.0001}] = 1,
) -> float:  # pragma: no cover
    """
    Compute the surface area of a mesh. Convenience wrapper around scikit-image's
    measure.mesh_surface_area:
    https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.mesh_surface_area

    Parameters
    ----------
    surface: tuple
        The surface tuple.
    Pixel_Size: float
        The edge length of the pixels in real-world units.

    Returns
    -------
    surf_area: float
        The surface area.
    """

    from skimage.measure import mesh_surface_area

    return mesh_surface_area(surface[0], surface[1]) * (Pixel_Size**2)


@register_operation(
    category="measure",
    display_name="Label Count",
    layer_type="labels",
    output_kind="value",
    recordable_toggle=True,
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

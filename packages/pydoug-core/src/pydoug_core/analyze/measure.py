"""
Measured and calculated values of images.
"""

from typing import Annotated, Literal

import cmasher  # noqa: F401 — imported for its expansion of matplotlib's color maps
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pydoug_core import util
from pydoug_core.registry import register_operation
from pydoug_core.results import PlotResult, ValueResult

__all__ = [
    "bulk",
    "surf",
    "heat_map",
    "label_count",
    "mesh_surface_area",
    "fractal_dim",
]


@register_operation(
    category="measure",
    display_name="Bulk Values",
    array_params=("Mask",),
    array_param_flags={"Mask": "Apply_Mask"},
    output_kind="value",
    recordable_toggle=True,
)
def bulk(
    imarray: np.ndarray,
    Measurement: Literal["Area", "Length", "Volume"] = "Volume",
    Pixel_Size: Annotated[float, {"min": 0, "step": 0.0001}] = 1,
    Units: str = "pixels",
    Normalize: bool = False,
    Normalization_Method: Literal["Phase", "Segmented", "Total"] = "Total",
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
) -> PlotResult | ValueResult:
    """
    Measure bulk values (area, length, volume) of segmented phases. If there are
    multiple phases, the measurement for each unique intensity is returned.
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
        The real-world units for "Pixel Size".
    Normalize: bool
        Whether to normalize the output by dividing it by a normalization factor
        specified in "Normalization Method" (True) or to leave the value absolute
        (False). Default is False.
    Normalization_Method: Literal
        If "Normalize" is "True", the normalization method to use. Default is "Total".
        - "Total": Divides each output by the total measurement for the dataset (e.g.,
        divide each volume by the dataset volume).
        - "Segmented": Same as "Total" but excludes background values from the value
        used for normalization.
        - "Phase": Divides each output by its total measurement. Will return 1 unless
        "Perform Slicewise" is "True" and "Normalize by Slice" is "False".
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
        each output by the values measured for each slice (True) or by the value for the
        entire dataset (False). Default is False.

    Returns
    -------
    bulk_dict: ValueResult
        A ValueResult containing the bulk measurement for each unique intensity in the
        dataset and units.
    bulk_plot: PlotResult
        The PlotResult object if "Perform Slicewise" is "True".
    """

    slice_key: str = "Slice"
    units: dict[str | int | float, str | None] = {slice_key: Units}
    if Normalize_by_Slice and not (Perform_Slicewise and util.is3d(imarray)):
        Normalize_by_Slice = False  # pragma: no cover

    def _bulk(values: np.ndarray) -> dict:

        unique_vals: np.ndarray = np.unique(values)
        bulk_dict: dict = {}
        for val in unique_vals:
            bulk_dict[val] = np.count_nonzero(values == val)

        return bulk_dict

    def _norm(bulk_dict: dict) -> dict:

        if Normalization_Method == "Total":
            dset_tot: int = (
                imarray[0].size
                if Normalize_by_Slice and util.is3d(imarray)
                else imarray.size
            )
            for key in bulk_dict.keys():
                if key != slice_key:
                    bulk_dict[key] /= dset_tot

        elif Normalization_Method == "Segmented":
            seg_tot: np.integer | np.ndarray = (
                np.count_nonzero(imarray, axis=(1, 2))
                if Normalize_by_Slice and util.is3d(imarray)
                else np.count_nonzero(imarray)
            )
            for key in bulk_dict.keys():
                if key != slice_key:
                    bulk_dict[key] /= seg_tot

        else:
            for key in bulk_dict.keys():
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
        display_units: str = r"\mu m" if Units == "um" else Units
        unique_vals: np.ndarray = (
            np.unique(imarray[util.prep_mask(imarray, Mask)])
            if Apply_Mask and Mask is not None
            else np.unique(imarray)
        )
        bulk_dict = {slice_key: np.arange(0, imarray.shape[0])}
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
            for key in bulk_dict.keys():
                units[key] = None if key != slice_key else units[slice_key]
            if Normalization_Method == "Total":
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
                y_label = rf"Area ($\mathrm{{{display_units}^{{2}}}}$)"
                for key in bulk_dict.keys():
                    if key != slice_key:
                        bulk_dict[key] *= Pixel_Size**2
                        units[key] = f"{Units}^2"
            elif Measurement == "Volume":
                y_label = rf"Volume ($\mathrm{{{display_units}^{{3}}}}$)"
                for key in bulk_dict.keys():
                    if key != slice_key:
                        bulk_dict[key] *= Pixel_Size**3
                        units[key] = f"{Units}^3"
            else:
                y_label = rf"Length ($\mathrm{{{display_units}}}$)"
                for key in bulk_dict.keys():
                    if key != slice_key:
                        bulk_dict[key] *= Pixel_Size
                        units[key] = Units

        from pydoug_core.analyze.plot_util import multi_line

        fig: Figure = Figure()
        ax: Axes = fig.add_subplot()
        y_vals: list = []
        data_labels: list = []
        for key in bulk_dict.keys():
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
            rf"Position ($\mathrm{{{display_units}}}$)",
            y_label,
            y_lims,
            data_labels,
            None,
            True,
        )
        fig.set_layout_engine("constrained")

        return PlotResult(fig, bulk_dict, (slice_key,), units=units)

    if Apply_Mask and Mask is not None:
        bulk_dict: dict = _bulk(imarray[util.prep_mask(imarray, Mask)])
    else:
        bulk_dict = _bulk(imarray)

    if Normalize:
        bulk_dict = _norm(bulk_dict)
        for key in bulk_dict.keys():
            units[key] = None if key != slice_key else units[slice_key]
    else:
        for key in bulk_dict.keys():
            if Measurement == "Area":
                bulk_dict[key] *= Pixel_Size**2
                units[key] = f"{Units}^2"
            elif Measurement == "Volume":
                bulk_dict[key] *= Pixel_Size**3
                units[key] = f"{Units}^3"
            else:
                bulk_dict[key] *= Pixel_Size
                units[key] = Units

    return ValueResult(bulk_dict, units)


@register_operation(
    category="measure",
    display_name="Surface Values",
    array_params=("Mask",),
    array_param_flags={"Mask": "Apply_Mask"},
    output_kind="value",
    recordable_toggle=True,
)
def surf(
    imarray: np.ndarray,
    Measurement: Literal["Area", "Perimeter"] = "Area",
    Include_Phase_Breakdown: bool = False,
    Pixel_Size: Annotated[float, {"min": 0, "step": 0.0001}] = 1,
    Units: str = "pixels",
    Correct_Overestimation: bool = True,
    Normalize: bool = False,
    Normalization_Method: Literal[
        "Phase", "Total", "Specific - Phase", "Specific - Segmented", "Specific - Total"
    ] = "Total",
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
    Normalize_by_Slice: bool = False,
) -> PlotResult | ValueResult:
    """
    Measure surface values (area or perimeter) of segmented phases. If there are
    multiple segmented phases, the measurement for each unique intensity is returned.
    Convenience wrapper around numpy's append, count_nonzero, hstack, vstack, and
    boolean indexing:
    https://numpy.org/doc/stable/reference/generated/numpy.append.html
    https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html
    https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
    https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
    https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing

    Parameters
    ----------
    imarray: np.ndarray
        The image to measure.
    Measurement: Literal
        The measurement to perform.
        - "Area": Multiplies number of dissimilar pixel borders by "Pixel Size" ^ 2.
        - "Perimeter": Multiplies number of dissimilar pixel borders by "Pixel Size".
    Include_Phase_Breakdown: bool
        Whether to include the portion of each phase's surface area in contact with each
        other phase (True) or just the total surface area of each phase (False). Default
        is False. Ignored if there are only two phases present.
    Pixel_Size: float
        The edge-length of pixels in real-world units.
    Units: str
        The real-world units for "Pixel Size".
    Correct_Overestimation: bool
        Whether to correct for overestimation of surface measurements inherent to
        pixelated data with Rajon's factors (True) or not (False). Default is True. If
        True, 3D values are multiplied by 2/3 while 2D values are multiplied by pi/4.
        Info: https://onlinelibrary.wiley.com/doi/abs/10.1118/1.1470207
    Normalize: bool
        Whether to normalize the output by dividing it by a normalization factor
        specified in "Normalization Method" (True) or to leave the value absolute
        (False). Default is False.
    Normalization_Method: Literal
        If "Normalize" is "True", the normalization method to use. Default is "Total".
        - "Phase": Divides the surface value of each phase by the phase's respective
        total surface value. Returns 1 if the operation is not performed slicewise and
        normalized by slice.
        - "Total": Divides the surface value of each phase by the total surface value.
        Returns 1 if there are only two phases (foreground/background) and the operation
        is not performed slicewise.
        - "Specific - Phase": Find the phase-specific surface value by dividing the
        surface value by the corresponding bulk value (volume for 3D surface area, area
        for 2D perimeter) for that phase.
        - "Specific - Segmented": Find the segmented-specific surface value by dividing
        the surface value by the corresponding bulk value (volume for 3D surface area,
        area for 2D perimeter) for the total segmented bulk value in the dataset.
        - "Specific - Total": Find the specific surface value by dividing the surface
        value by the corresponding bulk value (volume for 3D surface area, area for 2D
        perimeter) for the total dataset.
    Apply_Mask: bool
        Whether to mask the region where the value is measured (True) or not (False).
    Mask: np.ndarray | None
        If "Apply Mask" is "True", the mask to use.
    Perform_Slicewise: bool
        Whether to perform the operation slicewise and produce a plot of the measured
        value along the z-axis (True) or to return a single value for the entire dataset
        (False). Default is False. Ignored for 2D images.
        IMPORTANTLY, slicewise surface value measurements lose inter-slice surfaces.
        These surfaces are subtracted from the total for normalization, but the total
        value will not be the same as the single value measurement alternative.
    Y_Lims: tuple[float, float]
        If "Perform Slicewise" is "True", the limits of the plot's y-axis in the form
        [min, max]. Ignored if both are 0.
    Normalize_by_Slice: bool
        If "Perform Slicewise" and "Normalize" are both "True", whether to normalize
        each output by the values measured for each slice (True) or by the value for the
        entire dataset (False). Default is False.

    Returns
    -------
    surf_dict: ValueResult
        A ValueResult containing the surface measurement for each unique intensity in
        the dataset and units.
    surf_plot: PlotResult
        The PlotResult object if "Perform Slicewise" is "True".
    """

    slice_key: str = "Slice"
    units: dict[str | int | float, str | None] = {slice_key: Units}
    if Normalize_by_Slice and not (Perform_Slicewise and util.is3d(imarray)):
        Normalize_by_Slice = False  # pragma: no cover

    def _surf(values: np.ndarray, apply_mask: bool, mask: np.ndarray | None) -> dict:

        if apply_mask and mask is not None:
            mask = util.prep_mask(values, mask)
            unique_vals: np.ndarray = np.unique(values[mask])
        else:
            unique_vals: np.ndarray = np.unique(values)
        if unique_vals.size == 1:
            return {unique_vals[0]: 0}
        elif unique_vals.size == 2:
            surf_dict: dict = {}
        else:
            surf_dict = {"Total": 0}

        for val in unique_vals:
            bw_values: np.ndarray = values == val
            shift_arrays: list[np.ndarray] = []
            tot_count = 0
            if util.is3d(values):
                shift_arrays.append(
                    np.append(bw_values[1:, :, :], bw_values[-1:, :, :], 0)
                )
                shift_arrays.append(
                    np.append(bw_values[:1, :, :], bw_values[:-1, :, :], 0)
                )
                shift_arrays.append(
                    np.append(bw_values[:, 1:, :], bw_values[:, -1:, :], 1)
                )
                shift_arrays.append(
                    np.append(bw_values[:, :1, :], bw_values[:, :-1, :], 1)
                )
                shift_arrays.append(
                    np.append(bw_values[:, :, 1:], bw_values[:, :, -1:], 2)
                )
                shift_arrays.append(
                    np.append(bw_values[:, :, :1], bw_values[:, :, :-1], 2)
                )
            else:
                shift_arrays.append(np.vstack((bw_values[1:, :], bw_values[-1:, :])))
                shift_arrays.append(np.vstack((bw_values[:1, :], bw_values[:-1, :])))
                shift_arrays.append(np.hstack((bw_values[:, 1:], bw_values[:, -1:])))
                shift_arrays.append(np.hstack((bw_values[:, :1], bw_values[:, :-1])))
            for array in shift_arrays:
                tot_count += np.count_nonzero(array * np.logical_not(bw_values))
            surf_dict[val] = tot_count
            if val != 0 and "Total" in surf_dict.keys():
                surf_dict["Total"] += tot_count
            if unique_vals.size > 2 and Include_Phase_Breakdown:
                for contact_val in unique_vals:
                    if (
                        contact_val != val
                        and f"{contact_val}/{val}" not in surf_dict.keys()
                        and f"{val}/{contact_val}" not in surf_dict.keys()
                    ):
                        cont_count = 0
                        for array in shift_arrays:
                            cont_count += np.count_nonzero(
                                array
                                & np.logical_not(bw_values)
                                & (values == contact_val)
                            )
                        surf_dict[f"{val}/{contact_val}"] = cont_count

        return surf_dict

    def _norm(surf_dict: dict) -> dict:

        norm_keys: list = list(surf_dict.keys())
        if slice_key in surf_dict.keys():
            norm_keys.remove(slice_key)
        if Normalization_Method in (
            "Specific - Phase",
            "Specific - Segmented",
            "Specific - Total",
        ):
            if Measurement == "Area":
                bulk_result: PlotResult | ValueResult = bulk(
                    imarray,
                    "Volume",
                    Pixel_Size=Pixel_Size,
                    Apply_Mask=Apply_Mask,
                    Mask=Mask,
                    Perform_Slicewise=(
                        Normalize_by_Slice and Perform_Slicewise and util.is3d(imarray)
                    ),
                )
            else:
                bulk_result: PlotResult | ValueResult = bulk(
                    imarray,
                    "Area",
                    Pixel_Size=Pixel_Size,
                    Apply_Mask=Apply_Mask,
                    Mask=Mask,
                    Perform_Slicewise=(
                        Normalize_by_Slice and Perform_Slicewise and util.is3d(imarray)
                    ),
                )
            if isinstance(bulk_result, ValueResult):
                bulk_dict: dict = bulk_result.values  # type: ignore
            else:
                bulk_dict: dict = bulk_result.data
            dset_tot = (
                0
                if not isinstance(bulk_dict[list(bulk_dict.keys())[0]], np.ndarray)
                else np.zeros(imarray.shape[0])
            )
            for val in bulk_dict.keys():
                if (
                    val != 0 or Normalization_Method == "Specific - Total"
                ) and val != slice_key:
                    dset_tot += bulk_dict[val]
            if Normalization_Method == "Specific - Phase":
                if "Total" in norm_keys:
                    norm_keys.remove("Total")
                    surf_dict.pop("Total")
                str_keys: list[str] = []
                str_keys_dict: dict[str, int] = {}
                if Include_Phase_Breakdown:
                    str_keys = [key for key in norm_keys if isinstance(key, str)]
                    for key in str_keys:
                        str_keys_dict[key] = 0
                for val in bulk_dict.keys():
                    if val in surf_dict.keys() and val != slice_key:
                        surf_dict[val] /= bulk_dict[val]
                        for key in str_keys:
                            if str(val) in key.split("/", 1) and str_keys_dict[key] < 2:
                                new_name: str = f"{key} by {val}"
                                str_keys_dict[key] += 1
                                if str_keys_dict[key] == 2:
                                    surf_dict[new_name] = surf_dict.pop(key)
                                else:
                                    surf_dict[new_name] = surf_dict[key].copy()
                                surf_dict[new_name] /= bulk_dict[val]
            else:
                for val in surf_dict.keys():
                    if val != slice_key:
                        surf_dict[val] /= dset_tot
        elif Normalization_Method == "Phase":
            if "Total" in norm_keys:
                norm_keys.remove("Total")
                surf_dict.pop("Total")
            str_keys: list[str] = []
            str_keys_dict: dict[str, int] = {}
            if Include_Phase_Breakdown:
                phase_keys: list = [
                    key for key in norm_keys if not isinstance(key, str)
                ]
                str_keys = [key for key in norm_keys if isinstance(key, str)]
                for key in str_keys:
                    str_keys_dict[key] = 0
            else:
                phase_keys = norm_keys
            for phase in phase_keys:
                phase_tot = (
                    np.sum(surf_dict[phase])
                    if (
                        isinstance(surf_dict[phase], np.ndarray)
                        and not Normalize_by_Slice
                    )
                    else surf_dict[phase].copy()
                )
                surf_dict[phase] /= phase_tot
                if Include_Phase_Breakdown:
                    for key in str_keys:
                        if (str(phase) in key.split("/", 1)) and str_keys_dict[key] < 2:
                            new_name: str = f"{key} by {phase}"
                            str_keys_dict[key] += 1
                            if str_keys_dict[key] == 2:
                                surf_dict[new_name] = surf_dict.pop(key)
                            else:
                                surf_dict[new_name] = surf_dict[key].copy()
                            surf_dict[new_name] /= phase_tot
        else:
            tot_key = "Total" if "Total" in surf_dict.keys() else 0
            dset_tot = (
                np.sum(surf_dict[tot_key])
                if (
                    isinstance(surf_dict[tot_key], np.ndarray)
                    and not Normalize_by_Slice
                )
                else surf_dict[tot_key].copy()
            )
            for key in norm_keys:
                surf_dict[key] /= dset_tot
        return surf_dict

    if Perform_Slicewise and util.is3d(imarray):
        display_units: str = r"\mu m" if Units == "um" else Units
        unique_vals: np.ndarray = (
            np.unique(imarray[util.prep_mask(imarray, Mask)])
            if Apply_Mask and Mask is not None
            else np.unique(imarray)
        )
        surf_dict = {slice_key: np.arange(0, imarray.shape[0])}
        if unique_vals.size > 2:
            surf_dict["Total"] = np.empty(imarray.shape[0])
        for val in unique_vals:
            surf_dict[val] = np.empty(imarray.shape[0])
            if unique_vals.size > 2 and Include_Phase_Breakdown:
                for contact_val in unique_vals:
                    if (
                        contact_val != val
                        and f"{contact_val}/{val}" not in surf_dict.keys()
                        and f"{val}/{contact_val}" not in surf_dict.keys()
                    ):
                        surf_dict[f"{val}/{contact_val}"] = np.empty(imarray.shape[0])

        if Apply_Mask and Mask is not None:
            Mask = util.prep_mask(imarray, Mask)
        for index in range(0, imarray.shape[0]):
            if Apply_Mask and Mask is not None:
                cur_surf_dict: dict = _surf(imarray[index], Apply_Mask, Mask[index])
            else:
                cur_surf_dict: dict = _surf(imarray[index], False, None)
            for key in cur_surf_dict.keys():
                surf_dict[key][index] = (
                    cur_surf_dict[key] if key in cur_surf_dict.keys() else 0
                )

        if Normalize:
            if Normalization_Method == "Total":
                y_label: str = (
                    f"Fraction of Total Surface {Measurement}"
                    if not Normalize_by_Slice
                    else f"Fraction of Positional Surface {Measurement}"
                )
                for key in list(surf_dict.keys()):
                    units[key] = None if key != slice_key else units[slice_key]
            elif Normalization_Method == "Phase":
                y_label = (
                    f"Fraction of Phase Surface {Measurement}"
                    if not Normalize_by_Slice
                    else f"Fraction of Positional Phase Surface {Measurement}"
                )
                for key in list(surf_dict.keys()):
                    units[key] = None if key != slice_key else units[slice_key]
            elif Normalization_Method == "Specific - Total":
                y_label = (
                    "Specific Surface "
                    rf"{Measurement} ($\mathrm{{{display_units}^{{-1}}}}$)"
                    if not Normalize_by_Slice
                    else "Positional Specific Surface "
                    rf"{Measurement} ($\mathrm{{{display_units}^{{-1}}}}$)"
                )
            elif Normalization_Method == "Specific - Segmented":
                y_label = (
                    "Segmented-Specific Surface "
                    rf"{Measurement} ($\mathrm{{{display_units}^{{-1}}}}$)"
                    if not Normalize_by_Slice
                    else "Positional Segmented-Specific Surface "
                    rf"{Measurement} ($\mathrm{{{display_units}^{{-1}}}}$)"
                )
            else:
                y_label = (
                    "Phase-Specific Surface "
                    rf"{Measurement} ($\mathrm{{{display_units}^{{-1}}}}$)"
                    if not Normalize_by_Slice
                    else "Positional Phase-Specific Surface "
                    rf"{Measurement} ($\mathrm{{{display_units}^{{-1}}}}$)"
                )
            if Normalization_Method in (
                "Specific - Phase",
                "Specific - Segmented",
                "Specific - Total",
            ):
                for key in surf_dict.keys():
                    if key != slice_key:
                        units[key] = (
                            f"{Units}^(-1)" if key != slice_key else units[slice_key]
                        )
                        if Correct_Overestimation and Measurement == "Area":
                            surf_dict[key] *= 2 / 3
                        elif Correct_Overestimation and Measurement == "Perimeter":
                            surf_dict[key] *= np.pi / 4

            surf_dict = _norm(surf_dict)

        else:
            if Measurement == "Area":
                y_label = rf"Surface Area ($\mathrm{{{display_units}^{{2}}}}$)"
                for key in surf_dict.keys():
                    if key != slice_key:
                        surf_dict[key] *= Pixel_Size**2
                        units[key] = f"{Units}^2"
                        if Correct_Overestimation:
                            surf_dict[key] *= 2 / 3
            else:
                y_label = rf"Surface Perimeter ($\mathrm{{{display_units}}}$)"
                for key in surf_dict.keys():
                    if key != slice_key:
                        surf_dict[key] *= Pixel_Size
                        units[key] = f"{Units}"
                        if Correct_Overestimation:
                            surf_dict[key] *= np.pi / 4

        from pydoug_core.analyze.plot_util import multi_line

        fig: Figure = Figure()
        ax: Axes = fig.add_subplot()
        y_vals: list = []
        data_labels: list = []
        for key in surf_dict.keys():
            if key != slice_key and key != 0:
                y_vals.append(surf_dict[key])
                data_labels.append(key)
        y_lims: tuple[float, float] | None = (
            None if (Y_Lims[0] == Y_Lims[1] == 0) else Y_Lims
        )
        ax = multi_line(
            surf_dict[slice_key],
            y_vals,
            ax,
            rf"Position ($\mathrm{{{display_units}}}$)",
            y_label,
            y_lims,
            data_labels,
            None,
            True,
        )
        fig.set_layout_engine("constrained")

        return PlotResult(fig, surf_dict, (slice_key,), units=units)

    surf_dict: dict = _surf(imarray, Apply_Mask, Mask)

    if Normalize:
        if Normalization_Method in (
            "Specific - Phase",
            "Specific - Segmented",
            "Specific - Total",
        ):
            for key in list(surf_dict.keys()):
                units[key] = f"{Units}^(-1)" if key != slice_key else units[slice_key]
                if Correct_Overestimation and Measurement == "Area":
                    surf_dict[key] *= 2 / 3
                elif Correct_Overestimation and Measurement == "Perimeter":
                    surf_dict[key] *= np.pi / 4
        else:
            for key in list(surf_dict.keys()):
                units[key] = None if key != slice_key else units[slice_key]

        surf_dict = _norm(surf_dict)

    else:
        for key in surf_dict.keys():
            if key != slice_key:
                if Measurement == "Area":
                    surf_dict[key] *= Pixel_Size**2
                    units[key] = f"{Units}^2"
                    if Correct_Overestimation:
                        surf_dict[key] *= 2 / 3
                else:
                    surf_dict[key] *= Pixel_Size
                    units[key] = Units
                    if Correct_Overestimation:
                        surf_dict[key] *= np.pi / 4

    return ValueResult(surf_dict, units)


@register_operation(
    category="measure",
    display_name="Heat Map",
    output_kind="plot",
    array_params=("Mask",),
    array_param_flags={"Mask": "Apply_Mask"},
    recordable_toggle=True,
    widget_options={"Color_Map": {"choices": list(colormaps)}},
)
def heat_map(
    imarray: np.ndarray,
    Color_Map: str = "inferno",
    Pixel_Size: Annotated[float, {"min": 0, "step": 0.0001}] = 1,
    Units: str = "pixels",
    Colorbar_Label: str = "Thickness",
    C_Lims: tuple[
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
    Create a heat map from a 2D projection (these can be created in the "Math" tab.)

    Parameters
    ----------
    imarray: np.ndarray
        The 2D projection image. 3D images not accepted.
    Color_Map: Literal
        The color map to use for the heat map.
    Pixel_Size: float
        The edge-length of pixels in real-world units. Note that this does not apply the
        pixel size to the values in the heat map (these are done when you create the 2D
        projection), but is instead used for the axis extents in the plot generation.
        This value should match the value used to create the projection.
    Units: str
        The real-world units for "Pixel Size".
    Colorbar_Label: str
        The label for the colorbar (units are added automatically from "Units").
    C_Lims: tuple[float, float]
        The limits of the map's colorbar in the form [min, max]. Ignored if both are 0.
    Apply_Mask: bool
        Whether to mask the region where the value is measured (True) or not (False).
    Mask: np.ndarray | None
        If "Apply Mask" is "True", the mask to use.
    """

    from pydoug_core.analyze.image_info import stats
    from pydoug_core.analyze.plot_util import add_cbar, hmap_ax

    heat_stats: dict[str, float | int] = stats(imarray, Apply_Mask, Mask)  # type: ignore
    heat_stats.pop("Dtype Min")
    heat_stats.pop("Dtype Max")
    if Apply_Mask and Mask is not None:
        mask = util.prep_mask(imarray, Mask)
        eval_array: np.ndarray = np.copy(imarray)
        eval_array[np.logical_not(mask)] = 0
        heat_stats["Min Fraction"] = (
            np.count_nonzero(eval_array[mask] == np.min(eval_array[mask]))
            / eval_array[mask].size
        )
        heat_stats["Max Fraction"] = (
            np.count_nonzero(eval_array[mask] == np.max(eval_array[mask]))
            / eval_array[mask].size
        )
    else:
        eval_array = imarray
        heat_stats["Min Fraction"] = (
            np.count_nonzero(eval_array == np.min(eval_array)) / eval_array.size
        )
        heat_stats["Max Fraction"] = (
            np.count_nonzero(eval_array == np.max(eval_array)) / eval_array.size
        )

    fig: Figure = Figure()
    ax: Axes = fig.add_subplot()
    c_lims: tuple[float, float] | None = (
        None if (C_Lims[0] == C_Lims[1] == 0) else C_Lims
    )
    display_units: str = r"\mu m" if Units == "um" else Units
    axes_label: str = rf"Position ($\mathrm{{{display_units}}}$)"
    cbar_label: str = rf"{Colorbar_Label} ($\mathrm{{{display_units}}}$)"
    ax, ax_im = hmap_ax(
        eval_array,
        ax,
        Pixel_Size,
        Color_Map,
        c_lims,
        axes_label,
    )
    fig.set_layout_engine(layout="constrained")
    fig = add_cbar(fig, ax_im, cbar_label)

    return PlotResult(fig, {}, (), {"statistics": heat_stats})


@register_operation(
    category="measure",
    display_name="Label Count",
    layer_type="labels",
    output_kind="value",
    recordable_toggle=True,
)
def label_count(imarray: np.ndarray) -> int:
    """
    Count the number of unique labels in a labeled segmented image. Does not count the
    background (0) label. Convenience wrapper around numpy's unique:
    https://numpy.org/doc/stable/reference/generated/numpy.unique.html

    Parameters
    ----------
    imarray: np.ndarray
        The labeled segmented image to count unique labels.

    Returns
    -------
    lab_count: int
        The unique label count.
    """

    labs = np.unique(imarray)
    return labs.size - 1 if 0 in labs else labs.size


@register_operation(
    category="measure",
    display_name="Tortuosity",
    output_kind="value",
    recordable_toggle=True,
)
def tortuosity(
    imarray: np.ndarray,
    Axis: Literal["X", "Y", "Z"] = "Z",
    Perform_Slicewise: bool = False,
    Pixel_Size: Annotated[float, {"min": 0, "step": 0.0001}] = 1,
    Units: str = "pixels",
    Y_Lims: tuple[
        Annotated[
            float, {"min": -1000000000000000, "max": 1000000000000000, "step": 0.0001}
        ],
        Annotated[
            float, {"min": -1000000000000000, "max": 1000000000000000, "step": 0.0001}
        ],
    ] = (0, 0),
) -> PlotResult | ValueResult:  # pragma: no cover
    """
    Find the tortuosity of a binary segmented image along an axis. Convenience wrapper
    around porespy's simulations.tortuosity_fd:
    https://porespy.org/autoapi/porespy/simulations/tortuosity_fd.html

    Parameters
    ----------
    imarray: np.ndarray
        The image to find the tortuosity for.
    Axis: Literal
        The axis along which to find the tortuosity. Image must be 3D if "Z" is chosen.
    Pixel_Size: float
        The edge-length of pixels in real-world units. Only used to plot the x-axis if
        "Perform Slicewise" is "True".
    Units: str
        The real-world units for "Pixel Size". Only used to plot the x-axis if
        "Perform Slicewise" is "True".
    Perform_Slicewise: bool
        Whether to perform the measurement slicewise along the z-axis to return a plot
        of the tortuosity of each slice (True) or to provide a single tortuosity value
        for the entire dataset (False). Default is False. Ignored for 2D images.
    Y_Lims: tuple[float, float]
        If "Perform Slicewise" is "True", the limits of the plot's y-axis in the form
        [min, max]. Ignored if both are 0.

    Returns
    -------
    tort_val: float
        The tortuosity of the image.
    """

    if (not util.is3d(imarray) and Axis == "Z") or (
        util.is3d(imarray) and Perform_Slicewise and Axis == "Z"
    ):
        raise ValueError("Z chosen as axis but image is not 3D!")

    from porespy.simulations import tortuosity_fd

    if Perform_Slicewise and util.is3d(imarray):
        display_units: str = r"\mu m" if Units == "um" else Units
        slice_key: str = "Slice"

        tort_dict: dict[str, np.ndarray] = {
            slice_key: np.arange(0, imarray.shape[0]) * Pixel_Size,
            "Tortuosity": np.empty(imarray.shape[0]),
        }
        for index in range(0, imarray.shape[0]):
            tort_dict["Tortuosity"][index] = tortuosity_fd(
                imarray[index].astype(np.bool), util.axstr2int(imarray[index], Axis)
            ).tortuosity  # type: ignore

        from pydoug_core.analyze.plot_util import line_ax

        fig: Figure = Figure()
        ax: Axes = fig.add_subplot()
        y_lims: tuple[float, float] | None = (
            None if (Y_Lims[0] == Y_Lims[1] == 0) else Y_Lims
        )
        ax = line_ax(
            tort_dict[slice_key],
            tort_dict["Tortuosity"],
            ax,
            y_lims,
            rf"Position ($\mathrm{{{display_units}}}$)",
            "Tortuosity",
            color="blue",
        )
        fig.set_layout_engine("constrained")

        return PlotResult(
            fig,
            tort_dict,  # type: ignore
            (slice_key,),
            {},
            {slice_key: Units, "Tortuosity": None},
        )

    tort_val: float = tortuosity_fd(
        imarray.astype(np.bool), util.axstr2int(imarray, Axis)
    ).tortuosity  # type: ignore

    return ValueResult({"Tortuosity": tort_val})


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
    display_name="Fractal Dimension",
    output_kind="plot",
    recordable_toggle=True,
)
def fractal_dim(
    imarray: np.ndarray,
    Bin_Count: Annotated[int, {"min": 1}] = 10,
    Y_Lims: tuple[
        Annotated[float, {"min": 0.0001, "max": 1000000000000000, "step": 0.0001}],
        Annotated[float, {"min": 0.0001, "max": 1000000000000000, "step": 0.0001}],
    ] = (1, 1),
    X_Lims: tuple[
        Annotated[float, {"min": 0.0001, "max": 1000000000000000, "step": 0.0001}],
        Annotated[float, {"min": 0.0001, "max": 1000000000000000, "step": 0.0001}],
    ] = (1, 1),
) -> PlotResult:
    """
    Calculate the fractal dimension of a binary segmented image with the box counting
    method. The fractal dimension is the negative of the resulting trendline slope. It
    is typically >= 1 for 2D images and >= 2 for 3D images, where 1 or 2 indicate an
    extremely smooth surface with a regular geometry (e.g., a square or cube).
    Non-binary images are first converted (all non-zero pixels are treated as the
    segmented phase). Image must not be RGB. Convenience wrapper around porespy's
    metrics.boxcount and scipy's stats.linregress:
    https://porespy.org/autoapi/porespy/metrics/boxcount.html

    Parameters
    ----------
    imarray: np.ndarray
        The segmented image.
    Bin_Count: int
        The number of bins to use for box counting. Default is 10.
    Y_Lims: tuple[float, float]
        The limits of the plot's y-axis in the form [min, max]. Ignored if both are 1.
    X_Lims: tuple[float, float]
        The limits of the plot's x-axis in the form [min, max]. Ignored if both are 1.

    Returns
    -------
    fd_result: PlotResult
        The fractal dimension result object.
    """

    if util.isrgb(imarray):
        raise ValueError("Image cannot be RGB!")

    from porespy.metrics import boxcount
    from scipy.stats import linregress

    def _frac(imslice) -> dict[str, np.ndarray | float]:

        if not np.any(imslice):
            return {
                "Size": np.array([1]),
                "Count": np.array([1]),
                "Slope": 1,
                "Intercept": 0,
                "R^2": 0,
            }

        results = boxcount(imslice != 0, Bin_Count)
        size: np.ndarray = results.size  # type: ignore
        count: np.ndarray = results.count  # type: ignore
        reg = linregress(np.log10(size), np.log10(count))
        slope: float = reg.slope  # type: ignore
        intercept: float = reg.intercept  # type: ignore
        cod: float = reg.rvalue**2  # type: ignore

        return {
            "Size": size,
            "Count": count,
            "Slope": slope,
            "Intercept": intercept,
            "R^2": cod,
        }

    frac_dict: dict[str, np.ndarray | float] = _frac(imarray)

    from pydoug_core.analyze.plot_util import scatter_ax

    fig: Figure = Figure()
    ax: Axes = fig.add_subplot()
    ax = scatter_ax(
        frac_dict["Size"],
        frac_dict["Count"],
        ax,
        "Size",
        "Count",
        None if Y_Lims[0] == Y_Lims[1] == 1 else Y_Lims,
        None if X_Lims[0] == X_Lims[1] == 1 else X_Lims,
        "blue",
        True,
        frac_dict["Slope"],  # type: ignore
        frac_dict["Intercept"],  # type: ignore
        frac_dict["R^2"],  # type: ignore
        True,
        True,
    )
    fig.set_layout_engine("constrained")

    return PlotResult(
        fig,
        {"Size": frac_dict["Size"], "Count": frac_dict["Count"]},  # type: ignore
        ("Size",),
        {
            "regression": {
                "Slope": frac_dict["Slope"],
                "Intercept": frac_dict["Intercept"],
                "R^2": frac_dict["R^2"],
            }
        },
    )

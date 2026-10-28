"""
Utilities for generating plots.
"""

import cmasher  # noqa: F401 — imported for its expansion of matplotlib's color maps
import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.contour import ContourSet
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

__all__ = ["line_ax", "multi_line", "hist_ax", "hmap_ax", "add_cbar", "stats_ax"]

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]


def line_ax(
    x_vals,
    y_vals,
    ax: Axes,
    y_lims: tuple | None,
    x_label: str,
    y_label: str,
    color="red",
) -> Axes:
    """
    Add a single line plot for any line data to an input Axes. Convenience wrapper
    around matplotlib's plot:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

    Parameters
    ----------
    x_vals
        Any iterable format of values for the x-axis.
    y_vals
        Any iterable format of values for the y-axis.
    ax: Axes
        The input axes.
    y_lims: tuple | None
        (Optional) The limits for the y-axis.
    x_label: str
        The label for the x-axis.
    y_label: str
        The label for the y-axis.
    color: str
        The color for the line. Default is "red".

    Returns
    -------
    ax: Axes
        The input Axes with the line plot added.
    """

    ax.plot(x_vals, y_vals, color=color, linewidth=2)
    ax.set_xlim(0, np.max(x_vals))
    ax.set_ylim((min(y_lims), max(y_lims)) if y_lims else None)
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    ax.tick_params("both", labelsize=13)

    return ax


def multi_line(
    x_vals: np.ndarray,
    y_vals: list[np.ndarray],
    ax: Axes,
    x_label: str,
    y_label: str,
    y_lims: tuple | None = None,
    data_labels: list[str] | list[None] | None = None,
    colors: np.ndarray | str | list[str] | list[np.ndarray] | None | list[None] = None,
    add_legend: bool = False,
) -> Axes:
    """
    Add as many line plots as required to an input Axes. Convenience wrapper around
    matplotlib's plot:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

    Parameters
    ----------
    x_vals: np.ndarray
        The x-axis values.
    y_vals: list[np.ndarray]
        List of same-shape arrays as the x-axis array to plot.
    ax: Axes
        The input axes.
    x_label: str
        The x-axis label.
    y_label: str
        The y-axis label.
    y_lims: tuple | None
        (Optional) The y-axis limits. Calculated automatically if not provided.
    data_labels: list[str]
        The labels for the data to use if "add_legend" is "True".
    colors: str | list[str]
        A list of colors to use for each plot or a single name denoting a colormap to
        gradiate each line with.
    add_legend: bool
        Whether or not to add a legend. Default is False.

    Returns
    -------
    ml_ax: Axes
        The input Axes with the multi line plot added.
    """

    if isinstance(colors, str):
        import cmasher  # noqa: F401 — imported for its colormaps addition effect

        if colors in list(matplotlib.colormaps):
            cmap = matplotlib.colormaps[colors]
            colors = cmap(np.linspace(0, 1, len(y_vals)))
        else:
            colors = [colors] * len(y_vals)

    if isinstance(colors, np.ndarray):
        if colors.ndim == 1:
            colors = [colors] * len(y_vals)
        else:
            if colors.shape[0] < len(y_vals):
                colors = np.vstack(
                    (colors, [colors[-1]] * (len(y_vals) - colors.shape[0]))
                )

    if isinstance(colors, list):
        if len(colors) < len(y_vals):
            colors += [colors[-1]] * (len(y_vals) - len(colors))  # type: ignore

    if colors is None:
        colors = [None] * len(y_vals)

    if not data_labels:
        data_labels = [None] * len(y_vals)

    for index, y_val in enumerate(y_vals):
        ax.plot(
            x_vals, y_val, linewidth=2, color=colors[index], label=data_labels[index]
        )

    ax.set_xlim(min(x_vals), max(x_vals))
    ax.set_ylim((min(y_lims), max(y_lims)) if y_lims else None)
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    ax.tick_params("both", labelsize=13)
    if add_legend and len(y_vals) > 1:
        ax.legend(fontsize=12, frameon=False)  # type: ignore
    return ax


def hist_ax(
    hist_vals: np.ndarray,
    bin_centers: np.ndarray,
    hist_ax: Axes,
    x_label: str = "Gray Value",
    y_label: str = "Counts",
    y_max: float = 0,
    x_lims: tuple[float, float] | None = None,
) -> Axes:
    """
    Add a histogram plot for the given values to an input Axes. Convenience wrapper
    around matplotlib's hist:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html

    Parameters
    ----------
    hist_vals: np.ndarray
        The values to generate a histogram for.
    bin_centers: np.ndarray
        The bin centers for each value in "hist_vals".
    hist_ax: Axes
        The axes to add the histogram to.
    y_max: float
        The maximum y-axis value to plot. Ignored if 0.
    x_lims: tuple[float, float] | None
        The x-axis limits (optional).

    Returns
    -------
    hist_ax: Axes
        The input Axes with the histogram added.
    """

    hist_ax.hist(bin_centers, bin_centers, weights=hist_vals, edgecolor="k")  # type: ignore
    hist_ax.set_xlim((min(x_lims), max(x_lims)) if x_lims else None)
    if y_max:
        hist_ax.set_ylim(0, y_max)
    hist_ax.set_xlabel(x_label, fontsize=15)
    hist_ax.set_ylabel(y_label, fontsize=15)
    hist_ax.tick_params("both", labelsize=13)
    return hist_ax


def hmap_ax(
    harray: np.ndarray,
    h_ax: Axes,
    pixel_size: float = 1,
    cmap: str = "inferno",
    c_lims: tuple | None = None,
    axes_label: str = "Position",
) -> tuple[Axes, AxesImage]:
    """
    Add a heat map of the input 2D image to an input Axes. Convenience wrapper around
    matplot lib's imshow and colorbar:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html

    Parameters
    ----------
    harray: np.ndarray
        The 2D projection image used to create the heat map.
    h_ax: Axes
        The Axes to add the heat map to.
    cmap: str
        The color map to use for the heat map.
    c_lims: tuple | None
        The limits for the heat map's color map.
    axes_label: str
        The label for the x and y axes of the heat map.

    Returns
    -------
    h_ax: Axes
        The input Axes with the heat map added.
    """

    ax_im: AxesImage = h_ax.imshow(
        np.flipud(harray),
        cmap=cmap,
        vmin=np.min(harray) if not c_lims else min(c_lims),
        vmax=np.max(harray) if not c_lims else max(c_lims),
        origin="lower",
        interpolation="none",
        extent=(0, (harray.shape[1] * pixel_size), 0, (harray.shape[0] * pixel_size)),
    )
    h_ax.set_xlabel(axes_label, fontsize=15)
    h_ax.set_ylabel(axes_label, fontsize=15)
    h_ax.tick_params(axis="both", labelsize=13)

    return h_ax, ax_im


def add_cbar(
    fig: Figure,
    mappable: AxesImage | ContourSet,
    cbar_label: str = "Thickness",
) -> Figure:
    """
    Add a colorbar to a figure.

    Parameters
    ----------
    fig: Figure
        The figure to add a colorbar to.
    mappable: AxesImage | ContourSet
        The mappable object described by the colorbar.
    cbar_label: str
        The label for the colorbar.

    Returns
    -------
    fig: Figure
        The figure after colorbar addition.
    """

    fig_cbar: Colorbar = fig.colorbar(mappable)
    fig_cbar.set_label(cbar_label, rotation=270, va="bottom", fontsize=15)
    fig_cbar.ax.tick_params(labelsize=13)

    return fig


def stats_ax(
    stats_dict: dict[str, np.ndarray],
    stats_ax: Axes,
    y_lims: tuple | None = None,
    y2_lims: tuple | None = None,
    plot_mean: bool = True,
    plot_median: bool = False,
    plot_mode: bool = False,
    plot_stdv: bool = True,
    plot_var: bool = False,
    plot_count: bool = False,
    plot_min: bool = True,
    plot_max: bool = True,
    plot_dtype_min: bool = False,
    plot_dtype_max: bool = False,
    include_legend: bool = True,
) -> Axes:
    """
    Add a line plot for the given statistics to an input Axes. Convenience wrapper
    around matplotlib's plot and fill_between:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill_between.html

    Parameters
    ----------
    stats_dict: dict[str, np.ndarray]
        A dictionary containing the slicewise statistics.
    stats_ax: Axes
        The axis to plot on.
    y_lims: tuple[float, float] | None
        The y-axis limits (optional).
    y2_lims: tuple[float, float] | None
        The second y-axis limits (optional).
    plot_mean, plot_median, plot_mode, plot_stdv, plot_var, plot_count, plot_min,
    plot_max, plot_dtype_min, plot_dtype_max: bool
        Whether or not to plot each element of the stats dictionary. Defaults plot mean,
        stdv, min, and max.
    include_legend: bool
        Whether to include a legend (True) or not (False). Default is True.

    Returns
    -------
    stats_ax: Axes
        The input Axes with the stats plot added.
    """

    if plot_mean:
        stats_ax.plot(
            stats_dict["Slice"],
            stats_dict["Mean"],
            "black",
            linewidth=2,
            label="Mean",
        )
    if plot_median:
        stats_ax.plot(
            stats_dict["Slice"],
            stats_dict["Median"],
            "green",
            linewidth=2,
            label="Median",
        )
    if plot_mode:
        stats_ax.plot(
            stats_dict["Slice"],
            stats_dict["Mode"],
            "magenta",
            linewidth=2,
            label="Mode",
        )
    if plot_stdv:
        stats_ax.fill_between(
            stats_dict["Slice"],
            y1=stats_dict["Mean"] + stats_dict["Std Dev"],
            y2=stats_dict["Mean"] - stats_dict["Std Dev"],
            color="gray",
            alpha=0.5,
            label="Std Dev",
        )
    if plot_var:
        stats_ax.fill_between(
            stats_dict["Slice"],
            y1=stats_dict["Mean"] + stats_dict["Variance"],
            y2=stats_dict["Mean"] - stats_dict["Variance"],
            color="gray",
            alpha=0.25,
            label="Variance",
        )
    if plot_min:
        stats_ax.plot(
            stats_dict["Slice"], stats_dict["Min"], "blue", linewidth=2, label="Min"
        )
    if plot_max:
        stats_ax.plot(
            stats_dict["Slice"], stats_dict["Max"], "red", linewidth=2, label="Max"
        )
    if plot_dtype_min:
        stats_ax.plot(
            stats_dict["Slice"],
            stats_dict["Dtype Min"],
            "cyan",
            linewidth=2,
            label="Dtype Min",
        )
    if plot_dtype_max:
        stats_ax.plot(
            stats_dict["Slice"],
            stats_dict["Dtype Max"],
            "yellow",
            linewidth=2,
            label="Dtype Max",
        )
    y_lims = (min(y_lims), max(y_lims)) if y_lims else None
    stats_ax.set_ylim(y_lims)
    stats_ax.set_xlim(0, stats_dict["Slice"][-1])
    stats_ax.set_xlabel("Slice", fontsize=15)
    stats_ax.set_ylabel("Gray Value", fontsize=15)
    stats_ax.tick_params("both", labelsize=13)
    lines1, labels1 = stats_ax.get_legend_handles_labels()
    if plot_count:
        stats_ax2: Axes = stats_ax.twinx()
        stats_ax2.plot(
            stats_dict["Slice"],
            stats_dict["Count"],
            "gray",
            linewidth=2,
            label="Count",
        )
        y2_lims = (min(y2_lims), max(y2_lims)) if y2_lims else None
        stats_ax2.set_ylim(y2_lims)
        stats_ax2.set_ylabel("Gray Count", fontsize=15, rotation=270, va="bottom")
        stats_ax2.tick_params("y", labelsize=13)
        lines2, labels2 = stats_ax2.get_legend_handles_labels()
        lines1 += lines2
        labels1 += labels2
    if include_legend:
        stats_ax.legend(lines1, labels1, fontsize=12, frameon=False)

    return stats_ax

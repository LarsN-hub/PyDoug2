"""
Utilities for generating plots.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

__all__ = ["hist_ax", "line_ax", "stats_ax"]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"


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
    Generate an axis with a minimal histogram plot for the given values. Convenience
    wrapper around matplotlib's hist:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html

    Parameters
    ----------
    hist_vals: np.ndarray
        The values to generate a histogram for.
    bin_centers: np.ndarray
        The bin centers for each value in "hist_vals".
    hist_ax: Axes
        The axis to add the histogram to.
    y_max: float
        The maximum y-axis value to plot. Ignored if 0.
    x_lims: tuple[float, float] | None
        The x-axis limits (optional).

    Returns
    -------
    hist_ax: Axes
        The input axis with the histogram plot generated.
    """

    hist_ax.hist(bin_centers, bin_centers, weights=hist_vals, edgecolor="k")  # type: ignore
    hist_ax.set_xlim((min(x_lims), max(x_lims)) if x_lims else None)
    if y_max:
        hist_ax.set_ylim(0, y_max)
    hist_ax.set_xlabel(x_label, fontsize=15)
    hist_ax.set_ylabel(y_label, fontsize=15)
    hist_ax.tick_params("both", labelsize=13)
    return hist_ax


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
    Generate an axis with a minimal single line plot for any line data. Convenience
    wrapper around matplotlib's plot:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

    Parameters
    ----------
    x_vals
        Any iterable format of values for the x-axis.
    y_vals
        Any iterable format of values for the y-axis.
    ax: Axes
        The input axis.
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
        The input axis with the line plot added.
    """

    ax.plot(x_vals, y_vals, color=color, linewidth=2)
    ax.set_xlim(0, np.max(x_vals))
    ax.set_ylim((min(y_lims), max(y_lims)) if y_lims else None)
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    ax.tick_params("both", labelsize=13)

    return ax


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
    Generate an axis with a minimal line plot for the given statistics. Convenience
    wrapper around matplotlib's plot and fill_between:
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
        The input axis with the stats plot generated.
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
        stats_ax.legend(lines1, labels1, fontsize=12)

    return stats_ax

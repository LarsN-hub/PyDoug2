"""
Tests for PyDoug's analyze.plot_util module.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from pydoug_core.analyze import plot_util


def test_line_ax() -> None:

    _, ax = plt.subplots(layout="constrained")
    ax = plot_util.line_ax(
        np.array([1, 2, 3]), np.array([2, 4, 6]), ax, None, "Test X", "Test Y"
    )
    assert isinstance(ax, Axes)


def test_multi_line() -> None:

    x_vals: np.ndarray = np.array([1, 2, 3, 4, 5])
    y_vals: list[np.ndarray] = [np.array([2, 4, 6, 8, 10]), np.array([3, 6, 8, 12, 15])]
    _, ax1 = plt.subplots(layout="constrained")
    _, ax2 = plt.subplots(layout="constrained")
    _, ax3 = plt.subplots(layout="constrained")
    _, ax4 = plt.subplots(layout="constrained")
    _, ax5 = plt.subplots(layout="constrained")
    _, ax6 = plt.subplots(layout="constrained")
    _, ax7 = plt.subplots(layout="constrained")
    _, ax8 = plt.subplots(layout="constrained")
    _, ax9 = plt.subplots(layout="constrained")
    _, ax10 = plt.subplots(layout="constrained")
    ax1 = plot_util.multi_line(x_vals, y_vals, ax1, "X", "Y", colors="blue")
    ax2 = plot_util.multi_line(
        x_vals, y_vals, ax2, "X", "Y", (0, 1), ["1", "2"], "magma", True
    )
    ax3 = plot_util.multi_line(x_vals, y_vals, ax3, "X", "Y", colors=["blue"])
    ax4 = plot_util.multi_line(x_vals, y_vals, ax4, "X", "Y", colors="cmr.amethyst")
    ax5 = plot_util.multi_line(
        x_vals, y_vals, ax5, "X", "Y", colors=np.array([0, 0, 1])
    )
    ax6 = plot_util.multi_line(
        x_vals, y_vals, ax6, "X", "Y", colors=[np.array([0, 0, 1])]
    )
    ax7 = plot_util.multi_line(
        x_vals, y_vals, ax7, "X", "Y", colors=np.array([[0, 0, 1], [1, 0, 0]])
    )
    ax8 = plot_util.multi_line(x_vals, y_vals, ax8, "X", "Y", colors=["blue", "red"])
    ax9 = plot_util.multi_line(
        x_vals, y_vals, ax9, "X", "Y", colors=np.array([[0, 0, 1]])
    )
    ax10 = plot_util.multi_line(x_vals, y_vals, ax10, "X", "Y")
    assert isinstance(ax1, Axes)
    assert isinstance(ax2, Axes)
    assert isinstance(ax3, Axes)
    assert isinstance(ax4, Axes)
    assert isinstance(ax5, Axes)
    assert isinstance(ax6, Axes)
    assert isinstance(ax7, Axes)
    assert isinstance(ax8, Axes)
    assert isinstance(ax9, Axes)
    assert isinstance(ax10, Axes)


def test_hist_ax(sample_3dgray_array) -> None:

    from pydoug_core.analyze.distrib import hist_distrib

    _, ax = plt.subplots(layout="constrained")
    hist_vals, bin_centers, _ = hist_distrib(sample_3dgray_array)
    ax: Axes = plot_util.hist_ax(hist_vals, bin_centers, ax, y_max=10)
    assert isinstance(ax, Axes)


def test_hmap_ax_and_add_cbar(sample_2dgray_array) -> None:

    fig, ax = plt.subplots(layout="constrained")
    ax, ax_im = plot_util.hmap_ax(sample_2dgray_array, ax)
    fig = plot_util.add_cbar(fig, ax_im)
    assert isinstance(ax, Axes)
    assert isinstance(ax_im, AxesImage)
    assert isinstance(fig, Figure)


def test_stats_ax() -> None:

    _, ax = plt.subplots(layout="constrained")
    stats_dict: dict[str, np.ndarray] = {
        "Slice": np.array([0, 1]),
        "Mean": np.array([5, 6]),
        "Median": np.array([6, 5]),
        "Mode": np.array([7, 5]),
        "Std Dev": np.array([3, 2]),
        "Variance": np.array([9, 4]),
        "Count": np.array([9, 7]),
        "Min": np.array([1, 2]),
        "Max": np.array([9, 8]),
        "Dtype Min": np.array([0, 0]),
        "Dtype Max": np.array([10, 10]),
    }
    ax: Axes = plot_util.stats_ax(
        stats_dict,
        ax,
        None,
        None,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    )
    assert isinstance(ax, Axes)

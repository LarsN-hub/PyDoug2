"""
Tests for PyDoug's analyze.plot_util module.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pydoug_core.analyze import plot_util


def test_hist_ax(sample_3dgray_array) -> None:

    from pydoug_core.analyze.distrib import hist_distrib

    fig, ax = plt.subplots(layout="constrained")
    hist_vals, bin_centers, _ = hist_distrib(sample_3dgray_array)
    ax: Axes = plot_util.hist_ax(hist_vals, bin_centers, ax, y_max=10)
    assert isinstance(ax, Axes)
    assert isinstance(fig, Figure)


def test_line_ax() -> None:

    fig, ax = plt.subplots(layout="constrained")
    ax = plot_util.line_ax(
        np.array([1, 2, 3]), np.array([2, 4, 6]), ax, None, "Test X", "Test Y"
    )
    assert isinstance(ax, Axes)
    assert isinstance(fig, Figure)


def test_stats_ax() -> None:

    fig, ax = plt.subplots(layout="constrained")
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
    assert isinstance(fig, Figure)

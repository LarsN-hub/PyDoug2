"""
Tests for PyDoug's analyze.plot_util module.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pydoug_core.analyze import plot_util


def test_hist_ax(sample_3dgray_array) -> None:

    fig1, ax1 = plt.subplots(layout="constrained")
    ax1: Axes = plot_util.hist_ax(sample_3dgray_array, ax1)
    assert isinstance(ax1, Axes)
    assert isinstance(fig1, Figure)
    fig2, ax2 = plt.subplots(layout="constrained")
    ax2: Axes = plot_util.hist_ax(sample_3dgray_array, ax2, source_range="values")
    assert isinstance(ax2, Axes)
    assert isinstance(fig2, Figure)

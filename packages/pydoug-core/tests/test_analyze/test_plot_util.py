"""
Tests for PyDoug's analyze.plot_util module.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pydoug_core.analyze import plot_util


def test_hist_ax(sample_3dgray_array) -> None:

    from pydoug_core.analyze.distrib import hist_distrib

    fig, ax = plt.subplots(layout="constrained")
    hist_vals, bin_centers = hist_distrib(sample_3dgray_array)
    ax: Axes = plot_util.hist_ax(hist_vals, bin_centers, ax)
    assert isinstance(ax, Axes)
    assert isinstance(fig, Figure)

"""
Tests for PyDoug's hist.hist module.
"""

from matplotlib.figure import Figure

from pydoug_core.hist import hist


def test_plot_histogram(sample_3dgray_array, sample_masks) -> None:

    fig1: Figure = hist.plot_histogram(sample_3dgray_array, Normalize=True)
    fig2: Figure = hist.plot_histogram(
        sample_3dgray_array, Apply_Mask=True, Mask=sample_masks["3dgray"]
    )
    assert isinstance(fig1, Figure)
    assert isinstance(fig2, Figure)

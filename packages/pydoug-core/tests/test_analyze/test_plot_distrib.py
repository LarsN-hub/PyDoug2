"""
Tests for PyDoug's analyze.plot_distrib module.
"""

import numpy as np
from skimage.measure import label

from pydoug_core.analyze import plot_distrib as distrib
from pydoug_core.results import PlotResult, ValueResult


def test_hist_distrib(sample_2dgray_array) -> None:

    vals1, bins1, stats1 = distrib.hist_distrib(sample_2dgray_array)
    vals2, bins2, stats2 = distrib.hist_distrib(
        sample_2dgray_array, source_range="values"
    )
    vals3, bins3, stats3 = distrib.hist_distrib(
        sample_2dgray_array[np.zeros(sample_2dgray_array.shape, np.bool)]
    )
    assert isinstance(vals1, np.ndarray)
    assert isinstance(bins1, np.ndarray)
    assert isinstance(stats1, dict)
    assert isinstance(vals2, np.ndarray)
    assert isinstance(bins2, np.ndarray)
    assert isinstance(stats2, dict)
    assert isinstance(vals3, np.ndarray)
    assert isinstance(bins3, np.ndarray)
    assert isinstance(stats3, dict)
    assert len(vals1) == len(bins1)
    assert len(vals2) == len(bins2)
    assert len(vals1) >= len(vals2)
    assert stats1 == stats2


def test_size_distrib(sample_masks) -> None:

    result00: PlotResult | ValueResult = distrib.size_distrib(np.zeros((5, 5, 5)))
    result01: PlotResult | ValueResult = distrib.size_distrib(
        label(sample_masks["2dgray"], connectivity=1)  # type: ignore
    )
    result02: PlotResult | ValueResult = distrib.size_distrib(
        label(sample_masks["3dgray"], connectivity=1)  # type: ignore
    )
    result03: PlotResult | ValueResult = distrib.size_distrib(
        label(sample_masks["2dgray"], connectivity=1),  # type: ignore
        "Equivalent Radius",
    )
    result04: PlotResult | ValueResult = distrib.size_distrib(
        label(sample_masks["3dgray"], connectivity=1),  # type: ignore
        "Area",
        Pixel_Size=2,
    )
    result05: PlotResult | ValueResult = distrib.size_distrib(
        label(sample_masks["3dgray"], connectivity=1),  # type: ignore
        "Volume",
        Pixel_Size=2,
    )
    result06: PlotResult | ValueResult = distrib.size_distrib(
        label(sample_masks["2dgray"], connectivity=1),  # type: ignore
        Apply_Mask=True,
        Mask=sample_masks["2dgray"],
    )

    assert isinstance(result00, ValueResult)
    assert isinstance(result01, PlotResult)
    assert isinstance(result02, PlotResult)
    assert isinstance(result03, PlotResult)
    assert isinstance(result04, PlotResult)
    assert isinstance(result05, PlotResult)
    assert isinstance(result06, PlotResult)
    assert np.all(
        result00.values["Bin Centers"] == result00.values["Counts"] == np.array([0])  # type: ignore
    )
    assert np.all(result01.data["Bin Centers"] == result03.data["Bin Centers"] * 2)  # type: ignore
    assert np.all(result01.data["Counts"] == result03.data["Counts"])  # type: ignore
    assert np.all(result04.data["Bin Centers"] * 2 == result05.data["Bin Centers"])  # type: ignore
    assert np.all(result04.data["Counts"] == result05.data["Counts"])


def test_aspect_ratio_distrib(sample_masks) -> None:

    result00: PlotResult | ValueResult = distrib.dimless_morph_distrib(
        np.zeros((5, 5, 5))
    )
    result01: PlotResult | ValueResult = distrib.dimless_morph_distrib(
        label(sample_masks["2dgray"], connectivity=1)  # type: ignore
    )
    result02: PlotResult | ValueResult = distrib.dimless_morph_distrib(
        label(sample_masks["3dgray"], connectivity=1)  # type: ignore
    )
    result03: PlotResult | ValueResult = distrib.dimless_morph_distrib(
        label(sample_masks["2dgray"], connectivity=1),  # type: ignore
        Apply_Mask=True,
        Mask=sample_masks["2dgray"],
    )

    assert isinstance(result00, ValueResult)
    assert isinstance(result01, PlotResult)
    assert isinstance(result02, PlotResult)
    assert isinstance(result03, PlotResult)
    assert np.all(result01.data["Bin Centers"] == result03.data["Bin Centers"])
    assert np.all(result01.data["Counts"] == result03.data["Counts"])


def test_sphericity_distrib(sample_masks) -> None:

    result00: PlotResult | ValueResult = distrib.dimless_morph_distrib(
        np.zeros((5, 5, 5)), "Sphericity"
    )
    result01: PlotResult | ValueResult = distrib.dimless_morph_distrib(
        label(sample_masks["2dgray"], connectivity=1),  # type: ignore
        "Sphericity",
    )
    result02: PlotResult | ValueResult = distrib.dimless_morph_distrib(
        label(sample_masks["3dgray"], connectivity=1),  # type: ignore
        "Sphericity",
    )
    result03: PlotResult | ValueResult = distrib.dimless_morph_distrib(
        label(sample_masks["2dgray"], connectivity=1),  # type: ignore
        "Sphericity",
        Apply_Mask=True,
        Mask=sample_masks["2dgray"],
    )

    assert isinstance(result00, ValueResult)
    assert isinstance(result01, PlotResult)
    assert isinstance(result02, PlotResult)
    assert isinstance(result03, PlotResult)
    assert np.all(result01.data["Bin Centers"] == result03.data["Bin Centers"])
    assert np.all(result01.data["Counts"] == result03.data["Counts"])

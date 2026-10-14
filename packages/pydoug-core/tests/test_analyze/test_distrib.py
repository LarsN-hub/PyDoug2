"""
Tests for PyDoug's analyze.distrib module.
"""

import numpy as np

from pydoug_core.analyze import distrib


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

"""
Tests for PyDoug's analyze.distrib module.
"""

import numpy as np

from pydoug_core.analyze import distrib


def test_hist_distrib(sample_2dgray_array) -> None:

    vals1, bins1 = distrib.hist_distrib(sample_2dgray_array)
    vals2, bins2 = distrib.hist_distrib(sample_2dgray_array, source_range="values")
    assert isinstance(vals1, np.ndarray)
    assert isinstance(bins1, np.ndarray)
    assert isinstance(vals2, np.ndarray)
    assert isinstance(bins2, np.ndarray)
    assert len(vals1) == len(bins1)
    assert len(vals2) == len(bins2)
    assert len(vals1) >= len(vals2)

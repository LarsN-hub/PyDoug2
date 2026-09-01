"""
Tests for PyDoug's trans.domain module.
"""

import numpy as np

from pydoug_core.trans import domain


def test_radon(sample_arrays) -> None:

    result1: np.ndarray = domain.radon(sample_arrays["2dgray"])
    result2: np.ndarray = domain.radon(sample_arrays["3dgray"])
    result3: np.ndarray = domain.radon(sample_arrays["3drgb"])
    assert result1.shape[1] == 180
    assert result2.shape[0] == sample_arrays["3dgray"].shape[0]
    assert result2.shape[2] == 180
    assert result2.shape[0] == sample_arrays["3dgray"].shape[0]
    assert result3.shape[2] == 180

"""
Tests for PyDoug's analyze.measure module.
"""

import numpy as np
from skimage.measure import label

from pydoug_core.analyze import measure


def test_label_count(sample_masks) -> None:

    lab_array: np.ndarray = label(sample_masks["3dgray"])  # type: ignore
    result = measure.label_count(lab_array)
    assert result == label(sample_masks["3dgray"], return_num=True)[-1]  # type: ignore

"""
Tests for PyDoug's math.logic module.
"""

import numpy as np

from pydoug_core.math import logic


def test_apply_mask(sample_arrays, sample_masks) -> None:

    result1: np.ndarray = logic.apply_mask(
        sample_arrays["2dgray"], sample_masks["2dgray"]
    )
    result2: np.ndarray = logic.apply_mask(
        sample_arrays["3dgray"], sample_masks["3dgray"]
    )
    result3: np.ndarray = logic.apply_mask(
        sample_arrays["2drgb"], sample_masks["2drgb"]
    )
    result4: np.ndarray = logic.apply_mask(
        sample_arrays["3drgb"], sample_masks["3drgb"]
    )
    assert np.all(result1[sample_masks["2dgray"]] >= 128)
    assert np.all(result2[sample_masks["3dgray"]] >= 128)
    assert np.all(result3[sample_masks["2drgb"]] >= 128)
    assert np.all(result4[sample_masks["3drgb"]] >= 128)

    result5: np.ndarray = logic.apply_mask(
        sample_arrays["2dgray"], sample_masks["2dgray"], "In"
    )
    result6: np.ndarray = logic.apply_mask(
        sample_arrays["3dgray"], sample_masks["3dgray"], "In"
    )
    result7: np.ndarray = logic.apply_mask(
        sample_arrays["2drgb"], sample_masks["2drgb"], "In"
    )
    result8: np.ndarray = logic.apply_mask(
        sample_arrays["3drgb"], sample_masks["3drgb"], "In"
    )
    assert np.all(result5[sample_masks["2dgray"]] < 128)
    assert np.all(result6[sample_masks["3dgray"]] < 128)
    assert np.all(result7[sample_masks["2drgb"]] < 128)
    assert np.all(result8[sample_masks["3drgb"]] < 128)

    result13: np.ndarray = logic.apply_mask(
        sample_arrays["2dgray"], sample_masks["2dgray"], "Out", 5
    )
    result14: np.ndarray = logic.apply_mask(
        sample_arrays["3dgray"], sample_masks["3dgray"], "Out", 5
    )
    result15: np.ndarray = logic.apply_mask(
        sample_arrays["2drgb"], sample_masks["2drgb"], "Out", 5
    )
    result16: np.ndarray = logic.apply_mask(
        sample_arrays["3drgb"], sample_masks["3drgb"], "Out", 5
    )
    assert np.all(result13[np.logical_not(sample_masks["2dgray"])] == 5)
    assert np.all(result14[np.logical_not(sample_masks["3dgray"])] == 5)
    assert np.all(result15[np.logical_not(sample_masks["2drgb"])] == 5)
    assert np.all(result16[np.logical_not(sample_masks["3drgb"])] == 5)

    result17: np.ndarray = logic.apply_mask(
        sample_arrays["2dgray"], sample_masks["2dgray"], "In", 5
    )
    result18: np.ndarray = logic.apply_mask(
        sample_arrays["3dgray"], sample_masks["3dgray"], "In", 5
    )
    result19: np.ndarray = logic.apply_mask(
        sample_arrays["2drgb"], sample_masks["2drgb"], "In", 5
    )
    result20: np.ndarray = logic.apply_mask(
        sample_arrays["3drgb"], sample_masks["3drgb"], "In", 5
    )
    assert np.all(result17[sample_masks["2dgray"]] == 5)
    assert np.all(result18[sample_masks["3dgray"]] == 5)
    assert np.all(result19[sample_masks["2drgb"]] == 5)
    assert np.all(result20[sample_masks["3drgb"]] == 5)

    gray2dcopy: np.ndarray = sample_arrays["2dgray"]
    gray3dcopy: np.ndarray = sample_arrays["3dgray"]
    rgb2dcopy: np.ndarray = sample_arrays["2drgb"]
    rgb3dcopy: np.ndarray = sample_arrays["3drgb"]
    result21: np.ndarray = logic.apply_mask(
        gray2dcopy, sample_masks["2dgray"], "Out", 5, True
    )
    result22: np.ndarray = logic.apply_mask(
        gray3dcopy, sample_masks["3dgray"], "Out", 5, True
    )
    result23: np.ndarray = logic.apply_mask(
        rgb2dcopy, sample_masks["2drgb"], "Out", 5, True
    )
    result24: np.ndarray = logic.apply_mask(
        rgb3dcopy, sample_masks["3drgb"], "Out", 5, True
    )
    assert np.all(result21 == gray2dcopy)
    assert np.all(result22 == gray3dcopy)
    assert np.all(result23 == rgb2dcopy)
    assert np.all(result24 == rgb3dcopy)

    gray2dcopy: np.ndarray = sample_arrays["2dgray"]
    gray3dcopy: np.ndarray = sample_arrays["3dgray"]
    rgb2dcopy: np.ndarray = sample_arrays["2drgb"]
    rgb3dcopy: np.ndarray = sample_arrays["3drgb"]
    result25: np.ndarray = logic.apply_mask(
        gray2dcopy, sample_masks["2dgray"], "In", 5, True
    )
    result26: np.ndarray = logic.apply_mask(
        gray3dcopy, sample_masks["3dgray"], "In", 5, True
    )
    result27: np.ndarray = logic.apply_mask(
        rgb2dcopy, sample_masks["2drgb"], "In", 5, True
    )
    result28: np.ndarray = logic.apply_mask(
        rgb3dcopy, sample_masks["3drgb"], "In", 5, True
    )
    assert np.all(result25 == gray2dcopy)
    assert np.all(result26 == gray3dcopy)
    assert np.all(result27 == rgb2dcopy)
    assert np.all(result28 == rgb3dcopy)


def test_logic() -> None:

    first_array: np.ndarray = np.zeros((10, 10), np.bool)
    first_array[2:7, 2:7] = True
    second_array: np.ndarray = np.zeros((10, 10), np.bool)
    second_array[4:9, 4:9] = True
    result1: np.ndarray = logic.logic(first_array, second_array, "Intersect")
    result2: np.ndarray = logic.logic(first_array, second_array, "Subtract")
    result3: np.ndarray = logic.logic(first_array, second_array)
    assert np.all(result1[4:7, 4:7])
    assert np.all(result2[2:4, 2:4])
    assert np.all(result3[2:7, 2:7])
    assert np.all(result3[4:9, 4:9])

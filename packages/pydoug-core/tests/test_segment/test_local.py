"""
Tests for PyDoug's segment.local module.
"""

import numpy as np

from pydoug_core.segment import local


def test_local_threshold(sample_2dgray_array, sample_3dgray_array) -> None:

    result1: np.ndarray = local.local_threshold(sample_2dgray_array, "Adaptive")
    result2: np.ndarray = local.local_threshold(sample_2dgray_array, "Mean")
    result3: np.ndarray = local.local_threshold(sample_2dgray_array, "Niblack")
    result4: np.ndarray = local.local_threshold(sample_2dgray_array, "Otsu")
    result5: np.ndarray = local.local_threshold(sample_2dgray_array, "Sauvola")
    result6: np.ndarray = local.local_threshold(sample_3dgray_array, "Mean")
    result7: np.ndarray = local.local_threshold(sample_3dgray_array, "Otsu")
    assert result1.dtype == np.uint8
    assert result2.dtype == np.uint8
    assert result3.dtype == np.uint8
    assert result4.dtype == np.uint8
    assert result5.dtype == np.uint8
    assert result6.dtype == np.uint8
    assert result7.dtype == np.uint8
    assert len(np.unique(result1)) == 2
    assert len(np.unique(result2)) == 2
    assert len(np.unique(result3)) == 2
    assert len(np.unique(result4)) == 2
    assert len(np.unique(result5)) == 2
    assert len(np.unique(result6)) == 2
    assert len(np.unique(result7)) == 2
    assert np.max(result1) == 255
    assert np.max(result2) == 255
    assert np.max(result3) == 255
    assert np.max(result4) == 255
    assert np.max(result5) == 255
    assert np.max(result6) == 255
    assert np.max(result7) == 255


def test_rand_walk_image_and_labels(sample_arrays) -> None:

    markers1: np.ndarray = np.copy(sample_arrays["2dgray"])
    markers2: np.ndarray = np.copy(sample_arrays["3dgray"])
    markers3: np.ndarray = np.copy(sample_arrays["2drgb"])
    markers4: np.ndarray = np.copy(sample_arrays["3drgb"])
    markers1[markers1 >= 175] = 255
    markers1[markers1 <= 75] = 50
    markers1[(markers1 > 75) & (markers1 < 175)] = 0
    markers2[markers2 >= 200] = 255
    markers2[markers2 <= 50] = 25
    markers5: np.ndarray = np.copy(markers2)
    markers2[(markers2 > 50) & (markers2 < 200)] = 0
    mid_mask: np.ndarray = (markers5 > 100) & (markers5 < 150)
    markers5[(markers5 > 50) & (markers5 < 200)] = 0
    markers5[mid_mask] = 150
    markers3[markers3 >= 175] = 255
    markers3[markers3 <= 75] = 50
    markers3[(markers3 > 75) & (markers3 < 175)] = 0
    markers4[markers4 >= 200] = 255
    markers4[markers4 <= 50] = 50
    markers4[(markers4 > 50) & (markers4 < 200)] = 0

    result1: np.ndarray = local.rand_walk_image(sample_arrays["2dgray"], markers1)
    result2: np.ndarray = local.rand_walk_image(sample_arrays["3dgray"], markers2)
    result3: np.ndarray = local.rand_walk_image(
        sample_arrays["2drgb"], markers3[..., 0]
    )
    result4: np.ndarray = local.rand_walk_image(
        sample_arrays["3drgb"], markers4[..., 0]
    )
    assert len(np.unique(result1)) == 2
    assert len(np.unique(result2)) == 2
    assert len(np.unique(result3)) == 2
    assert len(np.unique(result4)) == 2
    assert result1.dtype == np.uint8
    assert result2.dtype == np.uint8
    assert result3.dtype == np.uint8
    assert result4.dtype == np.uint8

    result5: np.ndarray = local.rand_walk_image(
        sample_arrays["2dgray"], markers1, "Brute Force"
    )
    result6: np.ndarray = local.rand_walk_image(
        sample_arrays["2dgray"], markers1, "Conjugate Gradient"
    )
    assert len(np.unique(result5)) == 2
    assert len(np.unique(result6)) == 2
    assert result5.dtype == np.uint8
    assert result6.dtype == np.uint8

    result7: np.ndarray = local.rand_walk_image(sample_arrays["3dgray"], markers5)
    assert len(np.unique(result7)) == 3
    assert result7.dtype == np.uint8

    result1_2: np.ndarray = local.rand_walk_labels(sample_arrays["2dgray"], markers1)
    result2_2: np.ndarray = local.rand_walk_labels(sample_arrays["3dgray"], markers2)
    result3_2: np.ndarray = local.rand_walk_labels(
        sample_arrays["2drgb"], markers3[..., 0]
    )
    result4_2: np.ndarray = local.rand_walk_labels(
        sample_arrays["3drgb"], markers4[..., 0]
    )
    assert len(np.unique(result1_2)) == 2
    assert len(np.unique(result2_2)) == 2
    assert len(np.unique(result3_2)) == 2
    assert len(np.unique(result4_2)) == 2
    assert result1_2.dtype == np.int32
    assert result2_2.dtype == np.int32
    assert result3_2.dtype == np.int32
    assert result4_2.dtype == np.int32

    result5_2: np.ndarray = local.rand_walk_labels(
        sample_arrays["2dgray"], markers1, "Brute Force"
    )
    result6_2: np.ndarray = local.rand_walk_labels(
        sample_arrays["2dgray"], markers1, "Conjugate Gradient"
    )
    assert len(np.unique(result5_2)) == 2
    assert len(np.unique(result6_2)) == 2
    assert result5_2.dtype == np.int32
    assert result6_2.dtype == np.int32

    result7_2: np.ndarray = local.rand_walk_labels(sample_arrays["3dgray"], markers5)
    assert len(np.unique(result7_2)) == 3
    assert result7_2.dtype == np.int32

    assert np.all(result1 == result1_2)
    assert np.all(result2 == result2_2)
    assert np.all(result3 == result3_2)
    assert np.all(result4 == result4_2)
    assert np.all(result5 == result5_2)
    assert np.all(result6 == result6_2)
    assert np.all(result7 == result7_2)


def test_chan_vese_algs(sample_arrays) -> None:

    sample_2dgray_morph: np.ndarray = np.random.randint(
        150, 160, (10, 10), dtype=np.uint8
    )
    sample_2dgray_morph[3:7, 3:7] += 8
    sample_3dgray_morph: np.ndarray = np.stack([sample_2dgray_morph] * 5, axis=0)
    sample_2drgb_morph: np.ndarray = np.stack([sample_2dgray_morph] * 3, axis=2)
    sample_3drgb_morph: np.ndarray = np.stack([sample_2drgb_morph] * 5, axis=0)

    result1: np.ndarray = local.chan_vese_algs(sample_2dgray_morph)
    result2: np.ndarray = local.chan_vese_algs(
        sample_2dgray_morph, Morphological_Snakes=False
    )
    result3: np.ndarray = local.chan_vese_algs(sample_3dgray_morph)
    result4: np.ndarray = local.chan_vese_algs(
        sample_3dgray_morph, Morphological_Snakes=False
    )
    result5: np.ndarray = local.chan_vese_algs(sample_2drgb_morph)
    result6: np.ndarray = local.chan_vese_algs(
        sample_2drgb_morph, Morphological_Snakes=False
    )
    result7: np.ndarray = local.chan_vese_algs(sample_3drgb_morph)
    result8: np.ndarray = local.chan_vese_algs(
        sample_3drgb_morph, Morphological_Snakes=False
    )
    assert result1.shape == sample_2dgray_morph.shape
    assert result2.shape == sample_2dgray_morph.shape
    assert result3.shape == sample_3dgray_morph.shape
    assert result4.shape == sample_3dgray_morph.shape
    assert result5.shape == sample_2drgb_morph.shape
    assert result6.shape == sample_2drgb_morph.shape
    assert result7.shape == sample_3drgb_morph.shape
    assert result8.shape == sample_3drgb_morph.shape
    assert result1.dtype == np.int8
    assert result2.dtype == np.bool
    assert result3.dtype == np.int8
    assert result4.dtype == np.bool
    assert result5.dtype == np.int8
    assert result6.dtype == np.bool
    assert result7.dtype == np.int8
    assert result8.dtype == np.bool

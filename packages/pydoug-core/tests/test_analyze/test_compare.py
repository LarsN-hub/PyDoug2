"""
Tests for PyDoug's analyze.compare module.
"""

import numpy as np

from pydoug_core.analyze import compare


def test_mse(sample_arrays, sample_masks) -> None:

    assert compare.mse(sample_arrays["2dgray"], sample_arrays["2dgray"]) == 0
    assert compare.mse(sample_arrays["2dgray"], sample_arrays["2dgray"], True) == 0
    assert compare.mse(sample_arrays["3dgray"], sample_arrays["3dgray"]) == 0
    assert compare.mse(sample_arrays["3dgray"], sample_arrays["3dgray"], True) == 0
    assert compare.mse(sample_arrays["2drgb"], sample_arrays["2drgb"]) == 0
    assert compare.mse(sample_arrays["2drgb"], sample_arrays["2drgb"], True) == 0
    assert compare.mse(sample_arrays["3drgb"], sample_arrays["3drgb"]) == 0
    assert compare.mse(sample_arrays["3drgb"], sample_arrays["3drgb"], True) == 0
    assert compare.mse(sample_masks["2dgray"], sample_masks["2dgray"]) == 0


def test_ssi(sample_masks) -> None:

    big_array_2dgray: np.ndarray = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
    big_array_3dgray: np.ndarray = np.stack([big_array_2dgray] * 15, axis=0)
    big_array_2drgb: np.ndarray = np.stack([big_array_2dgray] * 3, axis=2)
    big_array_3drgb: np.ndarray = np.stack([big_array_2drgb] * 15, axis=0)
    assert compare.ssi(big_array_2dgray, big_array_2dgray) == 1
    assert compare.ssi(big_array_3dgray, big_array_3dgray) == 1
    assert compare.ssi(big_array_2drgb, big_array_2drgb) == 1
    assert compare.ssi(big_array_3drgb, big_array_3drgb) == 1
    assert compare.ssi(sample_masks["2dgray"], sample_masks["2dgray"]) == 1


def test_psnr(sample_arrays, sample_masks) -> None:

    off_arrays = sample_arrays.copy()
    if off_arrays["2dgray"][0, 0] == 0:  # pragma: no cover
        off_arrays["2dgray"][0, 0] += 1
    else:  # pragma: no cover
        off_arrays["2dgray"][0, 0] -= 1
    if off_arrays["3dgray"][0, 0, 0] == 0:  # pragma: no cover
        off_arrays["3dgray"][0, 0, 0] += 1
    else:  # pragma: no cover
        off_arrays["3dgray"][0, 0, 0] -= 1
    if off_arrays["2drgb"][0, 0, 0] == 0:  # pragma: no cover
        off_arrays["2drgb"][0, 0, 0] += 1
    else:  # pragma: no cover
        off_arrays["3drgb"][0, 0, 0, 0] -= 1
    if off_arrays["3drgb"][0, 0, 0, 0] == 0:  # pragma: no cover
        off_arrays["3drgb"][0, 0, 0, 0] += 1
    else:  # pragma: no cover
        off_arrays["3drgb"][0, 0, 0, 0] -= 1
    off_2dmask: np.ndarray = np.copy(sample_masks["2dgray"])
    if off_2dmask[0, 0] == 0:  # pragma: no cover
        off_2dmask[0, 0] += 1
    else:  # pragma: no cover
        off_2dmask[0, 0] -= 1
    assert isinstance(
        compare.psnr(sample_arrays["2dgray"], off_arrays["2dgray"]), float
    )
    assert isinstance(
        compare.psnr(sample_arrays["3dgray"], off_arrays["3dgray"]), float
    )
    assert isinstance(compare.psnr(sample_arrays["2drgb"], off_arrays["2drgb"]), float)
    assert isinstance(compare.psnr(sample_arrays["3drgb"], off_arrays["3drgb"]), float)
    assert isinstance(compare.psnr(sample_masks["2dgray"], off_2dmask), float)


def test_are(sample_masks) -> None:

    assert compare.are(sample_masks["2dgray"], sample_masks["2dgray"]) == {
        "Error": 0,
        "Precision": 1,
        "Recall": 1,
    }
    assert compare.are(sample_masks["3dgray"], sample_masks["3dgray"]) == {
        "Error": 0,
        "Precision": 1,
        "Recall": 1,
    }
    assert compare.are(sample_masks["2drgb"], sample_masks["2drgb"]) == {
        "Error": 0,
        "Precision": 1,
        "Recall": 1,
    }
    assert compare.are(sample_masks["3drgb"], sample_masks["3drgb"]) == {
        "Error": 0,
        "Precision": 1,
        "Recall": 1,
    }


def test_voi(sample_masks) -> None:

    assert compare.voi(sample_masks["2dgray"], sample_masks["2dgray"]) == {
        "Splits": 0,
        "Merges": 0,
    }
    assert compare.voi(sample_masks["3dgray"], sample_masks["3dgray"]) == {
        "Splits": 0,
        "Merges": 0,
    }
    assert compare.voi(sample_masks["2drgb"], sample_masks["2drgb"]) == {
        "Splits": 0,
        "Merges": 0,
    }
    assert compare.voi(sample_masks["3drgb"], sample_masks["3drgb"]) == {
        "Splits": 0,
        "Merges": 0,
    }


def test_nmi(sample_arrays, sample_masks) -> None:

    assert compare.nmi(sample_arrays["2dgray"], sample_arrays["2dgray"]) == 2
    assert compare.nmi(sample_arrays["3dgray"], sample_arrays["3dgray"]) == 2
    assert compare.nmi(sample_arrays["2drgb"], sample_arrays["2drgb"]) == 2
    assert compare.nmi(sample_arrays["3drgb"], sample_arrays["3drgb"]) == 2
    assert compare.nmi(sample_masks["2dgray"], sample_masks["2dgray"]) == 2


def test_intersection(sample_arrays, sample_masks) -> None:

    first_array: np.ndarray = np.zeros((10, 10), np.uint8)
    first_array[5:, :] = 128
    second_array: np.ndarray = np.zeros((10, 10), np.uint8)
    second_array[0:6, :] = 255
    assert compare.intersection(sample_arrays["2dgray"], sample_arrays["2dgray"]) == 1
    assert compare.intersection(sample_arrays["3dgray"], sample_arrays["3dgray"]) == 1
    assert compare.intersection(sample_arrays["2drgb"], sample_arrays["2drgb"]) == 1
    assert compare.intersection(sample_arrays["3drgb"], sample_arrays["3drgb"]) == 1
    assert compare.intersection(sample_masks["2dgray"], sample_masks["2dgray"]) == 1
    assert compare.intersection(sample_masks["3dgray"], sample_masks["3dgray"]) == 1
    assert compare.intersection(sample_masks["2drgb"], sample_masks["2drgb"]) == 1
    assert compare.intersection(sample_masks["3drgb"], sample_masks["3drgb"]) == 1
    assert compare.intersection(first_array, second_array) == 0.2


def test_hausdorff_dist(sample_masks, sample_arrays) -> None:

    assert compare.hausdorff_dist(sample_masks["2dgray"], sample_masks["2dgray"]) == 0
    assert compare.hausdorff_dist(sample_masks["3dgray"], sample_masks["3dgray"]) == 0
    assert compare.hausdorff_dist(sample_masks["2drgb"], sample_masks["2drgb"]) == 0
    assert compare.hausdorff_dist(sample_masks["3drgb"], sample_masks["3drgb"]) == 0
    assert compare.hausdorff_dist(sample_masks["3dgray"], sample_masks["3dgray"]) == 0

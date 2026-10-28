"""
Tests for PyDoug's manipulate.dims module.
"""

import numpy as np

from pydoug_core.manipulate import dims


def test_refine_pad_bounds(sample_arrays) -> None:

    result01: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["2dgray"], False, [0, 0], [0, 0], [0, 0]
    )
    result02: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["3dgray"], False, [0, 0], [0, 0], [0, 0]
    )
    result03: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["2drgb"], False, [0, 0], [0, 0], [0, 0]
    )
    result04: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["3drgb"], False, [0, 0], [0, 0], [0, 0]
    )
    result05: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["2dgray"], True, [0, 0], [0, 0], [0, 0]
    )
    result06: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["3dgray"], True, [0, 0], [0, 0], [0, 0]
    )
    result07: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["2drgb"], True, [0, 0], [0, 0], [0, 0]
    )
    result08: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["3drgb"], True, [0, 0], [0, 0], [0, 0]
    )

    assert result01 == ([0, 0], [0, 0], [0, 0])
    assert result02 == ([0, 0], [0, 0], [0, 0])
    assert result03 == ([0, 0], [0, 0], [0, 0])
    assert result04 == ([0, 0], [0, 0], [0, 0])
    assert result05 == ([0, 0], [0, 0], [0, 0])
    assert result06 == ([0, 0], [0, 0], [0, 0])
    assert result07 == ([0, 0], [0, 0], [0, 0])
    assert result08 == ([0, 0], [0, 0], [0, 0])

    result09: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["2dgray"], False, [-1, -1], [-1, -1], [-1, -1]
    )
    result10: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["3dgray"], False, [-1, -1], [-1, -1], [-1, -1]
    )
    result11: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["2drgb"], False, [-1, -1], [-1, -1], [-1, -1]
    )
    result12: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["3drgb"], False, [-1, -1], [-1, -1], [-1, -1]
    )
    result13: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["2dgray"], True, [-1, -1], [-1, -1], [-1, -1]
    )
    result14: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["3dgray"], True, [-1, -1], [-1, -1], [-1, -1]
    )
    result15: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["2drgb"], True, [-1, -1], [-1, -1], [-1, -1]
    )
    result16: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["3drgb"], True, [-1, -1], [-1, -1], [-1, -1]
    )

    assert result09 == ([0, 0], [0, 0], [0, 0])
    assert result10 == ([0, 0], [0, 0], [0, 0])
    assert result11 == ([0, 0], [0, 0], [0, 0])
    assert result12 == ([0, 0], [0, 0], [0, 0])
    assert result13 == ([0, 0], [0, 0], [0, 0])
    assert result14 == ([0, 0], [0, 0], [0, 0])
    assert result15 == ([0, 0], [0, 0], [0, 0])
    assert result16 == ([0, 0], [0, 0], [0, 0])

    result17: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["2dgray"], False, [1, 2], [1, 2], [1, 2]
    )
    result18: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["3dgray"], False, [1, 2], [1, 2], [1, 2]
    )
    result19: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["2drgb"], False, [1, 2], [1, 2], [1, 2]
    )
    result20: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["3drgb"], False, [1, 2], [1, 2], [1, 2]
    )
    result21: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["2dgray"], True, [1, 2], [1, 2], [1, 2]
    )
    result22: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["3dgray"], True, [1, 2], [1, 2], [1, 2]
    )
    result23: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["2drgb"], True, [1, 2], [1, 2], [1, 2]
    )
    result24: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["3drgb"], True, [1, 2], [1, 2], [1, 2]
    )

    assert result17 == ([1, 2], [1, 2], [1, 2])
    assert result18 == ([1, 2], [1, 2], [1, 2])
    assert result19 == ([1, 2], [1, 2], [1, 2])
    assert result20 == ([1, 2], [1, 2], [1, 2])
    assert result21 == ([0, 0], [0, 0], [0, 1])
    assert result22 == ([0, 0], [0, 0], [0, 0])
    assert result23 == ([0, 0], [0, 0], [0, 1])
    assert result24 == ([0, 0], [0, 0], [0, 0])

    result25: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["2dgray"], False, [11, 15], [11, 15], [11, 15]
    )
    result26: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["3dgray"], False, [11, 15], [11, 15], [11, 15]
    )
    result27: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["2drgb"], False, [11, 15], [11, 15], [11, 15]
    )
    result28: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["3drgb"], False, [11, 15], [11, 15], [11, 15]
    )
    result29: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["2dgray"], True, [11, 15], [11, 15], [11, 15]
    )
    result30: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["3dgray"], True, [11, 15], [11, 15], [11, 15]
    )
    result31: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["2drgb"], True, [11, 15], [11, 15], [11, 15]
    )
    result32: tuple[list[int], list[int], list[int]] = dims.refine_pad_bounds(
        sample_arrays["3drgb"], True, [11, 15], [11, 15], [11, 15]
    )

    assert result25 == ([11, 15], [11, 15], [11, 15])
    assert result26 == ([11, 15], [11, 15], [11, 15])
    assert result27 == ([11, 15], [11, 15], [11, 15])
    assert result28 == ([11, 15], [11, 15], [11, 15])
    assert result29 == ([2, 3], [2, 3], [7, 7])
    assert result30 == ([2, 3], [2, 3], [5, 5])
    assert result31 == ([2, 3], [2, 3], [7, 7])
    assert result32 == ([2, 3], [2, 3], [5, 5])


def test_trim_pad_pad(sample_arrays) -> None:

    in_place_2dgray: np.ndarray = np.copy(sample_arrays["2dgray"])
    in_place_3dgray: np.ndarray = np.copy(sample_arrays["3dgray"])
    in_place_2drgb: np.ndarray = np.copy(sample_arrays["2drgb"])
    in_place_3drgb: np.ndarray = np.copy(sample_arrays["3drgb"])

    result01: np.ndarray = dims.trim_pad(
        sample_arrays["2dgray"], "Pad", False, (0, 0), (0, 0), (0, 0), 0, False
    )
    result02: np.ndarray = dims.trim_pad(
        sample_arrays["3dgray"], "Pad", False, (0, 0), (0, 0), (0, 0), 0, False
    )
    result03: np.ndarray = dims.trim_pad(
        sample_arrays["2drgb"], "Pad", False, (0, 0), (0, 0), (0, 0), 0, False
    )
    result04: np.ndarray = dims.trim_pad(
        sample_arrays["3drgb"], "Pad", False, (0, 0), (0, 0), (0, 0), 0, False
    )
    result05: np.ndarray = dims.trim_pad(
        sample_arrays["2dgray"], "Pad", True, (0, 0), (0, 0), (0, 0), 0, False
    )
    result06: np.ndarray = dims.trim_pad(
        sample_arrays["3dgray"], "Pad", True, (0, 0), (0, 0), (0, 0), 0, False
    )
    result07: np.ndarray = dims.trim_pad(
        sample_arrays["2drgb"], "Pad", True, (0, 0), (0, 0), (0, 0), 0, False
    )
    result08: np.ndarray = dims.trim_pad(
        sample_arrays["3drgb"], "Pad", True, (0, 0), (0, 0), (0, 0), 0, False
    )

    assert result01.shape == sample_arrays["2dgray"].shape
    assert np.all(result01 == sample_arrays["2dgray"])
    assert result02.shape == sample_arrays["3dgray"].shape
    assert np.all(result02 == sample_arrays["3dgray"])
    assert result03.shape == sample_arrays["2drgb"].shape
    assert np.all(result03 == sample_arrays["2drgb"])
    assert result04.shape == sample_arrays["3drgb"].shape
    assert np.all(result04 == sample_arrays["3drgb"])
    assert result05.shape == sample_arrays["2dgray"].shape
    assert np.all(result05 == sample_arrays["2dgray"])
    assert result06.shape == sample_arrays["3dgray"].shape
    assert np.all(result06 == sample_arrays["3dgray"])
    assert result07.shape == sample_arrays["2drgb"].shape
    assert np.all(result07 == sample_arrays["2drgb"])
    assert result08.shape == sample_arrays["3drgb"].shape
    assert np.all(result08 == sample_arrays["3drgb"])

    result09: np.ndarray = dims.trim_pad(
        sample_arrays["2dgray"], "Pad", False, (-1, -1), (-1, -1), (-1, -1), 0, False
    )
    result10: np.ndarray = dims.trim_pad(
        sample_arrays["3dgray"], "Pad", False, (-1, -1), (-1, -1), (-1, -1), 0, False
    )
    result11: np.ndarray = dims.trim_pad(
        sample_arrays["2drgb"], "Pad", False, (-1, -1), (-1, -1), (-1, -1), 0, False
    )
    result12: np.ndarray = dims.trim_pad(
        sample_arrays["3drgb"], "Pad", False, (-1, -1), (-1, -1), (-1, -1), 0, False
    )
    result13: np.ndarray = dims.trim_pad(
        sample_arrays["2dgray"], "Pad", True, (-1, -1), (-1, -1), (-1, -1), 0, False
    )
    result14: np.ndarray = dims.trim_pad(
        sample_arrays["3dgray"], "Pad", True, (-1, -1), (-1, -1), (-1, -1), 0, False
    )
    result15: np.ndarray = dims.trim_pad(
        sample_arrays["2drgb"], "Pad", True, (-1, -1), (-1, -1), (-1, -1), 0, False
    )
    result16: np.ndarray = dims.trim_pad(
        sample_arrays["3drgb"], "Pad", True, (-1, -1), (-1, -1), (-1, -1), 0, False
    )

    assert result09.shape == sample_arrays["2dgray"].shape
    assert np.all(result09 == sample_arrays["2dgray"])
    assert result10.shape == sample_arrays["3dgray"].shape
    assert np.all(result10 == sample_arrays["3dgray"])
    assert result11.shape == sample_arrays["2drgb"].shape
    assert np.all(result11 == sample_arrays["2drgb"])
    assert result12.shape == sample_arrays["3drgb"].shape
    assert np.all(result12 == sample_arrays["3drgb"])
    assert result13.shape == sample_arrays["2dgray"].shape
    assert np.all(result13 == sample_arrays["2dgray"])
    assert result14.shape == sample_arrays["3dgray"].shape
    assert np.all(result14 == sample_arrays["3dgray"])
    assert result15.shape == sample_arrays["2drgb"].shape
    assert np.all(result15 == sample_arrays["2drgb"])
    assert result16.shape == sample_arrays["3drgb"].shape
    assert np.all(result16 == sample_arrays["3drgb"])

    result17: np.ndarray = dims.trim_pad(
        sample_arrays["2dgray"], "Pad", False, (1, 2), (1, 2), (1, 2), 0, False
    )
    result18: np.ndarray = dims.trim_pad(
        sample_arrays["3dgray"], "Pad", False, (1, 2), (1, 2), (1, 2), 0, False
    )
    result19: np.ndarray = dims.trim_pad(
        sample_arrays["2drgb"], "Pad", False, (1, 2), (1, 2), (1, 2), 0, False
    )
    result20: np.ndarray = dims.trim_pad(
        sample_arrays["3drgb"], "Pad", False, (1, 2), (1, 2), (1, 2), 0, False
    )
    result21: np.ndarray = dims.trim_pad(
        sample_arrays["2dgray"], "Pad", True, (1, 2), (1, 2), (1, 2), 0, False
    )
    result22: np.ndarray = dims.trim_pad(
        sample_arrays["3dgray"], "Pad", True, (1, 2), (1, 2), (1, 2), 0, False
    )
    result23: np.ndarray = dims.trim_pad(
        sample_arrays["2drgb"], "Pad", True, (1, 2), (1, 2), (1, 2), 0, False
    )
    result24: np.ndarray = dims.trim_pad(
        sample_arrays["3drgb"], "Pad", True, (1, 2), (1, 2), (1, 2), 0, False
    )

    assert result17.shape == (4, 13, 13)
    assert np.all(result17[1, 1:11, 1:11] == sample_arrays["2dgray"])
    assert result18.shape == (8, 13, 13)
    assert np.all(result18[1:6, 1:11, 1:11] == sample_arrays["3dgray"])
    assert result19.shape == (4, 13, 13, 3)
    assert np.all(result19[1, 1:11, 1:11] == sample_arrays["2drgb"])
    assert result20.shape == (8, 13, 13, 3)
    assert np.all(result20[1:6, 1:11, 1:11] == sample_arrays["3drgb"])
    assert result21.shape == (2, 10, 10)
    assert np.all(result21[0] == sample_arrays["2dgray"])
    assert result22.shape == sample_arrays["3dgray"].shape
    assert np.all(result22 == sample_arrays["3dgray"])
    assert result23.shape == (2, 10, 10, 3)
    assert np.all(result23[0] == sample_arrays["2drgb"])
    assert result24.shape == sample_arrays["3drgb"].shape
    assert np.all(result24 == sample_arrays["3drgb"])

    result25: np.ndarray = dims.trim_pad(
        sample_arrays["2dgray"], "Pad", False, (11, 15), (11, 15), (11, 15), 0, False
    )
    result26: np.ndarray = dims.trim_pad(
        sample_arrays["3dgray"], "Pad", False, (11, 15), (11, 15), (11, 15), 0, False
    )
    result27: np.ndarray = dims.trim_pad(
        sample_arrays["2drgb"], "Pad", False, (11, 15), (11, 15), (11, 15), 0, False
    )
    result28: np.ndarray = dims.trim_pad(
        sample_arrays["3drgb"], "Pad", False, (11, 15), (11, 15), (11, 15), 0, False
    )
    result29: np.ndarray = dims.trim_pad(
        sample_arrays["2dgray"], "Pad", True, (11, 15), (11, 15), (11, 15), 0, False
    )
    result30: np.ndarray = dims.trim_pad(
        sample_arrays["3dgray"], "Pad", True, (11, 15), (11, 15), (11, 15), 0, False
    )
    result31: np.ndarray = dims.trim_pad(
        sample_arrays["2drgb"], "Pad", True, (11, 15), (11, 15), (11, 15), 0, False
    )
    result32: np.ndarray = dims.trim_pad(
        sample_arrays["3drgb"], "Pad", True, (11, 15), (11, 15), (11, 15), 0, False
    )

    assert result25.shape == (27, 36, 36)
    assert np.all(result25[11, 11:21, 11:21] == sample_arrays["2dgray"])
    assert result26.shape == (31, 36, 36)
    assert np.all(result26[11:16, 11:21, 11:21] == sample_arrays["3dgray"])
    assert result27.shape == (27, 36, 36, 3)
    assert np.all(result27[11, 11:21, 11:21] == sample_arrays["2drgb"])
    assert result28.shape == (31, 36, 36, 3)
    assert np.all(result28[11:16, 11:21, 11:21] == sample_arrays["3drgb"])
    assert result29.shape == (15, 15, 15)
    assert np.all(result29[7, 2:12, 2:12] == sample_arrays["2dgray"])
    assert result30.shape == (15, 15, 15)
    assert np.all(result30[5:10, 2:12, 2:12] == sample_arrays["3dgray"])
    assert result31.shape == (15, 15, 15, 3)
    assert np.all(result31[7, 2:12, 2:12] == sample_arrays["2drgb"])
    assert result32.shape == (15, 15, 15, 3)
    assert np.all(result32[5:10, 2:12, 2:12] == sample_arrays["3drgb"])

    result33: np.ndarray = dims.trim_pad(
        in_place_2dgray, "Pad", False, (11, 15), (11, 15), (11, 15), 0, True
    )
    result34: np.ndarray = dims.trim_pad(
        in_place_3dgray, "Pad", False, (11, 15), (11, 15), (11, 15), 0, True
    )
    result35: np.ndarray = dims.trim_pad(
        in_place_2drgb, "Pad", True, (11, 15), (11, 15), (11, 15), 0, True
    )
    result36: np.ndarray = dims.trim_pad(
        in_place_3drgb, "Pad", True, (11, 15), (11, 15), (11, 15), 0, True
    )

    assert result33.shape == result25.shape
    assert np.all(result33 == result25)
    assert result34.shape == result26.shape
    assert np.all(result34 == result26)
    assert result35.shape == result31.shape
    assert np.all(result35 == result31)
    assert result36.shape == result32.shape
    assert np.all(result36 == result32)


def test_refine_trim_bounds(sample_arrays) -> None:

    result01: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2dgray"], False, [0, 0], [0, 0], [0, 0]
    )
    result02: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3dgray"], False, [0, 0], [0, 0], [0, 0]
    )
    result03: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2drgb"], False, [0, 0], [0, 0], [0, 0]
    )
    result04: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3drgb"], False, [0, 0], [0, 0], [0, 0]
    )
    result05: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2dgray"], True, [0, 0], [0, 0], [0, 0]
    )
    result06: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3dgray"], True, [0, 0], [0, 0], [0, 0]
    )
    result07: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2drgb"], True, [0, 0], [0, 0], [0, 0]
    )
    result08: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3drgb"], True, [0, 0], [0, 0], [0, 0]
    )

    assert result01 == ([0, 10], [0, 10], [0, 0])
    assert result02 == ([0, 10], [0, 10], [0, 5])
    assert result03 == ([0, 10], [0, 10], [0, 0])
    assert result04 == ([0, 10], [0, 10], [0, 5])
    assert result05 == ([0, 10], [0, 10], [0, 0])
    assert result06 == ([0, 10], [0, 10], [0, 5])
    assert result07 == ([0, 10], [0, 10], [0, 0])
    assert result08 == ([0, 10], [0, 10], [0, 5])

    result09: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2dgray"], False, [-1, -1], [-1, -1], [-1, -1]
    )
    result10: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3dgray"], False, [-1, -1], [-1, -1], [-1, -1]
    )
    result11: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2drgb"], False, [-1, -1], [-1, -1], [-1, -1]
    )
    result12: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3drgb"], False, [-1, -1], [-1, -1], [-1, -1]
    )
    result13: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2dgray"], True, [-1, -1], [-1, -1], [-1, -1]
    )
    result14: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3dgray"], True, [-1, -1], [-1, -1], [-1, -1]
    )
    result15: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2drgb"], True, [-1, -1], [-1, -1], [-1, -1]
    )
    result16: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3drgb"], True, [-1, -1], [-1, -1], [-1, -1]
    )

    assert result09 == ([4, 5], [4, 5], [0, 0])
    assert result10 == ([4, 5], [4, 5], [2, 3])
    assert result11 == ([4, 5], [4, 5], [0, 0])
    assert result12 == ([4, 5], [4, 5], [2, 3])
    assert result13 == ([0, 1], [0, 1], [0, 0])
    assert result14 == ([0, 1], [0, 1], [0, 1])
    assert result15 == ([0, 1], [0, 1], [0, 0])
    assert result16 == ([0, 1], [0, 1], [0, 1])

    result17: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2dgray"], False, [11, 11], [11, 11], [11, 11]
    )
    result18: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3dgray"], False, [11, 11], [11, 11], [11, 11]
    )
    result19: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2drgb"], False, [11, 11], [11, 11], [11, 11]
    )
    result20: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3drgb"], False, [11, 11], [11, 11], [11, 11]
    )
    result21: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2dgray"], True, [11, 11], [11, 11], [11, 11]
    )
    result22: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3dgray"], True, [11, 11], [11, 11], [11, 11]
    )
    result23: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2drgb"], True, [11, 11], [11, 11], [11, 11]
    )
    result24: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3drgb"], True, [11, 11], [11, 11], [11, 11]
    )

    assert result17 == ([0, 10], [0, 10], [0, 0])
    assert result18 == ([0, 10], [0, 10], [0, 5])
    assert result19 == ([0, 10], [0, 10], [0, 0])
    assert result20 == ([0, 10], [0, 10], [0, 5])
    assert result21 == ([9, 10], [9, 10], [0, 0])
    assert result22 == ([9, 10], [9, 10], [4, 5])
    assert result23 == ([9, 10], [9, 10], [0, 0])
    assert result24 == ([9, 10], [9, 10], [4, 5])

    result25: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2dgray"], False, [3, 8], [3, 8], [3, 8]
    )
    result26: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3dgray"], False, [3, 8], [3, 8], [3, 8]
    )
    result27: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2drgb"], False, [3, 8], [3, 8], [3, 8]
    )
    result28: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3drgb"], False, [3, 8], [3, 8], [3, 8]
    )
    result29: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2dgray"], True, [3, 8], [3, 8], [3, 8]
    )
    result30: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3dgray"], True, [3, 8], [3, 8], [3, 8]
    )
    result31: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2drgb"], True, [3, 8], [3, 8], [3, 8]
    )
    result32: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3drgb"], True, [3, 8], [3, 8], [3, 8]
    )

    assert result25 == ([1, 9], [1, 9], [0, 0])
    assert result26 == ([1, 9], [1, 9], [0, 5])
    assert result27 == ([1, 9], [1, 9], [0, 0])
    assert result28 == ([1, 9], [1, 9], [0, 5])
    assert result29 == ([3, 8], [3, 8], [0, 0])
    assert result30 == ([3, 8], [3, 8], [3, 5])
    assert result31 == ([3, 8], [3, 8], [0, 0])
    assert result32 == ([3, 8], [3, 8], [3, 5])

    result33: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2dgray"], True, [3, 3], [3, 3], [3, 3]
    )
    result34: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3dgray"], True, [3, 3], [3, 3], [3, 3]
    )
    result35: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2drgb"], True, [3, 3], [3, 3], [3, 3]
    )
    result36: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3drgb"], True, [3, 3], [3, 3], [3, 3]
    )
    result37: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2dgray"], True, [3, 4], [3, 4], [3, 4]
    )
    result38: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3dgray"], True, [3, 4], [3, 4], [3, 4]
    )
    result39: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["2drgb"], True, [3, 4], [3, 4], [3, 4]
    )
    result40: tuple[list[int], list[int], list[int]] = dims.refine_trim_bounds(
        sample_arrays["3drgb"], True, [3, 4], [3, 4], [3, 4]
    )

    assert result33 == ([3, 4], [3, 4], [0, 0])
    assert result34 == ([3, 4], [3, 4], [3, 4])
    assert result35 == ([3, 4], [3, 4], [0, 0])
    assert result36 == ([3, 4], [3, 4], [3, 4])
    assert result37 == ([3, 4], [3, 4], [0, 0])
    assert result38 == ([3, 4], [3, 4], [3, 4])
    assert result39 == ([3, 4], [3, 4], [0, 0])
    assert result40 == ([3, 4], [3, 4], [3, 4])


def test_trim_pad_trim(sample_arrays) -> None:

    in_place_2dgray: np.ndarray = np.copy(sample_arrays["2dgray"])
    in_place_3dgray: np.ndarray = np.copy(sample_arrays["3dgray"])
    in_place_2drgb: np.ndarray = np.copy(sample_arrays["2drgb"])
    in_place_3drgb: np.ndarray = np.copy(sample_arrays["3drgb"])

    result01: np.ndarray = dims.trim_pad(
        sample_arrays["2dgray"], "Trim", False, (0, 0), (0, 0), (0, 0), 0, False
    )
    result02: np.ndarray = dims.trim_pad(
        sample_arrays["3dgray"], "Trim", False, (0, 0), (0, 0), (0, 0), 0, False
    )
    result03: np.ndarray = dims.trim_pad(
        sample_arrays["2drgb"], "Trim", False, (0, 0), (0, 0), (0, 0), 0, False
    )
    result04: np.ndarray = dims.trim_pad(
        sample_arrays["3drgb"], "Trim", False, (0, 0), (0, 0), (0, 0), 0, False
    )
    result05: np.ndarray = dims.trim_pad(
        sample_arrays["2dgray"], "Trim", True, (0, 0), (0, 0), (0, 0), 0, False
    )
    result06: np.ndarray = dims.trim_pad(
        sample_arrays["3dgray"], "Trim", True, (0, 0), (0, 0), (0, 0), 0, False
    )
    result07: np.ndarray = dims.trim_pad(
        sample_arrays["2drgb"], "Trim", True, (0, 0), (0, 0), (0, 0), 0, False
    )
    result08: np.ndarray = dims.trim_pad(
        sample_arrays["3drgb"], "Trim", True, (0, 0), (0, 0), (0, 0), 0, False
    )

    assert result01.shape == sample_arrays["2dgray"].shape
    assert np.all(result01 == sample_arrays["2dgray"])
    assert result02.shape == sample_arrays["3dgray"].shape
    assert np.all(result02 == sample_arrays["3dgray"])
    assert result03.shape == sample_arrays["2drgb"].shape
    assert np.all(result03 == sample_arrays["2drgb"])
    assert result04.shape == sample_arrays["3drgb"].shape
    assert np.all(result04 == sample_arrays["3drgb"])
    assert result05.shape == sample_arrays["2dgray"].shape
    assert np.all(result05 == sample_arrays["2dgray"])
    assert result06.shape == sample_arrays["3dgray"].shape
    assert np.all(result06 == sample_arrays["3dgray"])
    assert result07.shape == sample_arrays["2drgb"].shape
    assert np.all(result07 == sample_arrays["2drgb"])
    assert result08.shape == sample_arrays["3drgb"].shape
    assert np.all(result08 == sample_arrays["3drgb"])

    result09: np.ndarray = dims.trim_pad(
        sample_arrays["2dgray"], "Trim", False, (-1, -1), (-1, -1), (-1, -1), 0, False
    )
    result10: np.ndarray = dims.trim_pad(
        sample_arrays["3dgray"], "Trim", False, (-1, -1), (-1, -1), (-1, -1), 0, False
    )
    result11: np.ndarray = dims.trim_pad(
        sample_arrays["2drgb"], "Trim", False, (-1, -1), (-1, -1), (-1, -1), 0, False
    )
    result12: np.ndarray = dims.trim_pad(
        sample_arrays["3drgb"], "Trim", False, (-1, -1), (-1, -1), (-1, -1), 0, False
    )
    result13: np.ndarray = dims.trim_pad(
        sample_arrays["2dgray"], "Trim", True, (-1, -1), (-1, -1), (-1, -1), 0, False
    )
    result14: np.ndarray = dims.trim_pad(
        sample_arrays["3dgray"], "Trim", True, (-1, -1), (-1, -1), (-1, -1), 0, False
    )
    result15: np.ndarray = dims.trim_pad(
        sample_arrays["2drgb"], "Trim", True, (-1, -1), (-1, -1), (-1, -1), 0, False
    )
    result16: np.ndarray = dims.trim_pad(
        sample_arrays["3drgb"], "Trim", True, (-1, -1), (-1, -1), (-1, -1), 0, False
    )

    assert result09.shape == (1, 1)
    assert np.all(result09 == sample_arrays["2dgray"][4:5, 4:5])
    assert result10.shape == (1, 1)
    assert np.all(result10 == sample_arrays["3dgray"][2, 4:5, 4:5])
    assert result11.shape == (1, 1, 3)
    assert np.all(result11 == sample_arrays["2drgb"][4:5, 4:5])
    assert result12.shape == (1, 1, 3)
    assert np.all(result12 == sample_arrays["3drgb"][2, 4:5, 4:5])
    assert result13.shape == (1, 1)
    assert np.all(result13 == sample_arrays["2dgray"][0:1, 0:1])
    assert result14.shape == (1, 1)
    assert np.all(result14 == sample_arrays["3dgray"][0, 0:1, 0:1])
    assert result15.shape == (1, 1, 3)
    assert np.all(result15 == sample_arrays["2drgb"][0:1, 0:1])
    assert result16.shape == (1, 1, 3)
    assert np.all(result16 == sample_arrays["3drgb"][0, 0:1, 0:1])

    result17: np.ndarray = dims.trim_pad(
        sample_arrays["2dgray"], "Trim", False, (11, 11), (11, 11), (11, 11), 0, False
    )
    result18: np.ndarray = dims.trim_pad(
        sample_arrays["3dgray"], "Trim", False, (11, 11), (11, 11), (11, 11), 0, False
    )
    result19: np.ndarray = dims.trim_pad(
        sample_arrays["2drgb"], "Trim", False, (11, 11), (11, 11), (11, 11), 0, False
    )
    result20: np.ndarray = dims.trim_pad(
        sample_arrays["3drgb"], "Trim", False, (11, 11), (11, 11), (11, 11), 0, False
    )
    result21: np.ndarray = dims.trim_pad(
        sample_arrays["2dgray"], "Trim", True, (11, 11), (11, 11), (11, 11), 0, False
    )
    result22: np.ndarray = dims.trim_pad(
        sample_arrays["3dgray"], "Trim", True, (11, 11), (11, 11), (11, 11), 0, False
    )
    result23: np.ndarray = dims.trim_pad(
        sample_arrays["2drgb"], "Trim", True, (11, 11), (11, 11), (11, 11), 0, False
    )
    result24: np.ndarray = dims.trim_pad(
        sample_arrays["3drgb"], "Trim", True, (11, 11), (11, 11), (11, 11), 0, False
    )

    assert result17.shape == sample_arrays["2dgray"].shape
    assert np.all(result17 == sample_arrays["2dgray"])
    assert result18.shape == sample_arrays["3dgray"].shape
    assert np.all(result18 == sample_arrays["3dgray"])
    assert result19.shape == sample_arrays["2drgb"].shape
    assert np.all(result19 == sample_arrays["2drgb"])
    assert result20.shape == sample_arrays["3drgb"].shape
    assert np.all(result20 == sample_arrays["3drgb"])
    assert result21.shape == (1, 1)
    assert np.all(result21 == sample_arrays["2dgray"][9:10, 9:10])
    assert result22.shape == (1, 1)
    assert np.all(result22 == sample_arrays["3dgray"][4, 9:10, 9:10])
    assert result23.shape == (1, 1, 3)
    assert np.all(result23 == sample_arrays["2drgb"][9:10, 9:10])
    assert result24.shape == (1, 1, 3)
    assert np.all(result24 == sample_arrays["3drgb"][4, 9:10, 9:10])

    result25: np.ndarray = dims.trim_pad(
        sample_arrays["2dgray"], "Trim", False, (3, 8), (3, 8), (3, 8), 0, False
    )
    result26: np.ndarray = dims.trim_pad(
        sample_arrays["3dgray"], "Trim", False, (3, 8), (3, 8), (3, 8), 0, False
    )
    result27: np.ndarray = dims.trim_pad(
        sample_arrays["2drgb"], "Trim", False, (3, 8), (3, 8), (3, 8), 0, False
    )
    result28: np.ndarray = dims.trim_pad(
        sample_arrays["3drgb"], "Trim", False, (3, 8), (3, 8), (3, 8), 0, False
    )
    result29: np.ndarray = dims.trim_pad(
        sample_arrays["2dgray"], "Trim", True, (3, 8), (3, 8), (3, 8), 0, False
    )
    result30: np.ndarray = dims.trim_pad(
        sample_arrays["3dgray"], "Trim", True, (3, 8), (3, 8), (3, 8), 0, False
    )
    result31: np.ndarray = dims.trim_pad(
        sample_arrays["2drgb"], "Trim", True, (3, 8), (3, 8), (3, 8), 0, False
    )
    result32: np.ndarray = dims.trim_pad(
        sample_arrays["3drgb"], "Trim", True, (3, 8), (3, 8), (3, 8), 0, False
    )

    assert result25.shape == (8, 8)
    assert np.all(result25 == sample_arrays["2dgray"][1:9, 1:9])
    assert result26.shape == (5, 8, 8)
    assert np.all(result26 == sample_arrays["3dgray"][:, 1:9, 1:9])
    assert result27.shape == (8, 8, 3)
    assert np.all(result27 == sample_arrays["2drgb"][1:9, 1:9])
    assert result28.shape == (5, 8, 8, 3)
    assert np.all(result28 == sample_arrays["3drgb"][:, 1:9, 1:9])
    assert result29.shape == (5, 5)
    assert np.all(result29 == sample_arrays["2dgray"][3:8, 3:8])
    assert result30.shape == (2, 5, 5)
    assert np.all(result30 == sample_arrays["3dgray"][3:, 3:8, 3:8])
    assert result31.shape == (5, 5, 3)
    assert np.all(result31 == sample_arrays["2drgb"][3:8, 3:8])
    assert result32.shape == (2, 5, 5, 3)
    assert np.all(result32 == sample_arrays["3drgb"][3:, 3:8, 3:8])

    result33: np.ndarray = dims.trim_pad(
        sample_arrays["2dgray"], "Trim", True, (3, 3), (3, 3), (3, 3), 0, False
    )
    result34: np.ndarray = dims.trim_pad(
        sample_arrays["3dgray"], "Trim", True, (3, 3), (3, 3), (3, 3), 0, False
    )
    result35: np.ndarray = dims.trim_pad(
        sample_arrays["2drgb"], "Trim", True, (3, 3), (3, 3), (3, 3), 0, False
    )
    result36: np.ndarray = dims.trim_pad(
        sample_arrays["3drgb"], "Trim", True, (3, 3), (3, 3), (3, 3), 0, False
    )
    result37: np.ndarray = dims.trim_pad(
        sample_arrays["2dgray"], "Trim", True, (3, 4), (3, 4), (3, 4), 0, False
    )
    result38: np.ndarray = dims.trim_pad(
        sample_arrays["3dgray"], "Trim", True, (3, 4), (3, 4), (3, 4), 0, False
    )
    result39: np.ndarray = dims.trim_pad(
        sample_arrays["2drgb"], "Trim", True, (3, 4), (3, 4), (3, 4), 0, False
    )
    result40: np.ndarray = dims.trim_pad(
        sample_arrays["3drgb"], "Trim", True, (3, 4), (3, 4), (3, 4), 0, False
    )

    assert result33.shape == (1, 1)
    assert np.all(result33 == sample_arrays["2dgray"][3:4, 3:4])
    assert result34.shape == (1, 1)
    assert np.all(result34 == sample_arrays["3dgray"][3, 3:4, 3:4])
    assert result35.shape == (1, 1, 3)
    assert np.all(result35 == sample_arrays["2drgb"][3:4, 3:4])
    assert result36.shape == (1, 1, 3)
    assert np.all(result36 == sample_arrays["3drgb"][3, 3:4, 3:4])
    assert result37.shape == (1, 1)
    assert np.all(result37 == sample_arrays["2dgray"][3:4, 3:4])
    assert result38.shape == (1, 1)
    assert np.all(result38 == sample_arrays["3dgray"][3, 3:4, 3:4])
    assert result39.shape == (1, 1, 3)
    assert np.all(result39 == sample_arrays["2drgb"][3:4, 3:4])
    assert result40.shape == (1, 1, 3)
    assert np.all(result40 == sample_arrays["3drgb"][3, 3:4, 3:4])

    result41: np.ndarray = dims.trim_pad(
        in_place_2dgray, "Trim", False, (3, 8), (3, 8), (3, 8), 0, True
    )
    result42: np.ndarray = dims.trim_pad(
        in_place_3dgray, "Trim", False, (3, 8), (3, 8), (3, 8), 0, True
    )
    result43: np.ndarray = dims.trim_pad(
        in_place_2drgb, "Trim", True, (3, 8), (3, 8), (3, 8), 0, True
    )
    result44: np.ndarray = dims.trim_pad(
        in_place_3drgb, "Trim", True, (3, 8), (3, 8), (3, 8), 0, True
    )

    assert result41.shape == result25.shape
    assert np.all(result41 == result25)
    assert result42.shape == result26.shape
    assert np.all(result42 == result26)
    assert result43.shape == result31.shape
    assert np.all(result43 == result31)
    assert result44.shape == result32.shape
    assert np.all(result44 == result32)


def test_make_3d(sample_arrays) -> None:

    result1: np.ndarray = dims.make_3d(sample_arrays["2dgray"])
    result2: np.ndarray = dims.make_3d(sample_arrays["3dgray"])
    result3: np.ndarray = dims.make_3d(sample_arrays["2drgb"])
    result4: np.ndarray = dims.make_3d(sample_arrays["3drgb"])
    assert result1.shape == (10, 10, 10)
    assert np.all(result2 == sample_arrays["3dgray"])
    assert result3.shape == (10, 10, 10, 3)
    assert np.all(result4 == sample_arrays["3drgb"])


def test_join(sample_arrays) -> None:

    result1: np.ndarray = dims.join(sample_arrays["2dgray"], sample_arrays["2dgray"])
    result2: np.ndarray = dims.join(sample_arrays["3dgray"], sample_arrays["2dgray"])
    result3: np.ndarray = dims.join(sample_arrays["2dgray"], sample_arrays["3dgray"])
    result4: np.ndarray = dims.join(sample_arrays["3dgray"], sample_arrays["3dgray"])
    result5: np.ndarray = dims.join(sample_arrays["2drgb"], sample_arrays["2drgb"])
    result6: np.ndarray = dims.join(sample_arrays["3drgb"], sample_arrays["2drgb"])
    result7: np.ndarray = dims.join(sample_arrays["2drgb"], sample_arrays["3drgb"])
    result8: np.ndarray = dims.join(sample_arrays["3drgb"], sample_arrays["3drgb"])
    assert result1.shape == (2, 10, 10)
    assert result2.shape == (6, 10, 10)
    assert result3.shape == (6, 10, 10)
    assert result4.shape == (10, 10, 10)
    assert result5.shape == (2, 10, 10, 3)
    assert result6.shape == (6, 10, 10, 3)
    assert result7.shape == (6, 10, 10, 3)
    assert result8.shape == (10, 10, 10, 3)

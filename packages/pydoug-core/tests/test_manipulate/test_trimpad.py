"""
Tests for PyDoug's manipulate.trimpad module.
"""

import numpy as np

from pydoug_core.manipulate import trimpad


def test_refine_pad_bounds(sample_arrays) -> None:

    result1: tuple = trimpad._refine_pad_bounds(
        sample_arrays["2dgray"], True, (0, 0), (0, 0), (0, 0)
    )
    assert result1[0] == (0, 0)
    assert result1[1] == (0, 0)
    assert result1[2] == (0, 0)

    result2: tuple = trimpad._refine_pad_bounds(
        sample_arrays["2dgray"],
        True,
        (2, 2),
        (3, 4),
        (-1, "test"),  # type: ignore
    )
    assert result2[0] == (2, 2)
    assert result2[1] == (3, 4)
    assert result2[2] == (-1, "test")

    result3: tuple = trimpad._refine_pad_bounds(
        sample_arrays["2dgray"], True, (-1, -1), (-1, -1), (0, 0)
    )
    assert result3[0] == (0, 0)
    assert result3[1] == (0, 0)

    result4: tuple = trimpad._refine_pad_bounds(
        sample_arrays["2dgray"],
        False,
        (2, 2),
        (3, sample_arrays["2dgray"].shape[0] + 2),
        (0, 0),
    )
    assert result4[0] == (0, 0)
    assert result4[1] == (1, 1)
    assert result4[2] == (0, 0)

    result5: tuple = trimpad._refine_pad_bounds(
        sample_arrays["3dgray"], True, (0, 0), (2, 2), (4, 4)
    )
    result6: tuple = trimpad._refine_pad_bounds(
        sample_arrays["3dgray"],
        False,
        (0, 0),
        (2, sample_arrays["3dgray"].shape[1] + 4),
        (4, sample_arrays["3dgray"].shape[0] + 8),
    )
    assert result5 == result6

    result7: tuple = trimpad._refine_pad_bounds(
        sample_arrays["3drgb"], True, (0, 0), (2, 2), (4, 4)
    )
    assert result7 == ((0, 0), (2, 2), (4, 4))


def test_refine_trim_bounds(sample_arrays) -> None:

    result1: tuple = trimpad._refine_trim_bounds(
        sample_arrays["2dgray"], True, (0, 0), (0, 0), (0, 0)
    )
    assert result1[0] == (0, sample_arrays["2dgray"].shape[1])
    assert result1[1] == (0, sample_arrays["2dgray"].shape[0])
    assert result1[2] == (0, 0)

    result2: tuple = trimpad._refine_trim_bounds(
        sample_arrays["2dgray"], True, (2, 2), (3, 4), (-1, 11)
    )
    assert result2[0] == (2, 8)
    assert result2[1] == (3, 6)
    assert result2[2] == (-1, 11)

    result3: tuple = trimpad._refine_trim_bounds(
        sample_arrays["2dgray"],
        True,
        (-1, -1),
        (6, 7),
        ("test", "test"),  # type: ignore
    )
    assert result3[0] == (0, sample_arrays["2dgray"].shape[1])
    assert result3[1] == (4, sample_arrays["2dgray"].shape[0] - 5)
    assert result3[2] == ("test", "test")

    result4: tuple = trimpad._refine_trim_bounds(
        sample_arrays["2dgray"],
        True,
        (0, sample_arrays["2dgray"].shape[1]),
        (sample_arrays["2dgray"].shape[0], sample_arrays["2dgray"].shape[0]),
        (0, 0),
    )
    assert result4[0] == (
        0,
        sample_arrays["2dgray"].shape[1] - (sample_arrays["2dgray"].shape[1] - 1),
    )
    assert result4[1] == (
        4,
        sample_arrays["2dgray"].shape[0] - (sample_arrays["2dgray"].shape[0] - 6),
    )

    result5: tuple = trimpad._refine_trim_bounds(
        sample_arrays["3dgray"], True, (2, 2), (2, 2), (2, 2)
    )
    assert result5[0] == (2, sample_arrays["3dgray"].shape[2] - 2)
    assert result5[1] == (2, sample_arrays["3dgray"].shape[1] - 2)
    assert result5[2] == (2, sample_arrays["3dgray"].shape[0] - 2)

    result6: tuple = trimpad._refine_trim_bounds(
        sample_arrays["2dgray"],
        False,
        (0, 0),
        (0, sample_arrays["2dgray"].shape[0]),
        (0, 0),
    )
    assert result6[0] == (4, 6)
    assert result6[1] == (0, sample_arrays["2dgray"].shape[0])
    assert result6[2] == (0, 0)

    result7: tuple = trimpad._refine_trim_bounds(
        sample_arrays["3dgray"],
        False,
        (0, 0),
        (0, sample_arrays["3dgray"].shape[1]),
        (0, sample_arrays["3dgray"].shape[0] + 1),
    )
    assert result7[0] == (4, 6)
    assert result7[1] == (0, sample_arrays["3dgray"].shape[1])
    assert result7[2] == (0, sample_arrays["3dgray"].shape[0])

    result8: tuple = trimpad._refine_trim_bounds(
        sample_arrays["3dgray"], True, (1, 1), (1, 1), (1, 1)
    )
    result9: tuple = trimpad._refine_trim_bounds(
        sample_arrays["3dgray"],
        False,
        (0, sample_arrays["3dgray"].shape[2] - 2),
        (0, sample_arrays["3dgray"].shape[1] - 2),
        (0, sample_arrays["3dgray"].shape[0] - 2),
    )
    assert result8 == result9

    result10: tuple = trimpad._refine_trim_bounds(
        sample_arrays["3drgb"], True, (0, 0), (4, 4), (2, 2)
    )
    assert result10 == (
        (0, sample_arrays["3drgb"].shape[2]),
        (4, sample_arrays["3drgb"].shape[1] - 4),
        (2, sample_arrays["3drgb"].shape[0] - 2),
    )


def test_trim_pad(sample_arrays) -> None:

    result1: np.ndarray = trimpad.trim_pad(
        sample_arrays["2dgray"], "Trim", True, (0, 0), (0, 0), (0, 0)
    )
    assert np.all(result1 == sample_arrays["2dgray"])

    result2: np.ndarray = trimpad.trim_pad(
        sample_arrays["3dgray"], "Trim", True, (0, 0), (0, 0), (0, 0)
    )
    assert np.all(result2 == sample_arrays["3dgray"])

    result3: np.ndarray = trimpad.trim_pad(
        sample_arrays["2dgray"], "Trim", True, (2, 2), (4, 4), (2, 2)
    )
    result4: np.ndarray = trimpad.trim_pad(
        sample_arrays["2dgray"],
        "Trim",
        False,
        (2, sample_arrays["2dgray"].shape[1] - 4),
        (4, sample_arrays["2dgray"].shape[0] - 8),
        (2, 2),
    )
    assert np.all(result3 == result4)

    result5: np.ndarray = trimpad.trim_pad(
        sample_arrays["3dgray"], "Trim", True, (2, 2), (4, 4), (2, 2)
    )
    result6: np.ndarray = trimpad.trim_pad(
        sample_arrays["3dgray"],
        "Trim",
        False,
        (2, sample_arrays["3dgray"].shape[2] - 4),
        (4, sample_arrays["3dgray"].shape[1] - 8),
        (2, sample_arrays["3dgray"].shape[0] - 4),
    )
    assert np.all(result5 == result6)

    result7: np.ndarray = trimpad.trim_pad(
        sample_arrays["2dgray"], "Pad", True, (0, 0), (0, 0), (0, 0)
    )
    assert np.all(result7 == sample_arrays["2dgray"])

    result8: np.ndarray = trimpad.trim_pad(
        sample_arrays["3dgray"], "Pad", True, (0, 0), (0, 0), (0, 0)
    )
    assert np.all(result8 == sample_arrays["3dgray"])

    result9: np.ndarray = trimpad.trim_pad(
        sample_arrays["2dgray"], "Pad", True, (2, 2), (4, 4), (2, 2)
    )
    result10: np.ndarray = trimpad.trim_pad(
        sample_arrays["2dgray"],
        "Pad",
        False,
        (2, sample_arrays["2dgray"].shape[1] + 4),
        (4, sample_arrays["2dgray"].shape[0] + 8),
        (2, 2),
    )
    assert np.all(result9 == result10)

    result11: np.ndarray = trimpad.trim_pad(
        sample_arrays["3dgray"], "Pad", True, (2, 2), (4, 4), (2, 2)
    )
    result12: np.ndarray = trimpad.trim_pad(
        sample_arrays["3dgray"],
        "Pad",
        False,
        (2, sample_arrays["3dgray"].shape[2] + 4),
        (4, sample_arrays["3dgray"].shape[1] + 8),
        (2, sample_arrays["3dgray"].shape[0] + 4),
    )
    assert np.all(result11 == result12)

    in_place_array: np.ndarray = np.copy(sample_arrays["2dgray"])
    result13: np.ndarray = trimpad.trim_pad(
        in_place_array, "Pad", True, (2, 2), (4, 4), (0, 0), In_Place=True
    )
    assert result13.shape == (18, 14)

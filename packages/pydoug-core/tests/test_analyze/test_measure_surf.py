"""
Tests for PyDoug's analyze.measure module's surf function.
"""

import numpy as np
import pytest

from pydoug_core.analyze.measure import surf
from pydoug_core.results import PlotResult, ValueResult


@pytest.fixture
def array_2d_3() -> np.ndarray:
    """
    Fixture to create a 2D array with 3 phases.

    Surf Counts:
    0:      9
    1:      9
    2:      6
    0/1:    6
    0/2:    3
    1/2:    3

    Bulk Counts:
    0:      4
    1:      7
    2:      5
    """

    return np.array(
        [
            [1, 1, 1, 2],
            [1, 0, 2, 2],
            [0, 0, 0, 2],
            [1, 1, 1, 2],
        ],
        np.uint8,
    )


@pytest.fixture
def array_2d_2(array_2d_3) -> np.ndarray:
    """
    Fixture to create a 2D array with two phases.

    Surf Counts:
    T/F:    9

    Bulk Counts:
    F:      4
    T:      12
    """

    return array_2d_3 >= 1


@pytest.fixture
def array_3d_3() -> np.ndarray:
    """
    Fixture to create a 3D array with three phases.

    Surf Counts:
    0:      32  9   17
    1:      32  9   16
    2:      24  6   11
    0/1:    20  6   11
    0/2:    12  3   6
    1/2:    12  3   5

    Bulk Counts:
    0:      10  4   6
    1:      13  7   6
    2:      9   5   4
    """

    return np.array(
        [
            [
                [1, 1, 1, 2],
                [1, 0, 2, 2],
                [0, 0, 0, 2],
                [1, 1, 1, 2],
            ],
            [
                [0, 2, 1, 0],
                [1, 0, 1, 2],
                [0, 1, 2, 0],
                [1, 0, 2, 1],
            ],
        ],
        np.uint8,
    )


@pytest.fixture
def array_3d_2(array_3d_3) -> np.ndarray:
    """
    Fixture to create a 3D array with two phases.

    Surf Counts:
    T/F:    32  9   17

    Bulk Counts:
    F:      10  4   6
    T:      22  12  10
    """

    return array_3d_3 >= 1


def test_surf_scalar(array_2d_2, array_2d_3, array_3d_2, array_3d_3) -> None:

    result001 = surf(array_2d_2, Units="um")
    result002 = surf(np.zeros((5, 5), np.bool))
    result01 = surf(array_2d_2, Correct_Overestimation=False)
    result02 = surf(array_3d_2, Correct_Overestimation=False)
    result03 = surf(array_2d_3, Correct_Overestimation=False)
    result04 = surf(array_3d_3, Correct_Overestimation=False)
    result05 = surf(
        array_2d_2, Correct_Overestimation=False, Include_Phase_Breakdown=True
    )
    result06 = surf(
        array_3d_2, Correct_Overestimation=False, Include_Phase_Breakdown=True
    )
    result07 = surf(
        array_2d_3, Correct_Overestimation=False, Include_Phase_Breakdown=True
    )
    result08 = surf(
        array_3d_3, Correct_Overestimation=False, Include_Phase_Breakdown=True
    )
    assert isinstance(result001, ValueResult)
    assert isinstance(result002, ValueResult)
    assert isinstance(result01, ValueResult)
    assert isinstance(result02, ValueResult)
    assert isinstance(result03, ValueResult)
    assert isinstance(result04, ValueResult)
    assert isinstance(result05, ValueResult)
    assert isinstance(result06, ValueResult)
    assert isinstance(result07, ValueResult)
    assert isinstance(result08, ValueResult)
    result002 = result002.values
    result01 = result01.values
    result02 = result02.values
    result03 = result03.values
    result04 = result04.values
    result05 = result05.values
    result06 = result06.values
    result07 = result07.values
    result08 = result08.values
    assert isinstance(result002, dict)
    assert isinstance(result01, dict)
    assert isinstance(result02, dict)
    assert isinstance(result03, dict)
    assert isinstance(result04, dict)
    assert isinstance(result05, dict)
    assert isinstance(result06, dict)
    assert isinstance(result07, dict)
    assert isinstance(result08, dict)
    assert list(result002.keys()) == [False]
    assert list(result01.keys()) == [0, 1]
    assert list(result02.keys()) == [0, 1]
    assert list(result03.keys()) == ["Total", 0, 1, 2]
    assert list(result04.keys()) == ["Total", 0, 1, 2]
    assert list(result05.keys()) == [0, 1]
    assert list(result06.keys()) == [0, 1]
    assert list(result07.keys()) == ["Total", 0, "0/1", "0/2", 1, "1/2", 2]
    assert list(result08.keys()) == ["Total", 0, "0/1", "0/2", 1, "1/2", 2]
    assert result002[False] == 0
    assert result01[0] == result01[1] == 9
    assert result02[0] == result02[1] == 32
    assert result03["Total"] == 15
    assert result03[0] == result03[1] == 9
    assert result03[2] == 6
    assert result04["Total"] == 56
    assert result04[0] == result04[1] == 32
    assert result04[2] == 24
    assert result05[0] == result05[1] == 9
    assert result06[0] == result06[1] == 32
    assert result07["Total"] == 15
    assert result07[0] == result07[1] == 9
    assert result07[2] == result07["0/1"] == 6
    assert result07["0/2"] == result07["1/2"] == 3
    assert result08["Total"] == 56
    assert result08[0] == result08[1] == 32
    assert result08[2] == 24
    assert result08["0/1"] == 20
    assert result08["0/2"] == result08["1/2"] == 12

    mask_2d: np.ndarray = array_2d_3 < 2
    mask_3d: np.ndarray = array_3d_3 < 2
    result09 = surf(
        array_2d_2, Correct_Overestimation=False, Apply_Mask=True, Mask=mask_2d
    )
    result10 = surf(
        array_3d_2, Correct_Overestimation=False, Apply_Mask=True, Mask=mask_3d
    )
    result11 = surf(
        array_2d_3, Correct_Overestimation=False, Apply_Mask=True, Mask=mask_2d
    )
    result12 = surf(
        array_3d_3, Correct_Overestimation=False, Apply_Mask=True, Mask=mask_3d
    )
    result13 = surf(
        array_2d_2,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Apply_Mask=True,
        Mask=mask_2d,
    )
    result14 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Apply_Mask=True,
        Mask=mask_3d,
    )
    result15 = surf(
        array_2d_3,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Apply_Mask=True,
        Mask=mask_2d,
    )
    result16 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Apply_Mask=True,
        Mask=mask_3d,
    )
    assert isinstance(result09, ValueResult)
    assert isinstance(result10, ValueResult)
    assert isinstance(result11, ValueResult)
    assert isinstance(result12, ValueResult)
    assert isinstance(result13, ValueResult)
    assert isinstance(result14, ValueResult)
    assert isinstance(result15, ValueResult)
    assert isinstance(result16, ValueResult)
    result09 = result09.values
    result10 = result10.values
    result11 = result11.values
    result12 = result12.values
    result13 = result13.values
    result14 = result14.values
    result15 = result15.values
    result16 = result16.values
    assert isinstance(result09, dict)
    assert isinstance(result10, dict)
    assert isinstance(result11, dict)
    assert isinstance(result12, dict)
    assert isinstance(result13, dict)
    assert isinstance(result14, dict)
    assert isinstance(result15, dict)
    assert isinstance(result16, dict)
    for result in (
        result09,
        result10,
        result11,
        result12,
        result13,
        result14,
        result15,
        result16,
    ):
        assert list(result.keys()) == [0, 1]
    for result in (result09, result11, result13, result15):
        assert result[0] == result[1] == 9
    for result in (result10, result12, result14, result16):
        assert result[0] == result[1] == 32

    result17 = surf(array_2d_2, "Perimeter")
    result18 = surf(array_3d_2)
    result19 = surf(array_2d_3, "Perimeter")
    result20 = surf(array_3d_3)
    result21 = surf(array_2d_2, "Perimeter", Include_Phase_Breakdown=True)
    result22 = surf(array_3d_2, Include_Phase_Breakdown=True)
    result23 = surf(array_2d_3, "Perimeter", Include_Phase_Breakdown=True)
    result24 = surf(array_3d_3, Include_Phase_Breakdown=True)
    assert isinstance(result17, ValueResult)
    assert isinstance(result18, ValueResult)
    assert isinstance(result19, ValueResult)
    assert isinstance(result20, ValueResult)
    assert isinstance(result21, ValueResult)
    assert isinstance(result22, ValueResult)
    assert isinstance(result23, ValueResult)
    assert isinstance(result24, ValueResult)
    result17 = result17.values
    result18 = result18.values
    result19 = result19.values
    result20 = result20.values
    result21 = result21.values
    result22 = result22.values
    result23 = result23.values
    result24 = result24.values
    assert isinstance(result17, dict)
    assert isinstance(result18, dict)
    assert isinstance(result19, dict)
    assert isinstance(result20, dict)
    assert isinstance(result21, dict)
    assert isinstance(result22, dict)
    assert isinstance(result23, dict)
    assert isinstance(result24, dict)
    assert list(result17.keys()) == [0, 1]
    assert list(result18.keys()) == [0, 1]
    assert list(result19.keys()) == ["Total", 0, 1, 2]
    assert list(result20.keys()) == ["Total", 0, 1, 2]
    assert list(result21.keys()) == [0, 1]
    assert list(result22.keys()) == [0, 1]
    assert list(result23.keys()) == ["Total", 0, "0/1", "0/2", 1, "1/2", 2]
    assert list(result24.keys()) == ["Total", 0, "0/1", "0/2", 1, "1/2", 2]
    assert result17[0] == result17[1] == 9 * (np.pi / 4)
    assert result18[0] == result18[1] == 32 * (2 / 3)
    assert result19["Total"] == 15 * (np.pi / 4)
    assert result19[0] == result19[1] == 9 * (np.pi / 4)
    assert result19[2] == 6 * (np.pi / 4)
    assert result20["Total"] == 56 * (2 / 3)
    assert result20[0] == result20[1] == 32 * (2 / 3)
    assert result20[2] == 24 * (2 / 3)
    assert result21[0] == result21[1] == 9 * (np.pi / 4)
    assert result22[0] == result22[1] == 32 * (2 / 3)
    assert result23["Total"] == 15 * (np.pi / 4)
    assert result23[0] == result23[1] == 9 * (np.pi / 4)
    assert result23[2] == result23["0/1"] == 6 * (np.pi / 4)
    assert result23["0/2"] == result23["1/2"] == 3 * (np.pi / 4)
    assert result24["Total"] == 56 * (2 / 3)
    assert result24[0] == result24[1] == 32 * (2 / 3)
    assert result24[2] == 24 * (2 / 3)
    assert result24["0/1"] == 20 * (2 / 3)
    assert result24["0/2"] == result24["1/2"] == 12 * (2 / 3)


def test_surf_scalar_norm(array_2d_2, array_2d_3, array_3d_2, array_3d_3) -> None:

    result001 = surf(
        array_2d_2,
        Correct_Overestimation=False,
        Normalize=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result01 = surf(array_2d_2, Correct_Overestimation=False, Normalize=True)
    result02 = surf(array_3d_2, Correct_Overestimation=False, Normalize=True)
    result03 = surf(array_2d_3, Correct_Overestimation=False, Normalize=True)
    result04 = surf(array_3d_3, Correct_Overestimation=False, Normalize=True)
    result05 = surf(
        array_2d_2,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
    )
    result06 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
    )
    result07 = surf(
        array_2d_3,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
    )
    result08 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
    )
    assert result001 == result01
    assert isinstance(result01, ValueResult)
    assert isinstance(result02, ValueResult)
    assert isinstance(result03, ValueResult)
    assert isinstance(result04, ValueResult)
    assert isinstance(result05, ValueResult)
    assert isinstance(result06, ValueResult)
    assert isinstance(result07, ValueResult)
    assert isinstance(result08, ValueResult)
    result01 = result01.values
    result02 = result02.values
    result03 = result03.values
    result04 = result04.values
    result05 = result05.values
    result06 = result06.values
    result07 = result07.values
    result08 = result08.values
    assert isinstance(result01, dict)
    assert isinstance(result02, dict)
    assert isinstance(result03, dict)
    assert isinstance(result04, dict)
    assert isinstance(result05, dict)
    assert isinstance(result06, dict)
    assert isinstance(result07, dict)
    assert isinstance(result08, dict)
    assert list(result01.keys()) == [0, 1]
    assert list(result02.keys()) == [0, 1]
    assert list(result03.keys()) == ["Total", 0, 1, 2]
    assert list(result04.keys()) == ["Total", 0, 1, 2]
    assert list(result05.keys()) == [0, 1]
    assert list(result06.keys()) == [0, 1]
    assert list(result07.keys()) == ["Total", 0, "0/1", "0/2", 1, "1/2", 2]
    assert list(result08.keys()) == ["Total", 0, "0/1", "0/2", 1, "1/2", 2]
    assert result01[0] == result01[1] == result02[0] == result02[1] == 1
    assert result05[0] == result05[1] == result06[0] == result06[1] == 1
    assert result03["Total"] == result04["Total"] == 1
    assert result03[0] == result03[1] == 9 / 15
    assert result03[2] == 6 / 15
    assert result04[0] == result04[1] == 32 / 56
    assert result04[2] == 24 / 56
    assert result07["Total"] == result08["Total"] == 1
    assert result07[0] == result07[1] == 9 / 15
    assert result07[2] == result07["0/1"] == 6 / 15
    assert result07["0/2"] == result07["1/2"] == 3 / 15
    assert result08[0] == result08[1] == 32 / 56
    assert result08[2] == 24 / 56
    assert result08["0/1"] == 20 / 56
    assert result08["0/2"] == result08["1/2"] == 12 / 56

    result09 = surf(
        array_2d_2,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Phase",
    )
    result10 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Phase",
    )
    result11 = surf(
        array_2d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Phase",
    )
    result12 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Phase",
    )
    result13 = surf(
        array_2d_2,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
        Normalization_Method="Phase",
    )
    result14 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
        Normalization_Method="Phase",
    )
    result15 = surf(
        array_2d_3,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
        Normalization_Method="Phase",
    )
    result16 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
        Normalization_Method="Phase",
    )
    assert isinstance(result09, ValueResult)
    assert isinstance(result10, ValueResult)
    assert isinstance(result11, ValueResult)
    assert isinstance(result12, ValueResult)
    assert isinstance(result13, ValueResult)
    assert isinstance(result14, ValueResult)
    assert isinstance(result15, ValueResult)
    assert isinstance(result16, ValueResult)
    result09 = result09.values
    result10 = result10.values
    result11 = result11.values
    result12 = result12.values
    result13 = result13.values
    result14 = result14.values
    result15 = result15.values
    result16 = result16.values
    assert isinstance(result09, dict)
    assert isinstance(result10, dict)
    assert isinstance(result11, dict)
    assert isinstance(result12, dict)
    assert isinstance(result13, dict)
    assert isinstance(result14, dict)
    assert isinstance(result15, dict)
    assert isinstance(result16, dict)
    assert list(result09.keys()) == [0, 1]
    assert list(result10.keys()) == [0, 1]
    assert list(result11.keys()) == [0, 1, 2]
    assert list(result12.keys()) == [0, 1, 2]
    assert list(result13.keys()) == [0, 1]
    assert list(result14.keys()) == [0, 1]
    assert list(result15.keys()) == [
        0,
        1,
        2,
        "0/1 by 0",
        "0/2 by 0",
        "0/1 by 1",
        "1/2 by 1",
        "0/2 by 2",
        "1/2 by 2",
    ]
    assert list(result16.keys()) == [
        0,
        1,
        2,
        "0/1 by 0",
        "0/2 by 0",
        "0/1 by 1",
        "1/2 by 1",
        "0/2 by 2",
        "1/2 by 2",
    ]
    assert result09[0] == result09[1] == result10[0] == result10[1] == 1
    assert result11[0] == result11[1] == result11[2] == 1
    assert result12[0] == result12[1] == result12[2] == 1
    assert result13[0] == result13[1] == result14[0] == result14[1] == 1
    assert result15[0] == result15[1] == result15[2] == 1
    assert result15["0/1 by 0"] == result15["0/1 by 1"] == 6 / 9
    assert result15["0/2 by 0"] == result15["1/2 by 1"] == 3 / 9
    assert result15["0/2 by 2"] == result15["1/2 by 2"] == 3 / 6
    assert result16[0] == result16[1] == result16[2] == 1
    assert result16["0/1 by 0"] == result16["0/1 by 1"] == 20 / 32
    assert result16["1/2 by 1"] == result16["0/2 by 0"] == 12 / 32
    assert result16["0/2 by 2"] == result16["1/2 by 2"] == 12 / 24

    result17 = surf(
        array_2d_2,
        "Perimeter",
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Total",
    )
    result18 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Total",
    )
    result19 = surf(
        array_2d_3,
        "Perimeter",
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Total",
    )
    result20 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Total",
    )
    result21 = surf(
        array_2d_2,
        "Perimeter",
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
        Normalization_Method="Specific - Total",
    )
    result22 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
        Normalization_Method="Specific - Total",
    )
    result23 = surf(
        array_2d_3,
        "Perimeter",
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
        Normalization_Method="Specific - Total",
    )
    result24 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
        Normalization_Method="Specific - Total",
    )
    assert isinstance(result17, ValueResult)
    assert isinstance(result18, ValueResult)
    assert isinstance(result19, ValueResult)
    assert isinstance(result20, ValueResult)
    assert isinstance(result21, ValueResult)
    assert isinstance(result22, ValueResult)
    assert isinstance(result23, ValueResult)
    assert isinstance(result24, ValueResult)
    result17 = result17.values
    result18 = result18.values
    result19 = result19.values
    result20 = result20.values
    result21 = result21.values
    result22 = result22.values
    result23 = result23.values
    result24 = result24.values
    assert isinstance(result17, dict)
    assert isinstance(result18, dict)
    assert isinstance(result19, dict)
    assert isinstance(result20, dict)
    assert isinstance(result21, dict)
    assert isinstance(result22, dict)
    assert isinstance(result23, dict)
    assert isinstance(result24, dict)
    assert list(result17.keys()) == [0, 1]
    assert list(result18.keys()) == [0, 1]
    assert list(result19.keys()) == ["Total", 0, 1, 2]
    assert list(result20.keys()) == ["Total", 0, 1, 2]
    assert list(result21.keys()) == [0, 1]
    assert list(result22.keys()) == [0, 1]
    assert list(result23.keys()) == ["Total", 0, "0/1", "0/2", 1, "1/2", 2]
    assert list(result24.keys()) == ["Total", 0, "0/1", "0/2", 1, "1/2", 2]
    assert result17[0] == result17[1] == 9 / 16
    assert result18[0] == result18[1] == 32 / 32
    assert result19[0] == result19[1] == 9 / 16
    assert result19[2] == 6 / 16
    assert result19["Total"] == 15 / 16
    assert result20[0] == result20[1] == 32 / 32
    assert result20[2] == 24 / 32
    assert result20["Total"] == 56 / 32
    assert result21[0] == result21[1] == 9 / 16
    assert result22[0] == result22[1] == 32 / 32
    assert result23[0] == result23[1] == 9 / 16
    assert result23[2] == result23["0/1"] == 6 / 16
    assert result23["0/2"] == result23["1/2"] == 3 / 16
    assert result23["Total"] == 15 / 16
    assert result24[0] == result24[1] == 32 / 32
    assert result24[2] == 24 / 32
    assert result24["0/1"] == 20 / 32
    assert result24["0/2"] == result24["1/2"] == 12 / 32
    assert result24["Total"] == 56 / 32

    result25 = surf(
        array_2d_2,
        "Perimeter",
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Segmented",
    )
    result26 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Segmented",
    )
    result27 = surf(
        array_2d_3,
        "Perimeter",
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Segmented",
    )
    result28 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Segmented",
    )
    result29 = surf(
        array_2d_2,
        "Perimeter",
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
        Normalization_Method="Specific - Segmented",
    )
    result30 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
        Normalization_Method="Specific - Segmented",
    )
    result31 = surf(
        array_2d_3,
        "Perimeter",
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
        Normalization_Method="Specific - Segmented",
    )
    result32 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
        Normalization_Method="Specific - Segmented",
    )
    assert isinstance(result25, ValueResult)
    assert isinstance(result26, ValueResult)
    assert isinstance(result27, ValueResult)
    assert isinstance(result28, ValueResult)
    assert isinstance(result29, ValueResult)
    assert isinstance(result30, ValueResult)
    assert isinstance(result31, ValueResult)
    assert isinstance(result32, ValueResult)
    result25 = result25.values
    result26 = result26.values
    result27 = result27.values
    result28 = result28.values
    result29 = result29.values
    result30 = result30.values
    result31 = result31.values
    result32 = result32.values
    assert isinstance(result25, dict)
    assert isinstance(result26, dict)
    assert isinstance(result27, dict)
    assert isinstance(result28, dict)
    assert isinstance(result29, dict)
    assert isinstance(result30, dict)
    assert isinstance(result31, dict)
    assert isinstance(result32, dict)
    assert list(result25.keys()) == [0, 1]
    assert list(result26.keys()) == [0, 1]
    assert list(result27.keys()) == ["Total", 0, 1, 2]
    assert list(result28.keys()) == ["Total", 0, 1, 2]
    assert list(result29.keys()) == [0, 1]
    assert list(result30.keys()) == [0, 1]
    assert list(result31.keys()) == ["Total", 0, "0/1", "0/2", 1, "1/2", 2]
    assert list(result32.keys()) == ["Total", 0, "0/1", "0/2", 1, "1/2", 2]
    assert result25[0] == result25[1] == 9 / 12
    assert result26[0] == result26[1] == 32 / 22
    assert result27[0] == result27[1] == 9 / 12
    assert result27[2] == 6 / 12
    assert result27["Total"] == 15 / 12
    assert result28[0] == result28[1] == 32 / 22
    assert result28[2] == 24 / 22
    assert result28["Total"] == 56 / 22
    assert result29[0] == result29[1] == 9 / 12
    assert result30[0] == result30[1] == 32 / 22
    assert result31[0] == result31[1] == 9 / 12
    assert result31[2] == result31["0/1"] == 6 / 12
    assert result31["0/2"] == result31["1/2"] == 3 / 12
    assert result31["Total"] == 15 / 12
    assert result32[0] == result32[1] == 32 / 22
    assert result32[2] == 24 / 22
    assert result32["0/1"] == 20 / 22
    assert result32["0/2"] == result32["1/2"] == 12 / 22
    assert result32["Total"] == 56 / 22

    result33 = surf(
        array_2d_2,
        "Perimeter",
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Phase",
    )
    result34 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Phase",
    )
    result35 = surf(
        array_2d_3,
        "Perimeter",
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Phase",
    )
    result36 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Phase",
    )
    result37 = surf(
        array_2d_2,
        "Perimeter",
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
        Normalization_Method="Specific - Phase",
    )
    result38 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
        Normalization_Method="Specific - Phase",
    )
    result39 = surf(
        array_2d_3,
        "Perimeter",
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
        Normalization_Method="Specific - Phase",
    )
    result40 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Normalize=True,
        Normalization_Method="Specific - Phase",
    )
    assert isinstance(result33, ValueResult)
    assert isinstance(result34, ValueResult)
    assert isinstance(result35, ValueResult)
    assert isinstance(result36, ValueResult)
    assert isinstance(result37, ValueResult)
    assert isinstance(result38, ValueResult)
    assert isinstance(result39, ValueResult)
    assert isinstance(result40, ValueResult)
    result33 = result33.values
    result34 = result34.values
    result35 = result35.values
    result36 = result36.values
    result37 = result37.values
    result38 = result38.values
    result39 = result39.values
    result40 = result40.values
    assert isinstance(result33, dict)
    assert isinstance(result34, dict)
    assert isinstance(result35, dict)
    assert isinstance(result36, dict)
    assert isinstance(result37, dict)
    assert isinstance(result38, dict)
    assert isinstance(result39, dict)
    assert isinstance(result40, dict)
    assert list(result33.keys()) == [0, 1]
    assert list(result34.keys()) == [0, 1]
    assert list(result35.keys()) == [0, 1, 2]
    assert list(result36.keys()) == [0, 1, 2]
    assert list(result37.keys()) == [0, 1]
    assert list(result38.keys()) == [0, 1]
    assert list(result39.keys()) == [
        0,
        1,
        2,
        "0/1 by 0",
        "0/2 by 0",
        "0/1 by 1",
        "1/2 by 1",
        "0/2 by 2",
        "1/2 by 2",
    ]
    assert list(result40.keys()) == [
        0,
        1,
        2,
        "0/1 by 0",
        "0/2 by 0",
        "0/1 by 1",
        "1/2 by 1",
        "0/2 by 2",
        "1/2 by 2",
    ]
    assert result33[0] == result37[0] == 9 / 4
    assert result33[1] == result37[1] == 9 / 12
    assert result34[0] == result38[0] == 32 / 10
    assert result34[1] == result38[1] == 32 / 22
    assert result35[0] == result39[0] == 9 / 4
    assert result35[1] == result39[1] == 9 / 7
    assert result35[2] == result39[2] == 6 / 5
    assert result36[0] == result40[0] == 32 / 10
    assert result36[1] == result40[1] == 32 / 13
    assert result36[2] == result40[2] == 24 / 9
    assert result39["0/1 by 0"] == 6 / 4
    assert result39["0/1 by 1"] == 6 / 7
    assert result39["0/2 by 0"] == 3 / 4
    assert result39["1/2 by 1"] == 3 / 7
    assert result39["0/2 by 2"] == result39["1/2 by 2"] == 3 / 5
    assert result40["0/1 by 0"] == 20 / 10
    assert result40["0/1 by 1"] == 20 / 13
    assert result40["0/2 by 0"] == 12 / 10
    assert result40["1/2 by 1"] == 12 / 13
    assert result40["0/2 by 2"] == result40["1/2 by 2"] == 12 / 9

    result41 = surf(
        array_2d_2,
        "Perimeter",
        Normalize=True,
        Normalization_Method="Specific - Phase",
    )
    result42 = surf(
        array_3d_2,
        Normalize=True,
        Normalization_Method="Specific - Phase",
    )
    assert isinstance(result41, ValueResult)
    assert isinstance(result42, ValueResult)
    result41 = result41.values
    result42 = result42.values
    assert isinstance(result41, dict)
    assert isinstance(result42, dict)
    assert list(result41.keys()) == [0, 1]
    assert list(result42.keys()) == [0, 1]
    assert round(result41[0], 5) == round((9 / 4) * (np.pi / 4), 5)
    assert round(result41[1], 5) == round((9 / 12) * (np.pi / 4), 5)
    assert round(result42[0], 5) == round((32 / 10) * (2 / 3), 5)
    assert round(result42[1], 5) == round((32 / 22) * (2 / 3), 5)


def test_surf_plot(array_2d_2, array_3d_2, array_3d_3) -> None:

    result001 = surf(
        array_2d_2,
        Correct_Overestimation=False,
        Perform_Slicewise=True,
    )
    result002 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result003 = surf(np.zeros((5, 5, 5), np.bool), Perform_Slicewise=True)
    result004 = surf(
        array_3d_2,
        "Perimeter",
        Correct_Overestimation=False,
        Perform_Slicewise=True,
    )
    result01 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Perform_Slicewise=True,
    )
    result02 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Perform_Slicewise=True,
    )
    result03 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
    )
    result04 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
    )
    assert isinstance(result001, ValueResult)
    assert isinstance(result002, PlotResult)
    assert isinstance(result003, PlotResult)
    assert isinstance(result004, PlotResult)
    assert isinstance(result01, PlotResult)
    assert isinstance(result02, PlotResult)
    assert isinstance(result03, PlotResult)
    assert isinstance(result04, PlotResult)
    result002 = result002.data
    result003 = result003.data
    result004 = result004.data
    result01 = result01.data
    result02 = result02.data
    result03 = result03.data
    result04 = result04.data
    assert isinstance(result002, dict)
    assert isinstance(result003, dict)
    assert isinstance(result004, dict)
    assert isinstance(result01, dict)
    assert isinstance(result02, dict)
    assert isinstance(result03, dict)
    assert isinstance(result04, dict)
    assert list(result002.keys()) == list(result01.keys())
    assert list(result003.keys()) == ["Slice", False]
    assert list(result004.keys()) == list(result01.keys())
    assert list(result01.keys()) == ["Slice", 0, 1]
    assert list(result02.keys()) == ["Slice", "Total", 0, 1, 2]
    assert list(result03.keys()) == ["Slice", 0, 1]
    assert list(result04.keys()) == ["Slice", "Total", 0, "0/1", "0/2", 1, "1/2", 2]
    for key in result002.keys():
        assert np.all(result002[key] == result01[key])
        assert np.all(result004[key] == result01[key])
    assert np.all(result01["Slice"] == result02["Slice"])
    assert np.all(result01["Slice"] == result03["Slice"])
    assert np.all(result01["Slice"] == result04["Slice"])
    assert np.all(result01[0] == np.array([9, 17]))  # type: ignore
    assert np.all(result01[1] == np.array([9, 17]))  # type: ignore
    assert np.all(result02[0] == np.array([9, 17]))  # type: ignore
    assert np.all(result02[1] == np.array([9, 16]))  # type: ignore
    assert np.all(result02[2] == np.array([6, 11]))  # type: ignore
    assert np.all(result02["Total"] == np.array([15, 27]))  # type: ignore
    assert np.all(result01[0] == result03[0])  # type: ignore
    assert np.all(result01[1] == result03[1])  # type: ignore
    assert np.all(result02[0] == result04[0])  # type: ignore
    assert np.all(result02[1] == result04[1])  # type: ignore
    assert np.all(result02[2] == result04[2])  # type: ignore
    assert np.all(result02["Total"] == result04["Total"])  # type: ignore
    assert np.all(result04["0/1"] == np.array([6, 11]))  # type: ignore
    assert np.all(result04["0/2"] == np.array([3, 6]))  # type: ignore
    assert np.all(result04["1/2"] == np.array([3, 5]))  # type: ignore

    mask: np.ndarray = array_3d_3 < 2
    result05 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Apply_Mask=True,
        Mask=mask,
        Perform_Slicewise=True,
    )
    result06 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Apply_Mask=True,
        Mask=mask,
        Perform_Slicewise=True,
    )
    result07 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Apply_Mask=True,
        Mask=mask,
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
    )
    result08 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Apply_Mask=True,
        Mask=mask,
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
    )
    assert isinstance(result05, PlotResult)
    assert isinstance(result06, PlotResult)
    assert isinstance(result07, PlotResult)
    assert isinstance(result08, PlotResult)
    result05 = result05.data
    result06 = result06.data
    result07 = result07.data
    result08 = result08.data
    assert isinstance(result05, dict)
    assert isinstance(result06, dict)
    assert isinstance(result07, dict)
    assert isinstance(result08, dict)
    assert list(result05.keys()) == ["Slice", 0, 1]
    assert list(result06.keys()) == ["Slice", 0, 1]
    assert list(result07.keys()) == ["Slice", 0, 1]
    assert list(result08.keys()) == ["Slice", 0, 1]
    assert np.all(result05[0] == np.array([9, 17]))  # type: ignore
    assert np.all(result05[1] == np.array([9, 17]))  # type: ignore
    assert np.all(result06[0] == np.array([9, 17]))  # type: ignore
    assert np.all(result06[1] == np.array([9, 16]))  # type: ignore
    assert np.all(result05[0] == result07[0])  # type: ignore
    assert np.all(result05[1] == result07[1])  # type: ignore
    assert np.all(result06[0] == result08[0])  # type: ignore
    assert np.all(result06[1] == result08[1])  # type: ignore

    result09 = surf(array_3d_2, Perform_Slicewise=True)
    result10 = surf(array_3d_3, Perform_Slicewise=True)
    result11 = surf(
        array_3d_2,
        "Perimeter",
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
    )
    result12 = surf(
        array_3d_3,
        "Perimeter",
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
    )
    assert isinstance(result09, PlotResult)
    assert isinstance(result10, PlotResult)
    assert isinstance(result11, PlotResult)
    assert isinstance(result12, PlotResult)
    result09 = result09.data
    result10 = result10.data
    result11 = result11.data
    result12 = result12.data
    assert isinstance(result09, dict)
    assert isinstance(result10, dict)
    assert isinstance(result11, dict)
    assert isinstance(result12, dict)
    assert list(result09.keys()) == ["Slice", 0, 1]
    assert list(result10.keys()) == ["Slice", "Total", 0, 1, 2]
    assert list(result11.keys()) == ["Slice", 0, 1]
    assert list(result12.keys()) == ["Slice", "Total", 0, "0/1", "0/2", 1, "1/2", 2]
    assert np.all(result09[0] == np.array([9, 17]) * (2 / 3))  # type: ignore
    assert np.all(result09[1] == np.array([9, 17]) * (2 / 3))  # type: ignore
    assert np.all(result10[0] == np.array([9, 17]) * (2 / 3))  # type: ignore
    assert np.all(result10[1] == np.array([9, 16]) * (2 / 3))  # type: ignore
    assert np.all(result10[2] == np.array([6, 11]) * (2 / 3))  # type: ignore
    assert np.all(result10["Total"] == np.array([15, 27]) * 2 / 3)  # type: ignore
    assert np.all(result09[0] * (3 / 2) == result11[0] * (4 / np.pi))  # type: ignore
    assert np.all(result09[1] * (3 / 2) == result11[1] * (4 / np.pi))  # type: ignore
    assert np.all(result10[0] * (3 / 2) == result12[0] * (4 / np.pi))  # type: ignore
    assert np.all(result10[1] * (3 / 2) == result12[1] * (4 / np.pi))  # type: ignore
    assert np.all(result10[2] * (3 / 2) == result12[2] * (4 / np.pi))  # type: ignore
    assert np.all(result10["Total"] * (3 / 2) == result12["Total"] * (4 / np.pi))  # type: ignore
    assert np.all(result12["0/1"] == np.array([6, 11]) * (np.pi / 4))  # type: ignore
    assert np.all(result12["0/2"] == np.array([3, 6]) * (np.pi / 4))  # type: ignore
    assert np.all(result12["1/2"] == np.array([3, 5]) * (np.pi / 4))  # type: ignore


def test_surf_plot_norm(array_3d_2, array_3d_3) -> None:

    result001 = result01 = surf(array_3d_2, Normalize=True, Perform_Slicewise=True)
    result01 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Normalize=True,
        Perform_Slicewise=True,
    )
    result02 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Perform_Slicewise=True,
    )
    result03 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Normalize=True,
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
    )
    result04 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
    )
    result05 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Normalize=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result06 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result07 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Normalize=True,
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result08 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    assert isinstance(result001, PlotResult)
    assert isinstance(result01, PlotResult)
    assert isinstance(result02, PlotResult)
    assert isinstance(result03, PlotResult)
    assert isinstance(result04, PlotResult)
    assert isinstance(result05, PlotResult)
    assert isinstance(result06, PlotResult)
    assert isinstance(result07, PlotResult)
    assert isinstance(result08, PlotResult)
    result001 = result001.data
    result01 = result01.data
    result02 = result02.data
    result03 = result03.data
    result04 = result04.data
    result05 = result05.data
    result06 = result06.data
    result07 = result07.data
    result08 = result08.data
    assert isinstance(result001, dict)
    assert isinstance(result01, dict)
    assert isinstance(result02, dict)
    assert isinstance(result03, dict)
    assert isinstance(result04, dict)
    assert isinstance(result05, dict)
    assert isinstance(result06, dict)
    assert isinstance(result07, dict)
    assert isinstance(result08, dict)
    assert list(result001.keys()) == list(result01.keys())
    assert list(result01.keys()) == ["Slice", 0, 1]
    assert list(result02.keys()) == ["Slice", "Total", 0, 1, 2]
    assert list(result03.keys()) == ["Slice", 0, 1]
    assert list(result04.keys()) == ["Slice", "Total", 0, "0/1", "0/2", 1, "1/2", 2]
    assert list(result05.keys()) == list(result01.keys())
    assert list(result06.keys()) == list(result02.keys())
    assert list(result07.keys()) == list(result03.keys())
    assert list(result08.keys()) == list(result04.keys())
    assert np.all(result01["Slice"] == result02["Slice"])
    assert np.all(result01["Slice"] == result03["Slice"])
    assert np.all(result01["Slice"] == result04["Slice"])
    assert np.all(result01["Slice"] == result05["Slice"])
    assert np.all(result01["Slice"] == result06["Slice"])
    assert np.all(result01["Slice"] == result07["Slice"])
    assert np.all(result01["Slice"] == result08["Slice"])
    for key in result01.keys():
        assert np.all(result01[key] == result001[key])
    assert np.all(result01[0] == np.array([9, 17]) / np.sum(np.array([9, 17])))  # type: ignore
    assert np.all(result01[1] == np.array([9, 17]) / np.sum(np.array([9, 17])))  # type: ignore
    assert np.all(result02[0] == np.array([9, 17]) / np.sum(np.array([15, 27])))  # type: ignore
    assert np.all(result02[1] == np.array([9, 16]) / np.sum(np.array([15, 27])))  # type: ignore
    assert np.all(result02[2] == np.array([6, 11]) / np.sum(np.array([15, 27])))  # type: ignore
    assert np.all(result02["Total"] == np.array([15, 27]) / np.sum(np.array([15, 27])))  # type: ignore
    assert np.all(result01[0] == result03[0])  # type: ignore
    assert np.all(result01[1] == result03[1])  # type: ignore
    assert np.all(result02[0] == result04[0])  # type: ignore
    assert np.all(result02[1] == result04[1])  # type: ignore
    assert np.all(result02[2] == result04[2])  # type: ignore
    assert np.all(result02["Total"] == result04["Total"])  # type: ignore
    assert np.all(result04["0/1"] == np.array([6, 11]) / np.sum(np.array([15, 27])))  # type: ignore
    assert np.all(result04["0/2"] == np.array([3, 6]) / np.sum(np.array([15, 27])))  # type: ignore
    assert np.all(result04["1/2"] == np.array([3, 5]) / np.sum(np.array([15, 27])))  # type: ignore
    assert np.all(result05[0] == np.array([9, 17]) / np.array([9, 17]))  # type: ignore
    assert np.all(result05[1] == np.array([9, 17]) / np.array([9, 17]))  # type: ignore
    assert np.all(result06[0] == np.array([9, 17]) / np.array([15, 27]))  # type: ignore
    assert np.all(result06[1] == np.array([9, 16]) / np.array([15, 27]))  # type: ignore
    assert np.all(result06[2] == np.array([6, 11]) / np.array([15, 27]))  # type: ignore
    assert np.all(result06["Total"] == np.array([15, 27]) / np.array([15, 27]))  # type: ignore
    assert np.all(result05[0] == result07[0])  # type: ignore
    assert np.all(result05[1] == result07[1])  # type: ignore
    assert np.all(result06[0] == result08[0])  # type: ignore
    assert np.all(result06[1] == result08[1])  # type: ignore
    assert np.all(result06[2] == result08[2])  # type: ignore
    assert np.all(result06["Total"] == result08["Total"])  # type: ignore
    assert np.all(result08["0/1"] == np.array([6, 11]) / np.array([15, 27]))  # type: ignore
    assert np.all(result08["0/2"] == np.array([3, 6]) / np.array([15, 27]))  # type: ignore
    assert np.all(result08["1/2"] == np.array([3, 5]) / np.array([15, 27]))  # type: ignore

    result09 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Phase",
        Perform_Slicewise=True,
    )
    result10 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Phase",
        Perform_Slicewise=True,
    )
    result11 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Phase",
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
    )
    result12 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Phase",
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
    )
    result13 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Phase",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result14 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Phase",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result15 = surf(
        array_3d_2,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Phase",
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result16 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Phase",
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    assert isinstance(result09, PlotResult)
    assert isinstance(result10, PlotResult)
    assert isinstance(result11, PlotResult)
    assert isinstance(result12, PlotResult)
    assert isinstance(result13, PlotResult)
    assert isinstance(result14, PlotResult)
    assert isinstance(result15, PlotResult)
    assert isinstance(result16, PlotResult)
    result09 = result09.data
    result10 = result10.data
    result11 = result11.data
    result12 = result12.data
    result13 = result13.data
    result14 = result14.data
    result15 = result15.data
    result16 = result16.data
    assert isinstance(result09, dict)
    assert isinstance(result10, dict)
    assert isinstance(result11, dict)
    assert isinstance(result12, dict)
    assert isinstance(result13, dict)
    assert isinstance(result14, dict)
    assert isinstance(result15, dict)
    assert isinstance(result16, dict)
    assert list(result09.keys()) == ["Slice", 0, 1]
    assert list(result10.keys()) == ["Slice", 0, 1, 2]
    assert list(result11.keys()) == ["Slice", 0, 1]
    assert list(result12.keys()) == [
        "Slice",
        0,
        1,
        2,
        "0/1 by 0",
        "0/2 by 0",
        "0/1 by 1",
        "1/2 by 1",
        "0/2 by 2",
        "1/2 by 2",
    ]
    assert list(result13.keys()) == list(result09.keys())
    assert list(result14.keys()) == list(result10.keys())
    assert list(result15.keys()) == list(result11.keys())
    assert list(result16.keys()) == list(result12.keys())
    assert np.all(result09[0] == np.array([9, 17]) / np.sum(np.array([9, 17])))  # type: ignore
    assert np.all(result09[1] == np.array([9, 17]) / np.sum(np.array([9, 17])))  # type: ignore
    assert np.all(result10[0] == np.array([9, 17]) / np.sum(np.array([9, 17])))  # type: ignore
    assert np.all(result10[1] == np.array([9, 16]) / np.sum(np.array([9, 16])))  # type: ignore
    assert np.all(result10[2] == np.array([6, 11]) / np.sum(np.array([6, 11])))  # type: ignore
    assert np.all(result09[0] == result11[0])  # type: ignore
    assert np.all(result09[1] == result11[1])  # type: ignore
    assert np.all(result10[0] == result12[0])  # type: ignore
    assert np.all(result10[1] == result12[1])  # type: ignore
    assert np.all(result10[2] == result12[2])  # type: ignore
    assert np.all(result12["0/1 by 0"] == np.array([6, 11]) / np.sum(np.array([9, 17])))  # type: ignore
    assert np.all(result12["0/1 by 1"] == np.array([6, 11]) / np.sum(np.array([9, 16])))  # type: ignore
    assert np.all(result12["0/2 by 0"] == np.array([3, 6]) / np.sum(np.array([9, 17])))  # type: ignore
    assert np.all(result12["0/2 by 2"] == np.array([3, 6]) / np.sum(np.array([6, 11])))  # type: ignore
    assert np.all(result12["1/2 by 1"] == np.array([3, 5]) / np.sum(np.array([9, 16])))  # type: ignore
    assert np.all(result12["1/2 by 2"] == np.array([3, 5]) / np.sum(np.array([6, 11])))  # type: ignore
    assert np.all(result13[0] == np.array([9, 17]) / np.array([9, 17]))  # type: ignore
    assert np.all(result13[1] == np.array([9, 17]) / np.array([9, 17]))  # type: ignore
    assert np.all(result14[0] == np.array([9, 17]) / np.array([9, 17]))  # type: ignore
    assert np.all(result14[1] == np.array([9, 16]) / np.array([9, 16]))  # type: ignore
    assert np.all(result14[2] == np.array([6, 11]) / np.array([6, 11]))  # type: ignore
    assert np.all(result13[0] == result15[0])  # type: ignore
    assert np.all(result13[1] == result15[1])  # type: ignore
    assert np.all(result14[0] == result16[0])  # type: ignore
    assert np.all(result14[1] == result16[1])  # type: ignore
    assert np.all(result14[2] == result16[2])  # type: ignore
    assert np.all(result16["0/1 by 0"] == np.array([6, 11]) / np.array([9, 17]))  # type: ignore
    assert np.all(result16["0/1 by 1"] == np.array([6, 11]) / np.array([9, 16]))  # type: ignore
    assert np.all(result16["0/2 by 0"] == np.array([3, 6]) / np.array([9, 17]))  # type: ignore
    assert np.all(result16["0/2 by 2"] == np.array([3, 6]) / np.array([6, 11]))  # type: ignore
    assert np.all(result16["1/2 by 1"] == np.array([3, 5]) / np.array([9, 16]))  # type: ignore
    assert np.all(result16["1/2 by 2"] == np.array([3, 5]) / np.array([6, 11]))  # type: ignore

    result17 = surf(
        array_3d_2,
        "Perimeter",
        Normalize=True,
        Normalization_Method="Specific - Total",
        Perform_Slicewise=True,
    )
    result18 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Total",
        Perform_Slicewise=True,
    )
    result19 = surf(
        array_3d_2,
        Normalize=True,
        Normalization_Method="Specific - Total",
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
    )
    result20 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Total",
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
    )
    result21 = surf(
        array_3d_2,
        "Perimeter",
        Normalize=True,
        Normalization_Method="Specific - Total",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result22 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Total",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result23 = surf(
        array_3d_2,
        Normalize=True,
        Normalization_Method="Specific - Total",
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result24 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Total",
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    assert isinstance(result17, PlotResult)
    assert isinstance(result18, PlotResult)
    assert isinstance(result19, PlotResult)
    assert isinstance(result20, PlotResult)
    assert isinstance(result21, PlotResult)
    assert isinstance(result22, PlotResult)
    assert isinstance(result23, PlotResult)
    assert isinstance(result24, PlotResult)
    result17 = result17.data
    result18 = result18.data
    result19 = result19.data
    result20 = result20.data
    result21 = result21.data
    result22 = result22.data
    result23 = result23.data
    result24 = result24.data
    assert isinstance(result17, dict)
    assert isinstance(result18, dict)
    assert isinstance(result19, dict)
    assert isinstance(result20, dict)
    assert isinstance(result21, dict)
    assert isinstance(result22, dict)
    assert isinstance(result23, dict)
    assert isinstance(result24, dict)
    assert list(result17.keys()) == ["Slice", 0, 1]
    assert list(result18.keys()) == ["Slice", "Total", 0, 1, 2]
    assert list(result19.keys()) == ["Slice", 0, 1]
    assert list(result20.keys()) == ["Slice", "Total", 0, "0/1", "0/2", 1, "1/2", 2]
    assert list(result21.keys()) == list(result17.keys())
    assert list(result22.keys()) == list(result18.keys())
    assert list(result23.keys()) == list(result19.keys())
    assert list(result24.keys()) == list(result20.keys())
    assert np.all(result17[0] == np.array([9, 17]) * (np.pi / 4) / 32)  # type: ignore
    assert np.all(result17[1] == np.array([9, 17]) * (np.pi / 4) / 32)  # type: ignore
    assert np.all(result18[0] == np.array([9, 17]) / 32)  # type: ignore
    assert np.all(result18[1] == np.array([9, 16]) / 32)  # type: ignore
    assert np.all(result18[2] == np.array([6, 11]) / 32)  # type: ignore
    assert np.all(result18["Total"] == np.array([15, 27]) / 32)  # type: ignore
    assert np.all(result19[0] == np.array([9, 17]) * (2 / 3) / 32)  # type: ignore
    assert np.all(result19[1] == np.array([9, 17]) * (2 / 3) / 32)  # type: ignore
    assert np.all(result20[0] == np.array([9, 17]) / 32)  # type: ignore
    assert np.all(result20[1] == np.array([9, 16]) / 32)  # type: ignore
    assert np.all(result20[2] == np.array([6, 11]) / 32)  # type: ignore
    assert np.all(result20["0/1"] == np.array([6, 11]) / 32)  # type: ignore
    assert np.all(result20["0/2"] == np.array([3, 6]) / 32)  # type: ignore
    assert np.all(result20["1/2"] == np.array([3, 5]) / 32)  # type: ignore
    assert np.all(result21[0] == np.array([9, 17]) * (np.pi / 4) / np.array([16, 16]))  # type: ignore
    assert np.all(result21[1] == np.array([9, 17]) * (np.pi / 4) / np.array([16, 16]))  # type: ignore
    assert np.all(result22[0] == np.array([9, 17]) / np.array([16, 16]))  # type: ignore
    assert np.all(result22[1] == np.array([9, 16]) / np.array([16, 16]))  # type: ignore
    assert np.all(result22[2] == np.array([6, 11]) / np.array([16, 16]))  # type: ignore
    assert np.all(result22["Total"] == np.array([15, 27]) / np.array([16, 16]))  # type: ignore
    assert np.all(result23[0] == np.array([9, 17]) * (2 / 3) / np.array([16, 16]))  # type: ignore
    assert np.all(result23[1] == np.array([9, 17]) * (2 / 3) / np.array([16, 16]))  # type: ignore
    assert np.all(result24[0] == np.array([9, 17]) / np.array([16, 16]))  # type: ignore
    assert np.all(result24[1] == np.array([9, 16]) / np.array([16, 16]))  # type: ignore
    assert np.all(result24[2] == np.array([6, 11]) / np.array([16, 16]))  # type: ignore
    assert np.all(result24["0/1"] == np.array([6, 11]) / np.array([16, 16]))  # type: ignore
    assert np.all(result24["0/2"] == np.array([3, 6]) / np.array([16, 16]))  # type: ignore
    assert np.all(result24["1/2"] == np.array([3, 5]) / np.array([16, 16]))  # type: ignore

    result25 = surf(
        array_3d_2,
        "Perimeter",
        Normalize=True,
        Normalization_Method="Specific - Segmented",
        Perform_Slicewise=True,
    )
    result26 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Segmented",
        Perform_Slicewise=True,
    )
    result27 = surf(
        array_3d_2,
        Normalize=True,
        Normalization_Method="Specific - Segmented",
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
    )
    result28 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Segmented",
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
    )
    result29 = surf(
        array_3d_2,
        "Perimeter",
        Normalize=True,
        Normalization_Method="Specific - Segmented",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result30 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Segmented",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result31 = surf(
        array_3d_2,
        Normalize=True,
        Normalization_Method="Specific - Segmented",
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result32 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Segmented",
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    assert isinstance(result25, PlotResult)
    assert isinstance(result26, PlotResult)
    assert isinstance(result27, PlotResult)
    assert isinstance(result28, PlotResult)
    assert isinstance(result29, PlotResult)
    assert isinstance(result30, PlotResult)
    assert isinstance(result31, PlotResult)
    assert isinstance(result32, PlotResult)
    result25 = result25.data
    result26 = result26.data
    result27 = result27.data
    result28 = result28.data
    result29 = result29.data
    result30 = result30.data
    result31 = result31.data
    result32 = result32.data
    assert isinstance(result25, dict)
    assert isinstance(result26, dict)
    assert isinstance(result27, dict)
    assert isinstance(result28, dict)
    assert isinstance(result29, dict)
    assert isinstance(result30, dict)
    assert isinstance(result31, dict)
    assert isinstance(result32, dict)
    assert list(result25.keys()) == ["Slice", 0, 1]
    assert list(result26.keys()) == ["Slice", "Total", 0, 1, 2]
    assert list(result27.keys()) == ["Slice", 0, 1]
    assert list(result28.keys()) == ["Slice", "Total", 0, "0/1", "0/2", 1, "1/2", 2]
    assert list(result29.keys()) == list(result25.keys())
    assert list(result30.keys()) == list(result26.keys())
    assert list(result31.keys()) == list(result27.keys())
    assert list(result32.keys()) == list(result28.keys())
    assert list(result25.keys()) == ["Slice", 0, 1]
    assert list(result26.keys()) == ["Slice", "Total", 0, 1, 2]
    assert list(result27.keys()) == ["Slice", 0, 1]
    assert list(result28.keys()) == ["Slice", "Total", 0, "0/1", "0/2", 1, "1/2", 2]
    assert list(result29.keys()) == list(result25.keys())
    assert list(result30.keys()) == list(result26.keys())
    assert list(result31.keys()) == list(result27.keys())
    assert list(result32.keys()) == list(result28.keys())
    assert np.all(
        result25[0] == np.array([9, 17]) * (np.pi / 4) / np.sum(np.array([12, 10]))  # type: ignore
    )
    assert np.all(
        result25[1] == np.array([9, 17]) * (np.pi / 4) / np.sum(np.array([12, 10]))  # type: ignore
    )
    assert np.all(result26[0] == np.array([9, 17]) / np.sum(np.array([12, 10])))  # type: ignore
    assert np.all(result26[1] == np.array([9, 16]) / np.sum(np.array([12, 10])))  # type: ignore
    assert np.all(result26[2] == np.array([6, 11]) / np.sum(np.array([12, 10])))  # type: ignore
    assert np.all(result26["Total"] == np.array([15, 27]) / np.sum(np.array([12, 10])))  # type: ignore
    assert np.all(
        result27[0] == np.array([9, 17]) * (2 / 3) / np.sum(np.array([12, 10]))  # type: ignore
    )
    assert np.all(
        result27[1] == np.array([9, 17]) * (2 / 3) / np.sum(np.array([12, 10]))  # type: ignore
    )
    assert np.all(result28[0] == np.array([9, 17]) / np.sum(np.array([12, 10])))  # type: ignore
    assert np.all(result28[1] == np.array([9, 16]) / np.sum(np.array([12, 10])))  # type: ignore
    assert np.all(result28[2] == np.array([6, 11]) / np.sum(np.array([12, 10])))  # type: ignore
    assert np.all(result28["0/1"] == np.array([6, 11]) / np.sum(np.array([12, 10])))  # type: ignore
    assert np.all(result28["0/2"] == np.array([3, 6]) / np.sum(np.array([12, 10])))  # type: ignore
    assert np.all(result28["1/2"] == np.array([3, 5]) / np.sum(np.array([12, 10])))  # type: ignore
    assert np.all(result29[0] == np.array([9, 17]) * (np.pi / 4) / np.array([12, 10]))  # type: ignore
    assert np.all(result29[1] == np.array([9, 17]) * (np.pi / 4) / np.array([12, 10]))  # type: ignore
    assert np.all(result30[0] == np.array([9, 17]) / np.array([12, 10]))  # type: ignore
    assert np.all(result30[1] == np.array([9, 16]) / np.array([12, 10]))  # type: ignore
    assert np.all(result30[2] == np.array([6, 11]) / np.array([12, 10]))  # type: ignore
    assert np.all(result30["Total"] == np.array([15, 27]) / np.array([12, 10]))  # type: ignore
    assert np.all(result31[0] == np.array([9, 17]) * (2 / 3) / np.array([12, 10]))  # type: ignore
    assert np.all(result31[1] == np.array([9, 17]) * (2 / 3) / np.array([12, 10]))  # type: ignore
    assert np.all(result32[0] == np.array([9, 17]) / np.array([12, 10]))  # type: ignore
    assert np.all(result32[1] == np.array([9, 16]) / np.array([12, 10]))  # type: ignore
    assert np.all(result32[2] == np.array([6, 11]) / np.array([12, 10]))  # type: ignore
    assert np.all(result32["0/1"] == np.array([6, 11]) / np.array([12, 10]))  # type: ignore
    assert np.all(result32["0/2"] == np.array([3, 6]) / np.array([12, 10]))  # type: ignore
    assert np.all(result32["1/2"] == np.array([3, 5]) / np.array([12, 10]))  # type: ignore

    result33 = surf(
        array_3d_2,
        "Perimeter",
        Normalize=True,
        Normalization_Method="Specific - Phase",
        Perform_Slicewise=True,
    )
    result34 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Phase",
        Perform_Slicewise=True,
    )
    result35 = surf(
        array_3d_2,
        Normalize=True,
        Normalization_Method="Specific - Phase",
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
    )
    result36 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Phase",
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
    )
    result37 = surf(
        array_3d_2,
        "Perimeter",
        Normalize=True,
        Normalization_Method="Specific - Phase",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result38 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Phase",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result39 = surf(
        array_3d_2,
        Normalize=True,
        Normalization_Method="Specific - Phase",
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result40 = surf(
        array_3d_3,
        Correct_Overestimation=False,
        Normalize=True,
        Normalization_Method="Specific - Phase",
        Include_Phase_Breakdown=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    assert isinstance(result33, PlotResult)
    assert isinstance(result34, PlotResult)
    assert isinstance(result35, PlotResult)
    assert isinstance(result36, PlotResult)
    assert isinstance(result37, PlotResult)
    assert isinstance(result38, PlotResult)
    assert isinstance(result39, PlotResult)
    assert isinstance(result40, PlotResult)
    result33 = result33.data
    result34 = result34.data
    result35 = result35.data
    result36 = result36.data
    result37 = result37.data
    result38 = result38.data
    result39 = result39.data
    result40 = result40.data
    assert isinstance(result33, dict)
    assert isinstance(result34, dict)
    assert isinstance(result35, dict)
    assert isinstance(result36, dict)
    assert isinstance(result37, dict)
    assert isinstance(result38, dict)
    assert isinstance(result39, dict)
    assert isinstance(result40, dict)
    assert list(result33.keys()) == ["Slice", 0, 1]
    assert list(result34.keys()) == ["Slice", 0, 1, 2]
    assert list(result35.keys()) == ["Slice", 0, 1]
    assert list(result36.keys()) == [
        "Slice",
        0,
        1,
        2,
        "0/1 by 0",
        "0/2 by 0",
        "0/1 by 1",
        "1/2 by 1",
        "0/2 by 2",
        "1/2 by 2",
    ]
    assert list(result37.keys()) == list(result33.keys())
    assert list(result38.keys()) == list(result34.keys())
    assert list(result39.keys()) == list(result35.keys())
    assert list(result40.keys()) == list(result36.keys())
    assert np.all(
        result33[0] == np.array([9, 17]) * (np.pi / 4) / np.sum(np.array([4, 6]))  # type: ignore
    )
    assert np.all(
        result33[1] == np.array([9, 17]) * (np.pi / 4) / np.sum(np.array([12, 10]))  # type: ignore
    )
    assert np.all(result34[0] == np.array([9, 17]) / np.sum(np.array([4, 6])))  # type: ignore
    assert np.all(result34[1] == np.array([9, 16]) / np.sum(np.array([7, 6])))  # type: ignore
    assert np.all(result34[2] == np.array([6, 11]) / np.sum(np.array([5, 4])))  # type: ignore
    assert np.all(
        np.round(result33[0] * (4 / np.pi), 5) == np.round(result35[0] * (3 / 2), 5)  # type: ignore
    )
    assert np.all(
        np.round(result33[1] * (4 / np.pi), 5) == np.round(result35[1] * (3 / 2), 5)  # type: ignore
    )
    assert np.all(result34[0] == result36[0])  # type: ignore
    assert np.all(result34[1] == result36[1])  # type: ignore
    assert np.all(result34[2] == result36[2])  # type: ignore
    assert np.all(result36["0/1 by 0"] == np.array([6, 11]) / np.sum(np.array([4, 6])))  # type: ignore
    assert np.all(result36["0/1 by 1"] == np.array([6, 11]) / np.sum(np.array([7, 6])))  # type: ignore
    assert np.all(result36["0/2 by 0"] == np.array([3, 6]) / np.sum(np.array([4, 6])))  # type: ignore
    assert np.all(result36["0/2 by 2"] == np.array([3, 6]) / np.sum(np.array([5, 4])))  # type: ignore
    assert np.all(result36["1/2 by 1"] == np.array([3, 5]) / np.sum(np.array([7, 6])))  # type: ignore
    assert np.all(result36["1/2 by 2"] == np.array([3, 5]) / np.sum(np.array([5, 4])))  # type: ignore

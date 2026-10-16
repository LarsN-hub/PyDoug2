"""
Tests for PyDoug's analyze.measure module.
"""

import numpy as np
from skimage.measure import label

from pydoug_core.analyze import measure
from pydoug_core.results import PlotResult, ValueResult


def test_bulk_scalar(sample_masks) -> None:

    result00 = measure.bulk(sample_masks["2dgray"], Units="um")
    result01 = measure.bulk(sample_masks["2dgray"], Pixel_Size=2)
    result02 = measure.bulk(sample_masks["2dgray"], "Area", 2)
    result03 = measure.bulk(sample_masks["2dgray"], "Length", 2)
    result04 = measure.bulk(sample_masks["3dgray"], Pixel_Size=2)
    result05 = measure.bulk(sample_masks["3dgray"], "Area", 2)
    result06 = measure.bulk(sample_masks["3dgray"], "Length", 2)

    assert isinstance(result00, ValueResult)
    assert isinstance(result01, ValueResult)
    assert isinstance(result02, ValueResult)
    assert isinstance(result03, ValueResult)
    assert isinstance(result04, ValueResult)
    assert isinstance(result05, ValueResult)
    assert isinstance(result06, ValueResult)
    result00 = result00.values
    result01 = result01.values
    result02 = result02.values
    result03 = result03.values
    result04 = result04.values
    result05 = result05.values
    result06 = result06.values
    assert isinstance(result00, dict)
    assert isinstance(result01, dict)
    assert isinstance(result02, dict)
    assert isinstance(result03, dict)
    assert isinstance(result04, dict)
    assert isinstance(result05, dict)
    assert isinstance(result06, dict)
    assert len(result01) == len(result02) == len(result03) == 2
    assert len(result04) == len(result05) == len(result06) == 2
    for key in list(result01.keys()):
        assert result01[key] >= 0
        assert result02[key] >= 0
        assert result03[key] >= 0
        assert result01[key] == result02[key] * 2
        assert result01[key] == result03[key] * 4
    for key in list(result04.keys()):
        assert result04[key] >= 0
        assert result05[key] >= 0
        assert result06[key] >= 0
        assert result04[key] == result05[key] * 2
        assert result04[key] == result06[key] * 4

    result07 = measure.bulk(sample_masks["2dgray"], Pixel_Size=2, Normalize=True)
    result08 = measure.bulk(sample_masks["2dgray"], "Area", 2, Normalize=True)
    result09 = measure.bulk(sample_masks["2dgray"], "Length", 2, Normalize=True)
    result10 = measure.bulk(sample_masks["3dgray"], Pixel_Size=2, Normalize=True)
    result11 = measure.bulk(sample_masks["3dgray"], "Area", 2, Normalize=True)
    result12 = measure.bulk(sample_masks["3dgray"], "Length", 2, Normalize=True)

    assert isinstance(result07, ValueResult)
    assert isinstance(result08, ValueResult)
    assert isinstance(result09, ValueResult)
    assert isinstance(result10, ValueResult)
    assert isinstance(result11, ValueResult)
    assert isinstance(result12, ValueResult)
    result07 = result07.values
    result08 = result08.values
    result09 = result09.values
    result10 = result10.values
    result11 = result11.values
    result12 = result12.values
    assert isinstance(result07, dict)
    assert isinstance(result08, dict)
    assert isinstance(result09, dict)
    assert isinstance(result10, dict)
    assert isinstance(result11, dict)
    assert isinstance(result12, dict)
    assert len(result07) == len(result08) == len(result09) == 2
    assert len(result10) == len(result11) == len(result12) == 2
    for key in list(result07.keys()):
        assert result07[key] >= 0
        assert result08[key] >= 0
        assert result09[key] >= 0
        assert result07[key] <= 1
        assert result08[key] <= 1
        assert result09[key] <= 1
        assert result07[key] == result08[key] == result09[key]
    for key in list(result10.keys()):
        assert result09[key] >= 0
        assert result10[key] >= 0
        assert result11[key] >= 0
        assert result09[key] <= 1
        assert result10[key] <= 1
        assert result11[key] <= 1
        assert result10[key] == result11[key] == result12[key]

    result13 = measure.bulk(
        sample_masks["2dgray"],
        Pixel_Size=2,
        Normalize=True,
        Normalization_Method="Segmented",
    )
    result14 = measure.bulk(
        sample_masks["2dgray"],
        "Area",
        2,
        Normalize=True,
        Normalization_Method="Segmented",
    )
    result15 = measure.bulk(
        sample_masks["2dgray"],
        "Length",
        2,
        Normalize=True,
        Normalization_Method="Segmented",
    )
    result16 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Normalize=True,
        Normalization_Method="Segmented",
    )
    result17 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Normalize=True,
        Normalization_Method="Segmented",
    )
    result18 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Normalize=True,
        Normalization_Method="Segmented",
    )

    assert isinstance(result13, ValueResult)
    assert isinstance(result14, ValueResult)
    assert isinstance(result15, ValueResult)
    assert isinstance(result16, ValueResult)
    assert isinstance(result17, ValueResult)
    assert isinstance(result18, ValueResult)
    result13 = result13.values
    result14 = result14.values
    result15 = result15.values
    result16 = result16.values
    result17 = result17.values
    result18 = result18.values
    assert isinstance(result13, dict)
    assert isinstance(result14, dict)
    assert isinstance(result15, dict)
    assert isinstance(result16, dict)
    assert isinstance(result17, dict)
    assert isinstance(result18, dict)
    assert len(result13) == len(result14) == len(result15) == 2
    assert len(result16) == len(result17) == len(result18) == 2
    for key in list(result13.keys()):
        if key != 0:
            assert result13[key] <= 1
            assert result14[key] <= 1
            assert result15[key] <= 1
        assert result13[key] >= 0
        assert result14[key] >= 0
        assert result15[key] >= 0
        assert result13[key] == result14[key] == result15[key]
    for key in list(result16.keys()):
        if key != 0:
            assert result16[key] <= 1
            assert result17[key] <= 1
            assert result18[key] <= 1
        assert result16[key] >= 0
        assert result17[key] >= 0
        assert result18[key] >= 0
        assert result16[key] == result17[key] == result18[key]

    result19 = measure.bulk(
        sample_masks["2dgray"],
        Pixel_Size=2,
        Normalize=True,
        Normalization_Method="Phase",
    )
    result20 = measure.bulk(
        sample_masks["2dgray"],
        "Area",
        2,
        Normalize=True,
        Normalization_Method="Phase",
    )
    result21 = measure.bulk(
        sample_masks["2dgray"],
        "Length",
        2,
        Normalize=True,
        Normalization_Method="Phase",
    )
    result22 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Normalize=True,
        Normalization_Method="Phase",
    )
    result23 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Normalize=True,
        Normalization_Method="Phase",
    )
    result24 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Normalize=True,
        Normalization_Method="Phase",
    )

    assert isinstance(result19, ValueResult)
    assert isinstance(result20, ValueResult)
    assert isinstance(result21, ValueResult)
    assert isinstance(result22, ValueResult)
    assert isinstance(result23, ValueResult)
    assert isinstance(result24, ValueResult)
    result19 = result19.values
    result20 = result20.values
    result21 = result21.values
    result22 = result22.values
    result23 = result23.values
    result24 = result24.values
    assert isinstance(result19, dict)
    assert isinstance(result20, dict)
    assert isinstance(result21, dict)
    assert isinstance(result22, dict)
    assert isinstance(result23, dict)
    assert isinstance(result24, dict)
    assert len(result19) == len(result20) == len(result21) == 2
    assert len(result22) == len(result23) == len(result24) == 2
    for key in list(result19.keys()):
        assert result19[key] == result20[key] == result21[key] == 1
    for key in list(result22.keys()):
        assert result22[key] == result23[key] == result24[key] == 1

    mask_2d: np.ndarray = np.zeros(sample_masks["2dgray"].shape, np.bool)
    mask_3d: np.ndarray = np.zeros(sample_masks["3dgray"].shape, np.bool)
    mask_2d[3:7, 3:7] = 1
    mask_3d[:, 3:7, 3:7] = 1
    result25 = measure.bulk(
        sample_masks["2dgray"],
        Pixel_Size=2,
        Apply_Mask=True,
        Mask=mask_2d,
    )
    result26 = measure.bulk(
        sample_masks["2dgray"],
        "Area",
        2,
        Apply_Mask=True,
        Mask=mask_2d,
    )
    result27 = measure.bulk(
        sample_masks["2dgray"],
        "Length",
        2,
        Apply_Mask=True,
        Mask=mask_2d,
    )
    result28 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Apply_Mask=True,
        Mask=mask_3d,
    )
    result29 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Apply_Mask=True,
        Mask=mask_3d,
    )
    result30 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Apply_Mask=True,
        Mask=mask_3d,
    )

    assert isinstance(result25, ValueResult)
    assert isinstance(result26, ValueResult)
    assert isinstance(result27, ValueResult)
    assert isinstance(result28, ValueResult)
    assert isinstance(result29, ValueResult)
    assert isinstance(result30, ValueResult)
    result25 = result25.values
    result26 = result26.values
    result27 = result27.values
    result28 = result28.values
    result29 = result29.values
    result30 = result30.values
    assert isinstance(result25, dict)
    assert isinstance(result26, dict)
    assert isinstance(result27, dict)
    assert isinstance(result28, dict)
    assert isinstance(result29, dict)
    assert isinstance(result30, dict)
    assert len(result25) == len(result26) == len(result27) == 2
    assert len(result28) == len(result29) == len(result30) == 2
    for key in list(result25.keys()):
        assert result25[key] >= 0
        assert result26[key] >= 0
        assert result27[key] >= 0
        assert result25[key] == result26[key] * 2
        assert result25[key] == result27[key] * 4
    for key in list(result28.keys()):
        assert result28[key] >= 0
        assert result29[key] >= 0
        assert result30[key] >= 0
        assert result28[key] == result29[key] * 2
        assert result28[key] == result30[key] * 4


def test_bulk_scalar_empty_masks(sample_masks) -> None:

    empty_mask: np.ndarray = np.zeros(sample_masks["2dgray"].shape)
    result01 = measure.bulk(
        sample_masks["2dgray"], Pixel_Size=2, Apply_Mask=True, Mask=empty_mask
    )
    result02 = measure.bulk(
        sample_masks["2dgray"], "Area", 2, Apply_Mask=True, Mask=empty_mask
    )
    result03 = measure.bulk(
        sample_masks["2dgray"], "Length", 2, Apply_Mask=True, Mask=empty_mask
    )
    result04 = measure.bulk(
        sample_masks["3dgray"], Pixel_Size=2, Apply_Mask=True, Mask=empty_mask
    )
    result05 = measure.bulk(
        sample_masks["3dgray"], "Area", 2, Apply_Mask=True, Mask=empty_mask
    )
    result06 = measure.bulk(
        sample_masks["3dgray"], "Length", 2, Apply_Mask=True, Mask=empty_mask
    )

    assert isinstance(result01, ValueResult)
    assert isinstance(result02, ValueResult)
    assert isinstance(result03, ValueResult)
    assert isinstance(result04, ValueResult)
    assert isinstance(result05, ValueResult)
    assert isinstance(result06, ValueResult)
    result01 = result01.values
    result02 = result02.values
    result03 = result03.values
    result04 = result04.values
    result05 = result05.values
    result06 = result06.values
    assert isinstance(result01, dict)
    assert isinstance(result02, dict)
    assert isinstance(result03, dict)
    assert isinstance(result04, dict)
    assert isinstance(result05, dict)
    assert isinstance(result06, dict)
    assert len(result01) == len(result02) == len(result03) == 0
    assert len(result04) == len(result05) == len(result06) == 0

    result07 = measure.bulk(
        sample_masks["2dgray"],
        Pixel_Size=2,
        Normalize=True,
        Apply_Mask=True,
        Mask=empty_mask,
    )
    result08 = measure.bulk(
        sample_masks["2dgray"],
        "Area",
        2,
        Normalize=True,
        Apply_Mask=True,
        Mask=empty_mask,
    )
    result09 = measure.bulk(
        sample_masks["2dgray"],
        "Length",
        2,
        Normalize=True,
        Apply_Mask=True,
        Mask=empty_mask,
    )
    result10 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Normalize=True,
        Apply_Mask=True,
        Mask=empty_mask,
    )
    result11 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Normalize=True,
        Apply_Mask=True,
        Mask=empty_mask,
    )
    result12 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Normalize=True,
        Apply_Mask=True,
        Mask=empty_mask,
    )

    assert isinstance(result07, ValueResult)
    assert isinstance(result08, ValueResult)
    assert isinstance(result09, ValueResult)
    assert isinstance(result10, ValueResult)
    assert isinstance(result11, ValueResult)
    assert isinstance(result12, ValueResult)
    result07 = result07.values
    result08 = result08.values
    result09 = result09.values
    result10 = result10.values
    result11 = result11.values
    result12 = result12.values
    assert isinstance(result07, dict)
    assert isinstance(result08, dict)
    assert isinstance(result09, dict)
    assert isinstance(result10, dict)
    assert isinstance(result11, dict)
    assert isinstance(result12, dict)
    assert len(result07) == len(result08) == len(result09) == 0
    assert len(result10) == len(result11) == len(result12) == 0

    result13 = measure.bulk(
        sample_masks["2dgray"],
        Pixel_Size=2,
        Normalize=True,
        Normalization_Method="Segmented",
        Apply_Mask=True,
        Mask=empty_mask,
    )
    result14 = measure.bulk(
        sample_masks["2dgray"],
        "Area",
        2,
        Normalize=True,
        Normalization_Method="Segmented",
        Apply_Mask=True,
        Mask=empty_mask,
    )
    result15 = measure.bulk(
        sample_masks["2dgray"],
        "Length",
        2,
        Normalize=True,
        Normalization_Method="Segmented",
        Apply_Mask=True,
        Mask=empty_mask,
    )
    result16 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Normalize=True,
        Normalization_Method="Segmented",
        Apply_Mask=True,
        Mask=empty_mask,
    )
    result17 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Normalize=True,
        Normalization_Method="Segmented",
        Apply_Mask=True,
        Mask=empty_mask,
    )
    result18 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Normalize=True,
        Normalization_Method="Segmented",
        Apply_Mask=True,
        Mask=empty_mask,
    )

    assert isinstance(result13, ValueResult)
    assert isinstance(result14, ValueResult)
    assert isinstance(result15, ValueResult)
    assert isinstance(result16, ValueResult)
    assert isinstance(result17, ValueResult)
    assert isinstance(result18, ValueResult)
    result13 = result13.values
    result14 = result14.values
    result15 = result15.values
    result16 = result16.values
    result17 = result17.values
    result18 = result18.values
    assert isinstance(result13, dict)
    assert isinstance(result14, dict)
    assert isinstance(result15, dict)
    assert isinstance(result16, dict)
    assert isinstance(result17, dict)
    assert isinstance(result18, dict)
    assert len(result13) == len(result14) == len(result15) == 0
    assert len(result16) == len(result17) == len(result18) == 0

    result19 = measure.bulk(
        sample_masks["2dgray"],
        Pixel_Size=2,
        Normalize=True,
        Normalization_Method="Phase",
        Apply_Mask=True,
        Mask=empty_mask,
    )
    result20 = measure.bulk(
        sample_masks["2dgray"],
        "Area",
        2,
        Normalize=True,
        Normalization_Method="Phase",
        Apply_Mask=True,
        Mask=empty_mask,
    )
    result21 = measure.bulk(
        sample_masks["2dgray"],
        "Length",
        2,
        Normalize=True,
        Normalization_Method="Phase",
        Apply_Mask=True,
        Mask=empty_mask,
    )
    result22 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Normalize=True,
        Normalization_Method="Phase",
        Apply_Mask=True,
        Mask=empty_mask,
    )
    result23 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Normalize=True,
        Normalization_Method="Phase",
        Apply_Mask=True,
        Mask=empty_mask,
    )
    result24 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Normalize=True,
        Normalization_Method="Phase",
        Apply_Mask=True,
        Mask=empty_mask,
    )

    assert isinstance(result19, ValueResult)
    assert isinstance(result20, ValueResult)
    assert isinstance(result21, ValueResult)
    assert isinstance(result22, ValueResult)
    assert isinstance(result23, ValueResult)
    assert isinstance(result24, ValueResult)
    result19 = result19.values
    result20 = result20.values
    result21 = result21.values
    result22 = result22.values
    result23 = result23.values
    result24 = result24.values
    assert isinstance(result19, dict)
    assert isinstance(result20, dict)
    assert isinstance(result21, dict)
    assert isinstance(result22, dict)
    assert isinstance(result23, dict)
    assert isinstance(result24, dict)
    assert len(result19) == len(result20) == len(result21) == 0
    assert len(result22) == len(result23) == len(result24) == 0


def test_bulk_plot(sample_masks) -> None:

    result00 = measure.bulk(sample_masks["2dgray"], Perform_Slicewise=True)
    result01 = measure.bulk(
        sample_masks["3dgray"], Pixel_Size=2, Perform_Slicewise=True
    )
    result02 = measure.bulk(sample_masks["3dgray"], "Area", 2, Perform_Slicewise=True)
    result03 = measure.bulk(sample_masks["3dgray"], "Length", 2, Perform_Slicewise=True)

    assert isinstance(result00, ValueResult)
    assert isinstance(result01, PlotResult)
    assert isinstance(result02, PlotResult)
    assert isinstance(result03, PlotResult)
    for index, key in enumerate(list(result01.data.keys())):
        assert isinstance(result01.data[key], np.ndarray)
        assert isinstance(result02.data[key], np.ndarray)
        assert isinstance(result03.data[key], np.ndarray)
        if index == 0:
            length: int = result01.data[key].size  # type: ignore
        assert result01.data[key].size == length  # type: ignore
        assert result02.data[key].size == length  # type: ignore
        assert result03.data[key].size == length  # type: ignore
        if key != "Slice":
            assert np.all(result01.data[key] >= 0)  # type: ignore
            assert np.all(result02.data[key] >= 0)  # type: ignore
            assert np.all(result03.data[key] >= 0)  # type: ignore
            assert np.all(result01.data[key] == result02.data[key] * 2)  # type: ignore
            assert np.all(result01.data[key] == result03.data[key] * 4)  # type: ignore
        else:
            assert np.all(
                result01.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore
            assert np.all(
                result02.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore
            assert np.all(
                result03.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore

    result04 = measure.bulk(
        sample_masks["3dgray"], Pixel_Size=2, Normalize=True, Perform_Slicewise=True
    )
    result05 = measure.bulk(
        sample_masks["3dgray"], "Area", 2, Normalize=True, Perform_Slicewise=True
    )
    result06 = measure.bulk(
        sample_masks["3dgray"], "Length", 2, Normalize=True, Perform_Slicewise=True
    )

    assert isinstance(result04, PlotResult)
    assert isinstance(result05, PlotResult)
    assert isinstance(result06, PlotResult)
    for index, key in enumerate(list(result04.data.keys())):
        assert isinstance(result04.data[key], np.ndarray)
        assert isinstance(result05.data[key], np.ndarray)
        assert isinstance(result06.data[key], np.ndarray)
        if index == 0:
            length: int = result04.data[key].size  # type: ignore
        assert result04.data[key].size == length  # type: ignore
        assert result05.data[key].size == length  # type: ignore
        assert result06.data[key].size == length  # type: ignore
        if key != "Slice":
            assert np.all(result04.data[key] >= 0)  # type: ignore
            assert np.all(result05.data[key] >= 0)  # type: ignore
            assert np.all(result06.data[key] >= 0)  # type: ignore
            assert np.all(result04.data[key] <= 1)  # type: ignore
            assert np.all(result05.data[key] <= 1)  # type: ignore
            assert np.all(result06.data[key] <= 1)  # type: ignore
            assert np.all(result04.data[key] == result05.data[key])  # type: ignore
            assert np.all(result04.data[key] == result06.data[key])  # type: ignore
        else:
            assert np.all(
                result04.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore
            assert np.all(
                result05.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore
            assert np.all(
                result06.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore

    result07 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Normalize=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result08 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Normalize=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result09 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Normalize=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )

    assert isinstance(result07, PlotResult)
    assert isinstance(result08, PlotResult)
    assert isinstance(result09, PlotResult)
    for index, key in enumerate(list(result07.data.keys())):
        assert isinstance(result07.data[key], np.ndarray)
        assert isinstance(result08.data[key], np.ndarray)
        assert isinstance(result09.data[key], np.ndarray)
        if index == 0:
            length: int = result07.data[key].size  # type: ignore
        assert result07.data[key].size == length  # type: ignore
        assert result08.data[key].size == length  # type: ignore
        assert result09.data[key].size == length  # type: ignore
        if key != "Slice":
            assert np.all(result07.data[key] >= 0)  # type: ignore
            assert np.all(result08.data[key] >= 0)  # type: ignore
            assert np.all(result09.data[key] >= 0)  # type: ignore
            assert np.all(result07.data[key] <= 1)  # type: ignore
            assert np.all(result08.data[key] <= 1)  # type: ignore
            assert np.all(result09.data[key] <= 1)  # type: ignore
            assert np.all(result07.data[key] >= result04.data[key])  # type: ignore
            assert np.all(result08.data[key] >= result05.data[key])  # type: ignore
            assert np.all(result09.data[key] >= result06.data[key])  # type: ignore
            assert np.all(result07.data[key] == result08.data[key])  # type: ignore
            assert np.all(result07.data[key] == result09.data[key])  # type: ignore
        else:
            assert np.all(
                result07.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore
            assert np.all(
                result08.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore
            assert np.all(
                result09.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore

    result10 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Normalize=True,
        Normalization_Method="Segmented",
        Perform_Slicewise=True,
    )
    result11 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Normalize=True,
        Normalization_Method="Segmented",
        Perform_Slicewise=True,
    )
    result12 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Normalize=True,
        Normalization_Method="Segmented",
        Perform_Slicewise=True,
    )

    assert isinstance(result10, PlotResult)
    assert isinstance(result11, PlotResult)
    assert isinstance(result12, PlotResult)
    for index, key in enumerate(list(result10.data.keys())):
        assert isinstance(result10.data[key], np.ndarray)
        assert isinstance(result11.data[key], np.ndarray)
        assert isinstance(result12.data[key], np.ndarray)
        if index == 0:
            length: int = result10.data[key].size  # type: ignore
        assert result10.data[key].size == length  # type: ignore
        assert result11.data[key].size == length  # type: ignore
        assert result12.data[key].size == length  # type: ignore
        if key != "Slice":
            assert np.all(result10.data[key] >= 0)  # type: ignore
            assert np.all(result11.data[key] >= 0)  # type: ignore
            assert np.all(result12.data[key] >= 0)  # type: ignore
            if key != 0:
                assert np.all(result10.data[key] <= 1)  # type: ignore
                assert np.all(result11.data[key] <= 1)  # type: ignore
                assert np.all(result12.data[key] <= 1)  # type: ignore
            assert np.all(result10.data[key] == result11.data[key])  # type: ignore
            assert np.all(result10.data[key] == result12.data[key])  # type: ignore
        else:
            assert np.all(
                result10.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore
            assert np.all(
                result11.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore
            assert np.all(
                result12.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore

    result13 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Normalize=True,
        Normalization_Method="Segmented",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result14 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Normalize=True,
        Normalization_Method="Segmented",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result15 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Normalize=True,
        Normalization_Method="Segmented",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )

    assert isinstance(result13, PlotResult)
    assert isinstance(result14, PlotResult)
    assert isinstance(result15, PlotResult)
    for index, key in enumerate(list(result13.data.keys())):
        assert isinstance(result13.data[key], np.ndarray)
        assert isinstance(result14.data[key], np.ndarray)
        assert isinstance(result15.data[key], np.ndarray)
        if index == 0:
            length: int = result13.data[key].size  # type: ignore
        assert result13.data[key].size == length  # type: ignore
        assert result14.data[key].size == length  # type: ignore
        assert result15.data[key].size == length  # type: ignore
        if key != "Slice":
            assert np.all(result13.data[key] >= 0)  # type: ignore
            assert np.all(result14.data[key] >= 0)  # type: ignore
            assert np.all(result15.data[key] >= 0)  # type: ignore
            assert np.all(result13.data[key] >= result10.data[key])  # type: ignore
            assert np.all(result14.data[key] >= result11.data[key])  # type: ignore
            assert np.all(result15.data[key] >= result12.data[key])  # type: ignore
            if key != 0:
                assert np.all(result13.data[key] == 1)  # type: ignore
                assert np.all(result14.data[key] == 1)  # type: ignore
                assert np.all(result15.data[key] == 1)  # type: ignore
        else:
            assert np.all(
                result13.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore
            assert np.all(
                result14.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore
            assert np.all(
                result15.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore

    result16 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Normalize=True,
        Normalization_Method="Phase",
        Perform_Slicewise=True,
    )
    result17 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Normalize=True,
        Normalization_Method="Phase",
        Perform_Slicewise=True,
    )
    result18 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Normalize=True,
        Normalization_Method="Phase",
        Perform_Slicewise=True,
    )

    assert isinstance(result16, PlotResult)
    assert isinstance(result17, PlotResult)
    assert isinstance(result18, PlotResult)
    for index, key in enumerate(list(result16.data.keys())):
        assert isinstance(result16.data[key], np.ndarray)
        assert isinstance(result17.data[key], np.ndarray)
        assert isinstance(result18.data[key], np.ndarray)
        if index == 0:
            length: int = result16.data[key].size  # type: ignore
        assert result16.data[key].size == length  # type: ignore
        assert result17.data[key].size == length  # type: ignore
        assert result18.data[key].size == length  # type: ignore
        if key != "Slice":
            assert np.all(result16.data[key] >= 0)  # type: ignore
            assert np.all(result17.data[key] >= 0)  # type: ignore
            assert np.all(result18.data[key] >= 0)  # type: ignore
            assert np.all(result16.data[key] <= 1)  # type: ignore
            assert np.all(result17.data[key] <= 1)  # type: ignore
            assert np.all(result18.data[key] <= 1)  # type: ignore
            assert np.all(result16.data[key] == result17.data[key])  # type: ignore
            assert np.all(result16.data[key] == result18.data[key])  # type: ignore
        else:
            assert np.all(
                result16.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore
            assert np.all(
                result17.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore
            assert np.all(
                result18.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore

    result19 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Normalize=True,
        Normalization_Method="Phase",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result20 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Normalize=True,
        Normalization_Method="Phase",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result21 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Normalize=True,
        Normalization_Method="Phase",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )

    assert isinstance(result19, PlotResult)
    assert isinstance(result20, PlotResult)
    assert isinstance(result21, PlotResult)
    for index, key in enumerate(list(result16.data.keys())):
        assert isinstance(result19.data[key], np.ndarray)
        assert isinstance(result20.data[key], np.ndarray)
        assert isinstance(result21.data[key], np.ndarray)
        if index == 0:
            length: int = result19.data[key].size  # type: ignore
        assert result19.data[key].size == length  # type: ignore
        assert result20.data[key].size == length  # type: ignore
        assert result21.data[key].size == length  # type: ignore
        if key != "Slice":
            assert np.all(result19.data[key] == 1)  # type: ignore
            assert np.all(result20.data[key] == 1)  # type: ignore
            assert np.all(result21.data[key] == 1)  # type: ignore
        else:
            assert np.all(
                result19.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore
            assert np.all(
                result20.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore
            assert np.all(
                result21.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore

    test_mask: np.ndarray = np.zeros(sample_masks["3dgray"].shape, np.bool)
    test_mask[:, 3:7, 3:7] = 1
    result22 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Apply_Mask=True,
        Mask=test_mask,
        Perform_Slicewise=True,
    )
    result23 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Apply_Mask=True,
        Mask=test_mask,
        Perform_Slicewise=True,
    )
    result24 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Apply_Mask=True,
        Mask=test_mask,
        Perform_Slicewise=True,
    )

    assert isinstance(result22, PlotResult)
    assert isinstance(result23, PlotResult)
    assert isinstance(result24, PlotResult)
    for index, key in enumerate(list(result22.data.keys())):
        assert isinstance(result22.data[key], np.ndarray)
        assert isinstance(result23.data[key], np.ndarray)
        assert isinstance(result24.data[key], np.ndarray)
        if index == 0:
            length: int = result22.data[key].size  # type: ignore
        assert result22.data[key].size == length  # type: ignore
        assert result23.data[key].size == length  # type: ignore
        assert result24.data[key].size == length  # type: ignore
        if key != "Slice":
            assert np.all(result22.data[key] >= 0)  # type: ignore
            assert np.all(result23.data[key] >= 0)  # type: ignore
            assert np.all(result24.data[key] >= 0)  # type: ignore
            assert np.all(result22.data[key] == result23.data[key] * 2)  # type: ignore
            assert np.all(result22.data[key] == result24.data[key] * 4)  # type: ignore
            assert np.all(result22.data[key] <= result01.data[key])  # type: ignore
            assert np.all(result23.data[key] <= result02.data[key])  # type: ignore
            assert np.all(result24.data[key] <= result03.data[key])  # type: ignore
        else:
            assert np.all(
                result22.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore
            assert np.all(
                result23.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore
            assert np.all(
                result24.data[key] == np.arange(0, sample_masks["3dgray"].shape[0])
            )  # type: ignore


def test_bulk_plot_empty_masks(sample_masks) -> None:

    empty_mask: np.ndarray = np.zeros(sample_masks["3dgray"].shape, np.bool)
    result01 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Apply_Mask=True,
        Mask=empty_mask,
        Perform_Slicewise=True,
    )
    result02 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Apply_Mask=True,
        Mask=empty_mask,
        Perform_Slicewise=True,
    )
    result03 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Apply_Mask=True,
        Mask=empty_mask,
        Perform_Slicewise=True,
    )

    assert isinstance(result01, PlotResult)
    assert isinstance(result02, PlotResult)
    assert isinstance(result03, PlotResult)
    assert len(list(result01.data.keys())) == 1
    assert len(list(result02.data.keys())) == 1
    assert len(list(result03.data.keys())) == 1

    result04 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Normalize=True,
        Apply_Mask=True,
        Mask=empty_mask,
        Perform_Slicewise=True,
    )
    result05 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Normalize=True,
        Apply_Mask=True,
        Mask=empty_mask,
        Perform_Slicewise=True,
    )
    result06 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Normalize=True,
        Apply_Mask=True,
        Mask=empty_mask,
        Perform_Slicewise=True,
    )

    assert isinstance(result04, PlotResult)
    assert isinstance(result05, PlotResult)
    assert isinstance(result06, PlotResult)
    assert len(list(result04.data.keys())) == 1
    assert len(list(result05.data.keys())) == 1
    assert len(list(result06.data.keys())) == 1

    result07 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Apply_Mask=True,
        Mask=empty_mask,
        Normalize=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result08 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Apply_Mask=True,
        Mask=empty_mask,
        Normalize=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result09 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Apply_Mask=True,
        Mask=empty_mask,
        Normalize=True,
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )

    assert isinstance(result07, PlotResult)
    assert isinstance(result08, PlotResult)
    assert isinstance(result09, PlotResult)
    assert len(list(result07.data.keys())) == 1
    assert len(list(result08.data.keys())) == 1
    assert len(list(result09.data.keys())) == 1

    result10 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Apply_Mask=True,
        Mask=empty_mask,
        Normalize=True,
        Normalization_Method="Segmented",
        Perform_Slicewise=True,
    )
    result11 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Apply_Mask=True,
        Mask=empty_mask,
        Normalize=True,
        Normalization_Method="Segmented",
        Perform_Slicewise=True,
    )
    result12 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Apply_Mask=True,
        Mask=empty_mask,
        Normalize=True,
        Normalization_Method="Segmented",
        Perform_Slicewise=True,
    )

    assert isinstance(result10, PlotResult)
    assert isinstance(result11, PlotResult)
    assert isinstance(result12, PlotResult)
    assert len(list(result10.data.keys())) == 1
    assert len(list(result11.data.keys())) == 1
    assert len(list(result12.data.keys())) == 1

    result13 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Apply_Mask=True,
        Mask=empty_mask,
        Normalize=True,
        Normalization_Method="Segmented",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result14 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Apply_Mask=True,
        Mask=empty_mask,
        Normalize=True,
        Normalization_Method="Segmented",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result15 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Apply_Mask=True,
        Mask=empty_mask,
        Normalize=True,
        Normalization_Method="Segmented",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )

    assert isinstance(result13, PlotResult)
    assert isinstance(result14, PlotResult)
    assert isinstance(result15, PlotResult)
    assert len(list(result13.data.keys())) == 1
    assert len(list(result14.data.keys())) == 1
    assert len(list(result15.data.keys())) == 1

    result16 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Apply_Mask=True,
        Mask=empty_mask,
        Normalize=True,
        Normalization_Method="Phase",
        Perform_Slicewise=True,
    )
    result17 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Apply_Mask=True,
        Mask=empty_mask,
        Normalize=True,
        Normalization_Method="Phase",
        Perform_Slicewise=True,
    )
    result18 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Apply_Mask=True,
        Mask=empty_mask,
        Normalize=True,
        Normalization_Method="Phase",
        Perform_Slicewise=True,
    )

    assert isinstance(result16, PlotResult)
    assert isinstance(result17, PlotResult)
    assert isinstance(result18, PlotResult)
    assert len(list(result16.data.keys())) == 1
    assert len(list(result17.data.keys())) == 1
    assert len(list(result18.data.keys())) == 1

    result19 = measure.bulk(
        sample_masks["3dgray"],
        Pixel_Size=2,
        Apply_Mask=True,
        Mask=empty_mask,
        Normalize=True,
        Normalization_Method="Phase",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result20 = measure.bulk(
        sample_masks["3dgray"],
        "Area",
        2,
        Apply_Mask=True,
        Mask=empty_mask,
        Normalize=True,
        Normalization_Method="Phase",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )
    result21 = measure.bulk(
        sample_masks["3dgray"],
        "Length",
        2,
        Apply_Mask=True,
        Mask=empty_mask,
        Normalize=True,
        Normalization_Method="Phase",
        Perform_Slicewise=True,
        Normalize_by_Slice=True,
    )

    assert isinstance(result19, PlotResult)
    assert isinstance(result20, PlotResult)
    assert isinstance(result21, PlotResult)
    assert len(list(result19.data.keys())) == 1
    assert len(list(result20.data.keys())) == 1
    assert len(list(result21.data.keys())) == 1


def test_label_count(sample_masks) -> None:

    lab_array: np.ndarray = label(sample_masks["3dgray"])  # type: ignore
    result = measure.label_count(lab_array)
    assert result == label(sample_masks["3dgray"], return_num=True)[-1]  # type: ignore

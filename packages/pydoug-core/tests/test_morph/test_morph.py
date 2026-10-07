"""
Tests for PyDoug's morph.morph module.
"""

import numpy as np
from skimage.measure import label

from pydoug_core.morph import morph


def test_dilation(sample_arrays) -> None:

    result1: np.ndarray = morph.dilation(sample_arrays["2dgray"])
    result2: np.ndarray = morph.dilation(sample_arrays["3dgray"])
    result3: np.ndarray = morph.dilation(sample_arrays["2drgb"])
    result4: np.ndarray = morph.dilation(sample_arrays["3drgb"])
    result5: np.ndarray = morph.dilation(
        sample_arrays["3dgray"], 3, Perform_Slicewise=True
    )
    assert result1.dtype == sample_arrays["2dgray"].dtype
    assert result2.dtype == sample_arrays["3dgray"].dtype
    assert result3.dtype == sample_arrays["2drgb"].dtype
    assert result4.dtype == sample_arrays["3drgb"].dtype
    assert result5.dtype == sample_arrays["3dgray"].dtype
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape == sample_arrays["2drgb"].shape
    assert result4.shape == sample_arrays["3drgb"].shape
    assert result5.shape == sample_arrays["3dgray"].shape


def test_erosion(sample_arrays) -> None:

    result1: np.ndarray = morph.erosion(sample_arrays["2dgray"])
    result2: np.ndarray = morph.erosion(sample_arrays["3dgray"])
    result3: np.ndarray = morph.erosion(sample_arrays["2drgb"])
    result4: np.ndarray = morph.erosion(sample_arrays["3drgb"])
    result5: np.ndarray = morph.erosion(
        sample_arrays["3dgray"], 3, Perform_Slicewise=True
    )
    assert result1.dtype == sample_arrays["2dgray"].dtype
    assert result2.dtype == sample_arrays["3dgray"].dtype
    assert result3.dtype == sample_arrays["2drgb"].dtype
    assert result4.dtype == sample_arrays["3drgb"].dtype
    assert result5.dtype == sample_arrays["3dgray"].dtype
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape == sample_arrays["2drgb"].shape
    assert result4.shape == sample_arrays["3drgb"].shape
    assert result5.shape == sample_arrays["3dgray"].shape


def test_closing(sample_arrays) -> None:

    result1: np.ndarray = morph.closing(sample_arrays["2dgray"])
    result2: np.ndarray = morph.closing(sample_arrays["3dgray"])
    result3: np.ndarray = morph.closing(sample_arrays["2drgb"])
    result4: np.ndarray = morph.closing(sample_arrays["3drgb"])
    result5: np.ndarray = morph.closing(
        sample_arrays["3dgray"], 3, Perform_Slicewise=True
    )
    assert result1.dtype == sample_arrays["2dgray"].dtype
    assert result2.dtype == sample_arrays["3dgray"].dtype
    assert result3.dtype == sample_arrays["2drgb"].dtype
    assert result4.dtype == sample_arrays["3drgb"].dtype
    assert result5.dtype == sample_arrays["3dgray"].dtype
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape == sample_arrays["2drgb"].shape
    assert result4.shape == sample_arrays["3drgb"].shape
    assert result5.shape == sample_arrays["3dgray"].shape


def test_opening(sample_arrays) -> None:

    result1: np.ndarray = morph.opening(sample_arrays["2dgray"])
    result2: np.ndarray = morph.opening(sample_arrays["3dgray"])
    result3: np.ndarray = morph.opening(sample_arrays["2drgb"])
    result4: np.ndarray = morph.opening(sample_arrays["3drgb"])
    result5: np.ndarray = morph.opening(
        sample_arrays["3dgray"], 3, Perform_Slicewise=True
    )
    assert result1.dtype == sample_arrays["2dgray"].dtype
    assert result2.dtype == sample_arrays["3dgray"].dtype
    assert result3.dtype == sample_arrays["2drgb"].dtype
    assert result4.dtype == sample_arrays["3drgb"].dtype
    assert result5.dtype == sample_arrays["3dgray"].dtype
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape == sample_arrays["2drgb"].shape
    assert result4.shape == sample_arrays["3drgb"].shape
    assert result5.shape == sample_arrays["3dgray"].shape


def test_black_tophat(sample_arrays) -> None:

    result1: np.ndarray = morph.black_tophat(sample_arrays["2dgray"])
    result2: np.ndarray = morph.black_tophat(sample_arrays["3dgray"])
    result3: np.ndarray = morph.black_tophat(sample_arrays["2drgb"])
    result4: np.ndarray = morph.black_tophat(sample_arrays["3drgb"])
    result5: np.ndarray = morph.black_tophat(
        sample_arrays["3dgray"], 3, Perform_Slicewise=True
    )
    assert result1.dtype == sample_arrays["2dgray"].dtype
    assert result2.dtype == sample_arrays["3dgray"].dtype
    assert result3.dtype == sample_arrays["2drgb"].dtype
    assert result4.dtype == sample_arrays["3drgb"].dtype
    assert result5.dtype == sample_arrays["3dgray"].dtype
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape == sample_arrays["2drgb"].shape
    assert result4.shape == sample_arrays["3drgb"].shape
    assert result5.shape == sample_arrays["3dgray"].shape


def test_white_tophat(sample_arrays) -> None:

    result1: np.ndarray = morph.white_tophat(sample_arrays["2dgray"])
    result2: np.ndarray = morph.white_tophat(sample_arrays["3dgray"])
    result3: np.ndarray = morph.white_tophat(sample_arrays["2drgb"])
    result4: np.ndarray = morph.white_tophat(sample_arrays["3drgb"])
    result5: np.ndarray = morph.white_tophat(
        sample_arrays["3dgray"], 3, Perform_Slicewise=True
    )
    assert result1.dtype == sample_arrays["2dgray"].dtype
    assert result2.dtype == sample_arrays["3dgray"].dtype
    assert result3.dtype == sample_arrays["2drgb"].dtype
    assert result4.dtype == sample_arrays["3drgb"].dtype
    assert result5.dtype == sample_arrays["3dgray"].dtype
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape == sample_arrays["2drgb"].shape
    assert result4.shape == sample_arrays["3drgb"].shape
    assert result5.shape == sample_arrays["3dgray"].shape


def test_remove_labeled_objects(sample_masks) -> None:

    test_2d: np.ndarray = label(sample_masks["2dgray"])  # type: ignore
    test_3d: np.ndarray = label(sample_masks["3dgray"])  # type: ignore
    result0: np.ndarray = morph.remove_labeled_objects(test_2d, Size=0)
    result0_2: np.ndarray = morph.remove_labeled_objects(
        test_2d, "Distance", Distance=0
    )
    result1: np.ndarray = morph.remove_labeled_objects(test_2d)
    result2: np.ndarray = morph.remove_labeled_objects(test_3d)
    result3: np.ndarray = morph.remove_labeled_objects(test_2d, "Distance")
    result4: np.ndarray = morph.remove_labeled_objects(test_3d, Remove_Above_Size=True)

    assert np.all(result0 == test_2d)
    assert np.all(result0_2 == test_2d)
    assert result1.dtype == test_2d.dtype
    assert result2.dtype == test_3d.dtype
    assert result3.dtype == test_2d.dtype
    assert result4.dtype == test_3d.dtype


def test_remove_holes(sample_masks) -> None:

    result0: np.ndarray = morph.remove_holes(sample_masks["2dgray"], Size=0)
    result1: np.ndarray = morph.remove_holes(sample_masks["2dgray"])
    result2: np.ndarray = morph.remove_holes(sample_masks["3dgray"])
    result3: np.ndarray = morph.remove_holes(
        sample_masks["3dgray"], Perform_Slicewise=True
    )
    result4: np.ndarray = morph.remove_holes(
        sample_masks["2dgray"], Perform_Slicewise=True
    )

    assert np.all(result0 == sample_masks["2dgray"])
    assert result1.dtype == sample_masks["2dgray"].dtype
    assert result2.dtype == sample_masks["3dgray"].dtype
    assert result3.dtype == sample_masks["3dgray"].dtype
    assert np.all(result4 == result1)


def test_convex_hull(sample_masks, sample_2drgb_array) -> None:

    result0: np.ndarray = morph.convex_hull(sample_2drgb_array)
    result1: np.ndarray = morph.convex_hull(sample_masks["2dgray"])
    result2: np.ndarray = morph.convex_hull(sample_masks["3dgray"])
    result3: np.ndarray = morph.convex_hull(sample_masks["2dgray"], False)
    result4: np.ndarray = morph.convex_hull(sample_masks["3dgray"], False)
    result5: np.ndarray = morph.convex_hull(
        sample_masks["2dgray"], Perform_Slicewise=True
    )
    result6: np.ndarray = morph.convex_hull(
        sample_masks["3dgray"], Perform_Slicewise=True
    )
    result7: np.ndarray = morph.convex_hull(
        sample_masks["2dgray"], False, Perform_Slicewise=True
    )
    result8: np.ndarray = morph.convex_hull(
        sample_masks["3dgray"], False, Perform_Slicewise=True
    )

    assert np.all(result0 == sample_2drgb_array)
    assert np.all(result1 == result5)
    assert np.all(result3 == result7)
    assert result1.dtype == np.bool
    assert result2.dtype == np.bool
    assert result3.dtype == np.bool
    assert result4.dtype == np.bool
    assert result5.dtype == np.bool
    assert result6.dtype == np.bool
    assert result7.dtype == np.bool
    assert result8.dtype == np.bool


def test_thin(sample_masks, sample_2drgb_array) -> None:

    result0: np.ndarray = morph.thin(sample_2drgb_array)
    result1: np.ndarray = morph.thin(sample_masks["2dgray"])
    result2: np.ndarray = morph.thin(sample_masks["3dgray"])
    result3: np.ndarray = morph.thin(sample_masks["2dgray"], False)
    result4: np.ndarray = morph.thin(sample_masks["3dgray"], False)
    result5: np.ndarray = morph.thin(sample_masks["2dgray"], Skeletonize_Method="Zhang")
    result6: np.ndarray = morph.thin(sample_masks["3dgray"], Skeletonize_Method="Zhang")
    result7: np.ndarray = morph.thin(sample_masks["2dgray"], Perform_Slicewise=True)
    result8: np.ndarray = morph.thin(sample_masks["3dgray"], Perform_Slicewise=True)
    result9: np.ndarray = morph.thin(
        sample_masks["2dgray"], False, Thinning_Iterations=10
    )
    result10: np.ndarray = morph.thin(
        sample_masks["3dgray"], False, Thinning_Iterations=10
    )
    result11: np.ndarray = morph.thin(
        sample_masks["2dgray"], False, Perform_Slicewise=True
    )
    result12: np.ndarray = morph.thin(
        sample_masks["3dgray"], False, Perform_Slicewise=True
    )
    result13: np.ndarray = morph.thin(
        sample_masks["2dgray"], Skeletonize_Method="Zhang", Perform_Slicewise=True
    )
    result14: np.ndarray = morph.thin(
        sample_masks["3dgray"], Skeletonize_Method="Zhang", Perform_Slicewise=True
    )

    assert np.all(result0 == sample_2drgb_array)
    assert np.all(result1 == result7)
    assert np.all(result3 == result11)
    assert np.all(result4 == result12)
    assert np.all(result5 == result13)
    assert np.all(result6 == result14)
    assert result1.dtype == np.bool
    assert result2.dtype == np.bool
    assert result3.dtype == np.bool
    assert result4.dtype == np.bool
    assert result5.dtype == np.bool
    assert result6.dtype == np.bool
    assert result7.dtype == np.bool
    assert result8.dtype == np.bool
    assert result9.dtype == np.bool
    assert result10.dtype == np.bool
    assert result11.dtype == np.bool
    assert result12.dtype == np.bool
    assert result13.dtype == np.bool
    assert result14.dtype == np.bool


def test_extrema(sample_arrays) -> None:

    result0: np.ndarray = morph.extrema(sample_arrays["2drgb"])
    result1: np.ndarray = morph.extrema(sample_arrays["2dgray"])
    result2: np.ndarray = morph.extrema(sample_arrays["3dgray"])
    result3: np.ndarray = morph.extrema(sample_arrays["2dgray"], Perform_Slicewise=True)
    result4: np.ndarray = morph.extrema(sample_arrays["3dgray"], Perform_Slicewise=True)
    result5: np.ndarray = morph.extrema(sample_arrays["2dgray"], False)
    result6: np.ndarray = morph.extrema(sample_arrays["3dgray"], False)
    result7: np.ndarray = morph.extrema(
        sample_arrays["2dgray"], False, Perform_Slicewise=True
    )
    result8: np.ndarray = morph.extrema(
        sample_arrays["3dgray"], False, Perform_Slicewise=True
    )
    result9: np.ndarray = morph.extrema(sample_arrays["2dgray"], Minimum_Difference=5)
    result10: np.ndarray = morph.extrema(sample_arrays["3dgray"], Minimum_Difference=5)
    result11: np.ndarray = morph.extrema(
        sample_arrays["2dgray"], Minimum_Difference=5, Perform_Slicewise=True
    )
    result12: np.ndarray = morph.extrema(
        sample_arrays["3dgray"], Minimum_Difference=5, Perform_Slicewise=True
    )
    result13: np.ndarray = morph.extrema(sample_arrays["2dgray"], False, 5)
    result14: np.ndarray = morph.extrema(sample_arrays["3dgray"], False, 5)
    result15: np.ndarray = morph.extrema(
        sample_arrays["2dgray"], False, 5, Perform_Slicewise=True
    )
    result16: np.ndarray = morph.extrema(
        sample_arrays["3dgray"], False, 5, Perform_Slicewise=True
    )

    assert np.all(result0 == sample_arrays["2drgb"])
    assert np.all(result1 == result3)
    assert np.all(result5 == result7)
    assert np.all(result9 == result11)
    assert np.all(result13 == result15)
    assert result1.dtype == np.bool
    assert result2.dtype == np.bool
    assert result3.dtype == np.bool
    assert result4.dtype == np.bool
    assert result5.dtype == np.bool
    assert result6.dtype == np.bool
    assert result7.dtype == np.bool
    assert result8.dtype == np.bool
    assert result9.dtype == np.bool
    assert result10.dtype == np.bool
    assert result11.dtype == np.bool
    assert result12.dtype == np.bool
    assert result13.dtype == np.bool
    assert result14.dtype == np.bool
    assert result15.dtype == np.bool
    assert result16.dtype == np.bool
    assert result1.shape == sample_arrays["2dgray"].shape
    assert result2.shape == sample_arrays["3dgray"].shape
    assert result3.shape == sample_arrays["2dgray"].shape
    assert result4.shape == sample_arrays["3dgray"].shape
    assert result5.shape == sample_arrays["2dgray"].shape
    assert result6.shape == sample_arrays["3dgray"].shape
    assert result7.shape == sample_arrays["2dgray"].shape
    assert result8.shape == sample_arrays["3dgray"].shape
    assert result9.shape == sample_arrays["2dgray"].shape
    assert result10.shape == sample_arrays["3dgray"].shape
    assert result11.shape == sample_arrays["2dgray"].shape
    assert result12.shape == sample_arrays["3dgray"].shape
    assert result13.shape == sample_arrays["2dgray"].shape
    assert result14.shape == sample_arrays["3dgray"].shape
    assert result15.shape == sample_arrays["2dgray"].shape
    assert result16.shape == sample_arrays["3dgray"].shape

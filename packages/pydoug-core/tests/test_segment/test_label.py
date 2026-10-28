"""
Tests for PyDoug's segment.label module.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.measure import label as conn_label
from skimage.morphology import ball, disk

from pydoug_core.segment import label


def test_connectivity(sample_3dgray_array) -> None:

    result: np.ndarray = label.connectivity(sample_3dgray_array >= 128)
    assert isinstance(result, np.ndarray)
    assert np.issubdtype(result.dtype, np.integer)
    assert len(np.unique(result)) > 1


def test_watershed(sample_masks) -> None:

    dist_2d: np.ndarray = distance_transform_edt(sample_masks["2dgray"])  # type: ignore
    dist_3d: np.ndarray = distance_transform_edt(sample_masks["3dgray"])  # type: ignore
    fp_2d: np.ndarray = disk(3)  # type: ignore
    fp_3d: np.ndarray = ball(3)  # type: ignore
    coords_2d: np.ndarray = peak_local_max(dist_2d, footprint=fp_2d)
    coords_3d: np.ndarray = peak_local_max(dist_3d, footprint=fp_3d)
    mask_2d: np.ndarray = np.zeros(dist_2d.shape, np.bool)
    mask_3d: np.ndarray = np.zeros(dist_3d.shape, np.bool)
    mask_2d[tuple(coords_2d.T)] = True
    mask_3d[tuple(coords_3d.T)] = True
    mark_2d: np.ndarray = conn_label(mask_2d)  # type: ignore
    mark_3d: np.ndarray = conn_label(mask_3d)  # type: ignore

    result0: np.ndarray = label.watershed(sample_masks["2drgb"])
    result1: np.ndarray = label.watershed(sample_masks["2dgray"])
    result2: np.ndarray = label.watershed(sample_masks["3dgray"])
    result3: np.ndarray = label.watershed(dist_2d, mark_2d, True)
    result4: np.ndarray = label.watershed(dist_3d, mark_3d, True)
    result5: np.ndarray = label.watershed(dist_2d, mark_2d, True, False)
    result6: np.ndarray = label.watershed(dist_3d, mark_3d, True, False)
    result7: np.ndarray = label.watershed(
        sample_masks["2dgray"], Perform_Slicewise=True
    )
    result8: np.ndarray = label.watershed(
        sample_masks["3dgray"], Perform_Slicewise=True
    )
    result9: np.ndarray = label.watershed(
        dist_2d, mark_2d, True, Perform_Slicewise=True
    )
    result10: np.ndarray = label.watershed(
        dist_3d, mark_3d, True, Perform_Slicewise=True
    )
    result11: np.ndarray = label.watershed(
        dist_2d, mark_2d, True, False, Perform_Slicewise=True
    )
    result12: np.ndarray = label.watershed(
        dist_3d, mark_3d, True, False, Perform_Slicewise=True
    )

    assert np.all(result0 == sample_masks["2drgb"])
    assert result1.dtype == np.int32
    assert result2.dtype == np.int32
    assert result3.dtype == np.int32
    assert result4.dtype == np.int32
    assert result5.dtype == np.int32
    assert result6.dtype == np.int32
    assert result7.dtype == np.int32
    assert result8.dtype == np.int32
    assert result9.dtype == np.int32
    assert result10.dtype == np.int32
    assert result11.dtype == np.int32
    assert result12.dtype == np.int32
    assert np.all(result1 == result3)
    assert np.all(result2 == result4)
    assert len(np.unique(result1)) >= 2
    assert len(np.unique(result2)) >= 2
    assert len(np.unique(result3)) >= 2
    assert len(np.unique(result4)) >= 2
    assert len(np.unique(result5)) >= 2
    assert len(np.unique(result6)) >= 2
    assert len(np.unique(result7)) >= 2
    assert len(np.unique(result8)) >= 2
    assert len(np.unique(result9)) >= 2
    assert len(np.unique(result10)) >= 2
    assert len(np.unique(result11)) >= 2
    assert len(np.unique(result12)) >= 2

"""
Tests for PyDoug's util module.
"""

import numpy as np

from pydoug_core import util


def test_is3d_gray_returns(
    sample_2dgray_array, sample_3dgray_array, sample_2drgb_array, sample_3drgb_array
) -> None:

    assert not util.is3d(sample_2dgray_array)
    assert util.is3d(sample_3dgray_array)
    assert not util.is3d(sample_2drgb_array)
    assert util.is3d(sample_3drgb_array)
    assert not util.is3d(imshape=sample_2dgray_array.shape)
    assert util.is3d(imshape=sample_3dgray_array.shape)
    assert not util.is3d(imshape=sample_2drgb_array.shape)
    assert util.is3d(imshape=sample_3drgb_array.shape)
    assert not util.is3d()


def test_isrgb_returns(
    sample_2dgray_array, sample_3dgray_array, sample_2drgb_array, sample_3drgb_array
) -> None:

    assert not util.isrgb(sample_2dgray_array)
    assert not util.isrgb(sample_3dgray_array)
    assert util.isrgb(sample_2drgb_array)
    assert util.isrgb(sample_3drgb_array)
    assert not util.isrgb(imshape=sample_2dgray_array.shape)
    assert not util.isrgb(imshape=sample_3dgray_array.shape)
    assert util.isrgb(imshape=sample_2drgb_array.shape)
    assert util.isrgb(imshape=sample_3drgb_array.shape)
    assert not util.isrgb()


def test_gray2rgb_returns(
    sample_2dgray_array, sample_3dgray_array, sample_2drgb_array, sample_3drgb_array
) -> None:

    assert util.isrgb(util.gray2rgb(sample_2dgray_array))
    assert util.isrgb(util.gray2rgb(sample_3dgray_array))
    assert np.all(util.gray2rgb(sample_2drgb_array) == sample_2drgb_array)
    assert np.all(util.gray2rgb(sample_3drgb_array) == sample_3drgb_array)

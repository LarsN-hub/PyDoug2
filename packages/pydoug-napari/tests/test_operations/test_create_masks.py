"""
Tests for PyDoug's napari plugin _operations.create_masks module.
"""

import numpy as np
import pytest
from napari.layers import Shapes

from pydoug_core.registry import OperationSpec, get_operation
from pydoug_napari._operations import create_masks


def _rectangle(
    y: tuple[int, int], x: tuple[int, int], z: int | None = None
) -> np.ndarray:
    """
    Four corners of a rectangle, on slice z if one is given.
    """

    corners: list[list[int]] = [[y[0], x[0]], [y[0], x[1]], [y[1], x[1]], [y[1], x[0]]]
    if z is not None:
        corners = [[z, *corner] for corner in corners]
    return np.array(corners, dtype=float)


def _layer(*rectangles: np.ndarray) -> Shapes:
    return Shapes(list(rectangles), shape_type="rectangle")


def test_2d_dimensions_are_read_as_x_y_z() -> None:

    layer: Shapes = _layer(_rectangle((10, 30), (20, 50)))
    mask: np.ndarray = create_masks.shapes_to_mask(
        layer, Dimensions=(60, 40, 0), Use_Image_Dimensions=False
    )

    assert mask.shape == (40, 60)
    assert mask.dtype == bool
    assert mask[20, 35]
    assert not mask[2, 2]


@pytest.mark.parametrize(
    "use_image_dimensions, image_layer",
    [(False, np.zeros((7, 7))), (True, None), (False, None)],
    ids=["flag_off_image_given", "flag_on_no_image", "flag_off_no_image"],
)
def test_dimensions_used_unless_image_dimensions_apply(
    use_image_dimensions, image_layer
) -> None:

    layer: Shapes = _layer(_rectangle((10, 30), (20, 50)))
    mask: np.ndarray = create_masks.shapes_to_mask(
        layer,
        Dimensions=(60, 40, 0),
        Use_Image_Dimensions=use_image_dimensions,
        Image_Layer=image_layer,
    )

    assert mask.shape == (40, 60)


@pytest.mark.parametrize(
    "image_shape, z",
    [((40, 60), None), ((5, 40, 60), 2)],
    ids=["2d_image", "3d_image"],
)
def test_image_dimensions_override_dimensions(image_shape, z) -> None:

    layer: Shapes = _layer(_rectangle((10, 30), (20, 50), z=z))
    mask: np.ndarray = create_masks.shapes_to_mask(
        layer,
        Dimensions=(7, 7, 7),
        Use_Image_Dimensions=True,
        Image_Layer=np.zeros(image_shape),
        Propagate_Through_Z=False,
    )

    point = (20, 35) if z is None else (z, 20, 35)
    assert mask.shape == image_shape
    assert mask[point]


def test_3d_dimensions_without_propagation_keep_the_drawn_slice_only() -> None:

    layer: Shapes = _layer(_rectangle((10, 30), (20, 50), z=2))
    mask: np.ndarray = create_masks.shapes_to_mask(
        layer,
        Dimensions=(60, 40, 5),
        Use_Image_Dimensions=False,
        Propagate_Through_Z=False,
    )

    assert mask.shape == (5, 40, 60)
    assert mask[2, 20, 35]
    assert not mask[[0, 1, 3, 4]].any()


@pytest.mark.parametrize("z", [0, 2, 4])
def test_propagation_copies_the_shape_to_every_slice(z) -> None:

    layer: Shapes = _layer(_rectangle((10, 30), (20, 50), z=z))
    mask: np.ndarray = create_masks.shapes_to_mask(
        layer,
        Dimensions=(60, 40, 5),
        Use_Image_Dimensions=False,
        Propagate_Through_Z=True,
    )

    assert mask.shape == (5, 40, 60)
    assert mask.dtype == bool
    assert mask[:, 20, 35].all()
    assert not mask[:, 2, 2].any()
    assert all(np.array_equal(mask[0], mask[i]) for i in range(1, 5))


def test_propagation_combines_shapes_drawn_on_different_slices() -> None:

    layer: Shapes = _layer(
        _rectangle((10, 30), (5, 25), z=0),
        _rectangle((10, 30), (35, 55), z=3),
    )
    mask: np.ndarray = create_masks.shapes_to_mask(
        layer, Dimensions=(60, 40, 5), Use_Image_Dimensions=False
    )

    assert mask[:, 20, 15].all()
    assert mask[:, 20, 45].all()
    assert not mask[:, 20, 30].any()


def test_registered_as_an_unrecorded_shapes_operation() -> None:

    spec: OperationSpec = get_operation("shapes_to_mask")

    assert spec.layer_type == "shapes"
    assert spec.recordable is False
    assert spec.array_params == ("Image_Layer",)
    assert spec.array_param_flags == {"Image_Layer": "Use_Image_Dimensions"}


def test_labels_to_mask(sample_arrays) -> None:

    labels_2d: np.ndarray = np.copy(sample_arrays["2dgray"])
    labels_2d[0:2, 0:2] = 0
    labels_3d: np.ndarray = np.copy(sample_arrays["3dgray"])
    labels_3d[0:2, 0:2, 0:2] = 0
    result1: np.ndarray = create_masks.labels_to_mask(labels_2d)
    result2: np.ndarray = create_masks.labels_to_mask(labels_3d)

    assert result1.dtype == np.bool
    assert result2.dtype == np.bool
    assert result1.shape == labels_2d.shape
    assert result2.shape == labels_3d.shape
    assert np.all((labels_2d == 0) == (result1 == 0))
    assert np.all((labels_3d == 0) == (result2 == 0))

"""
Shared fixtures for pydoug-napari tests.
"""

from typing import Any

import numpy as np
import pytest
from matplotlib.figure import Figure
from napari.layers import Shapes
from skimage.measure import marching_cubes
from skimage.morphology import ball

from pydoug_core.registry import register_operation, unregister_operation
from pydoug_core.results import PlotResult, ValueResult, VisualResult

pytest_plugins: list[str] = ["pydoug_core.testing"]


@pytest.fixture
def registered_half():
    """
    Fixture to create a registered "half" operation.
    """

    @register_operation(
        category="testing",
        display_name="Half",
        native_options={"factor": {"setDecimals": 3}},
    )
    def _half(
        imarray: np.ndarray, factor: float = 0.5
    ) -> np.ndarray:  # pragma: no cover
        return np.round(imarray * factor).astype(imarray.dtype)

    yield "_half"
    unregister_operation("_half")


@pytest.fixture
def registered_mask():
    """
    Fixture to create a registered "mask" operation.
    """

    @register_operation(
        category="testing",
        display_name="Mask",
        array_params=("maskarray",),
        array_param_flags={"Mask": "Apply_Mask"},
        allow_in_place=True,
    )
    def _mask(
        imarray: np.ndarray,
        maskarray: np.ndarray,
        Apply_Mask: bool = False,
        In_Place: bool = False,
    ) -> np.ndarray:  # pragma: no cover
        if In_Place:
            if Apply_Mask:
                imarray[np.logical_not(maskarray)] = 0
            return imarray
        output: np.ndarray = np.copy(imarray)
        if Apply_Mask:
            output[np.logical_not(maskarray)] = 0
            return output
        return output

    yield "_mask"
    unregister_operation("_mask")


@pytest.fixture
def registered_bool():
    """
    Fixture to create a registered "bool" operation.
    """

    @register_operation(
        category="testing",
        display_name="Mask",
        array_params=("maskarray",),
        array_param_flags={"Mask": "Apply_Mask"},
    )
    def _bool(imarray: np.ndarray, thresh: int = 128) -> np.ndarray:  # pragma: no cover
        output: np.ndarray = np.copy(imarray)
        output[output > thresh] = 255
        output[output <= thresh] = 0
        return np.astype(output, np.bool)

    yield "_mask"
    unregister_operation("_mask")


@pytest.fixture
def registered_calc():
    """
    Fixture to create a registered "calc" operation.
    """

    @register_operation(category="testing", display_name="Calc", output_kind="value")
    def _calc(imarray: np.ndarray) -> np.float64:  # pragma: no cover
        return np.mean(imarray)

    yield "_calc"
    unregister_operation("_calc")


@pytest.fixture
def registered_calc_vr():
    """
    Fixture to create a registered "calc vr" operation (vr = ValueResult).
    """

    @register_operation(category="testing", display_name="Calc VR", output_kind="value")
    def _calc_vr(imarray: np.ndarray) -> ValueResult:  # pragma: no cover
        return ValueResult(np.mean(imarray))

    yield "_calc_vr"
    unregister_operation("_calc_vr")


@pytest.fixture
def registered_plot():
    """
    Fixture to create a registered "plot" operation.
    """

    @register_operation(category="testing", display_name="Plot", output_kind="plot")
    def _plot(imarray: np.ndarray) -> PlotResult:  # pragma: no cover
        return PlotResult(
            Figure(),
            {"x": [1, 2, 3], "y": [2, 4, 6]},
            ("x",),
            {"stats": {"mean": 4}},
        )

    yield "_plot"
    unregister_operation("_plot")


@pytest.fixture
def registered_threshold():
    """
    Fixture to create a registerd "threshold" operation.
    """

    @register_operation(category="testing", display_name="Threshold", live_preview=True)
    def _threshold(
        imarray: np.ndarray, threshold: int = 128
    ) -> np.ndarray:  # pragma: no cover
        thresh_array: np.ndarray = np.zeros(imarray.shape, np.uint8)
        thresh_array[imarray > threshold] = 255
        return thresh_array

    yield "_threshold"
    unregister_operation("_threshold")


@pytest.fixture
def registered_label_count():
    """
    Fixture to create a registered "label_count" operation.
    """

    @register_operation(
        category="testing",
        display_name="Label Count",
        layer_type="labels",
        output_kind="value",
    )
    def _label_count(imarray: np.ndarray) -> int:  # pragma: no cover
        return np.max(imarray)

    yield "_label_count"
    unregister_operation("_label_count")


@pytest.fixture
def registered_capture_screen_image():
    """
    Fixture to create a registered "capture screen image" operation.
    """

    @register_operation(
        category="testing",
        display_name="Capture Screen Image",
        captures_viewer_settings=True,
    )
    def _capture_screen_image(
        imarray: np.ndarray, Settings: dict[str, Any]
    ) -> np.ndarray:  # pragma: no cover
        return imarray

    yield "_capture_screen_image"
    unregister_operation("_capture_screen_image")


@pytest.fixture
def registered_capture_screen_labels():
    """
    Fixture to create a registered "capture screen labels" operation.
    """

    @register_operation(
        category="testing",
        display_name="Capture Screen Labels",
        captures_viewer_settings=True,
        layer_type="labels",
    )
    def _capture_screen_labels(
        imarray: np.ndarray, Settings: dict[str, Any]
    ) -> np.ndarray:  # pragma: no cover
        return imarray

    yield "_capture_screen_labels"
    unregister_operation("_capture_screen_labels")


@pytest.fixture
def registered_generator():
    """
    Fixture to create a registered "dummy generator" operation.
    """

    @register_operation(
        category="testing",
        display_name="Dummy Generator",
        requires_layer=False,
    )
    def _dummy_generator() -> np.ndarray:  # pragma: no cover
        return np.random.randint(0, 256, (10, 10), dtype=np.uint8)

    yield "_dummy_generator"
    unregister_operation("_dummy_generator")


@pytest.fixture
def registered_surface():
    """
    Fixture to create a registered "dummy surface" operation.
    """

    @register_operation(
        category="testing",
        display_name="Dummy Surface",
        requires_layer=False,
        output_kind="surface",
    )
    def _dummy_surface() -> tuple:  # pragma: no cover
        array: np.ndarray = ball(25)  # type: ignore
        verts, faces, _, values = marching_cubes(array)
        return verts, faces, values

    yield "_dummy_surface"
    unregister_operation("_dummy_surface")


@pytest.fixture
def registered_surface_calc():
    """
    Fixture to create a registered "dummy surface calc" operation.
    """

    @register_operation(
        category="testing",
        display_name="Dummy Surface Calc",
        layer_type="surface",
        output_kind="value",
    )
    def _dummy_surface_calc(surface: tuple) -> int:  # pragma: no cover
        return 2

    yield "_dummy_surface_calc"
    unregister_operation("_dummy_surface_calc")


@pytest.fixture
def registered_shapes():
    """
    Fixture to create a registered "dummy shapes" operation.
    """

    @register_operation(
        category="testing",
        display_name="Dummy Shapes",
        layer_type="shapes",
        recordable=False,
    )
    def _dummy_shapes(shapes: Shapes) -> np.ndarray:  # pragma: no cover
        return shapes.to_labels(labels_shape=(10, 10)) > 1

    yield "_dummy_shapes"
    unregister_operation("_dummy_shapes")


@pytest.fixture
def registered_shapes_primary():
    """
    Fixture to create a registered "dummy shapes primary" operation.
    """

    @register_operation(
        category="testing",
        display_name="Dummy Shapes Primary",
        layer_type="shapes",
        array_params=("imarray",),
        primary_params=("imarray",),
        output_kind="value",
    )
    def _dummy_shapes_primary(
        shapes: Shapes, imarray: np.ndarray
    ) -> int:  # pragma: no cover
        return 2

    yield "_dummy_shapes_primary"
    unregister_operation("_dummy_shapes_primary")


@pytest.fixture
def registered_visual_result():
    """
    Fixture to create a registered "visual result" operation.
    """

    @register_operation(
        category="testing",
        display_name="Dummy Visual Result",
        output_kind="visual",
    )
    def _dummy_visual_result(imarray: np.ndarray) -> VisualResult:  # pragma: no cover
        return VisualResult(imarray, "image", {})

    yield "_dummy_visual_result"
    unregister_operation("_dummy_visual_result")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)

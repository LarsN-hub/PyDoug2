"""
Shared fixtures for pydoug-napari tests.
"""

import numpy as np
import pytest
from matplotlib.figure import Figure

from pydoug_core.registry import register_operation, unregister_operation

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
    def _half(imarray: np.ndarray, factor: float = 0.5) -> np.ndarray:
        return np.round(imarray * factor).astype(imarray.dtype)  # pragma: no cover

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
    ) -> np.ndarray:
        if In_Place:  # pragma: no cover
            if Apply_Mask:  # pragma: no cover
                imarray[np.logical_not(maskarray)] = 0  # pragma: no cover
            return imarray  # pragma: no cover
        output: np.ndarray = np.copy(imarray)  # pragma: no cover
        if Apply_Mask:  # pragma: no cover
            output[np.logical_not(maskarray)] = 0  # pragma: no cover
            return output  # pragma: no cover
        return output  # pragma: no cover

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
    def _bool(imarray: np.ndarray, thresh: int = 128) -> np.ndarray:
        output: np.ndarray = np.copy(imarray)  # pragma: no cover
        output[output > thresh] = 255  # pragma: no cover
        output[output <= thresh] = 0  # pragma: no cover
        return np.astype(output, np.bool)  # pragma: no cover

    yield "_mask"
    unregister_operation("_mask")


@pytest.fixture
def registered_calc():
    """
    Fixture to create a registered "calc" operation.
    """

    @register_operation(category="testing", display_name="Calc", output_kind="value")
    def _calc(imarray: np.ndarray) -> np.float64:
        return np.mean(imarray)  # pragma: no cover

    yield "_calc"
    unregister_operation("_calc")


@pytest.fixture
def registered_plot():
    """
    Fixture to create a registered "plot" operation.
    """

    @register_operation(category="testing", display_name="Plot", output_kind="plot")
    def _plot(imarray: np.ndarray) -> Figure:
        return Figure()  # pragma: no cover

    yield "_plot"
    unregister_operation("_plot")


@pytest.fixture
def registered_threshold():
    """
    Fixture to create a registerd "threshold" operation.
    """

    @register_operation(category="testing", display_name="Threshold", live_preview=True)
    def _threshold(imarray: np.ndarray, threshold: int) -> np.ndarray:
        thresh_array: np.ndarray = np.zeros(imarray.shape, np.uint8)  # pragma: no cover
        thresh_array[imarray > threshold] = 255  # pragma: no cover
        return thresh_array  # pragma: no cover

    yield "_threshold"
    unregister_operation("_threshold")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)

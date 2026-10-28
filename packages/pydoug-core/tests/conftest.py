"""
Shared fixtures for pydoug-core tests.
"""

import numpy as np
import pytest
from matplotlib.figure import Figure

from pydoug_core.payloads import ShapesData
from pydoug_core.registry import register_operation, unregister_operation

pytest_plugins: list[str] = ["pydoug_core.testing"]


@pytest.fixture
def registered_nothing():
    """
    Fixture to create a registered "nothing" operation.
    """

    @register_operation(category="testing", display_name="Nothing")
    def _nothing(
        imarray: np.ndarray,
        extra_array: np.ndarray | None = None,
    ) -> np.ndarray:  # pragma: no cover
        return np.copy(imarray)

    yield "_nothing"
    unregister_operation("_nothing")


@pytest.fixture
def registered_nothing_no_layer():
    """
    Fixture to create a registered "nothing_no_layer" operation.
    """

    @register_operation(
        category="testing",
        display_name="Nothing No Layer",
        requires_layer=False,
    )
    def _nothing_no_layer() -> np.ndarray:  # pragma: no cover
        return np.random.randint(0, 256, (10, 10), dtype=np.uint8)

    yield "_nothing_no_layer"
    unregister_operation("_nothing_no_layer")


@pytest.fixture
def registered_nothing_plot():
    """
    Fixture to create a registered "nothing_plot" operation.
    """

    @register_operation(
        category="testing",
        display_name="Nothing Plot",
        output_kind="plot",
    )
    def _nothing_plot(imarray: np.ndarray) -> Figure:  # pragma: no cover
        return Figure()

    yield "_nothing_plot"
    unregister_operation("_nothing_plot")


@pytest.fixture
def registered_nothing_calc():
    """
    Fixture to create a registered "nothing_calc" operation.
    """

    @register_operation(
        category="testing", display_name="Nothing Calc", output_kind="value"
    )
    def _nothing_calc(imarray: np.ndarray) -> dict[str, int]:  # pragma: no cover
        return {"test": 2}

    yield "_nothing_calc"
    unregister_operation("_nothing_calc")


@pytest.fixture
def registered_nothing_surface():
    """
    Fixture to create a registered "nothing_surface" operation.
    """

    @register_operation(
        category="testing",
        display_name="Nothing Surface",
        array_params=("dummy_surface",),
        array_param_types={"dummy_surface": "surface"},
        output_kind="surface",
    )
    def _nothing_surface(
        imarray: np.ndarray, dummy_surface: tuple
    ) -> tuple:  # pragma: no cover
        return (np.array([[1, 2, 3]]), np.array([[4, 5, 6]]), np.array([7]))

    yield "_nothing_surface"
    unregister_operation("_nothing_surface")


@pytest.fixture
def registered_nothing_shapes():
    """
    Fixture to create a registered "nothing_shapes" operation.
    """

    @register_operation(
        category="testing",
        display_name="Nothing Shapes",
        array_params=("imarray",),
        primary_params=("imarray",),
        output_kind="value",
        layer_type="shapes",
    )
    def _nothing_shapes(
        dummy_shapes: ShapesData, imarray: np.ndarray
    ) -> int:  # pragma: no cover
        return 2

    yield "_nothing_shapes"
    unregister_operation("_nothing_shapes")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)

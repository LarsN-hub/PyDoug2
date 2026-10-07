"""
Shared fixtures for pydoug-core tests.
"""

import numpy as np
import pytest
from matplotlib.figure import Figure

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
    ) -> np.ndarray:
        return np.copy(imarray)  # pragma: no cover

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
    def _nothing_no_layer() -> np.ndarray:
        return np.random.randint(0, 256, (10, 10), dtype=np.uint8)  # pragma: no cover

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
    def _nothing_plot(imarray: np.ndarray) -> Figure:
        return Figure()  # pragma: no cover

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
    def _nothing_calc(imarray: np.ndarray) -> dict[str, int]:
        return {"test": 2}  # pragma: no cover

    yield "_nothing_calc"
    unregister_operation("_nothing_calc")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)

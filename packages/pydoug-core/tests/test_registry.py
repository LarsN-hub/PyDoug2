"""
Tests for PyDoug's core registry module.
"""

import inspect
from collections.abc import Callable

from pydoug_core import registry
from pydoug_core.trans import geom


def test_register_operations() -> None:

    result1: Callable = registry.register_operation("transformation")
    assert isinstance(result1, Callable)
    result2: Callable = result1(geom.mirror)
    assert isinstance(result2, Callable)


def test_get_operation_and_unregister() -> None:

    result1: registry.OperationSpec = registry.get_operation("reslice")
    assert isinstance(result1, registry.OperationSpec)
    assert result1.name == "reslice"
    assert result1.display_name == "Reslice"
    assert result1.category == "transform"
    assert result1.func == geom.reslice
    assert result1.signature == inspect.signature(geom.reslice)
    registry.unregister_operation("reslice")
    assert "reslice" not in list(registry._REGISTRY.keys())


def test_list_operations() -> None:

    result1: list[registry.OperationSpec] = registry.list_operations()
    result2: list[registry.OperationSpec] = registry.list_operations("transform")
    assert isinstance(result1, list)
    assert isinstance(result1[0], registry.OperationSpec)
    assert isinstance(result2, list)
    assert isinstance(result2[0], registry.OperationSpec)
    assert result2[0].category == "transform"

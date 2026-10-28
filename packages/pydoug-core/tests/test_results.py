"""
Tests for PyDoug's results module.
"""

from collections.abc import Sequence
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from pydoug_core import results


def test_ValueResult() -> None:

    values1: results.ValueResult = results.ValueResult(2)
    assert values1.values == 2
    assert values1.units is None
    assert values1.display() == 2

    values2: results.ValueResult = results.ValueResult(2, "a.u.")
    assert values2.values == 2
    assert values2.units == "a.u."
    assert values2.display() == "2 a.u."

    values3: results.ValueResult = results.ValueResult({2: 2, 3: 3}, "a.u.")
    assert values3.values == {2: 2, 3: 3}
    assert values3.units == "a.u."
    assert values3.display() == {2: "2 a.u.", 3: "3 a.u."}

    values4: results.ValueResult = results.ValueResult(
        {2: 2, 3: 3}, {2: "a.u.1", 3: "a.u.2"}
    )
    assert values4.values == {2: 2, 3: 3}
    assert values4.units == {2: "a.u.1", 3: "a.u.2"}
    assert values4.display() == {2: "2 a.u.1", 3: "3 a.u.2"}


def test_PlotResult(tmp_path) -> None:

    data: dict[str, Sequence] = {
        "1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "3": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    plot_result: results.PlotResult = results.PlotResult(Figure(), data)
    plot_result.write_csv(Path(tmp_path / "test_write_csv.csv"))
    assert Path(tmp_path / "test_write_csv.csv").exists()


def test_PlotResult_errors() -> None:

    data1: dict[str, Sequence] = {
        "1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "2": [1, 2, 3, 4, 5],
        "3": [1, 2, 3, 4, 5, 6, 7, 8],
    }
    data2: dict[str, Sequence] = {
        "1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "3": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    with pytest.raises(ValueError):
        _: results.PlotResult = results.PlotResult(Figure(), data1)
    with pytest.raises(ValueError):
        _: results.PlotResult = results.PlotResult(Figure(), data2, ("test",))
    with pytest.raises(ValueError):
        _: results.PlotResult = results.PlotResult(
            Figure(), data2, ("1",), {"plot_data": {"test": 1}}
        )
    with pytest.raises(ValueError):
        _: results.PlotResult = results.PlotResult(
            Figure(), data2, ("1",), {"test": {"test": [1, 2]}}
        )

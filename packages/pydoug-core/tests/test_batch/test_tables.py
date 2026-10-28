"""
Tests for PyDoug's batch._tables module.
"""

import csv
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from pydoug_core.batch import _tables
from pydoug_core.results import PlotResult, ValueResult


def test_plain() -> None:

    assert isinstance(_tables._plain(np.float64(1.5)), float)
    assert isinstance(_tables._plain({"test": 1}), dict)


def test_is_value_result() -> None:

    assert _tables._is_value_result(1)
    assert _tables._is_value_result({"test": 1, "test2": 1.5})
    assert not _tables._is_value_result([1, 2])


def test_ValueTable_scalar_no_units(tmp_path) -> None:

    table: _tables._ValueTable = _tables._ValueTable()
    table.add("test1", 1)
    table.add("test2", 2)
    assert len(table.rows) == 2
    table.write(Path(tmp_path / "test_VT_scalar_no_units.csv"))
    assert Path(tmp_path / "test_VT_scalar_no_units.csv").exists
    with open(Path(tmp_path / "test_VT_scalar_no_units.csv")) as file:
        reader = csv.reader(file)
        rows: list = []
        for row in reader:
            rows.append(row)
        assert len(rows) == 3
        assert len(rows[0]) == len(rows[1]) == len(rows[2]) == 2
        assert rows[0] == ["dataset", "value"]
        assert rows[1] == ["test1", "1"]
        assert rows[2] == ["test2", "2"]


def test_ValueTable_scalar_units(tmp_path) -> None:

    table: _tables._ValueTable = _tables._ValueTable(units="a.u.")
    table.add("test1", 1)
    table.add("test2", 2)
    assert len(table.rows) == 2
    table.write(Path(tmp_path / "test_VT_scalar_units.csv"))
    assert Path(tmp_path / "test_VT_scalar_units.csv").exists
    with open(Path(tmp_path / "test_VT_scalar_units.csv")) as file:
        reader = csv.reader(file)
        rows: list = []
        for row in reader:
            rows.append(row)
        assert len(rows) == 3
        assert len(rows[0]) == len(rows[1]) == len(rows[2]) == 3
        assert rows[0] == ["dataset", "value", "units"]
        assert rows[1] == ["test1", "1", "a.u."]
        assert rows[2] == ["test2", "2", "a.u."]


def test_ValueTable_dict_no_units(tmp_path) -> None:

    table: _tables._ValueTable = _tables._ValueTable()
    table.add("test1", {"test_val1": 1, "test_val2": 2})
    table.add("test2", {"test_val1": 3, "test_val2": 4})
    assert len(table.rows) == 2
    table.write(Path(tmp_path / "test_VT_dict_no_units.csv"))
    assert Path(tmp_path / "test_VT_dict_no_units.csv").exists
    with open(Path(tmp_path / "test_VT_dict_no_units.csv")) as file:
        reader = csv.reader(file)
        rows: list = []
        for row in reader:
            rows.append(row)
        assert len(rows) == 3
        assert len(rows[0]) == len(rows[1]) == len(rows[2]) == 3
        assert rows[0] == ["dataset", "test_val1", "test_val2"]
        assert rows[1] == ["test1", "1", "2"]
        assert rows[2] == ["test2", "3", "4"]


def test_ValueTable_dict_single_units(tmp_path) -> None:

    table: _tables._ValueTable = _tables._ValueTable(units="a.u.")
    table.add("test1", {"test_val1": 1, "test_val2": 2})
    table.add("test2", {"test_val1": 3, "test_val2": 4})
    assert len(table.rows) == 2
    table.write(Path(tmp_path / "test_VT_dict_single_units.csv"))
    assert Path(tmp_path / "test_VT_dict_single_units.csv").exists
    with open(Path(tmp_path / "test_VT_dict_single_units.csv")) as file:
        reader = csv.reader(file)
        rows: list = []
        for row in reader:
            rows.append(row)
        assert len(rows) == 3
        assert len(rows[0]) == len(rows[1]) == len(rows[2]) == 3
        assert rows[0] == ["dataset", "test_val1 (a.u.)", "test_val2 (a.u.)"]
        assert rows[1] == ["test1", "1", "2"]
        assert rows[2] == ["test2", "3", "4"]


def test_ValueTable_dict_multi_units(tmp_path) -> None:

    table: _tables._ValueTable = _tables._ValueTable(
        units={"test_val1": "a.u.1", "test_val2": "a.u.2"}
    )
    table.add("test1", {"test_val1": 1, "test_val2": 2})
    table.add("test2", {"test_val1": 3, "test_val2": 4})
    assert len(table.rows) == 2
    table.write(Path(tmp_path / "test_VT_dict_multi_units.csv"))
    assert Path(tmp_path / "test_VT_dict_multi_units.csv").exists
    with open(Path(tmp_path / "test_VT_dict_multi_units.csv")) as file:
        reader = csv.reader(file)
        rows: list = []
        for row in reader:
            rows.append(row)
        assert len(rows) == 3
        assert len(rows[0]) == len(rows[1]) == len(rows[2]) == 3
        assert rows[0] == ["dataset", "test_val1 (a.u.1)", "test_val2 (a.u.2)"]
        assert rows[1] == ["test1", "1", "2"]
        assert rows[2] == ["test2", "3", "4"]


def test_BatchTables_handles() -> None:

    table: _tables.BatchTables = _tables.BatchTables()
    table.handles(
        PlotResult(
            Figure(),
            {
                "1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                "3": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
            },
            ("1",),
            {"test": {"test_val1": 1}},
        )
    )
    assert table.handles(ValueResult(2))
    assert table.handles(1)
    assert not table.handles([1, 2])


def test_BatchTables_values(tmp_path) -> None:

    table: _tables.BatchTables = _tables.BatchTables()
    table.add(Path(tmp_path / "test_BT_values_no_units"), "test1", 1)
    table.add(Path(tmp_path / "test_BT_values_units"), "test1", 1, "a.u.")
    table.add(Path(tmp_path / "test_BT_values_no_units"), "test2", 2)
    table.add(Path(tmp_path / "test_BT_values_units"), "test2", 2, "a.u.")
    assert list(table._values.keys())[0] == (
        Path(tmp_path / "test_BT_values_no_units"),
        "values",
    )
    assert list(table._values.keys())[1] == (
        Path(tmp_path / "test_BT_values_units"),
        "values",
    )
    assert (
        len(table._values[(Path(tmp_path / "test_BT_values_no_units"), "values")].rows)
        == 2
    )
    assert (
        len(table._values[(Path(tmp_path / "test_BT_values_units"), "values")].rows)
        == 2
    )

    table.write()
    assert Path(tmp_path / "test_BT_values_no_units/values.csv").exists()
    assert Path(tmp_path / "test_BT_values_units/values.csv").exists()

    with open(Path(tmp_path / "test_BT_values_no_units/values.csv")) as file:
        reader = csv.reader(file)
        rows: list = []
        for row in reader:
            rows.append(row)
        assert len(rows) == 3
        assert len(rows[0]) == len(rows[1]) == len(rows[2]) == 2
        assert rows[0] == ["dataset", "value"]
        assert rows[1] == ["test1", "1"]
        assert rows[2] == ["test2", "2"]

    with open(Path(tmp_path / "test_BT_values_units/values.csv")) as file:
        reader = csv.reader(file)
        rows: list = []
        for row in reader:
            rows.append(row)
        assert len(rows) == 3
        assert len(rows[0]) == len(rows[1]) == len(rows[2]) == 3
        assert rows[0] == ["dataset", "value", "units"]
        assert rows[1] == ["test1", "1", "a.u."]
        assert rows[2] == ["test2", "2", "a.u."]


def test_BatchTables_ValueResult_no_units(tmp_path) -> None:

    table: _tables.BatchTables = _tables.BatchTables()
    table.add(tmp_path, "test1", ValueResult(1))
    table.add(tmp_path, "test2", ValueResult(2))
    assert len(table._values) == 1

    table.write()
    assert Path(tmp_path / "values.csv").exists()

    with open(Path(tmp_path / "values.csv")) as file:
        reader = csv.reader(file)
        rows: list = []
        for row in reader:
            rows.append(row)
        assert len(rows) == 3
        assert len(rows[0]) == len(rows[1]) == len(rows[2]) == 2
        assert rows[0] == ["dataset", "value"]
        assert rows[1] == ["test1", "1"]
        assert rows[2] == ["test2", "2"]


def test_BatchTables_ValueResult_units(tmp_path) -> None:

    table: _tables.BatchTables = _tables.BatchTables()
    table.add(tmp_path, "test1", ValueResult(1, "a.u."))
    table.add(tmp_path, "test2", ValueResult(2, "a.u."))
    assert len(table._values) == 1

    table.write()
    assert Path(tmp_path / "values.csv").exists()

    with open(Path(tmp_path / "values.csv")) as file:
        reader = csv.reader(file)
        rows: list = []
        for row in reader:
            rows.append(row)
        assert len(rows) == 3
        assert len(rows[0]) == len(rows[1]) == len(rows[2]) == 3
        assert rows[0] == ["dataset", "value", "units"]
        assert rows[1] == ["test1", "1", "a.u."]
        assert rows[2] == ["test2", "2", "a.u."]


def test_BatchTables_ValueResult_dict_no_units(tmp_path) -> None:

    table: _tables.BatchTables = _tables.BatchTables()
    table.add(tmp_path, "test1", ValueResult({1: 1, 2: 2}))
    table.add(tmp_path, "test2", ValueResult({1: 3, 2: 4}))
    assert len(table._values) == 1

    table.write()
    assert Path(tmp_path / "values.csv").exists()

    with open(Path(tmp_path / "values.csv")) as file:
        reader = csv.reader(file)
        rows: list = []
        for row in reader:
            rows.append(row)
        assert len(rows) == 3
        assert len(rows[0]) == len(rows[1]) == len(rows[2]) == 3
        assert rows[0] == ["dataset", "1", "2"]
        assert rows[1] == ["test1", "1", "2"]
        assert rows[2] == ["test2", "3", "4"]


def test_BatchTables_ValueResult_dict_one_unit(tmp_path) -> None:

    table: _tables.BatchTables = _tables.BatchTables()
    table.add(tmp_path, "test1", ValueResult({1: 1, 2: 2}, "a.u."))
    table.add(tmp_path, "test2", ValueResult({1: 3, 2: 4}, "a.u."))
    assert len(table._values) == 1

    table.write()
    assert Path(tmp_path / "values.csv").exists()

    with open(Path(tmp_path / "values.csv")) as file:
        reader = csv.reader(file)
        rows: list = []
        for row in reader:
            rows.append(row)
        assert len(rows) == 3
        assert len(rows[0]) == len(rows[1]) == len(rows[2]) == 3
        assert rows[0] == ["dataset", "1 (a.u.)", "2 (a.u.)"]
        assert rows[1] == ["test1", "1", "2"]
        assert rows[2] == ["test2", "3", "4"]


def test_BatchTables_ValueResult_dict_units(tmp_path) -> None:

    table: _tables.BatchTables = _tables.BatchTables()
    table.add(tmp_path, "test1", ValueResult({1: 1, 2: 2}, {1: "a.u.1", 2: "a.u.2"}))
    table.add(tmp_path, "test2", ValueResult({1: 3, 2: 4}, {1: "a.u.1", 2: "a.u.2"}))
    assert len(table._values) == 1

    table.write()
    assert Path(tmp_path / "values.csv").exists()

    with open(Path(tmp_path / "values.csv")) as file:
        reader = csv.reader(file)
        rows: list = []
        for row in reader:
            rows.append(row)
        assert len(rows) == 3
        assert len(rows[0]) == len(rows[1]) == len(rows[2]) == 3
        assert rows[0] == ["dataset", "1 (a.u.1)", "2 (a.u.2)"]
        assert rows[1] == ["test1", "1", "2"]
        assert rows[2] == ["test2", "3", "4"]


def test_BatchTables_plots_x_cols(tmp_path) -> None:

    table: _tables.BatchTables = _tables.BatchTables()
    table.add(
        tmp_path,
        "test1",
        PlotResult(
            Figure(),
            {
                "1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            },
            ("1",),
            {"test_table": {"test_val": 2}},
        ),
    )
    table.add(
        tmp_path,
        "test2",
        PlotResult(
            Figure(),
            {
                "1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "2": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
            },
            ("1",),
            {"test_table": {"test_val": 3}},
        ),
    )
    assert len(table._values) == 1
    assert len(table._plots) == 1
    assert list(table._values.keys())[0] == (tmp_path, "test_table")
    assert len(table._values[(tmp_path, "test_table")].rows) == 2
    assert Path(tmp_path / "test1.png").exists()
    assert Path(tmp_path / "test2.png").exists()

    table.write()
    assert Path(tmp_path / "plot_data.csv").exists()
    assert Path(tmp_path / "test_table.csv").exists()

    with open(Path(tmp_path / "test_table.csv")) as file:
        reader = csv.reader(file)
        rows: list = []
        for row in reader:
            rows.append(row)
        assert len(rows) == 3
        assert len(rows[0]) == len(rows[1]) == 2
        assert rows[0] == ["dataset", "test_val"]
        assert rows[1] == ["test1", "2"]
        assert rows[2] == ["test2", "3"]

    with open(Path(tmp_path / "plot_data.csv")) as file:
        reader = csv.reader(file)
        rows: list = []
        for row in reader:
            rows.append(row)
        assert len(rows) == 11
        assert len(rows[0]) == len(rows[1]) == 3
        assert rows[0] == ["1", "test1", "test2"]
        assert rows[1] == ["1", "2", "3"]
        assert rows[2] == ["2", "4", "6"]


def test_BatchTables_plots_no_x_cols(tmp_path) -> None:

    table: _tables.BatchTables = _tables.BatchTables()
    table.add(
        tmp_path,
        "test1",
        PlotResult(
            Figure(),
            {
                "1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            },
            (),
            {"test_table": {"test_val": 2}},
        ),
    )
    table.add(
        tmp_path,
        "test2",
        PlotResult(
            Figure(),
            {
                "1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "2": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
            },
            (),
            {"test_table": {"test_val": 3}},
        ),
    )
    assert len(table._values) == 1
    assert len(table._plots) == 0
    assert Path(tmp_path / "test1.png").exists()
    assert Path(tmp_path / "test2.png").exists()

    table.write()
    assert Path(tmp_path / "test1.csv").exists()
    assert Path(tmp_path / "test2.csv").exists()
    assert Path(tmp_path / "test_table.csv").exists()

    with open(Path(tmp_path / "test_table.csv")) as file:
        reader = csv.reader(file)
        rows: list = []
        for row in reader:
            rows.append(row)
        assert len(rows) == 3
        assert len(rows[0]) == len(rows[1]) == 2
        assert rows[0] == ["dataset", "test_val"]
        assert rows[1] == ["test1", "2"]
        assert rows[2] == ["test2", "3"]

    with open(Path(tmp_path / "test1.csv")) as file:
        reader = csv.reader(file)
        rows: list = []
        for row in reader:
            rows.append(row)
        assert len(rows) == 11
        assert len(rows[0]) == len(rows[1]) == 2
        print(rows)
        assert rows[0] == ["1", "2"]
        assert rows[1] == ["1", "2"]
        assert rows[2] == ["2", "4"]

    with open(Path(tmp_path / "test2.csv")) as file:
        reader = csv.reader(file)
        rows: list = []
        for row in reader:
            rows.append(row)
        assert len(rows) == 11
        assert len(rows[0]) == len(rows[1]) == 2
        assert rows[0] == ["1", "2"]
        assert rows[1] == ["1", "3"]
        assert rows[2] == ["2", "6"]

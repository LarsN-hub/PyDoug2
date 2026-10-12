"""
Tables for storing and writing plot and value data.
"""

import csv
from pathlib import Path
from typing import Any

import numpy as np

from pydoug_core.results import SCALAR_TYPES, PlotResult


def _plain(value: Any) -> Any:
    """
    Remove the numpy datatype from a value if it is generic numpy instance, otherwise
    return the value.

    Parameters
    ----------
    value: Any
        The value to de-numpy.

    Returns
    -------
    dnpy_val: Any
        The de-numpyd value.
    """

    return value.item() if isinstance(value, np.generic) else value


def _is_value_result(result: Any) -> bool:
    """
    Return whether or not a result or (its values) is/are in SCALAR_TYPES.

    Parameters
    ----------
    result: Any
        The result to type validate.

    Returns
    -------
    is_value_result: bool
        True if in SCALAR_TYPES, else False.
    """

    if isinstance(result, SCALAR_TYPES):
        return True
    return isinstance(result, dict) and all(
        isinstance(v, SCALAR_TYPES) for v in result.values()
    )


class _ValueTable:
    """
    Table containing rows of data.

    Attributes
    ----------
    units: str | dict[str, str] | None
        The units for the data.
    """

    def __init__(self, units: str | dict[str, str] | None = None) -> None:

        self.units = units
        self.rows: list[tuple[str, dict]] = []

    def add(self, dataset: str, result: Any) -> None:
        """
        Add a result to the rows of _ValueTable.

        Parameters
        ----------
        dataset: str
            The name of the dataset.
        result: Any
            The result to add.
        """

        self.rows.append(
            (dataset, result if isinstance(result, dict) else {"value": result})
        )

    def _header(self, key: str) -> str:
        """
        Create the header for a csv column with units if present.

        Parameters
        ----------
        key: str
            The name of the column's data.

        Returns
        -------
        header_name: str
            The header for the column.
        """

        unit: str | None = (
            self.units.get(key) if isinstance(self.units, dict) else self.units
        )
        return f"{key} ({unit})" if unit else key

    def write(self, path: Path) -> None:
        """
        Write a _ValueTable's rows to a csv file.

        path: Path
            The path to the csv file.
        """

        keys: list = list(dict.fromkeys(k for _, row in self.rows for k in row))
        scalar: bool = keys == ["value"]
        unit: str = self.units if scalar and isinstance(self.units, str) else ""
        with open(path, "w", newline="") as file:
            writer = csv.writer(file)
            if scalar:
                writer.writerow(["dataset", "value"] + (["units"] if unit else []))
                for dataset, row in self.rows:
                    writer.writerow(
                        [dataset, _plain(row["value"])] + ([unit] if unit else [])
                    )
            else:
                writer.writerow(["dataset"] + [self._header(k) for k in keys])
                for dataset, row in self.rows:
                    writer.writerow([dataset] + [_plain(row.get(k, "")) for k in keys])


class BatchTables:
    """
    Collects per-dataset value and plot-data results so each step writes one file.

    Attributes
    ----------
    _values: dict[tuple[Path, str], _ValueTable]
        A dictionary with a tuple containing (the path to the save folder and a string
        with the filename) as the keys and _ValueTables as the values.
    _plots: dict[Path, list[dict]]
        A dictionary containing the path to the save folder as the keys and a list of
        dictionaries as the values.
    """

    def __init__(self) -> None:

        self._values: dict[tuple[Path, str], _ValueTable] = {}
        self._plots: dict[Path, list[dict]] = {}

    @staticmethod
    def handles(result: Any) -> bool:
        """
        Return whether or not a result is a PlotResult (containing a Figure and data)
        or a valid type in SCALAR_TYPES.

        Parameters
        ----------
        result: Any
            The result to evaluate.

        Returns
        -------
        bool_res: bool
            True if the result is a PlotResult or in SCALAR_TYPES, else False.
        """

        return isinstance(result, PlotResult) or _is_value_result(result)

    def _add_table(
        self,
        folder: Path,
        name: str,
        dataset: str,
        table: dict[str, dict],
        units: str | dict[str, str] | None,
    ) -> None:
        """
        Add data to the rows of a _ValueTable object.

        Parameters
        ----------
        folder: Path
            The path to the save folder.
        name: str
            The name of the file.
        dataset: str
            The name of the dataset.
        table: dict[str, dict]
            The table from the PlotResult.
        units: str | dict[str, str] | None
            The units for the data.
        """

        self._values.setdefault((folder, name), _ValueTable(units)).add(dataset, table)

    def _add_plot(
        self,
        folder: Path,
        dataset: str,
        result: PlotResult,
        units: str | dict[str, str] | None = None,
    ) -> None:
        """
        Save a plot from a PlotResult and add its data tables to the _ValueTable object.
        Leave out the x-axis if not needed.

        Parameters
        ----------
        folder: Path
            The path to the save folder.
        dataset: str
            The name of the dataset.
        result: PlotResult
            The PlotResult object containing a Figure and data to be written.
        units: str | dict[str, str] | None
            The units for the data.
        """

        result.figure.savefig(folder / f"{dataset}.png")
        for name, table in result.tables.items():
            self._add_table(folder, name, dataset, table, units)

        if not result.x_columns:
            result.write_csv(folder / f"{dataset}.csv")
            return

        x: dict[str, np.ndarray] = {
            c: np.asarray(result.data[c]) for c in result.x_columns
        }
        y: dict[str, np.ndarray] = {
            c: np.asarray(v) for c, v in result.data.items() if c not in x
        }
        groups: list[dict] = self._plots.setdefault(folder, [])
        for group in groups:
            if group["x"].keys() == x.keys() and all(
                np.array_equal(group["x"][c], x[c]) for c in x
            ):
                group["y"][dataset] = y
                return
        groups.append({"x": x, "y": {dataset: y}})

    def add(
        self,
        folder: Path,
        dataset: str,
        result: Any,
        units: str | dict[str, str] | None = None,
    ) -> None:
        """
        Add plot and tabular data to the BatchTables to save.

        Parameters
        ----------
        folder: Path
            The path to the save folder.
        dataset: str
            The name of the dataset.
        result: Any
            The result to be saved.
        units: str | dict[str, str] | None
            The units for the data.
        """

        if isinstance(result, PlotResult):
            self._add_plot(folder, dataset, result, units)
        else:
            self._add_table(folder, "values", dataset, result, units)

    def write(self) -> None:
        """
        Write the data in _values and _plots.
        """

        for (folder, name), table in self._values.items():
            folder.mkdir(parents=True, exist_ok=True)
            table.write(folder / f"{name}.csv")
        for folder, groups in self._plots.items():
            folder.mkdir(parents=True, exist_ok=True)
            for number, group in enumerate(groups, start=1):
                path = folder / (
                    "plot_data.csv" if number == 1 else f"plot_data_{number}.csv"
                )
                headers = list(group["x"])
                columns = [group["x"][c].tolist() for c in group["x"]]
                for dataset, y in group["y"].items():
                    for column, values in y.items():
                        headers.append(
                            dataset if len(y) == 1 else f"{dataset} {column}"
                        )
                        columns.append(values.tolist())
                with open(path, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(headers)
                    writer.writerows(zip(*columns))

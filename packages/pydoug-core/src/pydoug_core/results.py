"""
Class for obtaining plot and value results from operations.
"""

import csv
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

SCALAR_TYPES: tuple = (int, float, str, bool, np.generic)


@dataclass
class PlotResult:
    """
    A plot plus the data it was drawn from, saved together in batch runs.

    Attributes
    ----------
    figure: Figure
        The plot figure.
    data: dict[str, Sequence]
        The data used to make the plot.
    x_columns: tuple[str, ...]
        The x-axis column(s) for the plot. Allows writing to same csv file without
        duplicating the x-axis column. If not specified, a unique file is written for
        each dataset.
    tables: dict[str, dict]
        Extra scalar value tables of data accompanying the plot data as desired.
    """

    figure: Figure
    data: dict[str, Sequence]
    x_columns: tuple[str, ...] = ()
    tables: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:

        lengths: dict[str, int] = {
            name: len(values) for name, values in self.data.items()
        }
        if len(set(lengths.values())) > 1:
            raise ValueError(f"All data columns must be the same length, got {lengths}")

        missing: set = set(self.x_columns) - set(self.data)
        if missing:
            raise ValueError(f"x_columns not found in data: {sorted(missing)}")

        for name, table in self.tables.items():
            if name == "plot_data" or name.startswith("plot_data_"):
                raise ValueError(f"Table name '{name}' is reserved for plot data files")
            if not all(isinstance(v, SCALAR_TYPES) for v in table.values()):
                raise ValueError(
                    f"Table '{name}' may only contain single numbers or text"
                )

    def write_csv(self, path: Path) -> None:
        """
        Write the figure's data to a csv file.

        Parameters
        ----------
        path: Path
            The path to the csv file.
        """

        columns: dict[str, list[bool]] = {
            name: np.asarray(values).tolist() for name, values in self.data.items()
        }
        with open(path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(list(columns))
            writer.writerows(zip(*columns.values()))

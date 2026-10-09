"""
Class for obtaining plot and value results from operations.
"""

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


@dataclass
class PlotResult:
    """
    A plot plus the data it was drawn from, saved together in batch runs.
    """

    figure: Figure
    data: dict[str, Sequence]  # column name -> values, all columns the same length

    def __post_init__(self) -> None:
        lengths: dict[str, int] = {
            name: len(values) for name, values in self.data.items()
        }
        if len(set(lengths.values())) > 1:
            raise ValueError(f"All data columns must be the same length, got {lengths}")

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

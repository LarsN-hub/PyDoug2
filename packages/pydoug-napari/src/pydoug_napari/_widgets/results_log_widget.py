"""
Widget for displaying values from operations for PyDoug's napari plugin.
"""

from numpy import ndarray
from psygnal import Signal
from qtpy.QtWidgets import QListWidget


class ResultsLog:
    """
    Log of non-image layer results.

    Attributes
    ----------
    entries: list[tuple[str, object]]
        Entries to the log.
    """

    entry_added: Signal = Signal(str, object)  # (operation display name, result)

    def __init__(self):
        self.entries: list[tuple[str, object]] = []

    def add(self, label: str, result: dict | float | int | str | ndarray) -> None:
        """
        Add an item to the results log.

        Parameters
        ----------
        label: str
            The label for the result.
        result: dict[str, float | int | str] | float | int | str
            The result.
        """

        self.entries.append((label, result))
        self.entry_added.emit(label, result)


class ResultsLogWidget(QListWidget):
    """
    The widget for the results log.

    Attributes
    ----------
    results_log: ResultsLog
        The results log object.
    """

    def __init__(self, results_log: ResultsLog) -> None:
        super().__init__()
        results_log.entry_added.connect(self._on_entry)

    def _on_entry(self, label: str, result: dict | float | int | str | ndarray) -> None:
        """
        How to add the result to the results log.

        Parameters
        ----------
        label: str
            The label for the result.
        result: dict[str, float | int | str] | float | int | str
            The operation's result.
        """

        if isinstance(result, dict):
            text: str = "\n".join(f"  {k}: {v}" for k, v in result.items())
        else:  # pragma: no cover
            text = str(result)
        self.addItem(f"{label}\n{text}")

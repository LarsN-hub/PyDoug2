"""
Widget for displaying values from operations for PyDoug's napari plugin.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from napari.utils.notifications import show_error, show_info
from psygnal import Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pydoug_core.batch._tables import BatchTables


def _safe_name(label: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", label).strip() or "results"


@dataclass
class LogEntry:
    """
    Entry for the results log with data recorded as needed for exporting.

    Attributes
    ----------
    label: str
        The result label.
    display: object
        What the list shows.
    result: object | None
        The exportable result; None means display-only.
    dataset: str
        Source layer name, used as the dataset name on export.
    units: object | None
        The operation's registered default units.
    """

    label: str
    display: object
    result: object | None = None
    dataset: str = ""
    units: object | None = None


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
        self.entries: list[LogEntry] = []

    def add(
        self,
        label: str,
        display: object,
        result: object | None = None,
        dataset: str = "",
        units: object | None = None,
    ) -> None:
        """
        Add an item to the results log.

        Parameters
        ----------
        label: str
            The result label.
        display: object
            What the list shows.
        result: object | None
            The exportable result; None means display-only.
        dataset: str
            Source layer name, used as the dataset name on export.
        units: object | None
            The operation's registered default units.
        """

        entry = LogEntry(label, display, result, dataset or label, units)
        self.entries.append(entry)
        self.entry_added.emit(entry)


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
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        results_log.entry_added.connect(self._on_entry)

    def _on_entry(self, entry: LogEntry) -> None:
        """
        How to add the result to the results log.

        Parameters
        ----------
        entry: LogEntry
            The LogEntry object being entered.
        """

        display: object = entry.display
        text = (
            "\n".join(f"  {k}: {v}" for k, v in display.items())
            if isinstance(display, dict)
            else str(display)
        )
        self.addItem(f"{entry.label}\n{text}")


class ResultsPanel(QWidget):
    """
    The results list widget and buttons that export the data behind its entries.

    Attributes
    ----------
    results_log: ResultsLog
        The ResultsLog object.
    log_widget: ResultsLogWidget
        The ResultsLogWidget object.
    """

    def __init__(self, results_log: ResultsLog, log_widget: ResultsLogWidget) -> None:
        super().__init__()
        self.results_log = results_log
        self.log_widget = log_widget
        self.export_selected_button: QPushButton = QPushButton("Export Selected...")
        self.export_all_button: QPushButton = QPushButton("Export All...")
        self.export_selected_button.clicked.connect(self._export_selected)
        self.export_all_button.clicked.connect(self._export_all)
        buttons: QHBoxLayout = QHBoxLayout()
        buttons.addWidget(self.export_selected_button)
        buttons.addWidget(self.export_all_button)
        layout: QVBoxLayout = QVBoxLayout(self)
        layout.addWidget(self.log_widget)
        layout.addLayout(buttons)

    def _export_selected(self) -> None:
        """
        Export the selected items using _export().
        """

        rows: list[int] = sorted(
            {index.row() for index in self.log_widget.selectedIndexes()}
        )
        self._export([self.results_log.entries[row] for row in rows])

    def _export_all(self) -> None:
        """
        Export all items using _export().
        """

        self._export(list(self.results_log.entries))

    def _export(self, entries: list[LogEntry]) -> None:
        """
        Export the given entries.

        Parameters
        ----------
        entries: list[LogEntry]
            The entries to be exported.
        """

        exportable: list[LogEntry] = [
            e for e in entries if e.result is not None and BatchTables.handles(e.result)
        ]

        if not exportable:  # pragma: no cover
            show_error("None of the selected entries have data to export.")
            return
        folder: str = QFileDialog.getExistingDirectory(self, "Export results to folder")
        if not folder:  # pragma: no cover
            return

        tables: BatchTables = BatchTables(save_figures=False)
        seen: dict[tuple[Path, str], int] = {}
        try:
            for entry in exportable:
                step_folder: Path = Path(folder) / _safe_name(entry.label)
                count: int = seen.get((step_folder, entry.dataset), 0) + 1
                seen[(step_folder, entry.dataset)] = count
                dataset: object = (
                    entry.dataset if count == 1 else f"{entry.dataset} ({count})"
                )
                tables.add(step_folder, dataset, entry.result, entry.units)  # type: ignore
            tables.write()
        except ValueError as exc:  # pragma: no cover
            show_error(f"Export failed: {exc}")
            return

        show_info(f"Exported {len(exportable)} result(s) to {folder}")

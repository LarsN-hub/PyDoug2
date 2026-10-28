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
from pydoug_core.batch.recorder import Recorder


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
    number: int
        Assigned by the log to prevent overwriting when saving similarly named steps.
    step_id: str | None
        The recorded workflow step this came from, if any.
    """

    label: str
    display: object
    result: object | None = None
    dataset: str = ""
    units: object | None = None
    number: int = 0
    step_id: str | None = None


class ResultsLog:
    """
    Log of non-image layer results.

    Attributes
    ----------
    entries: list[tuple[str, object]]
        Entries to the log.
    _count: int
        The unique index for this result.
    """

    entry_added: Signal = Signal(str, object)  # (operation display name, result)
    entries_removed = Signal()

    def __init__(self):
        self.entries: list[LogEntry] = []
        self._count: int = 0

    def add(
        self,
        label: str,
        display: object,
        result: object | None = None,
        dataset: str = "",
        units: object | None = None,
        step_id: str | None = None,
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
        step_id: str | None
            The recorded workflow step this came from, if any.
        """

        self._count += 1
        entry: LogEntry = LogEntry(
            label, display, result, dataset or label, units, self._count, step_id
        )
        self.entries.append(entry)
        self.entry_added.emit(entry)

    def remove(self, doomed: list[LogEntry]) -> None:
        """
        Remove a step from the results log.

        Parameters
        ----------
        doomed: list[LogEntry]
            The LogEntry objects to be removed.
        """

        self.entries = [e for e in self.entries if e not in doomed]
        self.entries_removed.emit()


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
        self._results_log: ResultsLog = results_log
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        results_log.entry_added.connect(self._on_entry)
        results_log.entries_removed.connect(self._rebuild)

    def _on_entry(self, entry: LogEntry) -> None:
        """
        How to add the result to the results log.

        Parameters
        ----------
        entry: LogEntry
            The LogEntry object being entered.
        """

        display: object = entry.display
        text: str = (
            "\n".join(f"  {k}: {v}" for k, v in display.items())
            if isinstance(display, dict)
            else str(display)
        )
        self.addItem(f"{entry.number}. {entry.label}\n{text}")

    def _rebuild(self) -> None:
        """
        Rebuild a results log (used after deleting entries).
        """

        self.clear()
        for entry in self._results_log.entries:
            self._on_entry(entry)  # pragma: no cover


class ResultsPanel(QWidget):
    """
    The results list widget and buttons that export the data behind its entries.

    Attributes
    ----------
    results_log: ResultsLog
        The ResultsLog object.
    log_widget: ResultsLogWidget
        The ResultsLogWidget object.
    recorder: Recorder
        The workflow Recorder object.
    """

    def __init__(
        self,
        results_log: ResultsLog,
        log_widget: ResultsLogWidget,
        recorder: Recorder,
    ) -> None:
        super().__init__()
        self.results_log: ResultsLog = results_log
        self.log_widget: ResultsLogWidget = log_widget
        self.recorder: Recorder = recorder
        self.export_selected_button: QPushButton = QPushButton("Export Selected...")
        self.export_all_button: QPushButton = QPushButton("Export All...")
        self.delete_button = QPushButton("Delete Selected")
        self.export_selected_button.clicked.connect(self._export_selected)
        self.export_all_button.clicked.connect(self._export_all)
        self.delete_button.clicked.connect(self._delete_selected)
        buttons: QHBoxLayout = QHBoxLayout()
        buttons.addWidget(self.export_selected_button)
        buttons.addWidget(self.export_all_button)
        buttons.addWidget(self.delete_button)
        layout: QVBoxLayout = QVBoxLayout(self)
        layout.addWidget(self.log_widget)
        layout.addLayout(buttons)

    def _selected_entries(self) -> list[LogEntry]:
        """
        Return the currently selected entries.

        Returns
        -------
        sel_entries: list[LogEntry]
            The list of selected LogEntry objects.
        """

        rows: list[int] = sorted(
            {index.row() for index in self.log_widget.selectedIndexes()}
        )
        return [self.results_log.entries[row] for row in rows]

    def _export_selected(self) -> None:
        """
        Export the selected items using _export().
        """

        self._export(self._selected_entries())

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
        for entry in exportable:
            save_name: str = f"{entry.number:02d}_{_safe_name(entry.label)}"
            step_folder: Path = Path(folder) / save_name
            tables.add(step_folder, entry.dataset, entry.result, entry.units)  # type: ignore
        tables.write()

        show_info(f"Exported {len(exportable)} result(s) to {folder}")

    def _delete_selected(self) -> None:
        """
        Delete the selected entries from the log.
        """

        selected: list[LogEntry] = self._selected_entries()
        if not selected:  # pragma: no cover
            return
        step_ids: set[str] = {e.step_id for e in selected if e.step_id is not None}
        doomed: list[LogEntry] = [
            e
            for e in self.results_log.entries
            if e in selected or e.step_id in step_ids
        ]
        for step_id in step_ids:
            self.recorder.remove_step(step_id)
        self.results_log.remove(doomed)

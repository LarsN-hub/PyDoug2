"""
Multi-folder selector dialog widget for PyDoug's napari plugin.
"""

from pathlib import Path

from qtpy.QtCore import QDir, QModelIndex
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileSystemModel,  # type: ignore
    QHBoxLayout,
    QPushButton,
    QTreeView,
    QVBoxLayout,
)


class MultiFolderDialog(QDialog):
    """
    Folder picker supporting multi-selection.

    Attributes
    ----------
    start_dir: str | None
        The starting directory.
    """

    def __init__(self, parent=None, start_dir: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Select Image Sequence Folders")
        self.resize(600, 400)

        layout: QVBoxLayout = QVBoxLayout(self)
        self.model: QFileSystemModel = QFileSystemModel()
        self.model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot)  # type: ignore
        self.model.setRootPath("")

        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(""))
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)  # type: ignore
        self.tree.setHeaderHidden(True)
        for i in range(1, 4):
            self.tree.hideColumn(i)
        layout.addWidget(self.tree)

        button_row: QHBoxLayout = QHBoxLayout()
        select_button: QPushButton = QPushButton("Select")
        select_button.clicked.connect(self.accept)
        cancel_button: QPushButton = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(select_button)
        button_row.addWidget(cancel_button)
        layout.addLayout(button_row)

    def selected_folders(self) -> list[Path]:
        """
        Return a list of selected filepaths.

        Returns
        -------
        file_list: list[Path]
            A list of file paths.
        """

        indexes: list[QModelIndex] = self.tree.selectionModel().selectedRows()
        return [Path(self.model.filePath(i)) for i in indexes]

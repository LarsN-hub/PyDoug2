"""
Tests for PyDoug's multi_folder_dialog_widget napari plugin module.
"""

from qtpy.QtWidgets import QWidget

from pydoug_napari._widgets.multi_folder_dialog_widget import MultiFolderDialog


def test_MultiFolderDialog(tmp_path, sample_2dgray_array) -> None:

    container: QWidget = QWidget()
    mfdialog: MultiFolderDialog = MultiFolderDialog(container)
    result: list = mfdialog.selected_folders()
    assert len(result) == 0

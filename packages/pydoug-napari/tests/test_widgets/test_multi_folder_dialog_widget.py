"""
Tests for PyDoug's multi_folder_dialog_widget napari plugin module.
"""

import os
from pathlib import Path

import imageio.v3 as iio
from qtpy.QtWidgets import QWidget

from pydoug_napari._widgets.multi_folder_dialog_widget import MultiFolderDialog


def test_MultiFolderDialog(tmp_path, sample_2dgray_array) -> None:

    os.mkdir(Path(tmp_path / "mfdialog_test"))
    os.mkdir(Path(tmp_path / "mfdialog_test/seq1"))
    os.mkdir(Path(tmp_path / "mfdialog_test/seq2"))
    iio.imwrite(Path(tmp_path / "mfdialog_test/seq1/test1.tiff"), sample_2dgray_array)
    iio.imwrite(Path(tmp_path / "mfdialog_test/seq1/test2.tiff"), sample_2dgray_array)
    iio.imwrite(Path(tmp_path / "mfdialog_test/seq2/test1.tiff"), sample_2dgray_array)
    iio.imwrite(Path(tmp_path / "mfdialog_test/seq2/test2.tiff"), sample_2dgray_array)
    container: QWidget = QWidget()
    mfdialog: MultiFolderDialog = MultiFolderDialog(container)
    result: list = mfdialog.selected_folders()
    assert len(result) == 0

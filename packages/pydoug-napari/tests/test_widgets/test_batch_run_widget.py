"""
Tests for PyDoug's napari plugin widgets._batch_run_widget module.
"""

import os
from pathlib import Path

from imageio.v3 import imwrite as iiowrite
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QWidget

from pydoug_core.batch.recorder import Recorder
from pydoug_napari._widgets import _batch_run_widget as batch_run_widget
from pydoug_napari._widgets.multi_folder_dialog_widget import MultiFolderDialog


def test_build_run_batch_widget_select_files_and_remove_selected(
    qtbot, monkeypatch
) -> None:

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileNames",
        lambda *a, **k: (["/fake/img1.tif", "/fake/img2.tif"], ""),
    )
    widget: QWidget = batch_run_widget.build_run_batch_widget()
    qtbot.addWidget(widget)
    qtbot.mouseClick(widget.add_files_btn, Qt.MouseButton.LeftButton)  # type: ignore
    assert widget.image_list.count() == 2  # type: ignore
    assert Path("/fake/img1.tif") in widget.image_paths  # type: ignore
    current_row: int = list(widget.image_paths).index(  # type: ignore
        Path("/fake/img1.tif")
    )
    widget.image_list.setCurrentRow(current_row)  # type: ignore
    qtbot.mouseClick(
        widget.remove_selected_btn,  # type: ignore
        Qt.MouseButton.LeftButton,
    )
    assert widget.image_list.count() == 1  # type: ignore
    assert Path("/fake/img1.tif") not in widget.image_paths  # type: ignore


def test_build_run_batch_widget_select_folders_and_clear_list(
    qtbot, monkeypatch
) -> None:

    monkeypatch.setattr(MultiFolderDialog, "exec", lambda self: 1)
    monkeypatch.setattr(
        MultiFolderDialog,
        "selected_folders",
        lambda self: [Path("/fake/seq1"), Path("/fake/seq2")],
    )
    widget: QWidget = batch_run_widget.build_run_batch_widget()
    qtbot.addWidget(widget)
    qtbot.mouseClick(
        widget.select_folders_btn,  # type: ignore
        Qt.MouseButton.LeftButton,
    )
    assert widget.image_list.count() == 2  # type: ignore
    assert Path("/fake/seq1") in widget.image_paths  # type: ignore
    qtbot.mouseClick(widget.clear_list_btn, Qt.MouseButton.LeftButton)  # type: ignore
    assert widget.image_list.count() == 0  # type: ignore


def test_build_run_batch_widget_select_folders_cancel_adds_nothing(
    qtbot, monkeypatch
) -> None:

    monkeypatch.setattr(MultiFolderDialog, "exec", lambda self: 0)
    widget: QWidget = batch_run_widget.build_run_batch_widget()
    qtbot.addWidget(widget)
    qtbot.mouseClick(
        widget.select_folders_btn,  # type: ignore
        Qt.MouseButton.LeftButton,
    )
    assert widget.image_list.count() == 0  # type: ignore


def test_build_run_batch_widget_workflow_and_output_folders(
    qtbot, monkeypatch, tmp_path
) -> None:

    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
    )
    widget: QWidget = batch_run_widget.build_run_batch_widget()
    qtbot.addWidget(widget)
    qtbot.mouseClick(
        widget.workflow_pick_btn,  # type: ignore
        Qt.MouseButton.LeftButton,
    )
    assert widget.workflow_edit.text() == str(tmp_path)  # type: ignore
    qtbot.mouseClick(
        widget.output_pick_btn,  # type: ignore
        Qt.MouseButton.LeftButton,
    )
    assert widget.output_edit.text() == str(tmp_path)  # type: ignore


def test_build_run_batch_widget_run(
    qtbot,
    tmp_path,
    sample_2dgray_array,
    monkeypatch,
    registered_threshold,
) -> None:

    record: Recorder = Recorder()
    record.record("_threshold", {}, {"layer": "primary"}, None)
    record.save(Path(tmp_path / "test_batch_run_workflow"))
    widget: QWidget = batch_run_widget.build_run_batch_widget()
    qtbot.addWidget(widget)
    widget.workflow_edit.setText(  # type: ignore
        str(Path(tmp_path / "test_batch_run_workflow"))
    )
    os.mkdir(Path(tmp_path / "test_batch_run_output1"))
    os.mkdir(Path(tmp_path / "test_batch_run_import"))
    iiowrite(Path(tmp_path / "test_batch_run_import/test_im.tiff"), sample_2dgray_array)
    widget.output_edit.setText(  # type: ignore
        str(Path(tmp_path / "test_batch_run_output1"))
    )
    widget.image_paths.append(  # type: ignore
        Path(tmp_path / "test_batch_run_import/test_im.tiff")
    )
    widget.image_list.addItem(  # type: ignore
        str(Path(tmp_path / "test_batch_run_import/test_im.tiff"))
    )
    assert Path(tmp_path / "test_batch_run_import/test_im.tiff") in widget.image_paths  # type: ignore
    assert widget.image_list.count() == 1  # type: ignore
    assert widget.output_edit.text() == str(  # type: ignore
        Path(tmp_path / "test_batch_run_output1")
    )
    assert widget.workflow_edit.text() == str(  # type: ignore
        Path(tmp_path / "test_batch_run_workflow")
    )

    qtbot.mouseClick(widget.run_btn, Qt.MouseButton.LeftButton)  # type: ignore
    qtbot.waitUntil(
        lambda: Path(tmp_path / "test_batch_run_output1/test_im.tiff").exists(),
        timeout=5000,
    )
    assert Path(tmp_path / "test_batch_run_output1/test_im.tiff").exists()

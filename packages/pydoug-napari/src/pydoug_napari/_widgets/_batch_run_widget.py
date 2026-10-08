"""
The batch script mechanics for the batch widget of PyDoug's napari plugin.
"""

from pathlib import Path

from qtpy.QtWidgets import QListWidget, QVBoxLayout, QWidget


def build_run_batch_widget() -> QWidget:
    container: QWidget = QWidget()
    layout: QVBoxLayout = QVBoxLayout()
    image_paths: list[Path] = []
    image_list: QListWidget = QListWidget()

    from napari.qt.threading import GeneratorWorker, thread_worker
    from napari.utils.notifications import show_error, show_info
    from qtpy.QtWidgets import (
        QCheckBox,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QProgressBar,
        QPushButton,
    )

    from pydoug_core.batch.runner import run_batch
    from pydoug_napari._widgets.multi_folder_dialog_widget import MultiFolderDialog

    def _select_files():
        files, _ = QFileDialog.getOpenFileNames(container, "Select image files")
        _add_paths(Path(f) for f in files)

    def _select_folders():
        dialog: MultiFolderDialog = MultiFolderDialog(container)
        if dialog.exec():
            _add_paths(dialog.selected_folders())

    def _add_paths(paths):
        for p in paths:
            if p not in image_paths:
                image_paths.append(p)
                image_list.addItem(str(p))

    def _remove_selected():
        for item in sorted(
            image_list.selectedItems(), key=image_list.row, reverse=True
        ):
            idx: int = image_list.row(item)
            image_list.takeItem(idx)
            del image_paths[idx]

    def _clear_list():
        image_list.clear()
        image_paths.clear()

    btn_row: QHBoxLayout = QHBoxLayout()
    buttons: dict[str, QPushButton] = {}
    for label, handler in [
        ("Add File(s)...", _select_files),
        ("Add Folder (sequence)...", _select_folders),
        ("Remove Selected", _remove_selected),
        ("Clear List", _clear_list),
    ]:
        btn: QPushButton = QPushButton(label)
        btn.clicked.connect(handler)
        btn_row.addWidget(btn)
        buttons[label] = btn

    layout.addWidget(QLabel("Images to process:"))
    layout.addLayout(btn_row)
    layout.addWidget(image_list)

    workflow_edit, output_edit = QLineEdit(), QLineEdit()
    dir_buttons: dict[str, QPushButton] = {}
    for label_text, edit, dialog_title in [
        ("Workflow folder:", workflow_edit, "Select workflow folder"),
        ("Output folder:", output_edit, "Select output folder"),
    ]:
        layout.addWidget(QLabel(label_text))
        row: QHBoxLayout = QHBoxLayout()
        row.addWidget(edit)
        pick_btn: QPushButton = QPushButton("Browse...")
        pick_btn.clicked.connect(
            lambda _, e=edit, t=dialog_title: e.setText(
                QFileDialog.getExistingDirectory(container, t) or e.text()
            )
        )
        row.addWidget(pick_btn)
        layout.addLayout(row)
        dir_buttons[label_text] = pick_btn

    save_every_step_checkbox: QCheckBox = QCheckBox(
        "Save every image step (not just final image)"
    )
    save_images_checkbox: QCheckBox = QCheckBox("Save images")
    layout.addWidget(save_every_step_checkbox)
    layout.addWidget(save_images_checkbox)
    save_images_checkbox.setChecked(True)

    run_btn: QPushButton = QPushButton("Run Batch")
    progress_bar: QProgressBar = QProgressBar()
    status_label: QLabel = QLabel("")
    layout.addWidget(run_btn)
    layout.addWidget(progress_bar)
    layout.addWidget(status_label)

    is_running: dict[str, bool] = {"value": False}

    def _on_yield(payload):
        index, total, message = payload
        progress_bar.setMaximum(total)
        progress_bar.setValue(index)
        status_label.setText(message)

    def _on_finished():
        is_running["value"] = False
        run_btn.setEnabled(True)
        show_info("Batch processing finished.")

    def _on_error(exc):
        is_running["value"] = False  # pragma: no cover
        run_btn.setEnabled(True)  # pragma: no cover
        import traceback  # pragma: no cover

        print(  # pragma: no cover
            "".join(  # pragma: no cover
                traceback.format_exception(  # pragma: no cover
                    type(exc),
                    exc,
                    exc.__traceback__,  # pragma: no cover
                )  # pragma: no cover
            )  # pragma: no cover
        )  # pragma: no cover
        show_error(f"Batch processing failed: {exc}")  # pragma: no cover

    def _run_clicked():
        if is_running["value"] or not image_paths:
            if not image_paths:  # pragma: no cover
                show_error("No images selected.")  # pragma: no cover
            return  # pragma: no cover
        if not workflow_edit.text() or not output_edit.text():
            show_error("Select both a workflow and output folder.")  # pragma: no cover
            return  # pragma: no cover

        is_running["value"] = True
        run_btn.setEnabled(False)
        status_label.setText("Starting...")

        @thread_worker(ignore_errors=True)
        def _do_batch():
            yield from run_batch(  # pragma: no cover
                list(image_paths),
                Path(workflow_edit.text()),
                Path(output_edit.text()),
                save_every_step=save_every_step_checkbox.isChecked(),
                save_images=save_images_checkbox.isChecked(),
            )

        worker: GeneratorWorker = _do_batch()  # type: ignore
        worker.yielded.connect(_on_yield)
        worker.finished.connect(_on_finished)  # type: ignore
        worker.errored.connect(_on_error)
        worker.start()

    run_btn.clicked.connect(_run_clicked)
    container.setLayout(layout)
    container.add_files_btn = buttons["Add File(s)..."]  # type: ignore
    container.select_folders_btn = buttons["Add Folder (sequence)..."]  # type: ignore
    container.remove_selected_btn = buttons["Remove Selected"]  # type: ignore
    container.clear_list_btn = buttons["Clear List"]  # type: ignore
    container.workflow_pick_btn = dir_buttons["Workflow folder:"]  # type: ignore
    container.output_pick_btn = dir_buttons["Output folder:"]  # type: ignore
    container.image_list = image_list  # type: ignore
    container.image_paths = image_paths  # type: ignore
    container.workflow_edit = workflow_edit  # type: ignore
    container.output_edit = output_edit  # type: ignore
    container.run_btn = run_btn  # type: ignore
    container.progress_bar = progress_bar  # type: ignore
    container.status_label = status_label  # type: ignore

    return container

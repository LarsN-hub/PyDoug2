"""
Batch processing widget for PyDoug's napari plugin.
"""

from qtpy.QtWidgets import QWidget


def make_batch_widget(recorder) -> QWidget:
    """
    Create the batch widget container.

    Parameters
    ----------
    recorder: Recorder
        The parameters recording object.

    Returns
    -------
    container: QWidget
        The batch processing container widget.
    """

    from pathlib import Path

    from magicclass.widgets import CollapsibleContainer, ScrollableContainer
    from magicgui import magicgui
    from qtpy.QtWidgets import QVBoxLayout

    from pydoug_core import util
    from pydoug_napari._widgets._batch_run_widget import build_run_batch_widget

    @magicgui(Parent_Folder={"mode": "d"}, call_button="Save Workflow")
    def save_widget(
        Parent_Folder: Path = Path(""), Workflow_Folder_Name: str = "Workflow"
    ) -> None:
        """
        Save the current workflow.

        Parameters
        ----------
        Save_Folder: Path
            The folder where the workflow folder should be saved.
        File_Name: str
            The name of the workflow folder.
        """

        tidiedpath: Path = util.tidypath(Parent_Folder / Workflow_Folder_Name, "dir")
        recorder.save(tidiedpath)

    @magicgui(call_button="Clear Workflow")
    def clear_widget() -> None:
        """
        Clear the current workflow.
        """

        recorder.clear()

    def _get_collapsible_widget(widget) -> CollapsibleContainer:
        return CollapsibleContainer(widgets=[widget], labels=False)

    widget_list: list[CollapsibleContainer] = []
    for index, w in enumerate([save_widget, clear_widget]):
        current_w: CollapsibleContainer = _get_collapsible_widget(w)
        current_w.text = ["Save Workflow", "Clear Workflow"][index]
        current_w.native.setObjectName("Widget")
        current_w.native.setStyleSheet(
            "QWidget#Widget { border: 1px solid #888; border-radius: 4px; }"
        )
        current_w.native.setContentsMargins(2, 2, 2, 6)
        widget_list.append(current_w)
    container: ScrollableContainer = ScrollableContainer(
        widgets=widget_list, labels=False
    )
    container.text = "Workflow Operations"

    outer: QWidget = QWidget()
    outer_layout: QVBoxLayout = QVBoxLayout()
    outer_layout.addWidget(container.native)
    outer_layout.addWidget(build_run_batch_widget())
    outer.setLayout(outer_layout)

    outer.container = container  # type: ignore
    outer.run_batch_widget = build_run_batch_widget  # type: ignore

    return outer

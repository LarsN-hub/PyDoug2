"""
Main widget for PyDoug's napari plugin.
"""

from napari.viewer import Viewer
from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from pydoug_core.batch.recorder import Recorder
from pydoug_core.registry import list_operations
from pydoug_napari._widgets.batch_widget import make_batch_widget
from pydoug_napari._widgets.operation_widget import make_operation_widget


def make_main_widget(viewer: Viewer) -> QTabWidget:
    """
    Factory for the napari plugin's single top-level dock widget.

    Parameters
    ----------
    viewer: Viewer
        The napari viewer object.
    """
    recorder: Recorder = Recorder()
    tabs: QTabWidget = QTabWidget()

    categories: list[str] = sorted({spec.category for spec in list_operations()})
    for category in categories:
        tabs.addTab(_build_category_tab(category, viewer, recorder), category.title())

    tabs.addTab(make_batch_widget(recorder), "Batch")
    return tabs


def _build_category_tab(category: str, viewer: Viewer, recorder: Recorder) -> QWidget:
    """
    Build a tab for a given category of widgets in the registry.

    Parameters
    ----------
    category: str
        The operation category.
    viewer: Viewer
        The napari viewer object.
    recorder: Recorder
        The parameters logging object.

    Returns
    -------
    container: Qwidget
        The container widget for all operations in the category.
    """

    container: QWidget = QWidget()
    layout: QVBoxLayout = QVBoxLayout()
    for spec in list_operations(category=category):
        layout.addWidget(make_operation_widget(spec, viewer, recorder).native)
    container.setLayout(layout)
    return container

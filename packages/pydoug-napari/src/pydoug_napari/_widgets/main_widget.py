"""
Main widget for PyDoug's napari plugin.
"""

from napari.viewer import Viewer
from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget

import pydoug_core  # noqa: F401 — imported for its registration side effect
from pydoug_core.batch.recorder import Recorder
from pydoug_core.registry import list_operations
from pydoug_napari._widgets.batch_widget import make_batch_widget
from pydoug_napari._widgets.operation_widget import make_operation_widget


class MainWidget(QTabWidget):
    """
    Top-level dock widget containing the shared recorder and all widget tabs.

    Attributes
    ----------
    recorder: Recorder
        The parameters recording object.
    """

    def __init__(self, viewer: Viewer, parent=None):
        super().__init__(parent)
        self.recorder = Recorder()

        categories: list[str] = sorted({spec.category for spec in list_operations()})

        for category in categories:
            self.addTab(self._build_category_tab(category, viewer), category.title())

        self.addTab(make_batch_widget(self.recorder), "Batch")

    def _build_category_tab(self, category: str, viewer: Viewer):
        """
        Build a tab for a given category of widgets in the registry.

        Parameters
        ----------
        category: str
            The operation category.
        viewer: Viewer
            The napari viewer object.

        Returns
        -------
        container: Qwidget
            The container widget for all operations in the category.
        """

        container = QWidget()
        layout = QVBoxLayout()
        for spec in list_operations(category=category):
            layout.addWidget(make_operation_widget(spec, viewer, self.recorder).native)
        container.setLayout(layout)
        return container

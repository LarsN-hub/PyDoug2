"""
Main widget for PyDoug's napari plugin.
"""

from magicclass.widgets import CollapsibleContainer, ScrollableContainer
from napari.viewer import Viewer
from qtpy.QtWidgets import QTabWidget

import pydoug_core  # noqa: F401 — imported for its registration side effect
from pydoug_core.batch.recorder import Recorder
from pydoug_core.registry import list_operations
from pydoug_napari._widgets.batch_widget import make_batch_widget
from pydoug_napari._widgets.operation_widget import make_operation_widget
from pydoug_napari._widgets.recorder_log_widget import RecordingLogWidget


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
        self.operation_widgets: dict[str, CollapsibleContainer] = {}
        categories: list[str] = self._specify_category_order(
            [spec.category for spec in list_operations()]
        )
        for category in categories:
            self.addTab(
                self._build_category_tab(category, viewer).native, category.title()
            )
        self.addTab(make_batch_widget(self.recorder).native, "Batch")
        viewer.layers.events.removed.connect(self._on_layer_removed)
        viewer.window.add_dock_widget(
            RecordingLogWidget(self.recorder), name="Workflow Log", area="bottom"
        )

    def _build_category_tab(self, category: str, viewer: Viewer) -> ScrollableContainer:
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
        container: ScrollableContainer
            The container widget for all operations in the category.
        """

        containers: list[CollapsibleContainer] = []
        for spec in list_operations(category=category):
            op_widget: CollapsibleContainer = make_operation_widget(
                spec, viewer, self.recorder
            )
            self.operation_widgets[spec.name] = op_widget
            containers.append(op_widget)
        return ScrollableContainer(widgets=containers, labels=False)

    def _on_layer_removed(self, event):
        """
        Connector method for defining what happens when a layer is removed.

        Parameters
        ----------
        event: Event
            The napari event that triggers this connector.
        """

        self.recorder.remove_steps_by_layer(event.value)

    def _specify_category_order(self, categories: list[str]) -> list[str]:
        """
        Return a list of sorted categories.

        Parameters
        ----------
        categories: list[str]
            The unsorted list of categories.

        Returns
        -------
        sorted_categories: list[str]
            The sorted list of categories.
        """

        sorted_categories: list[str] = []
        if "transformations" in categories:
            sorted_categories.append("transformations")
            categories.pop(categories.index("transformations"))
        if "measurements" in categories:
            sorted_categories.append("measurements")
            categories.pop(categories.index("measurements"))
        if "utilities" in categories:
            sorted_categories.append("utilities")
            categories.pop(categories.index("utilities"))
        return sorted_categories + sorted(categories)

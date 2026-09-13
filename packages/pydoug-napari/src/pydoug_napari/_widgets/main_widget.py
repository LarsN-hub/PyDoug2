"""
Main widget for PyDoug's napari plugin.
"""

from magicclass.widgets import CollapsibleContainer, ScrollableContainer
from napari.layers import Layer
from napari.viewer import Viewer
from qtpy.QtWidgets import QMessageBox, QTabWidget

import pydoug_core  # noqa: F401 — imported for its registration side effect
from pydoug_core.batch.recorder import RecordedStep, Recorder
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
        self.viewer: Viewer = viewer
        self.recorder: Recorder = Recorder()
        self._cascading: bool = False
        self.operation_widgets: dict[str, CollapsibleContainer] = {}

        categories: list[str] = self._specify_category_order(
            {spec.category for spec in list_operations()}
        )
        for category in categories:
            self.addTab(
                self._build_category_tab(category, viewer).native, category.title()
            )

        self.addTab(make_batch_widget(self.recorder).native, "Batch")

        viewer.window.add_dock_widget(
            RecordingLogWidget(self.recorder), name="Workflow Log", area="bottom"
        )

        viewer.layers.events.removed.connect(self._on_layer_removed)

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

        if self._cascading:
            return

        removed: Layer = event.value
        removed_steps = [s for s in self.recorder.steps if s.layer_ref is removed]
        if not removed_steps:
            return

        dependent_ids = {
            dep.step_id
            for step in removed_steps
            for dep in self.recorder.dependents_of(step.step_id)
        }
        dependent_steps: list[RecordedStep] = [
            s for s in self.recorder.steps if s.step_id in dependent_ids
        ]

        if dependent_steps:
            names: str = ", ".join(s.operation_name for s in dependent_steps)
            reply = QMessageBox.question(
                self,
                "Remove dependent steps?\n\n",
                f"Removing this layer also affects {len(dependent_steps)} later "
                f"step(s) that used it as an input: {names}.\n\n"
                "Remove these dependent steps too? Choosing No will leave them "
                "in the recording referencing a step that no longer exists, "
                "which could break batch processing with this workflow.",
                QMessageBox.Yes | QMessageBox.No,  # type: ignore
            )
            if reply == QMessageBox.Yes:  # type: ignore
                self._cascading = True
                try:
                    for dep in dependent_steps:
                        if (
                            dep.layer_ref is not None
                            and dep.layer_ref in self.viewer.layers
                        ):
                            self.viewer.layers.remove(dep.layer_ref)
                        self.recorder.remove_step(dep.step_id)
                finally:
                    self._cascading = False

        self.recorder.remove_steps_by_layer(removed)

    def _specify_category_order(self, categories: set[str]) -> list[str]:
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
            categories.remove("transformations")
        if "mask" in categories:
            sorted_categories.append("mask")
            categories.remove("mask")
        if "histogram" in categories:
            sorted_categories.append("histogram")
            categories.remove("histogram")
        if "measurements" in categories:
            sorted_categories.append("measurements")
            categories.remove("measurements")
        if "color" in categories:
            sorted_categories.append("color")
            categories.remove("color")
        return sorted_categories + sorted(categories)

"""
Plot display widget for PyDoug's napari plugin.
"""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import QMainWindow

_open_plot_windows: list = []  # keeps window alive


def default_display_plot(fig: Figure) -> None:
    """
    Widget to display a matplotlib plot.

    Parameters
    ----------
    fig: Figure
        The matplotlib plot's figure object.
    """

    canvas: FigureCanvasQTAgg = FigureCanvasQTAgg(fig)
    window: QMainWindow = QMainWindow()
    window.setCentralWidget(canvas)
    window.resize
    window.show()
    _open_plot_windows.append(window)

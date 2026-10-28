"""
Plot display widget for PyDoug's napari plugin.
"""

from matplotlib.figure import Figure

_open_plot_windows: list = []  # keeps window alive


def default_display_plot(fig: Figure) -> None:
    """
    Widget to display a matplotlib plot.

    Parameters
    ----------
    fig: Figure
        The matplotlib plot's figure object.
    """

    from matplotlib.backends.backend_qt import NavigationToolbar2QT
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from qtpy.QtWidgets import QMainWindow

    canvas: FigureCanvasQTAgg = FigureCanvasQTAgg(fig)
    window: QMainWindow = QMainWindow()
    window.setCentralWidget(canvas)
    window.addToolBar(NavigationToolbar2QT(canvas, window))
    window.resize
    window.show()
    _open_plot_windows.append(window)

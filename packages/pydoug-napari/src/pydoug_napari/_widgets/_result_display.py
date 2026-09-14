"""
Widget for displaying results from operations that calculate/measure values.
"""

from matplotlib.figure import Figure
from qtpy.QtWidgets import QMessageBox


def default_display_result(
    result: dict[str, float | int] | Figure | float | int | str,
) -> None:
    """
    Create a message box widget to display a result from an operation.

    Parameters
    ----------
    result: dict[str, float | int] | float | int | str
        The results to display.
    """

    text = (
        "\n".join(f"{k}: {v}" for k, v in result.items())
        if isinstance(result, dict)
        else str(result)
    )
    QMessageBox.information(None, "Result", text)

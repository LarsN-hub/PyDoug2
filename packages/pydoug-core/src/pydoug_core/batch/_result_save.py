"""
Export results from operations that produce figures or values rather than images.
"""

import json
from pathlib import Path

from matplotlib.figure import Figure


def default_save_result(
    result: dict[str, float | int | str] | Figure | float | int | str,
    path_without_ext: Path,
) -> None:
    """
    Export results from an operation, figures as .png, dictionaries as .json, and all
    else as .txt.

    Parameters
    ----------
    result: dict[str, float | int | str] | Figure | float | int | str
        The results to save
    path_without_ext: Path
        The path (sans extension) where the results should be exported to.
    """

    if isinstance(result, Figure):
        result.savefig(path_without_ext.with_suffix(".png"))
        return
    if isinstance(result, dict):
        path_without_ext.with_suffix(".json").write_text(json.dumps(result, indent=2))
    else:
        path_without_ext.with_suffix(".txt").write_text(str(result))

"""
Export results from operations that produce figures or values rather than images.
"""

import json
from pathlib import Path

from matplotlib.figure import Figure
from numpy import generic, ndarray


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

        def _default(o):  # pragma: no cover
            if isinstance(o, generic):  # pragma: no cover
                return o.item()  # pragma: no cover
            if isinstance(o, ndarray):  # pragma: no cover
                return o.tolist()  # pragma: no cover
            raise TypeError(
                f"Object of type {type(o)} is not JSON serializable"
            )  # pragma: no cover

        path_without_ext.with_suffix(".json").write_text(
            json.dumps(result, indent=2, default=_default)
        )

    else:
        path_without_ext.with_suffix(".txt").write_text(str(result))

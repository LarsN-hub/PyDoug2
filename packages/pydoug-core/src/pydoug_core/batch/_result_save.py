"""
Export results from operations that produce figures or values rather than images.
"""

import json
from pathlib import Path

from matplotlib.figure import Figure
from numpy import generic, ndarray, savez

from pydoug_core.results import PlotResult


def default_save_result(
    result: dict | Figure | float | int | str | tuple | PlotResult,
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

    if isinstance(result, PlotResult):
        result.figure.savefig(path_without_ext.with_suffix(".png"))
        result.write_csv(path_without_ext.with_suffix(".csv"))
        return

    if isinstance(result, Figure):
        result.savefig(path_without_ext.with_suffix(".png"))
        return

    if (
        isinstance(result, tuple)
        and result
        and all(isinstance(a, ndarray) for a in result)
    ):
        savez(path_without_ext.with_suffix(".npz"), *result)
        return

    if isinstance(result, dict):

        def _default(o):  # pragma: no cover
            if isinstance(o, generic):
                return o.item()
            if isinstance(o, ndarray):
                return o.tolist()
            raise TypeError(f"Object of type {type(o)} is not JSON serializable")

        path_without_ext.with_suffix(".json").write_text(
            json.dumps(result, indent=2, default=_default)
        )

    else:
        path_without_ext.with_suffix(".txt").write_text(str(result))

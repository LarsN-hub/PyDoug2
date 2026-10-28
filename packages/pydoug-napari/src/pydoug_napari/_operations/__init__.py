"""
Auto-imports every module in this package for discovery by widget registry.
"""

import importlib
import sys

if getattr(sys, "frozen", False):  # pragma: no cover
    from pydoug_napari._operations._manifest import OPERATION_MODULES

    for name in OPERATION_MODULES:
        importlib.import_module(name)
else:
    import pkgutil

    for _, module_name, _ in pkgutil.walk_packages(__path__, prefix=f"{__name__}."):
        if module_name.rsplit(".", 1)[-1].startswith("_"):
            continue
        importlib.import_module(module_name)

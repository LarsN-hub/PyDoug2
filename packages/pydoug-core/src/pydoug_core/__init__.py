"""
Auto-imports every module in this package for discovery by widget registry.
"""

import importlib
import pkgutil

for _, module_name, _ in pkgutil.walk_packages(__path__, prefix=f"{__name__}."):
    importlib.import_module(module_name)

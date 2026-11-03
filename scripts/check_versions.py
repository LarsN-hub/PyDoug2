"""
Fail a pypi publishing run if a package's version doesn't match the release tag.
"""

import sys
from pathlib import Path

import tomllib

tag = sys.argv[1].removeprefix("v")
problems = []
for package in ("pydoug-core", "pydoug-napari"):
    data = tomllib.loads(Path(f"packages/{package}/pyproject.toml").read_text())
    version = data["project"]["version"]
    if version != tag:
        problems.append(f"{package} is {version} but the tag is {tag}")
if problems:
    sys.exit("Version mismatch:\n  " + "\n  ".join(problems))
print(f"All packages are at {tag}")

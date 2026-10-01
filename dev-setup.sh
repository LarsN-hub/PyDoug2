#!/usr/bin/env bash
set -e

echo "==> Installing local packages in editable mode..."
pip install -e "packages/pydoug-core[testing]"
pip install -e "packages/pydoug-napari[testing]"
pip install -e "packages/proj_name-standalone"

echo "==> Installing project-wide dev tools..."
pip install -r requirements-dev.txt

echo "==> Installing pre-commit hooks..."
pre-commit install

echo "==> Done. Environment ready."

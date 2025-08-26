#!/usr/bin/env bash
set -euo pipefail

# install the root project's deps from pyproject.toml / uv.lock
uv sync

# then install submodules' requirements files into the same env
uv pip install -r submodules/mod_a/requirements.txt
uv pip install -r submodules/mod_b/requirements.txt
# ...add more as needed


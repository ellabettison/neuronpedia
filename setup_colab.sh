#!/usr/bin/env bash
# Colab setup. Colab can't run Docker containers (read-only cgroups, no network namespaces),
# so on Colab the inference server runs directly in a poetry virtualenv instead.
# For a normal server use setup.sh instead.
set -euo pipefail
cd "$(dirname "$0")"

pip install -q poetry uv
make inference-localhost-install

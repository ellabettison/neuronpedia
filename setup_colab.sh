#!/usr/bin/env bash
# Colab setup. Colab can't run Docker containers (read-only cgroups, no network namespaces),
# so on Colab the inference server runs directly in a poetry virtualenv instead.
# For a normal server use setup.sh instead. Safe to rerun.
set -euo pipefail
cd "$(dirname "$0")"

# Never install into a caller's active venv (e.g. sae-jailbreaks/.venv): poetry would put the
# inference deps there, clobbering its torch. Drop it so poetry uses its own env.
if [ -n "${VIRTUAL_ENV:-}" ]; then PATH="${PATH//$VIRTUAL_ENV\/bin:/}"; unset VIRTUAL_ENV; fi

command -v poetry >/dev/null || python3 -m pip install -q poetry
cd apps/inference
# pyproject.toml already has the local inference client as a path dep, so a plain install is enough
# (make inference-localhost-install removes/re-adds it, rewriting pyproject.toml and the lock each run).
poetry check --lock >/dev/null 2>&1 || poetry lock
poetry install
poetry run python -c "import torch; assert torch.cuda.is_available(), 'inference env: torch sees no GPU'"

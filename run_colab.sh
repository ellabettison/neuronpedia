#!/usr/bin/env bash
# Colab: run the inference server for a MODEL_SOURCESET directly with poetry (run setup_colab.sh first).
# Colab can't run Docker containers (read-only cgroups, no bridge networking), so this mirrors what
# docker/compose.yaml sets up for the inference container. Server listens on localhost:5002.
# Usage: ./run_colab.sh <sourceset>   e.g. ./run_colab.sh gemma-2-9b-it.gemmascope-res-16k
# Set HF_TOKEN in the environment (or .env) for gated models.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_SOURCESET="${1:?usage: $0 <sourceset> (see make inference-list-configs)}"
if [ ! -f ".env.inference.$MODEL_SOURCESET" ]; then
  echo "Error: .env.inference.$MODEL_SOURCESET not found. Run 'make inference-list-configs' to see available configurations."
  exit 1
fi

# Same as the compose bind mount: use the patched SAE registry
SAE_LENS_DIR=$(cd apps/inference && poetry run python -c "import os, sae_lens; print(os.path.dirname(sae_lens.__file__))" 2>/dev/null)
cp apps/inference/patches/pretrained_saes.yaml "$SAE_LENS_DIR/pretrained_saes.yaml"

# Same env as the container: model config, plus HF_TOKEN and SECRET
set -a
source ".env.inference.$MODEL_SOURCESET"
[ -f .env ] && source .env
set +a
export SECRET="${SECRET:-$(grep -E '^INFERENCE_SERVER_SECRET=' .env.localhost | cut -d= -f2-)}"
export HF_TOKEN="${HF_TOKEN:-}"

cd apps/inference
exec poetry run python start.py

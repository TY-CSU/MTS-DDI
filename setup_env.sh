#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/environment.yml"
ENV_NAME="${MTS_DDI_ENV_NAME:-mts-ddi}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required; install Miniconda or Mambaforge first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f "$ENV_FILE"
else
  conda env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

conda run -n "$ENV_NAME" python -m pip install -r "$(dirname "$ENV_FILE")/requirements.txt"
echo "Environment ready. Activate with: conda activate $ENV_NAME"

#!/usr/bin/env bash
set -euo pipefail

PY="${PYTHON:-python3}"
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
fi

echo "Using interpreter: $PY"
"$PY" -m pytest -q

for notebook in \
  notebooks/01_data_audit.ipynb \
  notebooks/02_exploratory_analysis.ipynb \
  notebooks/03_baseline_models.ipynb \
  notebooks/04_model_comparison.ipynb \
  notebooks/05_feature_analysis.ipynb
do
  echo "Executing $notebook"
  "$PY" -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 "$notebook"
done

echo "Pipeline complete. See outputs/."

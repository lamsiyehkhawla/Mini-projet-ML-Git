# mlops-ml-project (baseline)

## Installation
python -m venv .venv
# activate venv
pip install -r requirements.txt

## Entraînement
python scripts/train.py

## Évaluation
python scripts/evaluate.py

## Artefacts générés
- artifacts/model.joblib
- artifacts/metrics.json
- artifacts/confusion_matrix.png
- artifacts/report.json

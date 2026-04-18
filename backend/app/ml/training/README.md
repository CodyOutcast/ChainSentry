# Graph ML Training Workspace

This folder now contains Student 2's end-to-end graph-model training pipeline.

Included workflow:

- `dataset.py`: deterministic synthetic dataset generation based on the current ChainSentry request schema
- `train_graph_model.py`: training, evaluation, threshold selection, and artifact export
- automatic bootstrap support when the backend starts with `CHAIN_SENTRY_PREDICTOR_BACKEND=graph-model`

Typical command:

```bash
PYTHONPATH=backend .venv/bin/python -m app.ml.training.train_graph_model \
  --artifact-path backend/app/ml/artifacts/graph-model.pt \
  --metrics-path backend/app/ml/artifacts/graph-model-metrics.json
```

Notes:

- The dataset is a reproducible student-project baseline built from synthetic transaction patterns and pseudo-labels produced by the current detectors.
- The saved artifact is consumed by `app.ml.inference.GraphModelPredictor`.
- The backend keeps the Student 1 heuristic path as a safety fallback if the model is unavailable.

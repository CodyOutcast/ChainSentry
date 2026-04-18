from __future__ import annotations

import os
from pathlib import Path


UNLIMITED_APPROVAL_THRESHOLD = 1_000_000_000
LARGE_APPROVAL_THRESHOLD = 10_000

PREDICTOR_BACKEND = os.getenv("CHAIN_SENTRY_PREDICTOR_BACKEND", "graph-model")
GRAPH_MODEL_ARCHITECTURE = os.getenv("CHAIN_SENTRY_GRAPH_MODEL_ARCHITECTURE", "rgcn")
GRAPH_MODEL_ARTIFACT_PATH = Path(
	os.getenv(
		"CHAIN_SENTRY_GRAPH_MODEL_PATH",
		str(Path(__file__).resolve().parent / "ml" / "artifacts" / "graph-model.pt"),
	)
)
GRAPH_MODEL_METRICS_PATH = Path(
    os.getenv(
        "CHAIN_SENTRY_GRAPH_MODEL_METRICS_PATH",
        str(Path(__file__).resolve().parent / "ml" / "artifacts" / "graph-model-metrics.json"),
    )
)
GRAPH_MODEL_TRAINING_SEED = int(os.getenv("CHAIN_SENTRY_GRAPH_MODEL_TRAINING_SEED", "17"))
GRAPH_MODEL_TRAINING_EPOCHS = int(os.getenv("CHAIN_SENTRY_GRAPH_MODEL_TRAINING_EPOCHS", "18"))

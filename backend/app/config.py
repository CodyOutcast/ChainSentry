from __future__ import annotations

import os
from pathlib import Path


UNLIMITED_APPROVAL_THRESHOLD = 1_000_000_000
LARGE_APPROVAL_THRESHOLD = 10_000

PREDICTOR_BACKEND = os.getenv("CHAIN_SENTRY_PREDICTOR_BACKEND", "heuristic-fallback")
GRAPH_MODEL_ARCHITECTURE = os.getenv("CHAIN_SENTRY_GRAPH_MODEL_ARCHITECTURE", "rgcn")
GRAPH_MODEL_ARTIFACT_PATH = Path(
	os.getenv(
		"CHAIN_SENTRY_GRAPH_MODEL_PATH",
		str(Path(__file__).resolve().parent / "ml" / "artifacts" / "graph-model.pt"),
	)
)

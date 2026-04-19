from __future__ import annotations

import argparse
from pathlib import Path

from app.config import GRAPH_MODEL_METRICS_PATH, GRAPH_MODEL_TRAINING_EPOCHS, GRAPH_MODEL_TRAINING_SEED
from app.ml.training.train_multidataset_model import (
    bootstrap_graph_model_artifacts,
    train_multidataset_model_artifact,
)


def train_graph_model_artifact(
    artifact_path: Path,
    metrics_path: Path | None = None,
    *,
    seed: int = GRAPH_MODEL_TRAINING_SEED,
    dataset_size: int = 640,
    epochs: int = GRAPH_MODEL_TRAINING_EPOCHS,
) -> dict[str, object]:
    return train_multidataset_model_artifact(
        artifact_path=artifact_path,
        metrics_path=metrics_path or GRAPH_MODEL_METRICS_PATH,
        seed=seed,
        dataset_size=dataset_size,
        epochs=epochs,
        size_profile="quick",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export the ChainSentry graph model artifact.")
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path, default=GRAPH_MODEL_METRICS_PATH)
    parser.add_argument("--seed", type=int, default=GRAPH_MODEL_TRAINING_SEED)
    parser.add_argument("--dataset-size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=GRAPH_MODEL_TRAINING_EPOCHS)
    args = parser.parse_args()
    train_graph_model_artifact(
        artifact_path=args.artifact_path,
        metrics_path=args.metrics_path,
        seed=args.seed,
        dataset_size=args.dataset_size,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path


def train_graph_model_artifact(*args, **kwargs):
    from app.ml.training.train_graph_model import train_graph_model_artifact as _train_graph_model_artifact

    return _train_graph_model_artifact(*args, **kwargs)


def bootstrap_graph_model_artifacts(artifact_path: Path):
    from app.ml.training.train_graph_model import bootstrap_graph_model_artifacts as _bootstrap_graph_model_artifacts

    return _bootstrap_graph_model_artifacts(artifact_path)


__all__ = ["bootstrap_graph_model_artifacts", "train_graph_model_artifact"]

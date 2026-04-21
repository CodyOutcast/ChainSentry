from __future__ import annotations

from pathlib import Path

from app.ml.training.external_datasets import (
    load_ethereum_fraud_by_activity,
    load_eth_labels,
    load_etherscamdb,
    load_forta_labels,
    load_forta_malicious_contracts,
    load_ptxphish,
    load_ptxphish_initial_addresses,
    load_raven,
    summarize_available_external_datasets,
)


def train_graph_model_artifact(*args, **kwargs):
    from app.ml.training.train_graph_model import train_graph_model_artifact as _train_graph_model_artifact

    return _train_graph_model_artifact(*args, **kwargs)


def train_multidataset_model_artifact(*args, **kwargs):
    from app.ml.training.train_multidataset_model import (
        train_multidataset_model_artifact as _train_multidataset_model_artifact,
    )

    return _train_multidataset_model_artifact(*args, **kwargs)


def bootstrap_graph_model_artifacts(artifact_path: Path):
    from app.ml.training.train_multidataset_model import (
        bootstrap_graph_model_artifacts as _bootstrap_graph_model_artifacts,
    )

    return _bootstrap_graph_model_artifacts(artifact_path)


__all__ = [
    "bootstrap_graph_model_artifacts",
    "train_graph_model_artifact",
    "train_multidataset_model_artifact",
    "load_ethereum_fraud_by_activity",
    "load_eth_labels",
    "load_etherscamdb",
    "load_forta_labels",
    "load_forta_malicious_contracts",
    "load_ptxphish",
    "load_ptxphish_initial_addresses",
    "load_raven",
    "summarize_available_external_datasets",
]

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from app.config import GRAPH_MODEL_METRICS_PATH, GRAPH_MODEL_TRAINING_EPOCHS, GRAPH_MODEL_TRAINING_SEED
from app.ml.model import (
    GraphModelMetadata,
    RelationAwareGraphModel,
    RelationAwareGraphModelConfig,
    save_graph_model_artifact,
)
from app.ml.training.dataset import TrainingExample, build_synthetic_dataset
from app.ml.vectorization import build_feature_vocabulary, encode_sample, fit_normalization


SEVERITY_LABELS = ("low", "medium", "high", "critical")
CATEGORY_LABELS = ("approval", "destination", "simulation")


@dataclass(frozen=True)
class EncodedTrainingExample:
    sample: object
    category_targets: torch.Tensor
    severity_target: int
    scenario_name: str


def train_graph_model_artifact(
    artifact_path: Path,
    metrics_path: Path | None = None,
    *,
    seed: int = GRAPH_MODEL_TRAINING_SEED,
    dataset_size: int = 640,
    epochs: int = GRAPH_MODEL_TRAINING_EPOCHS,
) -> dict[str, object]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = Random(seed)

    examples = build_synthetic_dataset(seed=seed, dataset_size=dataset_size)
    graphs = [example.graph for example in examples]
    feature_sets = [example.features for example in examples]
    vocabulary = build_feature_vocabulary(graphs, feature_sets)
    normalization = fit_normalization(feature_sets, vocabulary.numeric_keys)
    encoded_examples = [_encode_training_example(example, vocabulary, normalization) for example in examples]

    severity_targets = [example.severity_target for example in encoded_examples]
    indices = list(range(len(encoded_examples)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.25,
        random_state=seed,
        stratify=severity_targets,
    )

    model_config = RelationAwareGraphModelConfig()
    model = RelationAwareGraphModel(
        num_node_types=len(vocabulary.node_types),
        num_edge_types=len(vocabulary.edge_types),
        categorical_cardinalities=[len(vocabulary.categorical_values[key]) for key in vocabulary.categorical_keys],
        numeric_dim=len(vocabulary.numeric_keys),
        boolean_dim=len(vocabulary.boolean_keys),
        config=model_config,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    category_loss_fn = torch.nn.BCEWithLogitsLoss()
    severity_loss_fn = torch.nn.CrossEntropyLoss()

    last_epoch_loss = 0.0
    for _epoch in range(epochs):
        model.train()
        shuffled_indices = train_indices[:]
        rng.shuffle(shuffled_indices)
        epoch_loss = 0.0
        for index in shuffled_indices:
            encoded = encoded_examples[index]
            category_logits, severity_logits = model(encoded.sample)
            category_loss = category_loss_fn(category_logits, encoded.category_targets)
            severity_loss = severity_loss_fn(severity_logits.unsqueeze(0), torch.tensor([encoded.severity_target]))
            loss = category_loss + severity_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        last_epoch_loss = epoch_loss / max(len(shuffled_indices), 1)

    test_examples = [encoded_examples[index] for index in test_indices]
    category_probabilities, category_targets, severity_predictions, severity_targets_eval = _collect_predictions(model, test_examples)
    thresholds = _select_thresholds(category_targets, category_probabilities)
    metrics = _build_metrics(
        category_probabilities=category_probabilities,
        category_targets=category_targets,
        severity_predictions=severity_predictions,
        severity_targets=severity_targets_eval,
        thresholds=thresholds,
        dataset_size=len(encoded_examples),
        train_size=len(train_indices),
        test_size=len(test_indices),
        last_epoch_loss=last_epoch_loss,
        seed=seed,
        epochs=epochs,
    )

    metadata = GraphModelMetadata(
        architecture="relation-aware-mlp-gnn",
        framework="pytorch",
        artifact_path=artifact_path,
    )
    save_graph_model_artifact(
        artifact_path=artifact_path,
        metadata=metadata,
        vocabulary=vocabulary,
        normalization=normalization,
        model_config=model_config,
        model_state=model.state_dict(),
        thresholds=thresholds,
        metrics=metrics,
    )

    metrics_output_path = metrics_path or GRAPH_MODEL_METRICS_PATH
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def bootstrap_graph_model_artifacts(artifact_path: Path) -> dict[str, object]:
    metrics_path = GRAPH_MODEL_METRICS_PATH
    return train_graph_model_artifact(
        artifact_path=artifact_path,
        metrics_path=metrics_path,
        seed=GRAPH_MODEL_TRAINING_SEED,
        dataset_size=360,
        epochs=GRAPH_MODEL_TRAINING_EPOCHS,
    )


def _encode_training_example(example: TrainingExample, vocabulary, normalization) -> EncodedTrainingExample:
    encoded = encode_sample(example.graph, example.features, vocabulary, normalization)
    category_targets = torch.tensor(
        [example.category_labels[label] for label in CATEGORY_LABELS],
        dtype=torch.float32,
    )
    severity_target = SEVERITY_LABELS.index(example.severity_label)
    return EncodedTrainingExample(
        sample=encoded,
        category_targets=category_targets,
        severity_target=severity_target,
        scenario_name=example.scenario_name,
    )


def _collect_predictions(model, test_examples: list[EncodedTrainingExample]):
    model.eval()
    category_probabilities: list[list[float]] = []
    category_targets: list[list[float]] = []
    severity_predictions: list[int] = []
    severity_targets: list[int] = []

    with torch.no_grad():
        for encoded in test_examples:
            category_logits, severity_logits = model(encoded.sample)
            category_probabilities.append(torch.sigmoid(category_logits).tolist())
            category_targets.append(encoded.category_targets.tolist())
            severity_predictions.append(int(torch.argmax(severity_logits).item()))
            severity_targets.append(encoded.severity_target)

    return (
        np.array(category_probabilities, dtype=np.float32),
        np.array(category_targets, dtype=np.float32),
        np.array(severity_predictions, dtype=np.int64),
        np.array(severity_targets, dtype=np.int64),
    )


def _select_thresholds(category_targets: np.ndarray, category_probabilities: np.ndarray) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    search_space = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6)
    for index, label in enumerate(CATEGORY_LABELS):
        best_threshold = 0.5
        best_score = -1.0
        target = category_targets[:, index].astype(int)
        for candidate in search_space:
            prediction = (category_probabilities[:, index] >= candidate).astype(int)
            score = f1_score(target, prediction, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = candidate
        thresholds[label] = best_threshold
    return thresholds


def _build_metrics(
    *,
    category_probabilities: np.ndarray,
    category_targets: np.ndarray,
    severity_predictions: np.ndarray,
    severity_targets: np.ndarray,
    thresholds: dict[str, float],
    dataset_size: int,
    train_size: int,
    test_size: int,
    last_epoch_loss: float,
    seed: int,
    epochs: int,
) -> dict[str, object]:
    category_metrics: dict[str, dict[str, float]] = {}
    for index, label in enumerate(CATEGORY_LABELS):
        prediction = (category_probabilities[:, index] >= thresholds[label]).astype(int)
        target = category_targets[:, index].astype(int)
        precision, recall, f1_value, support = precision_recall_fscore_support(
            target,
            prediction,
            average="binary",
            zero_division=0,
        )
        category_metrics[label] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1_value), 4),
            "support": int(target.sum()),
            "threshold": thresholds[label],
        }

    return {
        "dataset": {
            "total_examples": dataset_size,
            "train_examples": train_size,
            "test_examples": test_size,
            "seed": seed,
            "epochs": epochs,
        },
        "training": {
            "last_epoch_loss": round(last_epoch_loss, 4),
        },
        "category_metrics": category_metrics,
        "severity_accuracy": round(float(accuracy_score(severity_targets, severity_predictions)), 4),
        "severity_macro_f1": round(float(f1_score(severity_targets, severity_predictions, average="macro")), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export the ChainSentry graph model artifact.")
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("backend/app/ml/artifacts/graph-model.pt"),
        help="Path to the exported PyTorch artifact.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("backend/app/ml/artifacts/graph-model-metrics.json"),
        help="Path to the exported metrics JSON.",
    )
    parser.add_argument("--seed", type=int, default=GRAPH_MODEL_TRAINING_SEED)
    parser.add_argument("--dataset-size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=GRAPH_MODEL_TRAINING_EPOCHS)
    args = parser.parse_args()

    metrics = train_graph_model_artifact(
        artifact_path=args.artifact_path,
        metrics_path=args.metrics_path,
        seed=args.seed,
        dataset_size=args.dataset_size,
        epochs=args.epochs,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

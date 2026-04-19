from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from app.config import GRAPH_MODEL_METRICS_PATH, GRAPH_MODEL_TRAINING_EPOCHS, GRAPH_MODEL_TRAINING_SEED
from app.ml.model import (
    MAIN_BINARY_HEADS,
    SEVERITY_LABELS,
    GraphModelMetadata,
    RelationAwareGraphModel,
    RelationAwareGraphModelConfig,
    save_graph_model_artifact,
)
from app.ml.training.multi_dataset import (
    MultiDatasetTrainingSet,
    WeightedDatasetSampler,
    build_data_loader,
    build_default_adaptors,
    build_split_dataset,
)
from app.ml.training.unified_sample import AUXILIARY_BINARY_HEADS, BINARY_HEADS, UnifiedTrainingSample
from app.ml.vectorization import build_feature_vocabulary, encode_sample, fit_normalization


@dataclass(frozen=True)
class MultiDatasetTrainingConfig:
    seed: int = GRAPH_MODEL_TRAINING_SEED
    epochs: int = GRAPH_MODEL_TRAINING_EPOCHS
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    train_samples_per_epoch: int | None = None
    dataset_weights: dict[str, float] | None = None
    binary_loss_weights: dict[str, float] | None = None
    dropout: float = 0.15
    hidden_dim: int = 64
    relation_layers: int = 2
    categorical_embedding_dim: int = 8
    feature_hidden_dim: int = 48
    head_hidden_dim: int = 64


SIZE_PROFILES = {
    "quick": {
        "train": {"forta": 120, "eth-labels": 120, "etherscamdb": 80, "ptxphish": 60, "raven": 120},
        "val": {"forta": 40, "eth-labels": 40, "etherscamdb": 30, "ptxphish": 20, "raven": 40},
        "test": {"forta": 40, "eth-labels": 40, "etherscamdb": 30, "ptxphish": 20, "raven": 40},
    },
    "standard": {
        "train": {"forta": 800, "eth-labels": 700, "etherscamdb": 450, "ptxphish": 120, "raven": 900},
        "val": {"forta": 180, "eth-labels": 180, "etherscamdb": 120, "ptxphish": 40, "raven": 180},
        "test": {"forta": 180, "eth-labels": 180, "etherscamdb": 120, "ptxphish": 40, "raven": 180},
    },
    "full": {
        "train": {"forta": 1800, "eth-labels": 1400, "etherscamdb": 900, "ptxphish": 160, "raven": 1800},
        "val": {"forta": 350, "eth-labels": 350, "etherscamdb": 240, "ptxphish": 50, "raven": 350},
        "test": {"forta": 350, "eth-labels": 350, "etherscamdb": 240, "ptxphish": 50, "raven": 350},
    },
}

DEFAULT_DATASET_WEIGHTS = {
    "forta": 0.24,
    "eth-labels": 0.22,
    "etherscamdb": 0.12,
    "ptxphish": 0.22,
    "raven": 0.20,
}
DEFAULT_BINARY_LOSS_WEIGHTS = {
    "approval": 1.0,
    "destination": 1.0,
    "simulation": 1.0,
    "address_malicious": 0.5,
    "failure_aux": 0.35,
}
DEFAULT_MULTICLASS_LOSS_WEIGHTS = {"severity": 0.75}


class SwanLabLogger:
    def __init__(self, *, enabled: bool, project: str, run_name: str | None, config: dict[str, Any]) -> None:
        self.enabled = False
        self._module = None
        self._run = None
        if not enabled:
            return
        import swanlab  # type: ignore

        api_key = os.getenv("SWANLAB_API_KEY")
        if api_key:
            swanlab.login(api_key=api_key)

        init_kwargs: dict[str, Any] = {"project": project, "config": config}
        signature = inspect.signature(swanlab.init)
        if run_name:
            if "experiment_name" in signature.parameters:
                init_kwargs["experiment_name"] = run_name
            elif "name" in signature.parameters:
                init_kwargs["name"] = run_name
        self._run = swanlab.init(**init_kwargs)
        self._module = swanlab
        self.enabled = True

    def log(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self._module is None:
            return
        self._module.log(payload)

    def finish(self) -> None:
        if not self.enabled:
            return
        if self._run is not None and hasattr(self._run, "finish"):
            self._run.finish()
        elif self._module is not None and hasattr(self._module, "finish"):
            self._module.finish()


def train_multidataset_model_artifact(
    artifact_path: Path,
    metrics_path: Path | None = None,
    *,
    seed: int = GRAPH_MODEL_TRAINING_SEED,
    epochs: int = GRAPH_MODEL_TRAINING_EPOCHS,
    size_profile: str = "standard",
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    dataset_size: int | None = None,
    swanlab_project: str | None = None,
    swanlab_run_name: str | None = None,
    enable_swanlab: bool = False,
) -> dict[str, object]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(seed)
    np.random.seed(seed)

    size_limits = _resolve_size_limits(size_profile=size_profile, dataset_size=dataset_size)
    train_config = MultiDatasetTrainingConfig(
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dataset_weights=DEFAULT_DATASET_WEIGHTS,
        binary_loss_weights=DEFAULT_BINARY_LOSS_WEIGHTS,
    )

    adaptors = build_default_adaptors()
    train_dataset = build_split_dataset(split="train", adaptors=adaptors, limits=size_limits["train"], seed=seed)
    val_dataset = build_split_dataset(split="val", adaptors=adaptors, limits=size_limits["val"], seed=seed)
    test_dataset = build_split_dataset(split="test", adaptors=adaptors, limits=size_limits["test"], seed=seed)

    vocabulary = build_feature_vocabulary(
        [sample.graph for sample in train_dataset.samples],
        [sample.features for sample in train_dataset.samples],
    )
    normalization = fit_normalization([sample.features for sample in train_dataset.samples], vocabulary.numeric_keys)

    model_config = RelationAwareGraphModelConfig(
        hidden_dim=train_config.hidden_dim,
        relation_layers=train_config.relation_layers,
        categorical_embedding_dim=train_config.categorical_embedding_dim,
        feature_hidden_dim=train_config.feature_hidden_dim,
        head_hidden_dim=train_config.head_hidden_dim,
        dropout=train_config.dropout,
    )
    model = RelationAwareGraphModel(
        num_node_types=len(vocabulary.node_types),
        num_edge_types=len(vocabulary.edge_types),
        categorical_cardinalities=[len(vocabulary.categorical_values[key]) for key in vocabulary.categorical_keys],
        numeric_dim=len(vocabulary.numeric_keys),
        boolean_dim=len(vocabulary.boolean_keys),
        auxiliary_binary_heads=AUXILIARY_BINARY_HEADS,
        config=model_config,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    binary_criteria = _build_binary_loss_fns(train_dataset.samples)
    severity_criterion = _build_severity_loss_fn(train_dataset.samples)
    train_sampler = WeightedDatasetSampler(
        train_dataset,
        dataset_weights=train_config.dataset_weights or DEFAULT_DATASET_WEIGHTS,
        num_samples=train_config.train_samples_per_epoch or max(len(train_dataset), batch_size),
        seed=seed,
    )
    train_loader = build_data_loader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = build_data_loader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = build_data_loader(test_dataset, batch_size=batch_size, shuffle=False)

    logger = SwanLabLogger(
        enabled=enable_swanlab,
        project=swanlab_project or "chainsentry",
        run_name=swanlab_run_name,
        config={
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "size_profile": size_profile,
            "size_limits": size_limits,
            "dataset_weights": train_config.dataset_weights or DEFAULT_DATASET_WEIGHTS,
            "binary_loss_weights": train_config.binary_loss_weights or DEFAULT_BINARY_LOSS_WEIGHTS,
        },
    )
    _log_dataset_summary(logger, train_dataset, val_dataset, test_dataset)

    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        epoch_start = time.perf_counter()
        train_metrics = _run_training_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            vocabulary=vocabulary,
            normalization=normalization,
            binary_criteria=binary_criteria,
            severity_criterion=severity_criterion,
            binary_loss_weights=train_config.binary_loss_weights or DEFAULT_BINARY_LOSS_WEIGHTS,
            gradient_clip_norm=train_config.gradient_clip_norm,
        )
        val_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            vocabulary=vocabulary,
            normalization=normalization,
            binary_criteria=binary_criteria,
            severity_criterion=severity_criterion,
            binary_thresholds={head_name: 0.5 for head_name in BINARY_HEADS},
        )
        epoch_seconds = time.perf_counter() - epoch_start
        epoch_log = {
            "epoch": epoch,
            "epoch_seconds": round(epoch_seconds, 4),
            **_prefix_metrics("train", train_metrics),
            **_prefix_metrics("val", val_metrics["scalar_metrics"]),
            "optimizer/lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_log)
        logger.log(epoch_log)

    val_predictions = collect_predictions(
        model=model,
        data_loader=val_loader,
        vocabulary=vocabulary,
        normalization=normalization,
    )
    thresholds = _select_thresholds(val_predictions["binary"])
    final_val_metrics = evaluate_model(
        model=model,
        data_loader=val_loader,
        vocabulary=vocabulary,
        normalization=normalization,
        binary_criteria=binary_criteria,
        severity_criterion=severity_criterion,
        binary_thresholds=thresholds,
    )
    final_test_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        vocabulary=vocabulary,
        normalization=normalization,
        binary_criteria=binary_criteria,
        severity_criterion=severity_criterion,
        binary_thresholds=thresholds,
    )
    logger.log(
        {
            "epoch": epochs,
            **_prefix_metrics("final_val", final_val_metrics["scalar_metrics"]),
            **_prefix_metrics("final_test", final_test_metrics["scalar_metrics"]),
        }
    )
    logger.finish()

    metadata = GraphModelMetadata(
        architecture="relation-aware-multihead-gnn",
        framework="pytorch",
        artifact_path=artifact_path,
        auxiliary_binary_heads=AUXILIARY_BINARY_HEADS,
    )
    save_graph_model_artifact(
        artifact_path=artifact_path,
        metadata=metadata,
        vocabulary=vocabulary,
        normalization=normalization,
        model_config=model_config,
        model_state=model.state_dict(),
        thresholds=thresholds,
        metrics={},  # filled below
    )

    metrics = {
        "dataset": {
            "train": asdict(train_dataset.summarize("train")),
            "val": asdict(val_dataset.summarize("val")),
            "test": asdict(test_dataset.summarize("test")),
        },
        "training_config": asdict(train_config),
        "size_profile": size_profile,
        "size_limits": size_limits,
        "thresholds": thresholds,
        "history": history,
        "validation_metrics": final_val_metrics,
        "test_metrics": final_test_metrics,
        "category_metrics": final_test_metrics["binary_metrics"]["main"],
        "auxiliary_metrics": final_test_metrics["binary_metrics"]["auxiliary"],
        "severity_metrics": final_test_metrics["multiclass_metrics"]["severity"],
        "severity_accuracy": final_test_metrics["multiclass_metrics"]["severity"]["accuracy"],
        "severity_macro_f1": final_test_metrics["multiclass_metrics"]["severity"]["macro_f1"],
    }
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


def evaluate_model(
    *,
    model: RelationAwareGraphModel,
    data_loader,
    vocabulary,
    normalization,
    binary_criteria: dict[str, torch.nn.BCEWithLogitsLoss],
    severity_criterion: torch.nn.CrossEntropyLoss,
    binary_thresholds: dict[str, float],
) -> dict[str, Any]:
    model.eval()
    prediction_store = collect_predictions(
        model=model,
        data_loader=data_loader,
        vocabulary=vocabulary,
        normalization=normalization,
    )
    scalar_metrics = {
        "loss_total": round(float(_compute_eval_loss(prediction_store, binary_criteria, severity_criterion)), 6),
    }
    binary_metrics = {
        "main": {},
        "auxiliary": {},
    }
    for head_name, payload in prediction_store["binary"].items():
        metrics = _compute_binary_metrics(
            targets=np.array(payload["targets"], dtype=np.float32),
            scores=np.array(payload["scores"], dtype=np.float32),
            threshold=binary_thresholds.get(head_name, 0.5),
        )
        scalar_metrics.update({f"{head_name}/{key}": value for key, value in metrics.items() if isinstance(value, (int, float))})
        if head_name in MAIN_BINARY_HEADS:
            binary_metrics["main"][head_name] = metrics
        else:
            binary_metrics["auxiliary"][head_name] = metrics

    severity_metrics = _compute_severity_metrics(
        targets=np.array(prediction_store["multiclass"]["severity"]["targets"], dtype=np.int64),
        predictions=np.array(prediction_store["multiclass"]["severity"]["predictions"], dtype=np.int64),
    )
    scalar_metrics.update({f"severity/{key}": value for key, value in severity_metrics.items() if isinstance(value, (int, float))})
    return {
        "scalar_metrics": scalar_metrics,
        "binary_metrics": binary_metrics,
        "multiclass_metrics": {"severity": severity_metrics},
    }


def collect_predictions(*, model: RelationAwareGraphModel, data_loader, vocabulary, normalization) -> dict[str, Any]:
    binary_payload = {head_name: {"targets": [], "scores": []} for head_name in BINARY_HEADS}
    severity_payload = {"targets": [], "predictions": [], "probabilities": []}
    with torch.no_grad():
        for batch in data_loader:
            for sample in batch:
                encoded = encode_sample(sample.graph, sample.features, vocabulary, normalization)
                outputs = model(encoded)
                for head_name in BINARY_HEADS:
                    if not sample.binary_target_mask.get(head_name, False):
                        continue
                    binary_payload[head_name]["targets"].append(sample.binary_targets[head_name])
                    score = torch.sigmoid(outputs.binary_logits[head_name]).item()
                    binary_payload[head_name]["scores"].append(score)
                if sample.multiclass_target_mask.get("severity", False):
                    probabilities = torch.softmax(outputs.multiclass_logits["severity"], dim=0)
                    severity_payload["targets"].append(sample.multiclass_targets["severity"])
                    severity_payload["predictions"].append(int(torch.argmax(probabilities).item()))
                    severity_payload["probabilities"].append(probabilities.tolist())
    return {"binary": binary_payload, "multiclass": {"severity": severity_payload}}


def bootstrap_graph_model_artifacts(artifact_path: Path) -> dict[str, object]:
    return train_multidataset_model_artifact(
        artifact_path=artifact_path,
        metrics_path=GRAPH_MODEL_METRICS_PATH,
        seed=GRAPH_MODEL_TRAINING_SEED,
        epochs=max(6, min(GRAPH_MODEL_TRAINING_EPOCHS, 10)),
        size_profile="quick",
    )


def _run_training_epoch(
    *,
    model: RelationAwareGraphModel,
    data_loader,
    optimizer,
    vocabulary,
    normalization,
    binary_criteria: dict[str, torch.nn.BCEWithLogitsLoss],
    severity_criterion: torch.nn.CrossEntropyLoss,
    binary_loss_weights: dict[str, float],
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    active_samples = 0
    head_loss_totals = {head_name: 0.0 for head_name in BINARY_HEADS}
    head_loss_totals["severity"] = 0.0
    head_counts = {head_name: 0 for head_name in BINARY_HEADS}
    head_counts["severity"] = 0
    grad_norm_total = 0.0
    optimizer.zero_grad()

    for batch in data_loader:
        batch_loss: torch.Tensor | None = None
        batch_active = 0
        for sample in batch:
            encoded = encode_sample(sample.graph, sample.features, vocabulary, normalization)
            outputs = model(encoded)
            sample_loss, sample_head_losses = _compute_sample_loss(
                sample=sample,
                outputs=outputs,
                binary_criteria=binary_criteria,
                severity_criterion=severity_criterion,
                binary_loss_weights=binary_loss_weights,
            )
            if sample_loss is None:
                continue
            batch_loss = sample_loss if batch_loss is None else batch_loss + sample_loss
            batch_active += 1
            active_samples += 1
            total_loss += float(sample_loss.detach().item())
            for head_name, loss_value in sample_head_losses.items():
                head_loss_totals[head_name] += float(loss_value)
                head_counts[head_name] += 1

        if batch_loss is None or batch_active == 0:
            continue
        (batch_loss / batch_active).backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm).item())
        grad_norm_total += grad_norm
        optimizer.step()
        optimizer.zero_grad()

    metrics = {
        "loss_total": round(total_loss / max(active_samples, 1), 6),
        "grad_norm": round(grad_norm_total / max(len(data_loader), 1), 6),
        "active_samples": float(active_samples),
    }
    for head_name, total in head_loss_totals.items():
        metrics[f"loss_{head_name}"] = round(total / max(head_counts[head_name], 1), 6)
    return metrics


def _compute_sample_loss(
    *,
    sample: UnifiedTrainingSample,
    outputs,
    binary_criteria: dict[str, torch.nn.BCEWithLogitsLoss],
    severity_criterion: torch.nn.CrossEntropyLoss,
    binary_loss_weights: dict[str, float],
) -> tuple[torch.Tensor | None, dict[str, float]]:
    total_loss: torch.Tensor | None = None
    head_losses: dict[str, float] = {}

    for head_name in BINARY_HEADS:
        if not sample.binary_target_mask.get(head_name, False):
            continue
        target = torch.tensor([sample.binary_targets[head_name]], dtype=torch.float32)
        loss = binary_criteria[head_name](outputs.binary_logits[head_name].unsqueeze(0), target)
        weighted_loss = loss * binary_loss_weights.get(head_name, 1.0) * sample.sample_weight
        total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        head_losses[head_name] = float(weighted_loss.detach().item())

    if sample.multiclass_target_mask.get("severity", False):
        target = torch.tensor([sample.multiclass_targets["severity"]], dtype=torch.long)
        loss = severity_criterion(outputs.multiclass_logits["severity"].unsqueeze(0), target)
        weighted_loss = loss * DEFAULT_MULTICLASS_LOSS_WEIGHTS["severity"] * sample.sample_weight
        total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        head_losses["severity"] = float(weighted_loss.detach().item())

    return total_loss, head_losses


def _build_binary_loss_fns(samples: list[UnifiedTrainingSample]) -> dict[str, torch.nn.BCEWithLogitsLoss]:
    loss_fns = {}
    for head_name in BINARY_HEADS:
        positives = 0.0
        total = 0
        for sample in samples:
            if not sample.binary_target_mask.get(head_name, False):
                continue
            positives += sample.binary_targets[head_name]
            total += 1
        negatives = max(total - positives, 1.0)
        positives = max(positives, 1.0)
        pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)
        loss_fns[head_name] = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fns


def _build_severity_loss_fn(samples: list[UnifiedTrainingSample]) -> torch.nn.CrossEntropyLoss:
    counts = np.ones(len(SEVERITY_LABELS), dtype=np.float32)
    for sample in samples:
        if not sample.multiclass_target_mask.get("severity", False):
            continue
        counts[sample.multiclass_targets["severity"]] += 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))


def _compute_eval_loss(predictions: dict[str, Any], binary_criteria, severity_criterion) -> float:
    total = 0.0
    count = 0
    for head_name, payload in predictions["binary"].items():
        for target, score in zip(payload["targets"], payload["scores"], strict=True):
            probability = min(max(score, 1e-6), 1 - 1e-6)
            logit = torch.logit(torch.tensor([probability], dtype=torch.float32))
            target_tensor = torch.tensor([target], dtype=torch.float32)
            total += float(binary_criteria[head_name](logit, target_tensor).item())
            count += 1
    for target, probabilities in zip(
        predictions["multiclass"]["severity"]["targets"],
        predictions["multiclass"]["severity"]["probabilities"],
        strict=True,
    ):
        logits = torch.log(torch.tensor([probabilities], dtype=torch.float32))
        target_tensor = torch.tensor([target], dtype=torch.long)
        total += float(severity_criterion(logits, target_tensor).item())
        count += 1
    return total / max(count, 1)


def _compute_binary_metrics(*, targets: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    if targets.size == 0:
        return {"threshold": threshold, "support": 0}
    predictions = (scores >= threshold).astype(int)
    precision, recall, f1_value, _ = precision_recall_fscore_support(
        targets.astype(int),
        predictions,
        average="binary",
        zero_division=0,
    )
    metrics = {
        "threshold": round(float(threshold), 4),
        "support": int(targets.sum()),
        "count": int(targets.shape[0]),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1_value), 4),
        "accuracy": round(float(accuracy_score(targets.astype(int), predictions)), 4),
        "positive_rate": round(float(targets.mean()), 4),
    }
    if len(np.unique(targets.astype(int))) > 1:
        metrics["average_precision"] = round(float(average_precision_score(targets, scores)), 4)
        metrics["roc_auc"] = round(float(roc_auc_score(targets, scores)), 4)
    return metrics


def _compute_severity_metrics(*, targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    if targets.size == 0:
        return {"count": 0}
    metrics = {
        "count": int(targets.shape[0]),
        "accuracy": round(float(accuracy_score(targets, predictions)), 4),
        "macro_f1": round(float(f1_score(targets, predictions, average="macro", zero_division=0)), 4),
    }
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=np.arange(len(SEVERITY_LABELS)),
        zero_division=0,
    )
    for index, label in enumerate(SEVERITY_LABELS):
        metrics[f"{label}_precision"] = round(float(per_class_precision[index]), 4)
        metrics[f"{label}_recall"] = round(float(per_class_recall[index]), 4)
        metrics[f"{label}_f1"] = round(float(per_class_f1[index]), 4)
        metrics[f"{label}_support"] = int(per_class_support[index])
    return metrics


def _select_thresholds(binary_predictions: dict[str, dict[str, list[float]]]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    search_space = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65)
    for head_name in MAIN_BINARY_HEADS:
        payload = binary_predictions[head_name]
        targets = np.array(payload["targets"], dtype=np.int64)
        scores = np.array(payload["scores"], dtype=np.float32)
        if targets.size == 0:
            thresholds[head_name] = 0.5
            continue
        best_threshold = 0.5
        best_score = -1.0
        for candidate in search_space:
            predictions = (scores >= candidate).astype(int)
            score = f1_score(targets, predictions, zero_division=0)
            if score > best_score:
                best_threshold = candidate
                best_score = score
        thresholds[head_name] = best_threshold
    for head_name in AUXILIARY_BINARY_HEADS:
        thresholds[head_name] = 0.5
    return thresholds


def _resolve_size_limits(*, size_profile: str, dataset_size: int | None) -> dict[str, dict[str, int]]:
    if dataset_size is not None:
        per_dataset = max(dataset_size // 5, 16)
        val_test = max(per_dataset // 3, 8)
        return {
            "train": {name: per_dataset for name in DEFAULT_DATASET_WEIGHTS},
            "val": {name: val_test for name in DEFAULT_DATASET_WEIGHTS},
            "test": {name: val_test for name in DEFAULT_DATASET_WEIGHTS},
        }
    if size_profile not in SIZE_PROFILES:
        raise ValueError(f"Unsupported size profile: {size_profile}")
    return SIZE_PROFILES[size_profile]


def _prefix_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def _log_dataset_summary(logger: SwanLabLogger, train_dataset: MultiDatasetTrainingSet, val_dataset: MultiDatasetTrainingSet, test_dataset: MultiDatasetTrainingSet) -> None:
    if not logger.enabled:
        return
    payload: dict[str, Any] = {
        "epoch": 0,
        "dataset/train_total": train_dataset.summarize("train").total_samples,
        "dataset/val_total": val_dataset.summarize("val").total_samples,
        "dataset/test_total": test_dataset.summarize("test").total_samples,
    }
    for split_name, dataset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        summary = dataset.summarize(split_name)
        for dataset_name, count in summary.dataset_counts.items():
            payload[f"dataset/{split_name}/{dataset_name}"] = count
    logger.log(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export the ChainSentry multi-dataset graph model artifact.")
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path, default=GRAPH_MODEL_METRICS_PATH)
    parser.add_argument("--seed", type=int, default=GRAPH_MODEL_TRAINING_SEED)
    parser.add_argument("--epochs", type=int, default=GRAPH_MODEL_TRAINING_EPOCHS)
    parser.add_argument("--size-profile", choices=tuple(SIZE_PROFILES), default="standard")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--enable-swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="chainsentry")
    parser.add_argument("--swanlab-run-name", default=None)
    args = parser.parse_args()

    train_multidataset_model_artifact(
        artifact_path=args.artifact_path,
        metrics_path=args.metrics_path,
        seed=args.seed,
        epochs=args.epochs,
        size_profile=args.size_profile,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        swanlab_project=args.swanlab_project,
        swanlab_run_name=args.swanlab_run_name,
        enable_swanlab=args.enable_swanlab,
    )


if __name__ == "__main__":
    main()

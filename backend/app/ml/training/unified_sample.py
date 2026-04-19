from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.ml.features import ScalarFeatureSet, extract_scalar_features
from app.ml.graph_builder import TransactionGraph, build_transaction_graph
from app.models import NormalizedTransaction, SimulationSummary, TransactionRequest
from app.services.parser import parse_transaction
from app.services.simulation import simulation_engine


SEVERITY_LABELS = ("low", "medium", "high", "critical")
MAIN_BINARY_HEADS = ("approval", "destination", "simulation")
AUXILIARY_BINARY_HEADS = ("address_malicious", "failure_aux")
BINARY_HEADS = MAIN_BINARY_HEADS + AUXILIARY_BINARY_HEADS
MULTICLASS_HEADS = ("severity",)

MetadataValue = str | int | float | bool | None


@dataclass(frozen=True)
class UnifiedTrainingSample:
    dataset_name: str
    sample_id: str
    normalized_transaction: NormalizedTransaction
    simulation: SimulationSummary
    graph: TransactionGraph
    features: ScalarFeatureSet
    binary_targets: dict[str, float] = field(default_factory=dict)
    binary_target_mask: dict[str, bool] = field(default_factory=dict)
    multiclass_targets: dict[str, int] = field(default_factory=dict)
    multiclass_target_mask: dict[str, bool] = field(default_factory=dict)
    sample_weight: float = 1.0
    metadata: dict[str, MetadataValue] = field(default_factory=dict)


def empty_binary_targets() -> dict[str, float]:
    return {head_name: 0.0 for head_name in BINARY_HEADS}


def empty_binary_mask() -> dict[str, bool]:
    return {head_name: False for head_name in BINARY_HEADS}


def empty_multiclass_targets() -> dict[str, int]:
    return {"severity": 0}


def empty_multiclass_mask() -> dict[str, bool]:
    return {head_name: False for head_name in MULTICLASS_HEADS}


def severity_label_to_index(label: str) -> int:
    normalized = label.strip().lower()
    if normalized not in SEVERITY_LABELS:
        raise ValueError(f"Unsupported severity label: {label}")
    return SEVERITY_LABELS.index(normalized)


def build_unified_training_sample(
    *,
    dataset_name: str,
    sample_id: str,
    request: TransactionRequest,
    binary_targets: dict[str, float] | None = None,
    binary_target_mask: dict[str, bool] | None = None,
    multiclass_targets: dict[str, int] | None = None,
    multiclass_target_mask: dict[str, bool] | None = None,
    sample_weight: float = 1.0,
    metadata: dict[str, MetadataValue] | None = None,
) -> UnifiedTrainingSample:
    normalized_transaction = parse_transaction(request)
    simulation = simulation_engine.simulate(normalized_transaction, request.simulation_profile)
    graph = build_transaction_graph(normalized_transaction, simulation)
    features = extract_scalar_features(normalized_transaction, simulation, graph)
    return UnifiedTrainingSample(
        dataset_name=dataset_name,
        sample_id=sample_id,
        normalized_transaction=normalized_transaction,
        simulation=simulation,
        graph=graph,
        features=features,
        binary_targets=_merge_binary_targets(binary_targets),
        binary_target_mask=_merge_binary_mask(binary_target_mask),
        multiclass_targets=_merge_multiclass_targets(multiclass_targets),
        multiclass_target_mask=_merge_multiclass_mask(multiclass_target_mask),
        sample_weight=sample_weight,
        metadata=dict(metadata or {}),
    )


def _merge_binary_targets(values: dict[str, float] | None) -> dict[str, float]:
    merged = empty_binary_targets()
    for key, value in (values or {}).items():
        if key not in merged:
            raise KeyError(f"Unknown binary target head: {key}")
        merged[key] = float(value)
    return merged


def _merge_binary_mask(values: dict[str, bool] | None) -> dict[str, bool]:
    merged = empty_binary_mask()
    for key, value in (values or {}).items():
        if key not in merged:
            raise KeyError(f"Unknown binary target mask: {key}")
        merged[key] = bool(value)
    return merged


def _merge_multiclass_targets(values: dict[str, int] | None) -> dict[str, int]:
    merged = empty_multiclass_targets()
    for key, value in (values or {}).items():
        if key not in merged:
            raise KeyError(f"Unknown multiclass target head: {key}")
        merged[key] = int(value)
    return merged


def _merge_multiclass_mask(values: dict[str, bool] | None) -> dict[str, bool]:
    merged = empty_multiclass_mask()
    for key, value in (values or {}).items():
        if key not in merged:
            raise KeyError(f"Unknown multiclass target mask: {key}")
        merged[key] = bool(value)
    return merged


def summarize_target_coverage(samples: list[UnifiedTrainingSample]) -> dict[str, Any]:
    binary_mask_totals = {head_name: 0 for head_name in BINARY_HEADS}
    binary_positive_totals = {head_name: 0.0 for head_name in BINARY_HEADS}
    multiclass_mask_totals = {head_name: 0 for head_name in MULTICLASS_HEADS}
    severity_label_totals = {label: 0 for label in SEVERITY_LABELS}

    for sample in samples:
        for head_name, enabled in sample.binary_target_mask.items():
            if not enabled:
                continue
            binary_mask_totals[head_name] += 1
            binary_positive_totals[head_name] += sample.binary_targets[head_name]
        for head_name, enabled in sample.multiclass_target_mask.items():
            if not enabled:
                continue
            multiclass_mask_totals[head_name] += 1
            if head_name == "severity":
                severity_label_totals[SEVERITY_LABELS[sample.multiclass_targets[head_name]]] += 1

    return {
        "binary_mask_totals": binary_mask_totals,
        "binary_positive_totals": binary_positive_totals,
        "multiclass_mask_totals": multiclass_mask_totals,
        "severity_label_totals": severity_label_totals,
    }

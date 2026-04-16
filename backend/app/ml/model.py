from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.ml.features import ScalarFeatureSet
from app.ml.graph_builder import TransactionGraph


@dataclass(frozen=True)
class GraphModelMetadata:
    architecture: str
    framework: str
    artifact_path: Path
    node_types: tuple[str, ...] = (
        "transaction",
        "address",
        "contract",
        "token",
        "effect",
    )
    edge_types: tuple[str, ...] = (
        "initiates",
        "targets",
        "approves",
        "requests_allowance_for",
        "transfers_value_to",
        "transfers_token_to",
        "routes_to",
        "grants_operator_to",
        "grants_privilege_to",
        "triggers_effect",
    )
    task_heads: tuple[str, ...] = (
        "approval",
        "destination",
        "simulation",
        "severity",
    )


@dataclass(frozen=True)
class GraphPredictionScores:
    category_scores: dict[str, float] = field(default_factory=dict)
    severity_scores: dict[str, float] = field(default_factory=dict)


class GraphRiskModel(Protocol):
    metadata: GraphModelMetadata

    def predict(self, graph: TransactionGraph, features: ScalarFeatureSet) -> GraphPredictionScores:
        ...


class PlaceholderRGCNModel:
    def __init__(self, metadata: GraphModelMetadata) -> None:
        self.metadata = metadata

    def predict(self, graph: TransactionGraph, features: ScalarFeatureSet) -> GraphPredictionScores:
        raise NotImplementedError(
            "No trained graph model artifact is included in the Student 1 handoff. "
            "Student 2 should replace PlaceholderRGCNModel with a trained relation-aware graph model."
        )
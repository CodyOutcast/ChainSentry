from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from app.config import GRAPH_MODEL_ARTIFACT_PATH, PREDICTOR_BACKEND
from app.content import explanation_templates
from app.ml.features import ScalarFeatureSet, extract_scalar_features
from app.ml.graph_builder import TransactionGraph, build_transaction_graph
from app.ml.model import GraphPredictionScores, LoadedGraphRiskModel
from app.ml.training import bootstrap_graph_model_artifacts
from app.models import (
    NormalizedTransaction,
    RiskCategory,
    RiskFinding,
    Severity,
    SimulationSummary,
    TransactionKind,
)
from app.services.detectors import run_detectors


@dataclass(frozen=True)
class PredictorResult:
    source: str
    graph: TransactionGraph
    features: ScalarFeatureSet
    findings: list[RiskFinding]
    scores: GraphPredictionScores | None = None


class RiskPredictor(Protocol):
    name: str

    def predict(
        self,
        transaction: NormalizedTransaction,
        simulation: SimulationSummary,
    ) -> PredictorResult:
        ...


class HeuristicFallbackPredictor:
    name = "heuristic-fallback"

    def predict(
        self,
        transaction: NormalizedTransaction,
        simulation: SimulationSummary,
    ) -> PredictorResult:
        graph = build_transaction_graph(transaction, simulation)
        features = extract_scalar_features(transaction, simulation, graph)
        findings = run_detectors(transaction, simulation)
        return PredictorResult(
            source=self.name,
            graph=graph,
            features=features,
            findings=findings,
            scores=None,
        )


class GraphModelPredictor:
    name = "graph-model"

    def __init__(self) -> None:
        if not GRAPH_MODEL_ARTIFACT_PATH.exists():
            bootstrap_graph_model_artifacts(GRAPH_MODEL_ARTIFACT_PATH)
        try:
            self._model = LoadedGraphRiskModel.load(GRAPH_MODEL_ARTIFACT_PATH)
        except Exception:
            bootstrap_graph_model_artifacts(GRAPH_MODEL_ARTIFACT_PATH)
            self._model = LoadedGraphRiskModel.load(GRAPH_MODEL_ARTIFACT_PATH)

    def predict(
        self,
        transaction: NormalizedTransaction,
        simulation: SimulationSummary,
    ) -> PredictorResult:
        graph = build_transaction_graph(transaction, simulation)
        features = extract_scalar_features(transaction, simulation, graph)
        scores = self._model.predict(graph, features)
        heuristic_findings = run_detectors(transaction, simulation)
        findings = _merge_model_and_findings(
            transaction=transaction,
            simulation=simulation,
            heuristic_findings=heuristic_findings,
            scores=scores,
            thresholds=self._model.thresholds,
            graph=graph,
        )
        return PredictorResult(
            source=self.name,
            graph=graph,
            features=features,
            findings=findings,
            scores=scores,
        )


@lru_cache(maxsize=2)
def get_predictor() -> RiskPredictor:
    if PREDICTOR_BACKEND == "heuristic-fallback":
        return HeuristicFallbackPredictor()
    try:
        return GraphModelPredictor()
    except Exception:
        return HeuristicFallbackPredictor()
    return HeuristicFallbackPredictor()


def _merge_model_and_findings(
    *,
    transaction: NormalizedTransaction,
    simulation: SimulationSummary,
    heuristic_findings: list[RiskFinding],
    scores: GraphPredictionScores,
    thresholds: dict[str, float],
    graph: TransactionGraph,
) -> list[RiskFinding]:
    grouped: dict[RiskCategory, list[RiskFinding]] = defaultdict(list)
    for finding in heuristic_findings:
        grouped[finding.category].append(finding)

    merged: list[RiskFinding] = []
    for category in (RiskCategory.approval, RiskCategory.destination, RiskCategory.simulation):
        category_findings = grouped.get(category, [])
        evidence = _model_evidence(category, scores, graph)
        if category_findings:
            merged.extend(_append_model_evidence(category_findings, evidence))
            continue
        probability = scores.category_scores.get(category.value, 0.0)
        if probability < thresholds.get(category.value, 0.5):
            continue
        generic = _build_generic_model_finding(
            transaction,
            simulation,
            category,
            probability,
            _dominant_model_severity(scores),
            evidence,
        )
        if generic is not None:
            merged.append(generic)

    return merged


def _append_model_evidence(findings: list[RiskFinding], evidence: list[str]) -> list[RiskFinding]:
    result: list[RiskFinding] = []
    for finding in findings:
        merged_evidence = finding.evidence + [item for item in evidence if item not in finding.evidence]
        result.append(finding.model_copy(update={"evidence": merged_evidence}))
    return result


def _build_generic_model_finding(
    transaction: NormalizedTransaction,
    simulation: SimulationSummary,
    category: RiskCategory,
    probability: float,
    severity: Severity,
    evidence: list[str],
) -> RiskFinding | None:
    if severity == Severity.low:
        return None
    if category == RiskCategory.approval and transaction.transaction_kind.value == "approval":
        return explanation_templates.model_approval_signal(transaction, probability, severity, evidence)
    if category == RiskCategory.destination and transaction.transaction_kind in {
        TransactionKind.approval,
        TransactionKind.transfer,
        TransactionKind.native_transfer,
        TransactionKind.swap,
    }:
        return explanation_templates.model_destination_signal(transaction, probability, severity, evidence)
    if category == RiskCategory.simulation and (simulation.triggered or simulation.profile.value != "none"):
        return explanation_templates.model_simulation_signal(transaction, probability, severity, evidence)
    return None


def _model_evidence(category: RiskCategory, scores: GraphPredictionScores, graph: TransactionGraph) -> list[str]:
    severity_profile = ", ".join(
        f"{label}:{score:.2f}" for label, score in scores.severity_scores.items()
    )
    return [
        f"Graph model {category.value} score: {scores.category_scores.get(category.value, 0.0):.2f}",
        f"Graph shape: {graph.node_count} nodes, {graph.edge_count} edges",
        f"Graph model severity profile: {severity_profile}",
    ]


def _dominant_model_severity(scores: GraphPredictionScores) -> Severity:
    if not scores.severity_scores:
        return Severity.low
    label = max(scores.severity_scores.items(), key=lambda item: item[1])[0]
    return Severity(label)

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.config import GRAPH_MODEL_ARCHITECTURE, GRAPH_MODEL_ARTIFACT_PATH, PREDICTOR_BACKEND
from app.ml.features import ScalarFeatureSet, extract_scalar_features
from app.ml.graph_builder import TransactionGraph, build_transaction_graph
from app.ml.model import GraphModelMetadata, PlaceholderRGCNModel
from app.models import NormalizedTransaction, RiskFinding, SimulationSummary
from app.services.detectors import run_detectors


@dataclass(frozen=True)
class PredictorResult:
    source: str
    graph: TransactionGraph
    features: ScalarFeatureSet
    findings: list[RiskFinding]


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
        )


class GraphModelPredictor:
    name = "graph-model"

    def __init__(self) -> None:
        self._model = PlaceholderRGCNModel(
            GraphModelMetadata(
                architecture=GRAPH_MODEL_ARCHITECTURE,
                framework="pytorch-geometric",
                artifact_path=GRAPH_MODEL_ARTIFACT_PATH,
            )
        )

    def predict(
        self,
        transaction: NormalizedTransaction,
        simulation: SimulationSummary,
    ) -> PredictorResult:
        graph = build_transaction_graph(transaction, simulation)
        features = extract_scalar_features(transaction, simulation, graph)
        self._model.predict(graph, features)
        raise NotImplementedError("Student 2 should replace GraphModelPredictor with trained graph-model inference.")


def get_predictor() -> RiskPredictor:
    if PREDICTOR_BACKEND == "graph-model":
        return GraphModelPredictor()
    return HeuristicFallbackPredictor()
from __future__ import annotations

from dataclasses import dataclass, field

from app.ml.graph_builder import TransactionGraph
from app.models import NormalizedTransaction, SimulationSummary


@dataclass(frozen=True)
class ScalarFeatureSet:
    numeric: dict[str, float] = field(default_factory=dict)
    categorical: dict[str, str] = field(default_factory=dict)
    boolean: dict[str, bool] = field(default_factory=dict)


def extract_scalar_features(
    transaction: NormalizedTransaction,
    simulation: SimulationSummary,
    graph: TransactionGraph,
) -> ScalarFeatureSet:
    return ScalarFeatureSet(
        numeric={
            "chain_id": float(transaction.chain_id),
            "value_eth": float(transaction.value_eth),
            "token_amount": float(transaction.token_amount or 0.0),
            "approval_amount": float(transaction.approval_amount or 0.0),
            "graph_node_count": float(graph.node_count),
            "graph_edge_count": float(graph.edge_count),
            "simulation_effect_count": float(len(simulation.effects)),
        },
        categorical={
            "transaction_kind": transaction.transaction_kind.value,
            "method_name": transaction.method_name.lower(),
            "selector": transaction.selector or "none",
            "simulation_profile": simulation.profile.value,
        },
        boolean={
            "has_spender": transaction.spender_address is not None,
            "has_token": transaction.token_symbol is not None,
            "simulation_triggered": simulation.triggered,
            "has_selector": transaction.selector is not None,
        },
    )
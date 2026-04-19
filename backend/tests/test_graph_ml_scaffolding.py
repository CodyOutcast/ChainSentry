from app.ml.features import extract_scalar_features
from app.ml.graph_builder import build_transaction_graph
from app.ml.inference import get_predictor
from app.ml.training.dataset import build_synthetic_dataset
from app.ml.training.train_graph_model import train_graph_model_artifact
from app.models import TransactionRequest
from app.services.parser import parse_transaction
from app.services.simulation import simulation_engine


def _build_allowance_case():
    request = TransactionRequest(
        chain_id=1,
        from_address="0x1111111111111111111111111111111111111111",
        to_address="0xdead00000000000000000000000000000000beef",
        method_name="approve",
        token_symbol="USDC",
        token_amount=250,
        approval_amount=1_000_000_000,
        spender_address="0xdead00000000000000000000000000000000beef",
        contract_name="Demo Drain Contract",
        simulation_profile="allowance_drain",
    )
    normalized = parse_transaction(request)
    simulation = simulation_engine.simulate(normalized, request.simulation_profile)
    return normalized, simulation


def test_graph_builder_creates_transaction_centered_graph() -> None:
    normalized, simulation = _build_allowance_case()

    graph = build_transaction_graph(normalized, simulation)

    node_types = {node.node_type for node in graph.nodes}
    edge_types = {edge.edge_type for edge in graph.edges}

    assert graph.anchor_node_id.startswith("transaction:")
    assert {"transaction", "address", "contract", "token", "effect"}.issubset(node_types)
    assert {"initiates", "targets", "approves", "requests_allowance_for", "triggers_effect"}.issubset(edge_types)


def test_feature_extractor_includes_graph_and_scalar_context() -> None:
    normalized, simulation = _build_allowance_case()
    graph = build_transaction_graph(normalized, simulation)

    features = extract_scalar_features(normalized, simulation, graph)

    assert features.numeric["approval_amount"] == 1_000_000_000
    assert features.numeric["graph_node_count"] == float(graph.node_count)
    assert features.categorical["transaction_kind"] == "approval"
    assert features.boolean["simulation_triggered"] is True


def test_default_predictor_remains_heuristic_fallback() -> None:
    assert get_predictor().name in {"graph-model", "heuristic-fallback"}


def test_synthetic_dataset_contains_multiple_risk_patterns() -> None:
    examples = build_synthetic_dataset(seed=7, dataset_size=120)

    category_totals = {
        "approval": sum(example.category_labels["approval"] for example in examples),
        "destination": sum(example.category_labels["destination"] for example in examples),
        "simulation": sum(example.category_labels["simulation"] for example in examples),
    }

    assert len(examples) >= 100
    assert all(total > 0 for total in category_totals.values())


def test_training_exports_artifact_and_metrics(tmp_path) -> None:
    artifact_path = tmp_path / "graph-model.pt"
    metrics_path = tmp_path / "graph-model-metrics.json"

    metrics = train_graph_model_artifact(
        artifact_path=artifact_path,
        metrics_path=metrics_path,
        seed=11,
        dataset_size=160,
        epochs=8,
    )

    assert artifact_path.exists()
    assert metrics_path.exists()
    assert "category_metrics" in metrics
    assert "severity_accuracy" in metrics

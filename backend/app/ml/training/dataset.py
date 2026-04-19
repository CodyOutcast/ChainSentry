from __future__ import annotations

from dataclasses import dataclass
from random import Random

from app.ml.features import ScalarFeatureSet, extract_scalar_features
from app.ml.graph_builder import TransactionGraph, build_transaction_graph
from app.models import NormalizedTransaction, RiskCategory, Severity, SimulationSummary, TransactionRequest
from app.services.detectors import run_detectors
from app.services.parser import parse_transaction
from app.services.simulation import simulation_engine


SEVERITY_ORDER = {
    Severity.low: 0,
    Severity.medium: 1,
    Severity.high: 2,
    Severity.critical: 3,
}

TOKENS = ("USDC", "USDT", "WETH", "DAI", "UNI")


@dataclass(frozen=True)
class TrainingExample:
    scenario_name: str
    normalized_transaction: NormalizedTransaction
    simulation: SimulationSummary
    graph: TransactionGraph
    features: ScalarFeatureSet
    category_labels: dict[str, float]
    severity_label: str


def build_synthetic_dataset(seed: int = 17, dataset_size: int = 640) -> list[TrainingExample]:
    rng = Random(seed)
    scenario_mix = (
        ("clean_transfer", 0.18, _build_clean_transfer),
        ("clean_contract_call", 0.10, _build_clean_contract_call),
        ("small_approval", 0.12, _build_small_approval),
        ("large_approval", 0.12, _build_large_approval),
        ("flagged_transfer", 0.08, _build_flagged_transfer),
        ("flagged_swap", 0.10, _build_flagged_swap),
        ("unlimited_approval_flagged", 0.12, _build_unlimited_approval_flagged),
        ("operator_takeover", 0.08, _build_operator_takeover),
        ("privilege_escalation", 0.05, _build_privilege_escalation),
        ("unexpected_outflow", 0.05, _build_unexpected_outflow),
    )

    examples: list[TrainingExample] = []
    for scenario_name, weight, builder in scenario_mix:
        count = max(8, int(dataset_size * weight))
        for index in range(count):
            request = builder(rng, index)
            examples.append(_materialize_training_example(scenario_name, request))
    return examples


def _materialize_training_example(scenario_name: str, request: TransactionRequest) -> TrainingExample:
    normalized = parse_transaction(request)
    simulation = simulation_engine.simulate(normalized, request.simulation_profile)
    graph = build_transaction_graph(normalized, simulation)
    features = extract_scalar_features(normalized, simulation, graph)
    findings = run_detectors(normalized, simulation)
    categories = {finding.category.value for finding in findings}
    severity = max((finding.severity for finding in findings), key=lambda item: SEVERITY_ORDER[item], default=Severity.low)
    return TrainingExample(
        scenario_name=scenario_name,
        normalized_transaction=normalized,
        simulation=simulation,
        graph=graph,
        features=features,
        category_labels={category.value: 1.0 if category.value in categories else 0.0 for category in RiskCategory},
        severity_label=severity.value,
    )


def _build_clean_transfer(rng: Random, index: int) -> TransactionRequest:
    token_symbol = TOKENS[index % len(TOKENS)]
    return TransactionRequest(
        chain_id=1 if index % 4 else 11155111,
        from_address=_wallet_address(index),
        to_address=_safe_address(rng, index),
        method_name="transfer",
        token_symbol=token_symbol,
        token_amount=round(rng.uniform(0.05, 4.5), 4),
        contract_name=f"Known wallet {index % 11}",
        interaction_label="Send funds to a known address",
        notes="Synthetic clean transfer example.",
        simulation_profile="none",
    )


def _build_clean_contract_call(rng: Random, index: int) -> TransactionRequest:
    return TransactionRequest(
        chain_id=1,
        from_address=_wallet_address(index + 500),
        to_address=_safe_address(rng, index + 400),
        method_name="contractCall",
        value_eth=round(rng.uniform(0.0, 0.3), 4),
        contract_name=f"Routine dApp {index % 9}",
        interaction_label="Routine contract interaction",
        notes="Synthetic low-risk contract call.",
        simulation_profile="none",
    )


def _build_small_approval(rng: Random, index: int) -> TransactionRequest:
    token_symbol = TOKENS[(index + 1) % len(TOKENS)]
    spender = _safe_address(rng, index + 700)
    return TransactionRequest(
        chain_id=1,
        from_address=_wallet_address(index + 1000),
        to_address=spender,
        method_name="approve",
        token_symbol=token_symbol,
        token_amount=round(rng.uniform(10, 250), 2),
        approval_amount=round(rng.uniform(50, 500), 2),
        spender_address=spender,
        contract_name=f"Trusted spender {index % 7}",
        interaction_label="Approve exact spend amount",
        notes="Synthetic small approval example.",
        simulation_profile="none",
    )


def _build_large_approval(rng: Random, index: int) -> TransactionRequest:
    token_symbol = TOKENS[(index + 2) % len(TOKENS)]
    spender = _safe_address(rng, index + 1400)
    return TransactionRequest(
        chain_id=1,
        from_address=_wallet_address(index + 1500),
        to_address=spender,
        method_name="approve",
        token_symbol=token_symbol,
        token_amount=round(rng.uniform(20, 350), 2),
        approval_amount=round(rng.uniform(20_000, 250_000), 2),
        spender_address=spender,
        contract_name=f"Broad spender {index % 5}",
        interaction_label="Approve liquidity or routing access",
        notes="Synthetic large approval example.",
        simulation_profile="none",
    )


def _build_flagged_transfer(rng: Random, index: int) -> TransactionRequest:
    to_address = "0x0bad000000000000000000000000000000c0de00"
    return TransactionRequest(
        chain_id=1,
        from_address=_wallet_address(index + 1900),
        to_address=to_address,
        method_name="transfer",
        token_symbol="ETH",
        token_amount=round(rng.uniform(0.3, 1.8), 4),
        contract_name="Demo Phishing Router",
        interaction_label="Send assets to a campaign address",
        notes="Synthetic flagged-destination transfer example.",
        simulation_profile="none",
    )


def _build_flagged_swap(rng: Random, index: int) -> TransactionRequest:
    router = "0x0bad000000000000000000000000000000c0de00"
    return TransactionRequest(
        chain_id=1 if index % 3 else 11155111,
        from_address=_wallet_address(index + 2200),
        to_address=router,
        method_name="swapExactTokensForTokens",
        token_symbol=TOKENS[index % len(TOKENS)],
        token_amount=round(rng.uniform(80, 800), 2),
        spender_address=router,
        contract_name="Demo Phishing Router",
        interaction_label="Swap via campaign router",
        notes="Synthetic flagged router example.",
        simulation_profile="unexpected_outflow" if index % 2 else "none",
    )


def _build_unlimited_approval_flagged(rng: Random, index: int) -> TransactionRequest:
    spender = "0xdead00000000000000000000000000000000beef"
    token_symbol = TOKENS[index % len(TOKENS)]
    return TransactionRequest(
        chain_id=1 if index % 2 else 11155111,
        from_address=_wallet_address(index + 2600),
        to_address=spender,
        method_name="approve",
        token_symbol=token_symbol,
        token_amount=round(rng.uniform(50, 450), 2),
        approval_amount=1_000_000_000,
        spender_address=spender,
        contract_name="Demo Drain Contract",
        interaction_label="Claim campaign rewards",
        notes="Synthetic unlimited approval to flagged spender example.",
        simulation_profile="allowance_drain",
    )


def _build_operator_takeover(rng: Random, index: int) -> TransactionRequest:
    return TransactionRequest(
        chain_id=1,
        from_address=_wallet_address(index + 3000),
        to_address=_safe_address(rng, index + 3010),
        method_name="setApprovalForAll",
        spender_address=_safe_address(rng, index + 3020),
        contract_name=f"Gallery contract {index % 6}",
        interaction_label="Approve collection-wide operator",
        notes="Synthetic operator control example.",
        simulation_profile="privilege_escalation",
    )


def _build_privilege_escalation(rng: Random, index: int) -> TransactionRequest:
    return TransactionRequest(
        chain_id=1,
        from_address=_wallet_address(index + 3300),
        to_address=_safe_address(rng, index + 3310),
        method_name="grantRole",
        contract_name=f"Vault admin module {index % 4}",
        interaction_label="Grant backend role",
        notes="Synthetic privilege-escalation example.",
        simulation_profile="privilege_escalation",
    )


def _build_unexpected_outflow(rng: Random, index: int) -> TransactionRequest:
    router = _safe_address(rng, index + 3600)
    return TransactionRequest(
        chain_id=1,
        from_address=_wallet_address(index + 3500),
        to_address=router,
        method_name="swapExactTokensForTokens",
        token_symbol=TOKENS[(index + 3) % len(TOKENS)],
        token_amount=round(rng.uniform(120, 950), 2),
        spender_address=router,
        contract_name=f"Shadow router {index % 8}",
        interaction_label="Swap through an unfamiliar route",
        notes="Synthetic unexpected outflow example.",
        simulation_profile="unexpected_outflow",
    )


def _wallet_address(index: int) -> str:
    return _hex_address(0x1111111111111111111111111111111111111111 + index)


def _safe_address(rng: Random, index: int) -> str:
    base = 0x7000000000000000000000000000000000000000 + index * 97 + rng.randint(1, 88)
    address = _hex_address(base)
    if address in {
        "0xdead00000000000000000000000000000000beef",
        "0x0bad000000000000000000000000000000c0de00",
    }:
        return _safe_address(rng, index + 1)
    return address


def _hex_address(value: int) -> str:
    return f"0x{value:040x}"[-42:]

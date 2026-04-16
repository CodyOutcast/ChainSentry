from __future__ import annotations

import json
from pathlib import Path

from app.config import LARGE_APPROVAL_THRESHOLD, UNLIMITED_APPROVAL_THRESHOLD
from app.content import explanation_templates
from app.models import EffectType, NormalizedTransaction, RiskFinding, Severity, SimulationSummary

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "flagged_contracts.json"


with DATA_FILE.open("r", encoding="utf-8") as file_handle:
    FLAGGED_CONTRACTS = json.load(file_handle)


def run_detectors(transaction: NormalizedTransaction, simulation: SimulationSummary) -> list[RiskFinding]:
    findings: list[RiskFinding] = []
    findings.extend(_check_approval(transaction))
    findings.extend(_check_flagged_destinations(transaction))
    findings.extend(_check_simulation(transaction, simulation))
    return findings


def _check_approval(transaction: NormalizedTransaction) -> list[RiskFinding]:
    if transaction.transaction_kind.value != "approval" or transaction.method_name.lower() == "setapprovalforall":
        return []

    approval_amount = transaction.approval_amount or 0
    evidence = [
        f"Method: {transaction.method_name}",
        f"Approval amount: {approval_amount}",
        f"Spender: {transaction.spender_address or transaction.to_address}",
    ]

    if approval_amount >= UNLIMITED_APPROVAL_THRESHOLD:
        return [explanation_templates.unlimited_approval(transaction, evidence)]

    if approval_amount >= LARGE_APPROVAL_THRESHOLD:
        return [explanation_templates.large_approval(transaction, evidence)]

    return []


def _check_flagged_destinations(transaction: NormalizedTransaction) -> list[RiskFinding]:
    findings: list[RiskFinding] = []
    seen: set[str] = set()

    for address_role, address in (("destination", transaction.to_address), ("spender", transaction.spender_address)):
        if not address or address in seen:
            continue
        entry = _lookup_flagged_contract(transaction.chain_id, address)
        if not entry:
            continue
        seen.add(address)
        severity = Severity(entry["severity"])
        evidence = [
            f"Flagged role: {address_role}",
            f"Address: {address}",
            f"Dataset label: {entry['label']}",
        ]
        findings.append(
            explanation_templates.flagged_destination(
                transaction,
                address_role=address_role,
                flagged_label=entry["label"],
                flagged_reason=entry["reason"],
                severity=severity,
                evidence=evidence,
            )
        )

    return findings


def _lookup_flagged_contract(chain_id: int, address: str) -> dict[str, str] | None:
    chain_table = FLAGGED_CONTRACTS.get(str(chain_id), {})
    return chain_table.get(address)


def _check_simulation(transaction: NormalizedTransaction, simulation: SimulationSummary) -> list[RiskFinding]:
    findings: list[RiskFinding] = []

    for effect in simulation.effects:
        evidence = [
            f"Simulation engine: {simulation.engine}",
            f"Simulation profile: {simulation.profile.value}",
            f"Simulated effect: {effect.summary}",
        ]
        if effect.effect_type == EffectType.allowance_grant:
            findings.append(explanation_templates.allowance_grant(transaction, evidence))
        elif effect.effect_type == EffectType.operator_control:
            findings.append(explanation_templates.operator_control(transaction, evidence))
        elif effect.effect_type == EffectType.unexpected_outflow:
            findings.append(explanation_templates.unexpected_outflow(transaction, evidence))
        elif effect.effect_type == EffectType.privilege_grant:
            findings.append(explanation_templates.privilege_escalation(transaction, evidence))

    return findings

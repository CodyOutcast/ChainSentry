from __future__ import annotations

from app.models import NormalizedTransaction, RiskCategory, RiskFinding, Severity


def short_address(address: str | None) -> str:
    if not address:
        return "unknown address"
    return f"{address[:6]}...{address[-4:]}"


def contract_label(transaction: NormalizedTransaction) -> str:
    return transaction.contract_name or short_address(transaction.to_address)


def spender_label(transaction: NormalizedTransaction) -> str:
    return transaction.spender_address or transaction.to_address


def format_token_amount(value: float | None, symbol: str | None) -> str:
    if value is None:
        return f"{symbol or 'tokens'}"
    if value >= 1_000_000_000:
        return f"unlimited {symbol or 'tokens'}"
    amount = int(value) if value.is_integer() else round(value, 4)
    if symbol:
        return f"{amount} {symbol}"
    return str(amount)


def unlimited_approval(transaction: NormalizedTransaction, evidence: list[str]) -> RiskFinding:
    spender = short_address(spender_label(transaction))
    token = format_token_amount(transaction.approval_amount, transaction.token_symbol)
    return RiskFinding(
        id="approval-unlimited",
        category=RiskCategory.approval,
        severity=Severity.high,
        expected_action=transaction.summary,
        risk_reason=f"The transaction gives {spender} a very large allowance ({token}).",
        possible_impact="If the spender contract is malicious or later compromised, it can move the approved tokens without asking again.",
        recommended_action="Reject unless the spender is trusted and a broad approval is required for the workflow.",
        evidence=evidence,
    )


def large_approval(transaction: NormalizedTransaction, evidence: list[str]) -> RiskFinding:
    spender = short_address(spender_label(transaction))
    token = format_token_amount(transaction.approval_amount, transaction.token_symbol)
    return RiskFinding(
        id="approval-large",
        category=RiskCategory.approval,
        severity=Severity.medium,
        expected_action=transaction.summary,
        risk_reason=f"The transaction approves a large allowance ({token}) for {spender}.",
        possible_impact="A larger allowance increases loss exposure if the spender behaves unexpectedly.",
        recommended_action="Inspect the spender and reduce the approval amount if a smaller approval works.",
        evidence=evidence,
    )


def flagged_destination(
    transaction: NormalizedTransaction,
    address_role: str,
    flagged_label: str,
    flagged_reason: str,
    severity: Severity,
    evidence: list[str],
) -> RiskFinding:
    return RiskFinding(
        id=f"destination-{address_role}",
        category=RiskCategory.destination,
        severity=severity,
        expected_action=transaction.summary,
        risk_reason=f"The {address_role} is listed in ChainSentry's demo flagged-contract set as {flagged_label}.",
        possible_impact=flagged_reason,
        recommended_action="Reject until the destination is independently verified and expected.",
        evidence=evidence,
    )


def operator_control(transaction: NormalizedTransaction, evidence: list[str]) -> RiskFinding:
    operator = short_address(spender_label(transaction))
    return RiskFinding(
        id="simulation-operator-control",
        category=RiskCategory.simulation,
        severity=Severity.high,
        expected_action=transaction.summary,
        risk_reason=f"Simulation shows that {operator} gains broad operator control, not just a one-time action.",
        possible_impact="The operator can move or manage approved assets later without another signature.",
        recommended_action="Reject unless the operator is trusted and collection-wide control is intentional.",
        evidence=evidence,
    )


def allowance_grant(transaction: NormalizedTransaction, evidence: list[str]) -> RiskFinding:
    spender = short_address(spender_label(transaction))
    return RiskFinding(
        id="simulation-allowance-grant",
        category=RiskCategory.simulation,
        severity=Severity.medium,
        expected_action=transaction.summary,
        risk_reason=f"Simulation confirms that {spender} receives a reusable allowance, not a one-time spend permission.",
        possible_impact="The approved amount can be spent later without another wallet confirmation while the allowance remains active.",
        recommended_action="Inspect the approval scope and reduce or revoke the allowance if persistent access is unnecessary.",
        evidence=evidence,
    )


def unexpected_outflow(transaction: NormalizedTransaction, evidence: list[str]) -> RiskFinding:
    return RiskFinding(
        id="simulation-unexpected-outflow",
        category=RiskCategory.simulation,
        severity=Severity.critical,
        expected_action=transaction.summary,
        risk_reason="Simulation predicts a downstream asset outflow that is larger or broader than the visible transaction inputs suggest.",
        possible_impact="Signing can enable later token drain or loss of assets beyond the amount a typical user expects.",
        recommended_action="Reject and inspect the destination contract, allowance scope, and expected asset movement before retrying.",
        evidence=evidence,
    )


def privilege_escalation(transaction: NormalizedTransaction, evidence: list[str]) -> RiskFinding:
    return RiskFinding(
        id="simulation-privilege-escalation",
        category=RiskCategory.simulation,
        severity=Severity.high,
        expected_action=transaction.summary,
        risk_reason="Simulation shows that the call grants elevated permissions instead of a narrow, one-time action.",
        possible_impact="The destination can gain standing authority over assets or system behavior after this signature.",
        recommended_action="Reject unless the permission change is expected, documented, and required.",
        evidence=evidence,
    )


def model_approval_signal(
    transaction: NormalizedTransaction,
    probability: float,
    severity: Severity,
    evidence: list[str],
) -> RiskFinding:
    spender = short_address(spender_label(transaction))
    return RiskFinding(
        id="model-approval-signal",
        category=RiskCategory.approval,
        severity=severity,
        expected_action=transaction.summary,
        risk_reason=(
            f"The trained graph model scored this approval pattern as risky (score {probability:.2f}) "
            f"based on the spender, token, and transaction relations."
        ),
        possible_impact="This pattern resembles approvals that leave reusable token access in place after the immediate action.",
        recommended_action=f"Inspect whether {spender} truly needs persistent approval before signing.",
        evidence=evidence,
    )


def model_destination_signal(
    transaction: NormalizedTransaction,
    probability: float,
    severity: Severity,
    evidence: list[str],
) -> RiskFinding:
    return RiskFinding(
        id="model-destination-signal",
        category=RiskCategory.destination,
        severity=severity,
        expected_action=transaction.summary,
        risk_reason=(
            f"The trained graph model assigned an elevated destination-risk score ({probability:.2f}) "
            "to this contract interaction pattern."
        ),
        possible_impact="Signing can route value or permissions to a destination that behaves more broadly than the wallet prompt suggests.",
        recommended_action="Inspect the destination contract, UI context, and expected asset movement before proceeding.",
        evidence=evidence,
    )


def model_simulation_signal(
    transaction: NormalizedTransaction,
    probability: float,
    severity: Severity,
    evidence: list[str],
) -> RiskFinding:
    return RiskFinding(
        id="model-simulation-signal",
        category=RiskCategory.simulation,
        severity=severity,
        expected_action=transaction.summary,
        risk_reason=(
            f"The trained graph model linked the simulated effects and transaction graph to a risky behavior pattern "
            f"(score {probability:.2f})."
        ),
        possible_impact="The transaction may enable follow-on effects or authority changes that are not obvious from the visible wallet prompt.",
        recommended_action="Inspect the simulated effects and reject the transaction if the broader outcome is unexpected.",
        evidence=evidence,
    )

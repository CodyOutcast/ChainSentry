from __future__ import annotations

from app.ml.inference import get_predictor
from app.models import AnalysisResponse, RecommendedAction, Severity, TransactionRequest
from app.services.parser import parse_transaction
from app.services.simulation import simulation_engine


SEVERITY_ORDER = {
    Severity.low: 0,
    Severity.medium: 1,
    Severity.high: 2,
    Severity.critical: 3,
}


def analyze_transaction(request: TransactionRequest) -> AnalysisResponse:
    normalized_transaction = parse_transaction(request)
    simulation = simulation_engine.simulate(normalized_transaction, request.simulation_profile)
    predictor_result = get_predictor().predict(normalized_transaction, simulation)
    findings = predictor_result.findings
    overall_severity = _overall_severity(findings)
    recommended_action = _recommended_action(overall_severity)
    summary = _summary(findings, overall_severity)

    return AnalysisResponse(
        normalized_transaction=normalized_transaction,
        overall_severity=overall_severity,
        recommended_action=recommended_action,
        summary=summary,
        findings=findings,
        simulation=simulation,
    )


def _overall_severity(findings):
    if not findings:
        return Severity.low
    return max(findings, key=lambda finding: SEVERITY_ORDER[finding.severity]).severity


def _recommended_action(severity: Severity) -> RecommendedAction:
    if severity == Severity.low:
        return RecommendedAction.proceed
    if severity == Severity.medium:
        return RecommendedAction.inspect_further
    return RecommendedAction.reject


def _summary(findings, severity: Severity) -> str:
    if not findings:
        return "No configured ChainSentry risk signals were detected for this transaction."

    count = len(findings)
    noun = "signal" if count == 1 else "signals"
    return f"ChainSentry found {count} risk {noun}. Highest severity: {severity.value}."

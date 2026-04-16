from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


ADDRESS_PATTERN = re.compile(r"^0x[a-fA-F0-9]{40}$")
CALLDATA_PATTERN = re.compile(r"^0x([a-fA-F0-9]{2})*$")


class Severity(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class RecommendedAction(StrEnum):
    proceed = "proceed"
    inspect_further = "inspect_further"
    reject = "reject"


class TransactionKind(StrEnum):
    approval = "approval"
    transfer = "transfer"
    swap = "swap"
    privilege = "privilege"
    native_transfer = "native_transfer"
    contract_call = "contract_call"


class SimulationProfile(StrEnum):
    none = "none"
    allowance_drain = "allowance_drain"
    privilege_escalation = "privilege_escalation"
    unexpected_outflow = "unexpected_outflow"


class RiskCategory(StrEnum):
    approval = "approval"
    destination = "destination"
    simulation = "simulation"


class EffectType(StrEnum):
    allowance_grant = "allowance_grant"
    operator_control = "operator_control"
    privilege_grant = "privilege_grant"
    unexpected_outflow = "unexpected_outflow"


class TransactionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    chain_id: int = Field(default=1, ge=1)
    from_address: str
    to_address: str
    method_name: str | None = None
    calldata: str | None = None
    value_eth: float = Field(default=0.0, ge=0.0)
    token_symbol: str | None = None
    token_amount: float | None = Field(default=None, ge=0.0)
    approval_amount: float | None = Field(default=None, ge=0.0)
    spender_address: str | None = None
    contract_name: str | None = None
    interaction_label: str | None = None
    notes: str | None = None
    simulation_profile: SimulationProfile = SimulationProfile.none

    @field_validator("from_address", "to_address", "spender_address", mode="before")
    @classmethod
    def normalize_address(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            if info.field_name in {"from_address", "to_address"}:
                raise ValueError("Address is required.")
            return None
        stripped = value.strip()
        if stripped == "":
            if info.field_name in {"from_address", "to_address"}:
                raise ValueError("Address is required.")
            return None
        if not ADDRESS_PATTERN.fullmatch(stripped):
            raise ValueError("Expected a 42-character hexadecimal address.")
        return stripped.lower()

    @field_validator("method_name", "contract_name", "interaction_label", "notes", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("token_symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped.upper() or None

    @field_validator("calldata", mode="before")
    @classmethod
    def normalize_calldata(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        stripped = value.strip()
        if not CALLDATA_PATTERN.fullmatch(stripped):
            raise ValueError("Calldata must be a hexadecimal string prefixed with 0x.")
        return stripped.lower()


class NormalizedTransaction(BaseModel):
    chain_id: int
    transaction_kind: TransactionKind
    from_address: str
    to_address: str
    spender_address: str | None = None
    contract_name: str | None = None
    method_name: str
    selector: str | None = None
    value_eth: float
    token_symbol: str | None = None
    token_amount: float | None = None
    approval_amount: float | None = None
    interaction_label: str | None = None
    summary: str


class SimulationEffect(BaseModel):
    effect_type: EffectType
    summary: str


class SimulationSummary(BaseModel):
    engine: str
    profile: SimulationProfile
    triggered: bool
    description: str | None = None
    effects: list[SimulationEffect] = Field(default_factory=list)


class RiskFinding(BaseModel):
    id: str
    category: RiskCategory
    severity: Severity
    expected_action: str
    risk_reason: str
    possible_impact: str
    recommended_action: str
    evidence: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    normalized_transaction: NormalizedTransaction
    overall_severity: Severity
    recommended_action: RecommendedAction
    summary: str
    findings: list[RiskFinding] = Field(default_factory=list)
    simulation: SimulationSummary

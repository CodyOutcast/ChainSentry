from __future__ import annotations

from app.models import EffectType, NormalizedTransaction, SimulationEffect, SimulationProfile, SimulationSummary, TransactionKind


class HeuristicSimulationEngine:
    name = "heuristic"

    def simulate(
        self,
        transaction: NormalizedTransaction,
        profile: SimulationProfile,
    ) -> SimulationSummary:
        effects: list[SimulationEffect] = []
        description: str | None = None

        if transaction.transaction_kind == TransactionKind.approval and transaction.method_name.lower() == "setapprovalforall":
            effects.append(
                SimulationEffect(
                    effect_type=EffectType.operator_control,
                    summary="The operator receives broad transfer authority over the linked asset collection.",
                )
            )
            description = "Collection-wide operator approval grants standing transfer power."

        if transaction.transaction_kind == TransactionKind.approval and transaction.approval_amount:
            effects.append(
                SimulationEffect(
                    effect_type=EffectType.allowance_grant,
                    summary="The spender receives a token allowance that can be reused without another signature.",
                )
            )
            description = description or "The transaction grants a reusable token allowance."

        if profile == SimulationProfile.allowance_drain:
            effects.append(
                SimulationEffect(
                    effect_type=EffectType.unexpected_outflow,
                    summary="A later contract call can draw down the approved balance without another signature.",
                )
            )
            description = "The approval enables later token movement outside the immediate action visible in the wallet."

        if profile == SimulationProfile.privilege_escalation or transaction.transaction_kind == TransactionKind.privilege:
            effects.append(
                SimulationEffect(
                    effect_type=EffectType.privilege_grant,
                    summary="The call grants elevated privileges or standing control to the destination.",
                )
            )
            description = "The simulated outcome changes permissions, not just asset balances."

        if profile == SimulationProfile.unexpected_outflow:
            effects.append(
                SimulationEffect(
                    effect_type=EffectType.unexpected_outflow,
                    summary="The simulated path shows asset movement beyond the amount a typical user expects from the visible prompt.",
                )
            )
            description = "The simulated outcome includes a broader asset outflow than the visible confirmation implies."

        triggered = bool(effects)
        return SimulationSummary(
            engine=self.name,
            profile=profile,
            triggered=triggered,
            description=description,
            effects=effects,
        )


simulation_engine = HeuristicSimulationEngine()

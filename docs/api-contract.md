# ChainSentry API Contract

The frontend and backend currently communicate through one endpoint.

Important: this contract is the current baseline interface from Student 1. Student 2 is expected to integrate a graph-based trained predictor behind this API where practical, so the frontend can continue working during the ML transition.

## Endpoint

`POST /api/v1/analyze`

Base URL during local development:

`http://localhost:8000`

## Request Shape

```json
{
  "chain_id": 1,
  "from_address": "0x1111111111111111111111111111111111111111",
  "to_address": "0xdead00000000000000000000000000000000beef",
  "method_name": "approve",
  "calldata": "0x095ea7b3",
  "value_eth": 0,
  "token_symbol": "USDC",
  "token_amount": 250,
  "approval_amount": 1000000000,
  "spender_address": "0xdead00000000000000000000000000000000beef",
  "contract_name": "Demo Drain Contract",
  "interaction_label": "Connect wallet to claim rewards",
  "notes": "The interface asks for a broad approval.",
  "simulation_profile": "allowance_drain"
}
```

## Request Field Notes

• `chain_id`: blockchain network identifier.
• `from_address`: sender address.
• `to_address`: destination contract or recipient.
• `method_name`: wallet-visible method or inferred contract action.
• `calldata`: optional raw calldata for selector inference.
• `value_eth`: native ETH value.
• `token_symbol`: token symbol for user-facing summaries.
• `token_amount`: visible token amount for transfers or swaps.
• `approval_amount`: allowance value for approval transactions.
• `spender_address`: spender or operator address when relevant.
• `contract_name`: user-facing label for the destination.
• `interaction_label`: short label describing the visible action.
• `notes`: optional analyst notes for demos or evaluation.
• `simulation_profile`: current baseline analysis-profile hint that can also be reused as a graph-context feature.

## Recommended Graph Construction Behind This API

The current request should be treated as the seed for a local heterogeneous graph in the backend.

Recommended graph nodes:

• Initiating wallet address.
• Destination contract.
• Spender or operator address when present.
• Token asset.
• Transaction anchor node.
• Optional simulation-effect or token-flow nodes.

Recommended graph edges:

• `initiates`
• `targets`
• `approves`
• `requests_allowance_for`
• `transfers_value_to`
• `transfers_token_to`
• `routes_to`
• `grants_operator_to` or `grants_privilege_to`
• `triggers_effect`

This design allows Student 2 to add graph ML without requiring a frontend payload rewrite.

Supported `simulation_profile` values:

• `none`
• `allowance_drain`
• `privilege_escalation`
• `unexpected_outflow`

## Response Shape

The example below reflects the current heuristic baseline implementation. Student 2 may change the internal scoring source to a graph-based trained predictor, but should preserve field names and types unless the frontend and documentation are updated together.

```json
{
  "normalized_transaction": {
    "chain_id": 1,
    "transaction_kind": "approval",
    "from_address": "0x1111111111111111111111111111111111111111",
    "to_address": "0xdead00000000000000000000000000000000beef",
    "spender_address": "0xdead00000000000000000000000000000000beef",
    "contract_name": "Demo Drain Contract",
    "method_name": "approve",
    "selector": null,
    "value_eth": 0,
    "token_symbol": "USDC",
    "token_amount": 250,
    "approval_amount": 1000000000,
    "interaction_label": "Connect wallet to claim rewards",
    "summary": "Approve 0xdead...beef to spend unlimited USDC."
  },
  "overall_severity": "critical",
  "recommended_action": "reject",
  "summary": "ChainSentry found 3 risk signals. Highest severity: critical.",
  "findings": [
    {
      "id": "approval-unlimited",
      "category": "approval",
      "severity": "high",
      "expected_action": "Approve 0xdead...beef to spend unlimited USDC.",
      "risk_reason": "The transaction gives 0xdead...beef a very large allowance (unlimited USDC).",
      "possible_impact": "If the spender contract is malicious or later compromised, it can move the approved tokens without asking again.",
      "recommended_action": "Reject unless the spender is trusted and a broad approval is required for the workflow.",
      "evidence": [
        "Method: approve",
        "Approval amount: 1000000000",
        "Spender: 0xdead00000000000000000000000000000000beef"
      ]
    }
  ],
  "simulation": {
    "engine": "heuristic",
    "profile": "allowance_drain",
    "triggered": true,
    "description": "The approval enables later token movement outside the immediate action visible in the wallet.",
    "effects": [
      {
        "effect_type": "allowance_grant",
        "summary": "The spender receives a token allowance that can be reused without another signature."
      },
      {
        "effect_type": "unexpected_outflow",
        "summary": "A later contract call can draw down the approved balance without another signature."
      }
    ]
  }
}
```

## Stable Response Fields

Student 2 can change graph construction, scoring internals, wording, demo scenarios, and UI layout, but these fields should remain stable unless both backend and frontend are updated together:

• `overall_severity`
• `recommended_action`
• `summary`
• `findings[].id`
• `findings[].category`
• `findings[].severity`
• `findings[].expected_action`
• `findings[].risk_reason`
• `findings[].possible_impact`
• `findings[].recommended_action`
• `findings[].evidence`
• `simulation.engine`
• `simulation.profile`
• `simulation.triggered`
• `simulation.description`
• `simulation.effects`
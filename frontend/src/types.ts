export type Severity = 'low' | 'medium' | 'high' | 'critical'
export type RecommendedAction = 'proceed' | 'inspect_further' | 'reject'
export type RiskCategory = 'approval' | 'destination' | 'simulation'
export type SimulationProfile = 'none' | 'allowance_drain' | 'privilege_escalation' | 'unexpected_outflow'
export type DraftErrors = Partial<Record<keyof TransactionDraft, string>>

export type TransactionRequest = {
  chain_id: number
  from_address: string
  to_address: string
  method_name?: string
  calldata?: string
  value_eth: number
  token_symbol?: string
  token_amount?: number
  approval_amount?: number
  spender_address?: string
  contract_name?: string
  interaction_label?: string
  notes?: string
  simulation_profile: SimulationProfile
}

export type TransactionDraft = {
  chain_id: string
  from_address: string
  to_address: string
  method_name: string
  calldata: string
  value_eth: string
  token_symbol: string
  token_amount: string
  approval_amount: string
  spender_address: string
  contract_name: string
  interaction_label: string
  notes: string
  simulation_profile: SimulationProfile
}

export type NormalizedTransaction = {
  chain_id: number
  transaction_kind: string
  from_address: string
  to_address: string
  spender_address?: string | null
  contract_name?: string | null
  method_name: string
  selector?: string | null
  value_eth: number
  token_symbol?: string | null
  token_amount?: number | null
  approval_amount?: number | null
  interaction_label?: string | null
  summary: string
}

export type SimulationEffect = {
  effect_type: 'allowance_grant' | 'operator_control' | 'privilege_grant' | 'unexpected_outflow'
  summary: string
}

export type SimulationSummary = {
  engine: string
  profile: SimulationProfile
  triggered: boolean
  description?: string | null
  effects: SimulationEffect[]
}

export type RiskFinding = {
  id: string
  category: RiskCategory
  severity: Severity
  expected_action: string
  risk_reason: string
  possible_impact: string
  recommended_action: string
  evidence: string[]
}

export type AnalysisResponse = {
  normalized_transaction: NormalizedTransaction
  overall_severity: Severity
  recommended_action: RecommendedAction
  summary: string
  findings: RiskFinding[]
  simulation: SimulationSummary
}

export type DemoScenario = {
  id: string
  title: string
  focus: string
  description: string
  transaction: TransactionRequest
}

export function transactionToDraft(transaction: TransactionRequest): TransactionDraft {
  return {
    chain_id: String(transaction.chain_id),
    from_address: transaction.from_address,
    to_address: transaction.to_address,
    method_name: transaction.method_name ?? '',
    calldata: transaction.calldata ?? '',
    value_eth: transaction.value_eth ? String(transaction.value_eth) : '',
    token_symbol: transaction.token_symbol ?? '',
    token_amount: transaction.token_amount !== undefined ? String(transaction.token_amount) : '',
    approval_amount: transaction.approval_amount !== undefined ? String(transaction.approval_amount) : '',
    spender_address: transaction.spender_address ?? '',
    contract_name: transaction.contract_name ?? '',
    interaction_label: transaction.interaction_label ?? '',
    notes: transaction.notes ?? '',
    simulation_profile: transaction.simulation_profile,
  }
}

export function draftToRequest(draft: TransactionDraft): TransactionRequest {
  return {
    chain_id: parseInteger(draft.chain_id, 1),
    from_address: draft.from_address.trim(),
    to_address: draft.to_address.trim(),
    method_name: optionalString(draft.method_name),
    calldata: optionalString(draft.calldata),
    value_eth: parseNumber(draft.value_eth, 0),
    token_symbol: optionalString(draft.token_symbol),
    token_amount: optionalNumber(draft.token_amount),
    approval_amount: optionalNumber(draft.approval_amount),
    spender_address: optionalString(draft.spender_address),
    contract_name: optionalString(draft.contract_name),
    interaction_label: optionalString(draft.interaction_label),
    notes: optionalString(draft.notes),
    simulation_profile: draft.simulation_profile,
  }
}

function parseInteger(value: string, fallback: number): number {
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) ? parsed : fallback
}

function parseNumber(value: string, fallback: number): number {
  const parsed = Number.parseFloat(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function optionalNumber(value: string): number | undefined {
  if (!value.trim()) {
    return undefined
  }
  const parsed = Number.parseFloat(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function optionalString(value: string): string | undefined {
  const trimmed = value.trim()
  return trimmed ? trimmed : undefined
}

import type { FormEvent } from 'react'

import type { DraftErrors, SimulationProfile, TransactionDraft } from '../types'

type TransactionFormProps = {
  draft: TransactionDraft
  isSubmitting: boolean
  errors: DraftErrors
  walletAddress?: string
  onFieldChange: (field: keyof TransactionDraft, value: string) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
}

const methodOptions = [
  'approve',
  'setApprovalForAll',
  'transfer',
  'swapExactTokensForTokens',
  'grantRole',
  'contractCall',
]

const simulationOptions: Array<{ value: SimulationProfile; label: string }> = [
  { value: 'none', label: 'No extra simulation signal' },
  { value: 'allowance_drain', label: 'Allowance drain path' },
  { value: 'privilege_escalation', label: 'Privilege escalation path' },
  { value: 'unexpected_outflow', label: 'Unexpected outflow path' },
]

export function TransactionForm({
  draft,
  isSubmitting,
  errors,
  walletAddress,
  onFieldChange,
  onSubmit,
}: TransactionFormProps) {
  return (
    <section className="panel form-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Transaction</p>
          <h2>Configure &amp; analyze</h2>
        </div>
      </div>

      <form onSubmit={onSubmit}>
        <div className="form-grid">
          <label>
            <span>Chain</span>
            <select value={draft.chain_id} onChange={(event) => onFieldChange('chain_id', event.target.value)}>
              <option value="1">Ethereum Mainnet</option>
              <option value="11155111">Sepolia Testnet</option>
            </select>
          </label>

          <label>
            <span>Method</span>
            <select value={draft.method_name} onChange={(event) => onFieldChange('method_name', event.target.value)}>
              {methodOptions.map((option) => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </label>

          <label className="field-span-2">
            <span>From Address</span>
            <input
              aria-invalid={Boolean(errors.from_address)}
              onChange={(event) => onFieldChange('from_address', event.target.value)}
              placeholder="0x..."
              value={draft.from_address}
            />
            <small>{walletAddress ? 'Auto-filled from connected wallet.' : 'Connect a wallet or enter manually.'}</small>
            {errors.from_address ? <small className="error-text">{errors.from_address}</small> : null}
          </label>

          <label className="field-span-2">
            <span>To Address</span>
            <input
              aria-invalid={Boolean(errors.to_address)}
              onChange={(event) => onFieldChange('to_address', event.target.value)}
              placeholder="0x..."
              value={draft.to_address}
            />
            {errors.to_address ? <small className="error-text">{errors.to_address}</small> : null}
          </label>

          <label>
            <span>Token Amount</span>
            <input
              aria-invalid={Boolean(errors.token_amount)}
              onChange={(event) => onFieldChange('token_amount', event.target.value)}
              placeholder="0.0"
              value={draft.token_amount}
            />
            {errors.token_amount ? <small className="error-text">{errors.token_amount}</small> : null}
          </label>

          <label>
            <span>Approval Amount</span>
            <input
              aria-invalid={Boolean(errors.approval_amount)}
              onChange={(event) => onFieldChange('approval_amount', event.target.value)}
              placeholder="0.0"
              value={draft.approval_amount}
            />
            {errors.approval_amount ? <small className="error-text">{errors.approval_amount}</small> : null}
          </label>
        </div>

        <details className="advanced-section">
          <summary>Advanced options</summary>
          <div className="advanced-grid">
            <label className="field-span-2">
              <span>Spender / Operator</span>
              <input
                aria-invalid={Boolean(errors.spender_address)}
                onChange={(event) => onFieldChange('spender_address', event.target.value)}
                placeholder="0x..."
                value={draft.spender_address}
              />
              {errors.spender_address ? <small className="error-text">{errors.spender_address}</small> : null}
            </label>

            <label>
              <span>Value (ETH)</span>
              <input
                aria-invalid={Boolean(errors.value_eth)}
                onChange={(event) => onFieldChange('value_eth', event.target.value)}
                placeholder="0.0"
                value={draft.value_eth}
              />
              {errors.value_eth ? <small className="error-text">{errors.value_eth}</small> : null}
            </label>

            <label>
              <span>Token Symbol</span>
              <input
                onChange={(event) => onFieldChange('token_symbol', event.target.value)}
                placeholder="e.g. USDC"
                value={draft.token_symbol}
              />
            </label>

            <label>
              <span>Contract Name</span>
              <input
                onChange={(event) => onFieldChange('contract_name', event.target.value)}
                placeholder="e.g. USDC Token"
                value={draft.contract_name}
              />
            </label>

            <label>
              <span>Interaction Label</span>
              <input
                onChange={(event) => onFieldChange('interaction_label', event.target.value)}
                placeholder="e.g. Uniswap Swap"
                value={draft.interaction_label}
              />
            </label>

            <label className="field-span-2">
              <span>Simulation Profile</span>
              <select
                value={draft.simulation_profile}
                onChange={(event) => onFieldChange('simulation_profile', event.target.value as SimulationProfile)}
              >
                {simulationOptions.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>

            <label className="field-span-2">
              <span>Calldata <small>(hex, optional)</small></span>
              <input
                aria-invalid={Boolean(errors.calldata)}
                onChange={(event) => onFieldChange('calldata', event.target.value)}
                placeholder="0x"
                value={draft.calldata}
              />
              {errors.calldata ? <small className="error-text">{errors.calldata}</small> : null}
            </label>

            <label className="field-span-2">
              <span>Notes <small>(optional)</small></span>
              <textarea
                onChange={(event) => onFieldChange('notes', event.target.value)}
                placeholder="Additional context for this transaction..."
                rows={2}
                value={draft.notes}
              />
            </label>
          </div>
        </details>

        <div className="form-actions">
          <button className="primary-button" disabled={isSubmitting} type="submit">
            {isSubmitting && <span className="btn-spinner" />}
            {isSubmitting ? 'Analyzing…' : 'Analyze Transaction'}
          </button>
        </div>
      </form>
    </section>
  )
}

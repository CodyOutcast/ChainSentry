import { startTransition, useEffect, useState, type FormEvent } from 'react'
import { useAccount } from 'wagmi'

import { analyzeTransaction } from './api/client'
import { RiskReport } from './components/RiskReport'
import { TransactionForm } from './components/TransactionForm'
import { WalletConnection } from './components/WalletConnection'
import { sampleScenarios } from './data/sampleTransactions'
import { draftToRequest, transactionToDraft, type AnalysisResponse, type DraftErrors, type TransactionDraft } from './types'

const initialScenario = sampleScenarios[0]!
const addressPattern = /^0x[a-fA-F0-9]{40}$/
const calldataPattern = /^0x([a-fA-F0-9]{2})*$/

function applyScenarioToDraft(address: string | undefined, scenarioId: string): TransactionDraft {
  const scenario = sampleScenarios.find((candidate) => candidate.id === scenarioId) ?? initialScenario
  return transactionToDraft({
    ...scenario.transaction,
    from_address: address ?? scenario.transaction.from_address,
  })
}

export default function App() {
  const { address } = useAccount()
  const [selectedScenarioId, setSelectedScenarioId] = useState(initialScenario.id)
  const [draft, setDraft] = useState<TransactionDraft>(() => applyScenarioToDraft(undefined, initialScenario.id))
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [validationErrors, setValidationErrors] = useState<DraftErrors>({})
  const [isSubmitting, setIsSubmitting] = useState(false)

  useEffect(() => {
    if (!address) {
      return
    }
    setDraft((current) => ({ ...current, from_address: address }))
  }, [address])

  const selectedScenario = sampleScenarios.find((scenario) => scenario.id === selectedScenarioId) ?? initialScenario

  function handleScenarioSelect(scenarioId: string) {
    setSelectedScenarioId(scenarioId)
    startTransition(() => {
      setDraft(applyScenarioToDraft(address, scenarioId))
      setAnalysis(null)
      setError(null)
      setValidationErrors({})
    })
  }

  function handleFieldChange(field: keyof TransactionDraft, value: string) {
    setDraft((current) => ({ ...current, [field]: value }))
    setValidationErrors((current) => {
      if (!current[field]) {
        return current
      }
      const nextErrors = { ...current }
      delete nextErrors[field]
      return nextErrors
    })
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const nextErrors = validateDraft(draft)
    if (Object.keys(nextErrors).length > 0) {
      setValidationErrors(nextErrors)
      setError('Fix the highlighted fields before running the analysis.')
      setAnalysis(null)
      return
    }

    setIsSubmitting(true)
    setError(null)
    setValidationErrors({})

    try {
      const response = await analyzeTransaction(draftToRequest(draft))
      setAnalysis(response)
    } catch (submissionError) {
      const message = submissionError instanceof Error ? submissionError.message : 'Unknown analysis error'
      setError(message)
      setAnalysis(null)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="page-shell">
      <header className="top-bar">
        <div className="brand-area">
          <div className="brand-icon">⬡</div>
          <div className="brand-text">
            <span className="brand-name">ChainSentry</span>
            <span className="brand-tagline">Pre-signature risk analysis</span>
          </div>
        </div>
        <WalletConnection />
      </header>

      <div className="scenario-row">
        {sampleScenarios.map((scenario) => (
          <button
            className={`scenario-chip ${scenario.id === selectedScenarioId ? 'active' : ''}`}
            key={scenario.id}
            onClick={() => handleScenarioSelect(scenario.id)}
            type="button"
          >
            <strong>{scenario.title}</strong>
            <span>{scenario.focus}</span>
          </button>
        ))}
      </div>

      <main className="main-grid">
        <TransactionForm
          draft={draft}
          isSubmitting={isSubmitting}
          errors={validationErrors}
          walletAddress={address}
          onFieldChange={handleFieldChange}
          onSubmit={handleSubmit}
        />
        <RiskReport analysis={analysis} error={error} scenario={selectedScenario} />
      </main>
    </div>
  )
}

function validateDraft(draft: TransactionDraft): DraftErrors {
  const errors: DraftErrors = {}

  if (!addressPattern.test(draft.from_address.trim())) {
    errors.from_address = 'Enter a valid 42-character sender address.'
  }

  if (!addressPattern.test(draft.to_address.trim())) {
    errors.to_address = 'Enter a valid 42-character destination address.'
  }

  if (draft.spender_address.trim() && !addressPattern.test(draft.spender_address.trim())) {
    errors.spender_address = 'Enter a valid 42-character spender or operator address.'
  }

  if (draft.calldata.trim() && !calldataPattern.test(draft.calldata.trim())) {
    errors.calldata = 'Calldata must be a hex string prefixed with 0x.'
  }

  for (const field of ['value_eth', 'token_amount', 'approval_amount'] as const) {
    const value = draft[field].trim()
    if (!value) {
      continue
    }
    const parsed = Number.parseFloat(value)
    if (!Number.isFinite(parsed) || parsed < 0) {
      errors[field] = 'Enter a non-negative number.'
    }
  }

  return errors
}

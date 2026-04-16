import type { AnalysisResponse, DemoScenario } from '../types'
import { SeverityBadge } from './SeverityBadge'

type RiskReportProps = {
  analysis: AnalysisResponse | null
  error: string | null
  scenario?: DemoScenario
}

export function RiskReport({ analysis, error, scenario }: RiskReportProps) {
  if (error) {
    return (
      <section className="panel report-panel error-panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Analysis Output</p>
            <h2>Analysis failed</h2>
          </div>
        </div>
        <p>{error}</p>
      </section>
    )
  }

  if (!analysis) {
    return (
      <section className="panel report-panel scenario-preview-panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Scenario</p>
            <h2>{scenario?.title ?? 'Select a scenario'}</h2>
          </div>
          {scenario ? <span className="focus-pill">{scenario.focus}</span> : null}
        </div>
        {scenario ? (
          <>
            <p className="scenario-description">{scenario.description}</p>
            <div className="analyze-hint">
              <span className="analyze-hint-icon">↓</span>
              <span>Review the pre-filled transaction on the left, then click <strong>Analyze Transaction</strong> to see the risk report.</span>
            </div>
          </>
        ) : (
          <p className="scenario-description">Pick one of the scenarios above to load a pre-built transaction.</p>
        )}
      </section>
    )
  }

  return (
    <section className="panel report-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Analysis Output</p>
          <h2>{analysis.summary}</h2>
        </div>
        <div className="report-summary">
          <SeverityBadge severity={analysis.overall_severity} />
          <span className="recommendation-pill">{analysis.recommended_action.replace('_', ' ')}</span>
        </div>
      </div>

      <article className="highlight-card">
        <h3>Expected action</h3>
        <p>{analysis.normalized_transaction.summary}</p>
      </article>

      <div className="findings-grid">
        {analysis.findings.length === 0 ? (
          <article className="finding-card safe-card">
            <h3>No risk patterns detected</h3>
            <p>This transaction did not match any approval, flagged destination, or simulation-based risk rules.</p>
          </article>
        ) : (
          analysis.findings.map((finding) => (
            <article className="finding-card" key={finding.id}>
              <header>
                <SeverityBadge severity={finding.severity} />
                <span className="category-tag">{finding.category}</span>
              </header>
              <div className="finding-copy">
                <p>
                  <strong>Expected action</strong> {finding.expected_action}
                </p>
                <p>
                  <strong>Risk reason</strong> {finding.risk_reason}
                </p>
                <p>
                  <strong>Possible impact</strong> {finding.possible_impact}
                </p>
                <p>
                  <strong>Recommended action</strong> {finding.recommended_action}
                </p>
              </div>
              {finding.evidence.length > 0 ? (
                <ul className="evidence-list">
                  {finding.evidence.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              ) : null}
            </article>
          ))
        )}
      </div>

      <div className="details-grid">
        <article className="detail-card">
          <h3>Simulation</h3>
          <p>{analysis.simulation.description ?? 'No simulation path required for this transaction.'}</p>
          {analysis.simulation.effects.length > 0 ? (
            <ul className="simulation-list">
              {analysis.simulation.effects.map((effect) => (
                <li key={`${effect.effect_type}-${effect.summary}`}>
                  <strong>{effect.effect_type}:</strong> {effect.summary}
                </li>
              ))}
            </ul>
          ) : null}
        </article>

        <article className="detail-card">
          <h3>Transaction details</h3>
          <ul className="transaction-list">
            <li>
              <strong>Kind</strong>
              <span>{analysis.normalized_transaction.transaction_kind}</span>
            </li>
            <li>
              <strong>Method</strong>
              <span>{analysis.normalized_transaction.method_name}</span>
            </li>
            <li>
              <strong>To</strong>
              <span>{analysis.normalized_transaction.to_address}</span>
            </li>
            <li>
              <strong>Spender</strong>
              <span>{analysis.normalized_transaction.spender_address ?? '—'}</span>
            </li>
            <li>
              <strong>Token</strong>
              <span>{analysis.normalized_transaction.token_symbol ?? '—'}</span>
            </li>
          </ul>
        </article>
      </div>
    </section>
  )
}

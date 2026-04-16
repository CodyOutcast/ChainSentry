import type { Severity } from '../types'

type SeverityBadgeProps = {
  severity: Severity
}

export function SeverityBadge({ severity }: SeverityBadgeProps) {
  return <span className={`severity-badge severity-${severity}`}>{severity}</span>
}


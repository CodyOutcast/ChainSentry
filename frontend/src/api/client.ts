import type { AnalysisResponse, TransactionRequest } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'
const REQUEST_TIMEOUT_MS = 15000

export async function analyzeTransaction(payload: TransactionRequest): Promise<AnalysisResponse> {
  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

  let response: Response

  try {
    response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('ChainSentry analysis timed out. Verify that the backend is running and try again.')
    }
    throw new Error('ChainSentry could not reach the backend service. Verify the API URL and try again.')
  } finally {
    window.clearTimeout(timeoutId)
  }

  if (!response.ok) {
    const message = await response.text()
    throw new Error(message || 'ChainSentry could not analyze the transaction.')
  }

  return response.json() as Promise<AnalysisResponse>
}

export { API_BASE_URL }

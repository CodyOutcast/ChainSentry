import { useAccount, useConnect, useDisconnect } from 'wagmi'

export function WalletConnection() {
  const { address, chain, isConnected } = useAccount()
  const { connect, connectors, error, isPending } = useConnect()
  const { disconnect } = useDisconnect()

  const availableConnector = connectors[0]

  if (isConnected && address) {
    return (
      <div className="wallet-widget wallet-connected">
        <span className="wallet-chain">{chain?.name ?? 'Unknown'}</span>
        <span className="wallet-address">{address.slice(0, 6)}…{address.slice(-4)}</span>
        <button className="wallet-disconnect" onClick={() => disconnect()} title="Disconnect" type="button">✕</button>
      </div>
    )
  }

  return (
    <div className="wallet-widget">
      {error ? <span className="wallet-error">{error.message}</span> : null}
      <button
        className="secondary-button wallet-connect-btn"
        disabled={!availableConnector || isPending}
        onClick={() => {
          if (availableConnector) {
            connect({ connector: availableConnector })
          }
        }}
        type="button"
      >
        {isPending ? 'Connecting…' : !availableConnector ? 'No wallet detected' : 'Connect Wallet'}
      </button>
    </div>
  )
}

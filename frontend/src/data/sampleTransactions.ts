import type { DemoScenario } from '../types'

const wallet = '0x1111111111111111111111111111111111111111'

export const sampleScenarios: DemoScenario[] = [
  {
    id: 'allowance-drain',
    title: 'Unlimited Approval To Drain Contract',
    focus: 'Approval + flagged spender + downstream drain risk',
    description:
      'A common wallet trap: the user thinks they are enabling a small token action, but the approval is effectively unlimited and points to a known-risk spender.',
    transaction: {
      chain_id: 1,
      from_address: wallet,
      to_address: '0xdead00000000000000000000000000000000beef',
      method_name: 'approve',
      value_eth: 0,
      token_symbol: 'USDC',
      token_amount: 250,
      approval_amount: 1000000000,
      spender_address: '0xdead00000000000000000000000000000000beef',
      contract_name: 'Demo Drain Contract',
      interaction_label: 'Connect wallet to claim rewards',
      notes: 'The interface advertises a simple rewards claim, but it asks for a broad approval.',
      simulation_profile: 'allowance_drain',
    },
  },
  {
    id: 'operator-takeover',
    title: 'Collection-Wide Operator Approval',
    focus: 'Simulation-driven operator control',
    description:
      'This scenario demonstrates a permission change that looks routine in a wallet prompt but actually grants broad control over a collection.',
    transaction: {
      chain_id: 1,
      from_address: wallet,
      to_address: '0x5555555555555555555555555555555555555555',
      method_name: 'setApprovalForAll',
      value_eth: 0,
      spender_address: '0xbad00000000000000000000000000000000c0de',
      contract_name: 'Shadow Gallery',
      interaction_label: 'List NFT for marketplace campaign',
      notes: 'The requested permission is broader than a one-time listing flow.',
      simulation_profile: 'privilege_escalation',
    },
  },
  {
    id: 'unexpected-outflow',
    title: 'Swap Route With Hidden Outflow',
    focus: 'Flagged destination + unexpected simulated outflow',
    description:
      'This demo scenario shows a routing contract that appears to perform a swap but produces broader asset movement in simulation.',
    transaction: {
      chain_id: 1,
      from_address: wallet,
      to_address: '0xbad00000000000000000000000000000000c0de',
      method_name: 'swapExactTokensForTokens',
      value_eth: 0,
      token_symbol: 'USDT',
      token_amount: 500,
      spender_address: '0xbad00000000000000000000000000000000c0de',
      contract_name: 'Demo Phishing Router',
      interaction_label: 'Swap stablecoins for campaign token',
      notes: 'The route and post-transaction asset movement are inconsistent with the visible prompt.',
      simulation_profile: 'unexpected_outflow',
    },
  },
  {
    id: 'clean-transfer',
    title: 'Standard Transfer',
    focus: 'Low-risk baseline scenario',
    description:
      'A simple transfer scenario used as a low-risk control case for demos and evaluation.',
    transaction: {
      chain_id: 1,
      from_address: wallet,
      to_address: '0x3333333333333333333333333333333333333333',
      method_name: 'transfer',
      value_eth: 0,
      token_symbol: 'ETH',
      token_amount: 0.15,
      contract_name: 'Friend Wallet',
      interaction_label: 'Send funds to a known address',
      notes: 'No flagged destination and no risky simulation effects are expected.',
      simulation_profile: 'none',
    },
  },
]

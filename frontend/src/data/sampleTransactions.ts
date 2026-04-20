import type { DemoScenario } from '../types'

const wallet = '0x1111111111111111111111111111111111111111'

export const sampleScenarios: DemoScenario[] = [
  {
    id: 'clean-transfer',
    title: 'Clean Transfer Baseline',
    focus: 'Low-risk control case',
    description:
      'A standard transfer to a known wallet that serves as the low-risk baseline for the demo.',
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
  {
    id: 'large-approval-benign',
    title: 'Large Approval To Benign Router',
    focus: 'Approval risk without blacklist hit',
    description:
      'A large token approval to an unflagged spender, used to highlight approval risk and reusable allowance behavior.',
    transaction: {
      chain_id: 1,
      from_address: wallet,
      to_address: '0x5555555555555555555555555555555555555555',
      method_name: 'approve',
      value_eth: 0,
      token_symbol: 'USDC',
      approval_amount: 50000,
      spender_address: '0x5555555555555555555555555555555555555555',
      contract_name: 'Demo Router',
      interaction_label: 'Approve router for token swap',
      notes: 'The spender is not in the demo blacklist, but the approval is unusually large.',
      simulation_profile: 'none',
    },
  },
  {
    id: 'flagged-transfer',
    title: 'Transfer To Flagged Destination',
    focus: 'Destination risk',
    description:
      'A token transfer to a destination that appears in the demo flagged-contract dataset.',
    transaction: {
      chain_id: 1,
      from_address: wallet,
      to_address: '0xdead00000000000000000000000000000000beef',
      method_name: 'transfer',
      value_eth: 0,
      token_symbol: 'USDT',
      token_amount: 1200,
      contract_name: 'Flagged Recipient',
      interaction_label: 'Send funds to flagged recipient',
      notes: 'This example should surface the destination finding path.',
      simulation_profile: 'none',
    },
  },
  {
    id: 'operator-control',
    title: 'Operator Control Approval',
    focus: 'Approval + destination + simulation signals',
    description:
      'A setApprovalForAll request that demonstrates broad operator control and multiple overlapping risk signals.',
    transaction: {
      chain_id: 1,
      from_address: wallet,
      to_address: '0x6666666666666666666666666666666666666666',
      method_name: 'setApprovalForAll',
      value_eth: 0,
      spender_address: '0x6666666666666666666666666666666666666666',
      contract_name: 'Collection Manager',
      interaction_label: 'Approve operator for NFT collection',
      notes: 'This should highlight broad operator permissions and simulated downstream loss.',
      simulation_profile: 'allowance_drain',
    },
  },
  {
    id: 'privilege-escalation',
    title: 'Grant Role Escalation',
    focus: 'Privilege escalation signal',
    description:
      'A grantRole transaction used to demonstrate how the backend exposes a privilege escalation scenario in the final report.',
    transaction: {
      chain_id: 1,
      from_address: wallet,
      to_address: '0x4444444444444444444444444444444444444444',
      method_name: 'grantRole',
      value_eth: 0,
      contract_name: 'Shadow Vault',
      interaction_label: 'Grant elevated role to manager contract',
      notes: 'This should primarily surface the simulation privilege escalation finding.',
      simulation_profile: 'privilege_escalation',
    },
  },
]

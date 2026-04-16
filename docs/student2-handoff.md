# Student 2 Handoff Guide

This handoff is for Student 2's ML, product, and evaluation work.

Important: the current backend is only a baseline heuristic prototype. The final project is expected to include a graph-based trained predictor model, and Student 2 owns that model work and its integration into the backend.

## What Is Already Implemented

### Backend
Implemented in `backend/`:

• FastAPI server with `POST /api/v1/analyze`.
• Transaction request and response models.
• Transaction parser and transaction-kind inference.
• Baseline approval risk detection.
• Baseline flagged destination checks.
• Baseline heuristic simulation layer.
• Explanation-ready findings with severity, impact, and recommendation fields.
• Focused backend tests.

Current limitation: the handoff now includes graph-construction and predictor scaffolding, but there is still no dataset pipeline, no training script, no saved model artifact, and no trained graph-predictor inference service yet.

### Frontend
Implemented in `frontend/`:

• Wallet-aware React app using wagmi.
• Editable transaction form.
• Demo scenario loader.
• API client wired to the backend.
• Risk report UI with findings and simulation output.
• Production build validation.

## Student 2 Primary Responsibility

Student 2 should treat the graph-based trained predictor as a required workstream, not an optional extension.

Student 2 is expected to:

• Define the transaction-centered graph schema.
• Define or finalize the label scheme for risky vs. safe transaction patterns.
• Prepare or curate the training dataset used for the predictor.
• Design node, edge, and scalar features for the model.
• Train and evaluate the graph predictor.
• Integrate graph-model inference into the backend while preserving the frontend contract where practical.
• Document model limitations, metrics, and retraining steps.

## Recommended Graph ML Design

Best recommended approach: use a transaction-centered heterogeneous graph rather than a global chain-wide graph.

Why:

• The current app analyzes one transaction at a time.
• The current `TransactionRequest` already provides the seed entities needed for graph construction.
• A compact local graph is easier to train and explain within project scope.

Recommended node types:

• Initiating wallet address.
• Destination contract.
• Spender or operator address when present.
• Token asset.
• Transaction anchor node.
• Optional token-flow or simulated-effect nodes.

Recommended edge types:

• `initiates`
• `targets`
• `approves`
• `requests_allowance_for`
• `transfers_value_to`
• `transfers_token_to`
• `routes_to`
• `grants_operator_to` or `grants_privilege_to`
• `triggers_effect`

Recommended first model:

• Start with an R-GCN or similarly relation-aware heterogeneous GNN.
• Pool around the transaction anchor node.
• Concatenate the pooled graph embedding with scalar transaction features such as amounts, selector, chain ID, and simulation profile.
• Use a multi-task head for risk-family prediction plus severity scoring.

Recommended integration approach:

• Keep `POST /api/v1/analyze` stable where practical.
• Treat the current parser as the graph-seeding preprocessor.
• Use the current heuristic detectors as a fallback, benchmark, or teacher baseline during transition.

## Files Student 2 Can Safely Edit

### Core backend analysis logic
Edit these files:

• `backend/app/services/analysis.py`
• `backend/app/services/detectors.py`
• `backend/app/services/simulation.py`

Use them to:

• Replace or augment baseline heuristic scoring.
• Route requests through the trained predictor.
• Combine model output with explainable findings.
• Keep the API response shape stable unless the frontend is updated too.

### New ML files Student 2 should extend
Scaffolded location: `backend/app/ml/`

Included scaffolding:

• `graph_builder.py`
• `features.py`
• `model.py`
• `inference.py`
• `training/`

Use them to:

• Build the heterogeneous graph from the parsed transaction.
• Define feature extraction for nodes, edges, and scalars.
• Train and save the graph model.
• Load the model and run inference inside the backend.

### Scenario content
Edit `frontend/src/data/sampleTransactions.ts`.

Use this file to:

• Add or remove demo transactions.
• Change scenario titles, descriptions, and focus labels.
• Prepare evaluation examples.

### User-facing backend explanation copy
Edit `backend/app/content/explanation_templates.py`.

Use this file to:

• Refine wording for expected action, risk reason, impact, and recommendation.
• Standardize tone across findings.
• Adjust wording for user comprehension tests.

Important: keep the output fields the same. Do not rename `expected_action`, `risk_reason`, `possible_impact`, or `recommended_action` unless both backend and frontend are updated together.

### Demo flagged contract dataset
Edit `backend/app/data/flagged_contracts.json`.

Use this file to:

• Add demo flagged addresses.
• Update labels and reasons.
• Create alternate evaluation cases for different networks.

### UI wording and layout
Edit these files:

• `frontend/src/components/RiskReport.tsx`
• `frontend/src/components/TransactionForm.tsx`
• `frontend/src/App.tsx`
• `frontend/src/styles.css`

Use them to:

• Refine layout and user-facing copy.
• Adjust emphasis for severity or recommendations.
• Improve clarity for evaluation sessions.

## Files Student 2 Should Usually Leave Alone

These files are better treated as stable integration surfaces unless Student 2 intentionally changes the schema or parser assumptions:

• `backend/app/models.py`
• `backend/app/services/parser.py`
• `frontend/src/api/client.ts`
• `frontend/src/types.ts`

Edit these only if the project decides to change the API contract, request shape, or response schema.

## Current Supported Risk Logic

The current Student 1 baseline analyzes three risk families:

• Approval risk: large or unlimited token approvals.
• Destination risk: transactions involving addresses listed in the demo flagged-contract dataset.
• Simulation risk: operator control, privilege escalation, and unexpected downstream outflow.

Student 2 should treat this as a baseline to beat or replace with a graph-based trained predictor, not as the final project endpoint.

## Current Demo Scenarios

The frontend includes these built-in demo scenarios:

• Unlimited approval to a flagged contract.
• Collection-wide operator approval.
• Swap route with hidden outflow.
• Clean transfer baseline.

## How To Extend The Project Safely

### To build the graph-based trained predictor
1. Keep the current API response fields stable while the model is being integrated, unless the frontend is updated in the same change.
2. Build a reproducible graph-construction and training path and document it in the repo.
3. Record evaluation metrics and the dataset assumptions used for training.
4. Preserve explanation quality by mapping graph-model output back into the existing finding structure.

### To add a new scenario
1. Add the scenario in `frontend/src/data/sampleTransactions.ts`.
2. Keep the same `TransactionRequest` shape.
3. Test it through the UI.

### To change explanation wording
1. Update the relevant function in `backend/app/content/explanation_templates.py`.
2. Run backend tests.
3. Re-run the scenario in the UI and check readability.

### To add a new flagged demo contract
1. Add the address to `backend/app/data/flagged_contracts.json`.
2. Re-run backend tests.
3. Add a matching frontend scenario if needed.

## Suggested Student 2 Work Plan

1. Define the dataset, labels, and feature strategy for the trained predictor.
2. Define the graph schema and train the first R-GCN-style baseline, then integrate inference into the backend.
3. Update explanation wording so model-driven findings remain understandable.
4. Expand or improve demo scenarios in `frontend/src/data/sampleTransactions.ts`.
5. Adjust the frontend report UI and documentation for the final ML-backed flow.

## Validation Checklist After Student 2 Changes

If the handoff folder was provided without generated dependencies, recreate them first:

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
cd frontend
npm install
```

Student 2 should also add and document the exact graph-model training and evaluation commands once the ML pipeline is created.

Backend:

```bash
.venv/bin/python -m pytest backend/tests -q
```

Frontend:

```bash
cd frontend
npm run build
```
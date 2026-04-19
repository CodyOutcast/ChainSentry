# Student 2 Final Handoff Guide

This handoff reflects the current repository state after the Student 2 graph-model integration.

Important: the repo is no longer only a heuristic scaffold. It now includes a reproducible graph-model training pipeline, saved artifacts, backend inference integration, and fallback behavior.

## What Is Already Implemented

### Backend
Implemented in `backend/`:

• FastAPI server with `POST /api/v1/analyze`
• Stable request/response models
• Transaction parser and transaction-kind inference
• Baseline approval, destination, and simulation detectors
• Heuristic simulation layer
• Transaction-centered graph construction
• Scalar feature extraction and vectorization
• Graph-model training, evaluation, and artifact export
• Backend graph-model inference with heuristic fallback
• Focused backend tests

### Frontend
Implemented in `frontend/`:

• Wallet-aware React app using wagmi
• Editable transaction form
• Demo scenario loader
• API client wired to the backend
• Risk report UI with findings and simulation output
• Production build validation

## What Student 2 Delivered

Student 2 now owns and has delivered the following workstreams in this repo:

• Local heterogeneous transaction graph design
• Scalar feature extraction and sample vectorization
• Multi-dataset adaptor layer and unified sample format
• Reproducible multi-dataset graph-model training, SwanLab logging, and metrics export
• Saved artifact loading and graph-model inference inside the backend
• Hybrid model-plus-rules finding generation
• Updated Chinese documentation for setup, usage, and evaluation

## Current Metrics

The current repository now trains through the multi-dataset path, and the exact numbers depend on the last completed run that wrote:

• `backend/app/ml/artifacts/graph-model-metrics.json`
• the paired SwanLab run for that training job

The historical synthetic baseline is only a legacy reference now.  
For the current path, report the metrics JSON and SwanLab dashboard produced by `train_multidataset_model.py`.

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

### Current ML implementation
Current location: `backend/app/ml/`

Key files:

• `graph_builder.py`
• `features.py`
• `vectorization.py`
• `model.py`
• `inference.py`
• `training/unified_sample.py`
• `training/multi_dataset.py`
• `training/adaptors/`
• `training/external_datasets.py`
• `training/dataset.py` (legacy synthetic helper, kept mainly for compatibility)
• `training/train_multidataset_model.py`
• `training/train_graph_model.py`

Use them to inspect or extend:

• graph construction
• feature design
• dataset generation
• training and artifact export
• inference integration

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

## Remaining Optional Work

The core Student 2 deliverables are already present. Only optional extensions remain:

1. Replace more shell-style supervision with richer transaction-level labeled data
2. Improve the simulation engine beyond heuristic effect templates
3. Run a formal user study instead of only scenario-based evaluation
4. Expand threat coverage beyond the current three risk families

## Validation Checklist

If the environment does not yet exist, recreate dependencies first:

```bash
conda create -y -p .conda-envs/chainsentry python=3.12
conda run -p .conda-envs/chainsentry python -m pip install -r backend/requirements.txt
cd frontend
npm install
```

Graph-model training:

```bash
PYTHONPATH=backend .conda-envs/chainsentry/bin/python -m app.ml.training.train_multidataset_model \
  --artifact-path backend/app/ml/artifacts/graph-model.pt \
  --metrics-path backend/app/ml/artifacts/graph-model-metrics.json \
  --epochs 18 \
  --size-profile standard
```

Backend tests:

```bash
PYTHONPATH=backend .conda-envs/chainsentry/bin/python -m pytest backend/tests -q
```

Frontend build:

```bash
cd frontend
npm run build
```

## Supporting Docs

• `docs/student2-使用说明.md`: Chinese Student 2 workflow guide
• `docs/用户使用说明.md`: Chinese end-user usage guide
• `docs/评估与交付总结.md`: final scope, metrics, and evaluation summary
• `docs/api-contract.md`: stable frontend/backend contract

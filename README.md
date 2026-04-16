# ChainSentry Prototype

ChainSentry is a two-part prototype:

• A FastAPI backend that currently parses transactions, runs baseline risk detectors, applies baseline simulation, and returns structured findings.
• A React frontend that captures transaction data, connects to an injected wallet, runs the analysis, and displays risk cards for demos.

## Project Structure

• `backend/`: FastAPI service, baseline risk logic, graph-ML integration surface, simulation prototype, and tests.
• `frontend/`: Vite React client, demo scenarios, wallet integration, and report UI.
• `docs/`: handoff and API documentation for Student 2.
• `info.md`: implementation plan and project reference.

## Final Project Direction

The current Student 1 codebase is a baseline prototype, not the final ML deliverable.

Student 2 is expected to build and integrate the required graph-based trained predictor model. The recommended design is a transaction-centered heterogeneous graph where addresses, contracts, tokens, and token-flow effects are represented as typed nodes and relations.

The best fit for this repo is to keep the existing `POST /api/v1/analyze` contract stable, build the graph inside the backend from the current request plus local context, and use a relation-aware graph model such as an R-GCN as the first ML baseline.

## First-Time Setup

Backend dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
```

Frontend dependencies:

```bash
cd frontend
npm install
```

The Student 1 handoff intentionally excludes generated folders such as `.venv/`, `frontend/node_modules/`, and `frontend/dist/`. Recreate them locally using the commands above.

## Run The Backend

From the workspace root:

```bash
.venv/bin/python -m uvicorn backend.app.main:app --reload
```

Backend URL:

• `http://localhost:8000`
• Health check: `http://localhost:8000/health`
• Analysis endpoint: `POST http://localhost:8000/api/v1/analyze`

## Run The Frontend

From the workspace root:

```bash
cd frontend
cp .env.example .env
npm run dev
```

Frontend URL:

• `http://localhost:5173`

## Validation Commands

Backend tests:

```bash
.venv/bin/python -m pytest backend/tests -q
```

Frontend build:

```bash
cd frontend
npm run build
```

## Student 2 Handoff

See `docs/student2-handoff.md` for the integration guide and `docs/api-contract.md` for the request and response shapes.
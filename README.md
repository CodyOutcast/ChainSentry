# ChainSentry Prototype

ChainSentry is an explainable pre-signature transaction risk prototype.

It currently ships as a hybrid system:

• A FastAPI backend that parses transactions, runs local simulation and heuristic checks, and augments them with a trained transaction-centered graph model.
• A React frontend that captures transaction data, connects to an injected wallet, runs the analysis, and displays short risk cards for demos.

## Project Structure

• `backend/`: FastAPI service, heuristic detectors, graph-model training/inference pipeline, simulation prototype, and tests.
• `frontend/`: Vite React client, demo scenarios, wallet integration, and risk report UI.
• `docs/`: API contract, Student 2 handoff, evaluation summary, and Chinese usage guides.
• `info.md`: implementation plan and project reference.

## Final Project Direction

The repo now reflects the narrowed final scope:

• Focus on a small set of common, explainable risks instead of a broad security platform.
• Keep the existing `POST /api/v1/analyze` contract stable.
• Use a transaction-centered local graph plus scalar features as the ML baseline.
• Preserve short user-facing explanations and actionable recommendations.

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
PYTHONPATH=backend .venv/bin/python -m uvicorn app.main:app --reload
```

If you are using a system Python environment instead of `.venv`, replace `.venv/bin/python` with `python3`.

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
PYTHONPATH=backend .venv/bin/python -m pytest backend/tests -q
```

Frontend build:

```bash
cd frontend
npm run build
```

## Student 2 Handoff

See `docs/student2-handoff.md` for the integration guide and `docs/api-contract.md` for the request and response shapes.

## Student 2 Usage

For the Student 2 graph-model workflow and Chinese usage notes, see `docs/student2-使用说明.md`.

## End-User Usage

For a clear Chinese guide to running the app and reading the risk report, see `docs/用户使用说明.md`.

## Evaluation Summary

For the final scope, metrics, limitations, and response to teacher feedback, see `docs/评估与交付总结.md`.

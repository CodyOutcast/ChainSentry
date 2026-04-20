# ChainSentry Prototype

ChainSentry is an explainable pre-signature transaction risk prototype.

It currently ships as a hybrid system:

• A FastAPI backend that parses transactions, runs local simulation and heuristic checks, and augments them with a trained transaction-centered graph model.
• A React frontend that captures transaction data, connects to an injected wallet, runs the analysis, and displays short risk cards for demos.

## Project Structure

• `backend/`: FastAPI service, heuristic detectors, packaged graph-model inference, simulation prototype, and tests.
• `frontend/`: Vite React client, demo scenarios, wallet integration, and risk report UI.
• `docs/`: final report, demo guide, project proposal, and supplementary notes.
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

The handoff intentionally excludes local dependency folders such as `.venv/` and `frontend/node_modules/`. The presentation package may include a prebuilt `frontend/dist/` bundle so the UI can be served on a fresh machine without rerunning the frontend build.

This handoff keeps the runtime analysis path and the packaged graph-model artifacts, but it does not include the full training workspace or the raw external training corpora. That keeps the repository smaller and avoids exposing partial training code paths that are not needed for the demo or runtime analysis.

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

The packaged backend expects `backend/app/ml/artifacts/graph-model.pt`. If that artifact is missing, the backend will fall back to the heuristic predictor instead of trying to regenerate a model locally.

## Run The Frontend

From the workspace root:

```bash
cd frontend
cp .env.example .env
npm run dev
```

Frontend URL:

• `http://localhost:5173`

## Presentation Frontend

For the final presentation, the frontend can be shipped prebuilt from `frontend/dist/`.

Build the presentation bundle:

```bash
cd frontend
npm run build:presentation
```

Serve the prebuilt bundle on any machine with Python installed:

```bash
cd frontend
npm run serve:presentation
```

That serves the static frontend from `http://localhost:4173` by default. The UI still expects the backend API to be reachable at `http://localhost:8000` unless you rebuild with a different `VITE_API_BASE_URL`.

## One-Command Presentation Mode

To copy this folder to another computer and run the demo with minimal setup, use the root launcher:

```bash
bash run-presentation.sh
```

What it does:

- creates `.venv/` locally if it does not exist
- attempts to install Python 3 automatically on common macOS and Linux setups if Python is missing
- installs only the backend runtime dependencies from `backend/requirements-presentation.txt`
- starts the FastAPI backend on `http://127.0.0.1:8000`
- serves the prebuilt frontend bundle from `frontend/dist/` on `http://127.0.0.1:4173`

What the target machine still needs:

- permission to install Python if it is not already present
- network access the first time, so `pip` can download the backend runtime packages

After the first successful run on that machine, rerunning the same command will reuse the local `.venv/`.

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

## Documentation

- `docs/ChainSentry_Project_Report.tex`: final project report.
- `docs/demo-scenarios-guide.md`: presentation speaking guide for the shipped demo scenarios.
- `docs/模型输入输出与5个案例说明.md`: concise Chinese note on model I/O and the current five demo cases.
- `docs/project proposal.txt`: original proposal text kept for reference.

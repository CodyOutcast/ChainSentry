from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models import AnalysisResponse, TransactionRequest
from app.services.analysis import analyze_transaction


app = FastAPI(
    title="ChainSentry API",
    version="0.1.0",
    description="Pre-signature risk analysis for blockchain transaction prototypes.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
def analyze(request: TransactionRequest) -> AnalysisResponse:
    return analyze_transaction(request)

"""FastAPI service scoring one transaction feature-vector at a time.

``POST /score`` takes the engineered per-transaction features (the stream consumer
or any feature pipeline computes them), returns the fraud probability, a decision
against the configured threshold and the top contributing features from XGBoost's
``pred_contribs`` (per-feature margin contributions, no SHAP dependency).
"""

from __future__ import annotations

import json
import math
from contextlib import asynccontextmanager
from typing import Annotated, Any

import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel, Field

from fraud_ml.config import Settings, get_settings

TOP_FEATURES = 3


class ScoreRequest(BaseModel):
    transaction_id: str | None = None
    amount: Annotated[float, Field(gt=0, description="transaction amount")]
    txn_count_1h: Annotated[int, Field(ge=0)] = 0
    txn_count_24h: Annotated[int, Field(ge=0)] = 0
    amount_zscore: float = 0.0
    seconds_since_last_txn: Annotated[float, Field(ge=0)] = 604800.0
    is_new_merchant: Annotated[int, Field(ge=0, le=1)] = 1
    geo_distance_km: Annotated[float, Field(ge=0)] = 0.0
    is_night: Annotated[int, Field(ge=0, le=1)] = 0


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class ScoreResponse(BaseModel):
    transaction_id: str | None
    fraud_probability: float
    decision: str
    threshold: float
    top_features: list[FeatureContribution]


class Scorer:
    """Wraps the committed booster plus the feature spec that pins input order."""

    def __init__(self, settings: Settings) -> None:
        spec = json.loads(settings.feature_spec_path.read_text())
        self.feature_names: list[str] = spec["features"]
        self.booster = xgb.Booster()
        self.booster.load_model(str(settings.model_path))
        self.threshold = settings.fraud_threshold

    def vector(self, req: ScoreRequest) -> np.ndarray:
        values = req.model_dump()
        values["log_amount"] = math.log1p(req.amount)
        return np.array([[float(values[name]) for name in self.feature_names]])

    def score(self, req: ScoreRequest) -> ScoreResponse:
        matrix = xgb.DMatrix(self.vector(req), feature_names=self.feature_names)
        probability = float(self.booster.predict(matrix)[0])
        contribs = self.booster.predict(matrix, pred_contribs=True)[0][:-1]  # drop bias term
        top = np.argsort(np.abs(contribs))[::-1][:TOP_FEATURES]
        return ScoreResponse(
            transaction_id=req.transaction_id,
            fraud_probability=round(probability, 6),
            decision="review" if probability >= self.threshold else "allow",
            threshold=self.threshold,
            top_features=[
                FeatureContribution(
                    feature=self.feature_names[i], contribution=round(float(contribs[i]), 4)
                )
                for i in top
            ],
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.scorer = Scorer(app_settings)
        yield

    app = FastAPI(title="fraud-ml scoring API", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        scorer: Scorer | None = getattr(app.state, "scorer", None)
        return {
            "status": "ok",
            "model_loaded": scorer is not None,
            "n_features": len(scorer.feature_names) if scorer else 0,
        }

    @app.post("/score", response_model=ScoreResponse)
    async def score(request: ScoreRequest) -> ScoreResponse:
        return app.state.scorer.score(request)

    return app


app = create_app()

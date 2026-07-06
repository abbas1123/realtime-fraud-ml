import json

import numpy as np
import pytest
import xgboost as xgb

from fraud_ml.data.generator import GeneratorConfig, generate
from fraud_ml.features.build import FEATURE_COLUMNS
from fraud_ml.train import XGBParams, precision_at_top, train_models

SMOKE_PARAMS = XGBParams(n_estimators=60, max_depth=4)


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory):
    df = generate(
        GeneratorConfig(n_transactions=30_000, n_users=400, n_merchants=150, days=30, seed=11)
    )
    models_dir = tmp_path_factory.mktemp("models")
    return train_models(df, models_dir=models_dir, xgb_params=SMOKE_PARAMS), models_dir


def test_precision_at_top() -> None:
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.4, 0.5, 0.9, 0.8])
    assert precision_at_top(y, scores, 0.2) == 1.0
    assert precision_at_top(y, scores, 0.4) == 0.5


def test_xgboost_beats_trivial_baseline(results) -> None:
    metrics, _ = results
    # a random or constant scorer sits at ROC-AUC 0.5 and PR-AUC ~= fraud rate (~0.02)
    assert metrics["xgboost"]["roc_auc"] > 0.85
    assert metrics["xgboost"]["pr_auc"] > 0.30
    assert metrics["xgboost"]["pr_auc"] > 10 * metrics["val_fraud_rate"]


def test_baseline_is_reported(results) -> None:
    metrics, _ = results
    for model in ("baseline", "xgboost"):
        for key in ("roc_auc", "pr_auc", "precision_at_1pct"):
            assert 0.0 <= metrics[model][key] <= 1.0


def test_artifacts_written_and_loadable(results) -> None:
    _, models_dir = results
    model_path = models_dir / "fraud_xgb.json"
    spec_path = models_dir / "feature_spec.json"
    assert model_path.exists()

    spec = json.loads(spec_path.read_text())
    assert spec["features"] == FEATURE_COLUMNS

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    row = np.array([[25.0, np.log1p(25.0), 1, 4, 0.2, 3600.0, 0, 1.0, 0]])
    prob = float(booster.predict(xgb.DMatrix(row, feature_names=FEATURE_COLUMNS))[0])
    assert 0.0 <= prob <= 1.0

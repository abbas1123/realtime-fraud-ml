"""Train and compare a logistic-regression baseline against XGBoost.

The split is strictly time-ordered (first 80% of events train, last 20% validate)
so the validation set never leaks future behavior into training. Class imbalance
is handled with ``class_weight="balanced"`` for the baseline and
``scale_pos_weight`` for XGBoost. Metrics reported on the validation window:
ROC-AUC, PR-AUC (average precision) and precision in the top 1% of scores.

Params, metrics and artifacts go to a local MLflow file store (``mlruns/``,
gitignored). The XGBoost booster is saved in native JSON format together with a
feature spec pinning the input order the model expects.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fraud_ml.config import get_settings
from fraud_ml.features.build import FEATURE_COLUMNS, build_features

VALIDATION_FRACTION = 0.2


@dataclass(frozen=True)
class XGBParams:
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.10
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: int = 5
    reg_lambda: float = 1.0


def precision_at_top(y_true: np.ndarray, scores: np.ndarray, fraction: float = 0.01) -> float:
    """Precision among the ``fraction`` highest-scored transactions."""
    k = max(1, int(len(scores) * fraction))
    top = np.argsort(scores)[::-1][:k]
    return float(np.asarray(y_true)[top].mean())


def evaluate(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "precision_at_1pct": precision_at_top(y_true, scores, 0.01),
    }


def time_split(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an event-time-sorted feature frame into train/validation windows."""
    cut = int(len(features) * (1.0 - VALIDATION_FRACTION))
    return features.iloc[:cut], features.iloc[cut:]


def train_models(
    df: pd.DataFrame,
    models_dir: Path | None = None,
    xgb_params: XGBParams | None = None,
    use_mlflow: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Build features, train both models and return metrics plus the booster."""
    xgb_params = xgb_params or XGBParams()

    features = build_features(df)
    train_df, val_df = time_split(features)
    x_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df["is_fraud"].to_numpy()
    x_val = val_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_val = val_df["is_fraud"].to_numpy()

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1)

    baseline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    baseline.fit(x_train, y_train)
    baseline_metrics = evaluate(y_val, baseline.predict_proba(x_val)[:, 1])

    xgb = XGBClassifier(
        **asdict(xgb_params),
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=seed,
    )
    xgb.fit(x_train, y_train)
    xgb_metrics = evaluate(y_val, xgb.predict_proba(x_val)[:, 1])

    results: dict[str, Any] = {
        "baseline": baseline_metrics,
        "xgboost": xgb_metrics,
        "train_rows": len(y_train),
        "val_rows": len(y_val),
        "val_fraud_rate": float(y_val.mean()),
        "scale_pos_weight": scale_pos_weight,
        "booster": xgb.get_booster(),
    }

    if models_dir is not None:
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "fraud_xgb.json"
        xgb.get_booster().save_model(str(model_path))
        spec = {
            "model_file": model_path.name,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "features": FEATURE_COLUMNS,
            "train_rows": len(y_train),
            "train_fraud_rate": float(y_train.mean()),
            "params": asdict(xgb_params) | {"scale_pos_weight": round(scale_pos_weight, 3)},
            "validation_metrics": {"baseline_logreg": baseline_metrics, "xgboost": xgb_metrics},
        }
        (models_dir / "feature_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
        results["model_path"] = model_path

    if use_mlflow:
        _log_to_mlflow(xgb_params, scale_pos_weight, baseline_metrics, xgb_metrics, models_dir)

    return results


def _log_to_mlflow(
    xgb_params: XGBParams,
    scale_pos_weight: float,
    baseline_metrics: dict[str, float],
    xgb_metrics: dict[str, float],
    models_dir: Path | None,
) -> None:
    import mlflow  # local import: keeps the training path usable without mlflow installed

    settings = get_settings()
    uri = settings.mlflow_tracking_uri
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run(run_name="baseline_logreg"):
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metrics(baseline_metrics)

    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_param("model", "xgboost")
        mlflow.log_params(asdict(xgb_params))
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 3))
        mlflow.log_metrics(xgb_metrics)
        if models_dir is not None:
            mlflow.log_artifact(str(models_dir / "fraud_xgb.json"))
            mlflow.log_artifact(str(models_dir / "feature_spec.json"))


def format_table(results: dict[str, Any]) -> str:
    header = f"{'model':<20}{'roc_auc':>10}{'pr_auc':>10}{'p@top1%':>10}"
    lines = [header, "-" * len(header)]
    for name, key in (("logreg (baseline)", "baseline"), ("xgboost", "xgboost")):
        m = results[key]
        lines.append(
            f"{name:<20}{m['roc_auc']:>10.4f}{m['pr_auc']:>10.4f}{m['precision_at_1pct']:>10.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud models on a generated dataset.")
    parser.add_argument("--data", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    print(f"loaded {len(df)} transactions, fraud_rate={df['is_fraud'].mean():.4f}")

    results = train_models(df, models_dir=args.models_dir, use_mlflow=not args.no_mlflow)
    print()
    print(format_table(results))
    print(
        f"\nval rows={results['val_rows']}  val fraud rate={results['val_fraud_rate']:.4f}  "
        f"scale_pos_weight={results['scale_pos_weight']:.1f}"
    )
    print(f"model saved to {results['model_path']}")


if __name__ == "__main__":
    main()

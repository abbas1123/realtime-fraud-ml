"""Runtime configuration, overridable via FRAUD_* environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FRAUD_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    model_path: Path = Path("models/fraud_xgb.json")
    feature_spec_path: Path = Path("models/feature_spec.json")
    threshold: float = 0.5  # decision boundary for the "review" verdict

    kafka_bootstrap_servers: str = "localhost:9092"
    transactions_topic: str = "transactions"
    scores_topic: str = "fraud-scores"
    consumer_group: str = "fraud-scorer"

    # Local tracking store; mlruns/ is gitignored. MLflow >= 3.14 requires a
    # database-backed store, so a SQLite file replaces the legacy ./mlruns layout.
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    mlflow_experiment: str = "fraud-ml"


def get_settings() -> Settings:
    return Settings()

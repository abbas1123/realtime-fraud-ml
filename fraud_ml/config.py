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
    fraud_threshold: float = 0.5

    kafka_bootstrap_servers: str = "localhost:9092"
    transactions_topic: str = "transactions"
    scores_topic: str = "fraud-scores"
    consumer_group: str = "fraud-scorer"

    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment: str = "fraud-ml"


def get_settings() -> Settings:
    return Settings()

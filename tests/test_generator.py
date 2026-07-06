import dataclasses

import pandas as pd
import pytest

from fraud_ml.data.generator import COLUMNS, FRAUD_TYPES, GeneratorConfig, generate, stream

SMALL = GeneratorConfig(n_transactions=8_000, n_users=300, n_merchants=120, days=21, seed=7)


@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    return generate(SMALL)


def test_same_seed_same_dataset(small_df: pd.DataFrame) -> None:
    again = generate(SMALL)
    pd.testing.assert_frame_equal(small_df, again)


def test_different_seed_different_dataset(small_df: pd.DataFrame) -> None:
    other = generate(dataclasses.replace(SMALL, seed=8))
    assert not small_df["amount"].equals(other["amount"])


def test_schema(small_df: pd.DataFrame) -> None:
    assert list(small_df.columns) == COLUMNS
    assert len(small_df) == SMALL.n_transactions
    assert small_df["transaction_id"].is_unique
    assert small_df["event_time"].is_monotonic_increasing
    assert (small_df["amount"] > 0).all()
    assert small_df["lat"].between(-90, 90).all()
    assert small_df["lon"].between(-180, 180).all()
    assert small_df.notna().all().all()


def test_fraud_labels(small_df: pd.DataFrame) -> None:
    rate = small_df["is_fraud"].mean()
    assert 0.5 * SMALL.fraud_rate <= rate <= 1.5 * SMALL.fraud_rate

    fraud = small_df[small_df["is_fraud"] == 1]
    legit = small_df[small_df["is_fraud"] == 0]
    assert set(fraud["fraud_type"]) == set(FRAUD_TYPES)
    assert (legit["fraud_type"] == "").all()

    # card-testing transactions are small by construction
    testing = fraud[fraud["fraud_type"] == "card_testing"]
    assert (testing["amount"] <= 3.5).all()
    # amount outliers are large by construction
    outliers = fraud[fraud["fraud_type"] == "amount_outlier"]
    assert (outliers["amount"] >= 900).all()


def test_stream_matches_batch(small_df: pd.DataFrame) -> None:
    records = list(stream(small_df.head(50)))
    assert len(records) == 50
    first = records[0]
    assert first["transaction_id"] == small_df.iloc[0]["transaction_id"]
    assert first["amount"] == small_df.iloc[0]["amount"]
    assert isinstance(first["event_time"], str)
    assert first["event_time"] == small_df.iloc[0]["event_time"].isoformat()


def test_drift_config_shifts_amounts() -> None:
    base = generate(GeneratorConfig(n_transactions=6_000, n_users=200, n_merchants=80, seed=3))
    drifted = generate(
        GeneratorConfig(n_transactions=6_000, n_users=200, n_merchants=80, seed=3, drift=True)
    )
    assert drifted["amount"].median() > base["amount"].median()

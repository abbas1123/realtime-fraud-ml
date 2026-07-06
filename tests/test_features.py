import numpy as np
import pandas as pd
import pytest

from fraud_ml.data.generator import GeneratorConfig, generate
from fraud_ml.features.build import FEATURE_COLUMNS, MAX_GAP_SECONDS, build_features, haversine_km
from fraud_ml.features.online import CardTracker


def _frame(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(
        rows, columns=["event_time", "card_id", "merchant_id", "amount", "lat", "lon"]
    )
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df


HANDCRAFTED = _frame(
    [
        ("2026-01-01 10:00:00", "c1", "m1", 10.0, 40.0, 50.0),
        ("2026-01-01 10:10:00", "c2", "m1", 5.0, 52.0, 13.0),
        ("2026-01-01 10:30:00", "c1", "m1", 20.0, 40.0, 50.0),
        ("2026-01-01 11:15:00", "c1", "m2", 30.0, 40.0, 50.0),
        ("2026-01-02 03:00:00", "c1", "m3", 100.0, 41.0, 51.0),
    ]
)


@pytest.fixture(scope="module")
def feats() -> pd.DataFrame:
    return build_features(HANDCRAFTED)


class TestHandcrafted:
    def test_first_transaction_defaults(self, feats: pd.DataFrame) -> None:
        row = feats.iloc[0]  # c1's first transaction
        assert row["txn_count_1h"] == 0
        assert row["txn_count_24h"] == 0
        assert row["amount_zscore"] == 0.0
        assert row["seconds_since_last_txn"] == MAX_GAP_SECONDS
        assert row["is_new_merchant"] == 1
        assert row["geo_distance_km"] == 0.0
        assert row["is_night"] == 0

    def test_other_card_does_not_interfere(self, feats: pd.DataFrame) -> None:
        row = feats.iloc[1]  # c2's only transaction, 10 minutes after c1's first
        assert row["txn_count_1h"] == 0
        assert row["txn_count_24h"] == 0
        assert row["is_new_merchant"] == 1
        assert row["geo_distance_km"] == 0.0

    def test_second_transaction(self, feats: pd.DataFrame) -> None:
        row = feats.iloc[2]  # c1, 10:30
        assert row["txn_count_1h"] == 1
        assert row["txn_count_24h"] == 1
        assert row["amount_zscore"] == 0.0  # only one prior amount, no std yet
        assert row["seconds_since_last_txn"] == 1800.0
        assert row["is_new_merchant"] == 0  # m1 seen at 10:00
        assert row["geo_distance_km"] == 0.0

    def test_third_transaction_window_boundary(self, feats: pd.DataFrame) -> None:
        row = feats.iloc[3]  # c1, 11:15 - the 10:00 txn left the 1h window
        assert row["txn_count_1h"] == 1
        assert row["txn_count_24h"] == 2
        # past amounts [10, 20]: mean 15, sample std ~7.0711
        assert row["amount_zscore"] == pytest.approx((30 - 15) / np.std([10, 20], ddof=1))
        assert row["seconds_since_last_txn"] == 2700.0
        assert row["is_new_merchant"] == 1  # m2 is new for c1

    def test_next_day_night_and_distance(self, feats: pd.DataFrame) -> None:
        row = feats.iloc[4]  # c1, 03:00 next day, 1 degree away
        assert row["txn_count_1h"] == 0
        assert row["txn_count_24h"] == 3
        # past amounts [10, 20, 30]: mean 20, sample std 10
        assert row["amount_zscore"] == pytest.approx(8.0)
        assert row["seconds_since_last_txn"] == pytest.approx(56_700.0)  # 15h45m
        assert row["is_new_merchant"] == 1
        assert row["geo_distance_km"] == pytest.approx(139.68, abs=0.05)
        assert row["is_night"] == 1


def test_haversine_known_distance() -> None:
    # Baku to Istanbul is roughly 1760 km
    assert haversine_km(40.4093, 49.8671, 41.0082, 28.9784) == pytest.approx(1760, abs=15)
    assert haversine_km(40.0, 50.0, 40.0, 50.0) == 0.0


def test_point_in_time_no_future_leakage() -> None:
    """Features of early transactions must not change when later ones are removed."""
    df = generate(GeneratorConfig(n_transactions=4_000, n_users=150, n_merchants=60, seed=5))
    full = build_features(df)
    half = build_features(df.iloc[: len(df) // 2])
    pd.testing.assert_frame_equal(
        full.iloc[: len(df) // 2].reset_index(drop=True)[FEATURE_COLUMNS],
        half.reset_index(drop=True)[FEATURE_COLUMNS],
    )


def test_online_tracker_matches_batch() -> None:
    """Replaying the stream through CardTracker reproduces the batch features."""
    df = generate(GeneratorConfig(n_transactions=3_000, n_users=120, n_merchants=60, seed=9))
    batch = build_features(df)
    tracker = CardTracker()

    online = np.array([tracker.vector(row) for row in batch.to_dict("records")])
    expected = batch[FEATURE_COLUMNS].to_numpy(dtype=float)
    np.testing.assert_allclose(online, expected, rtol=1e-9, atol=1e-9)

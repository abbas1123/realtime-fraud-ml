"""Point-in-time-correct per-card rolling features.

Every feature for a transaction at time ``t`` is computed strictly from that card's
transactions *before* ``t`` (window semantics: previous rows with timestamp in
``[t - window, t)``, plus earlier rows sharing the exact timestamp). The current
row never contributes to its own feature values, so there is no label leakage and
the same numbers can be reproduced event-by-event in a stream consumer.

Definitions (see ``FEATURE_COLUMNS`` for model input order):

* ``amount``                 - raw transaction amount
* ``log_amount``             - ``log1p(amount)``
* ``txn_count_1h``           - prior transactions on this card in the last hour
* ``txn_count_24h``          - prior transactions on this card in the last 24 hours
* ``amount_zscore``          - (amount - mean of past amounts) / sample std of past
                               amounts; 0.0 until the card has 2 prior transactions
                               or while the past std is 0
* ``seconds_since_last_txn`` - gap to the card's previous transaction, capped at 7
                               days (also the value for a card's first transaction)
* ``is_new_merchant``        - 1 if the card never paid this merchant before
* ``geo_distance_km``        - haversine distance from the card's previous
                               transaction location; 0.0 for the first transaction
* ``is_night``               - 1 if the transaction hour (UTC) is 0-5
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "amount",
    "log_amount",
    "txn_count_1h",
    "txn_count_24h",
    "amount_zscore",
    "seconds_since_last_txn",
    "is_new_merchant",
    "geo_distance_km",
    "is_night",
]

MAX_GAP_SECONDS = 7 * 24 * 3600.0
EARTH_RADIUS_KM = 6371.0


def haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray | float:
    """Great-circle distance in kilometers."""
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2)
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _window_counts(times: np.ndarray, window_seconds: float) -> np.ndarray:
    """For each element of a sorted time array, count prior elements within the window."""
    left = np.searchsorted(times, times - window_seconds, side="left")
    return np.arange(len(times)) - left


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` (event-time order) with feature columns appended.

    Expects columns: ``event_time`` (datetime64), ``card_id``, ``merchant_id``,
    ``amount``, ``lat``, ``lon``. Rows may arrive in any order; output is sorted by
    ``event_time`` (stable), matching the order a stream consumer would see.
    """
    out = df.sort_values(["card_id", "event_time"], kind="stable").copy()
    t = out["event_time"].to_numpy(dtype="datetime64[ns]").view("int64") / 1e9
    amount = out["amount"].to_numpy(dtype=float)

    # --- per-card window counts ---------------------------------------------------
    n = len(out)
    count_1h = np.zeros(n, dtype=np.int64)
    count_24h = np.zeros(n, dtype=np.int64)
    for idx in out.groupby("card_id", sort=False).indices.values():
        card_times = t[idx]
        count_1h[idx] = _window_counts(card_times, 3600.0)
        count_24h[idx] = _window_counts(card_times, 86400.0)

    # --- z-score of amount vs the card's past amounts (expanding, shifted) ---------
    grouped = out.groupby("card_id", sort=False)
    prior_n = grouped.cumcount().to_numpy(dtype=float)
    cum_sum = grouped["amount"].cumsum().to_numpy(dtype=float) - amount
    amount_sq = out["amount"] ** 2
    cum_sq = (
        amount_sq.groupby(out["card_id"], sort=False).cumsum().to_numpy(dtype=float) - amount**2
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        past_mean = cum_sum / prior_n
        past_var = (cum_sq - prior_n * past_mean**2) / (prior_n - 1.0)
        past_std = np.sqrt(np.maximum(past_var, 0.0))
        zscore = (amount - past_mean) / past_std
    zscore = np.where((prior_n >= 2) & (past_std > 0), zscore, 0.0)

    # --- gap to previous transaction ------------------------------------------------
    prev_t = grouped["event_time"].shift(1).to_numpy(dtype="datetime64[ns]").view("int64") / 1e9
    gap = np.where(prior_n > 0, t - prev_t, MAX_GAP_SECONDS)
    gap = np.minimum(gap, MAX_GAP_SECONDS)

    # --- new merchant for this card -------------------------------------------------
    seen_before = out.groupby(["card_id", "merchant_id"], sort=False).cumcount().to_numpy()
    is_new_merchant = (seen_before == 0).astype(np.int64)

    # --- distance from the card's previous transaction ------------------------------
    prev_lat = grouped["lat"].shift(1).to_numpy(dtype=float)
    prev_lon = grouped["lon"].shift(1).to_numpy(dtype=float)
    lat = out["lat"].to_numpy(dtype=float)
    lon = out["lon"].to_numpy(dtype=float)
    dist = np.where(
        prior_n > 0,
        haversine_km(np.nan_to_num(prev_lat), np.nan_to_num(prev_lon), lat, lon),
        0.0,
    )

    out["log_amount"] = np.log1p(amount)
    out["txn_count_1h"] = count_1h
    out["txn_count_24h"] = count_24h
    out["amount_zscore"] = zscore
    out["seconds_since_last_txn"] = gap
    out["is_new_merchant"] = is_new_merchant
    out["geo_distance_km"] = dist.astype(float)
    out["is_night"] = (out["event_time"].dt.hour < 6).astype(np.int64)

    return out.sort_values("event_time", kind="stable").reset_index(drop=True)

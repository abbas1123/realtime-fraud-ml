"""Incremental, per-card feature computation for stream scoring.

``CardTracker`` reproduces exactly the numbers ``fraud_ml.features.build`` computes
in batch, but one event at a time: call :meth:`CardTracker.features` with a
transaction *before* recording it, then :meth:`CardTracker.update` to fold it into
the card's history. ``tests/test_features.py`` asserts batch/online parity on a
generated sample.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fraud_ml.features.build import FEATURE_COLUMNS, MAX_GAP_SECONDS, haversine_km

__all__ = ["FEATURE_COLUMNS", "CardTracker"]


@dataclass
class _CardState:
    times: deque[float] = field(default_factory=deque)  # epoch seconds, ascending
    merchants: set[str] = field(default_factory=set)
    n: int = 0
    amount_sum: float = 0.0
    amount_sq_sum: float = 0.0
    last_t: float = 0.0
    last_lat: float = 0.0
    last_lon: float = 0.0


def _epoch_seconds(event_time: str | datetime) -> float:
    """Epoch seconds; naive timestamps are treated as UTC (matching the batch path)."""
    if isinstance(event_time, str):
        event_time = datetime.fromisoformat(event_time)
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=UTC)
    return event_time.timestamp()


class CardTracker:
    """Keeps rolling state for every card seen on the stream."""

    def __init__(self, max_window_seconds: float = 86400.0) -> None:
        self._window = max_window_seconds
        self._cards: dict[str, _CardState] = {}

    def _state(self, card_id: str) -> _CardState:
        state = self._cards.get(card_id)
        if state is None:
            state = _CardState()
            self._cards[card_id] = state
        return state

    def features(self, txn: dict[str, Any]) -> dict[str, float]:
        """Feature vector for ``txn`` using only previously recorded transactions."""
        state = self._state(txn["card_id"])
        t = _epoch_seconds(txn["event_time"])
        amount = float(txn["amount"])

        # Drop history that left the widest window (keeps memory bounded).
        while state.times and state.times[0] < t - self._window:
            state.times.popleft()
        count_24h = sum(1 for x in state.times if x >= t - 86400.0)
        count_1h = sum(1 for x in state.times if x >= t - 3600.0)

        if state.n >= 2:
            mean = state.amount_sum / state.n
            var = (state.amount_sq_sum - state.n * mean * mean) / (state.n - 1)
            std = math.sqrt(max(var, 0.0))
            zscore = (amount - mean) / std if std > 0 else 0.0
        else:
            zscore = 0.0

        if state.n > 0:
            gap = min(t - state.last_t, MAX_GAP_SECONDS)
            dist = float(
                haversine_km(state.last_lat, state.last_lon, float(txn["lat"]), float(txn["lon"]))
            )
        else:
            gap = MAX_GAP_SECONDS
            dist = 0.0

        hour = int((t % 86400.0) // 3600.0)
        return {
            "amount": amount,
            "log_amount": math.log1p(amount),
            "txn_count_1h": float(count_1h),
            "txn_count_24h": float(count_24h),
            "amount_zscore": zscore,
            "seconds_since_last_txn": gap,
            "is_new_merchant": 0.0 if txn["merchant_id"] in state.merchants else 1.0,
            "geo_distance_km": dist,
            "is_night": 1.0 if hour < 6 else 0.0,
        }

    def update(self, txn: dict[str, Any]) -> None:
        """Record ``txn`` into its card's history."""
        state = self._state(txn["card_id"])
        t = _epoch_seconds(txn["event_time"])
        amount = float(txn["amount"])
        state.times.append(t)
        state.merchants.add(txn["merchant_id"])
        state.n += 1
        state.amount_sum += amount
        state.amount_sq_sum += amount * amount
        state.last_t = t
        state.last_lat = float(txn["lat"])
        state.last_lon = float(txn["lon"])

    def vector(self, txn: dict[str, Any]) -> list[float]:
        """Features as a list in ``FEATURE_COLUMNS`` order, then record the txn."""
        feats = self.features(txn)
        self.update(txn)
        return [feats[name] for name in FEATURE_COLUMNS]

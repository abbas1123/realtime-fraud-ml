"""Seeded synthetic card-transaction generator with labeled fraud injection.

Produces a population of users, cards and merchants, then simulates legitimate
spending (per-category log-normal amounts, daytime-weighted timestamps, home-region
geo coordinates with occasional travel) and injects four labeled fraud patterns:

* ``card_testing``   - burst of many small rapid transactions at one online merchant
* ``geo_jump``       - stolen-card usage far from the user's home region
* ``amount_outlier`` - single purchases far above the card's normal spend
* ``velocity_spree`` - rapid medium/high-value purchases across many merchants

Everything is driven by one ``numpy`` RNG, so a given ``GeneratorConfig`` always
yields the identical dataset. The ``drift`` flag shifts spend mix and amount
distributions to emulate the kind of behavior change a drift monitor should catch.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EPOCH_START = pd.Timestamp("2026-05-01 00:00:00")

COLUMNS = [
    "transaction_id",
    "event_time",
    "card_id",
    "user_id",
    "merchant_id",
    "merchant_category",
    "mcc",
    "amount",
    "lat",
    "lon",
    "is_fraud",
    "fraud_type",
]

FRAUD_TYPES = ("card_testing", "geo_jump", "amount_outlier", "velocity_spree")


@dataclass(frozen=True)
class Category:
    name: str
    mcc: int
    log_mu: float
    log_sigma: float
    online: bool = False


CATEGORIES: tuple[Category, ...] = (
    Category("grocery", 5411, 3.60, 0.55),
    Category("fuel", 5541, 3.40, 0.40),
    Category("restaurant", 5812, 3.10, 0.60),
    Category("coffee", 5814, 1.80, 0.40),
    Category("online_retail", 5969, 3.90, 0.90, online=True),
    Category("electronics", 5732, 5.20, 0.80),
    Category("travel", 4722, 5.90, 0.70, online=True),
    Category("entertainment", 7832, 2.80, 0.50),
    Category("pharmacy", 5912, 3.00, 0.70),
    Category("utilities", 4900, 4.30, 0.35, online=True),
)

# (city, lat, lon) - home regions users are anchored to.
REGIONS: tuple[tuple[str, float, float], ...] = (
    ("baku", 40.4093, 49.8671),
    ("istanbul", 41.0082, 28.9784),
    ("berlin", 52.5200, 13.4050),
    ("warsaw", 52.2297, 21.0122),
    ("london", 51.5074, -0.1278),
    ("madrid", 40.4168, -3.7038),
    ("dubai", 25.2048, 55.2708),
    ("new_york", 40.7128, -74.0060),
    ("singapore", 1.3521, 103.8198),
    ("tbilisi", 41.7151, 44.8271),
)

# Relative volume of legitimate transactions per hour of day (UTC).
HOUR_WEIGHTS = np.array(
    [1, 1, 1, 1, 2, 3, 6, 9, 11, 12, 13, 14, 14, 13, 12, 12, 13, 14, 13, 11, 8, 5, 3, 2],
    dtype=float,
)


@dataclass(frozen=True)
class GeneratorConfig:
    n_transactions: int = 200_000
    n_users: int = 2_000
    n_merchants: int = 600
    days: int = 60
    seed: int = 42
    fraud_rate: float = 0.02
    travel_share: float = 0.02
    drift: bool = False
    start: pd.Timestamp = field(default=EPOCH_START)


def _geo_jitter(rng: np.random.Generator, scale: float, size: int) -> np.ndarray:
    return rng.normal(0.0, scale, size)


def _sample_hours(rng: np.random.Generator, size: int, weights: np.ndarray) -> np.ndarray:
    p = weights / weights.sum()
    return rng.choice(24, size=size, p=p)


class _World:
    """Static population of users, cards and merchants for one dataset."""

    def __init__(self, cfg: GeneratorConfig, rng: np.random.Generator) -> None:
        n_cat = len(CATEGORIES)

        self.user_region = rng.integers(0, len(REGIONS), cfg.n_users)
        region_lat = np.array([r[1] for r in REGIONS])
        region_lon = np.array([r[2] for r in REGIONS])
        self.user_home_lat = region_lat[self.user_region] + rng.normal(0, 0.15, cfg.n_users)
        self.user_home_lon = region_lon[self.user_region] + rng.normal(0, 0.15, cfg.n_users)

        prefs = rng.dirichlet(np.full(n_cat, 1.3), size=cfg.n_users)
        if cfg.drift:
            # Behavior shift: online categories take a much larger share of spend.
            online_idx = [i for i, c in enumerate(CATEGORIES) if c.online]
            prefs[:, online_idx] *= 1.9
            prefs /= prefs.sum(axis=1, keepdims=True)
        self.user_prefs = prefs

        # 1-2 cards per user.
        extra = rng.random(cfg.n_users) < 0.35
        owners = np.concatenate([np.arange(cfg.n_users), np.flatnonzero(extra)])
        owners.sort()
        self.card_owner = owners
        self.n_cards = len(owners)

        self.merchant_cat = rng.integers(0, n_cat, cfg.n_merchants)
        self.merchant_region = rng.integers(0, len(REGIONS), cfg.n_merchants)
        self.merchants_by_cat: dict[int, np.ndarray] = {
            c: np.flatnonzero(self.merchant_cat == c) for c in range(n_cat)
        }
        # Guarantee every category has at least one merchant.
        for c in range(n_cat):
            if len(self.merchants_by_cat[c]) == 0:
                self.merchant_cat[c] = c
                self.merchants_by_cat[c] = np.array([c])
        self.online_merchants = np.flatnonzero(
            np.array([CATEGORIES[c].online for c in self.merchant_cat])
        )

        # Per-user favorite merchants per category (habitual spending).
        self.favorites = np.empty((cfg.n_users, n_cat, 2), dtype=np.int64)
        for c in range(n_cat):
            pool = self.merchants_by_cat[c]
            self.favorites[:, c, :] = pool[rng.integers(0, len(pool), (cfg.n_users, 2))]


def _legit_amounts(rng: np.random.Generator, cat_idx: np.ndarray, drift: bool) -> np.ndarray:
    mu = np.array([c.log_mu for c in CATEGORIES])[cat_idx]
    sigma = np.array([c.log_sigma for c in CATEGORIES])[cat_idx]
    if drift:
        mu = mu + 0.22  # inflation-like shift in ticket size
    return np.round(np.maximum(rng.lognormal(mu, sigma), 0.5), 2)


def _generate_legit(cfg: GeneratorConfig, rng: np.random.Generator, world: _World) -> dict[str, np.ndarray]:
    n = int(cfg.n_transactions * (1.0 - cfg.fraud_rate))

    card_weights = rng.lognormal(0.0, 0.6, world.n_cards)
    card_weights /= card_weights.sum()
    card_idx = rng.choice(world.n_cards, size=n, p=card_weights)
    user_idx = world.card_owner[card_idx]

    day = rng.integers(0, cfg.days, n)
    hour_weights = HOUR_WEIGHTS.copy()
    if cfg.drift:
        hour_weights[0:6] *= 2.0  # more night-time activity
    hour = _sample_hours(rng, n, hour_weights)
    sec = rng.uniform(0, 3600, n)
    t = day * 86400.0 + hour * 3600.0 + sec

    # Category per transaction, sampled from each user's preference vector.
    cat_idx = np.empty(n, dtype=np.int64)
    order = np.argsort(user_idx, kind="stable")
    sorted_users = user_idx[order]
    boundaries = np.flatnonzero(np.diff(sorted_users)) + 1
    for block in np.split(order, boundaries):
        u = user_idx[block[0]]
        cat_idx[block] = rng.choice(len(CATEGORIES), size=len(block), p=world.user_prefs[u])

    # Merchant: mostly favorites, otherwise any merchant of the category.
    merchant_idx = np.empty(n, dtype=np.int64)
    use_fav = rng.random(n) < 0.75
    fav_slot = rng.integers(0, 2, n)
    merchant_idx[use_fav] = world.favorites[user_idx[use_fav], cat_idx[use_fav], fav_slot[use_fav]]
    rest = np.flatnonzero(~use_fav)
    for i in rest:
        pool = world.merchants_by_cat[cat_idx[i]]
        merchant_idx[i] = pool[rng.integers(0, len(pool))]

    amount = _legit_amounts(rng, cat_idx, cfg.drift)

    # Geo: near home, except an occasional trip to another region.
    travel_share = cfg.travel_share * (2.5 if cfg.drift else 1.0)
    traveling = rng.random(n) < travel_share
    lat = world.user_home_lat[user_idx] + _geo_jitter(rng, 0.05, n)
    lon = world.user_home_lon[user_idx] + _geo_jitter(rng, 0.05, n)
    if traveling.any():
        trip_region = rng.integers(0, len(REGIONS), traveling.sum())
        lat[traveling] = np.array([REGIONS[r][1] for r in trip_region]) + _geo_jitter(
            rng, 0.1, traveling.sum()
        )
        lon[traveling] = np.array([REGIONS[r][2] for r in trip_region]) + _geo_jitter(
            rng, 0.1, traveling.sum()
        )

    return {
        "t": t,
        "card_idx": card_idx,
        "user_idx": user_idx,
        "merchant_idx": merchant_idx,
        "cat_idx": cat_idx,
        "amount": amount,
        "lat": lat,
        "lon": lon,
        "is_fraud": np.zeros(n, dtype=np.int64),
        "fraud_type": np.array([""] * n, dtype=object),
    }


def _episode_rows(
    cfg: GeneratorConfig,
    rng: np.random.Generator,
    world: _World,
    fraud_budget: int,
) -> list[dict[str, Any]]:
    """Sample fraud episodes until the transaction budget is spent."""
    rows: list[dict[str, Any]] = []
    horizon = cfg.days * 86400.0

    while len(rows) < fraud_budget:
        kind = rng.choice(FRAUD_TYPES, p=[0.25, 0.30, 0.25, 0.20])
        card = int(rng.integers(0, world.n_cards))
        user = int(world.card_owner[card])
        home_lat = float(world.user_home_lat[user])
        home_lon = float(world.user_home_lon[user])
        t0 = float(rng.uniform(86400.0, horizon))  # leave day one for card history

        if kind == "card_testing":
            n_tx = int(rng.integers(8, 26))
            merchant = int(rng.choice(world.online_merchants))
            t = t0
            for _ in range(n_tx):
                t += float(rng.uniform(3.0, 45.0))
                rows.append(
                    {
                        "t": t,
                        "card_idx": card,
                        "user_idx": user,
                        "merchant_idx": merchant,
                        "cat_idx": int(world.merchant_cat[merchant]),
                        "amount": round(float(rng.uniform(0.5, 3.5)), 2),
                        "lat": home_lat + float(rng.normal(0, 0.05)),
                        "lon": home_lon + float(rng.normal(0, 0.05)),
                        "fraud_type": kind,
                    }
                )
        elif kind == "geo_jump":
            n_tx = int(rng.integers(2, 7))
            far = [r for r in range(len(REGIONS)) if r != world.user_region[user]]
            region = int(rng.choice(np.array(far)))
            r_lat, r_lon = REGIONS[region][1], REGIONS[region][2]
            t = t0
            for _ in range(n_tx):
                t += float(rng.uniform(180.0, 2400.0))
                cat = int(rng.choice([4, 5, 6, 2]))  # online, electronics, travel, restaurant
                pool = world.merchants_by_cat[cat]
                rows.append(
                    {
                        "t": t,
                        "card_idx": card,
                        "user_idx": user,
                        "merchant_idx": int(pool[rng.integers(0, len(pool))]),
                        "cat_idx": cat,
                        "amount": round(
                            float(rng.lognormal(CATEGORIES[cat].log_mu, CATEGORIES[cat].log_sigma))
                            * float(rng.uniform(1.4, 2.8)),
                            2,
                        ),
                        "lat": r_lat + float(rng.normal(0, 0.1)),
                        "lon": r_lon + float(rng.normal(0, 0.1)),
                        "fraud_type": kind,
                    }
                )
        elif kind == "amount_outlier":
            n_tx = int(rng.integers(1, 3))
            t = t0
            for _ in range(n_tx):
                t += float(rng.uniform(60.0, 5400.0))
                cat = int(rng.choice([5, 6, 4]))  # electronics, travel, online
                pool = world.merchants_by_cat[cat]
                rows.append(
                    {
                        "t": t,
                        "card_idx": card,
                        "user_idx": user,
                        "merchant_idx": int(pool[rng.integers(0, len(pool))]),
                        "cat_idx": cat,
                        "amount": round(float(rng.uniform(900.0, 4200.0)), 2),
                        "lat": home_lat + float(rng.normal(0, 0.15)),
                        "lon": home_lon + float(rng.normal(0, 0.15)),
                        "fraud_type": kind,
                    }
                )
        else:  # velocity_spree
            n_tx = int(rng.integers(5, 13))
            t = t0
            for _ in range(n_tx):
                t += float(rng.uniform(120.0, 720.0))
                cat = int(rng.choice([4, 5, 7, 2]))
                pool = world.merchants_by_cat[cat]
                rows.append(
                    {
                        "t": t,
                        "card_idx": card,
                        "user_idx": user,
                        "merchant_idx": int(pool[rng.integers(0, len(pool))]),
                        "cat_idx": cat,
                        "amount": round(
                            float(rng.lognormal(CATEGORIES[cat].log_mu, CATEGORIES[cat].log_sigma))
                            * float(rng.uniform(1.2, 2.4)),
                            2,
                        ),
                        "lat": home_lat + float(rng.normal(0, 0.3)),
                        "lon": home_lon + float(rng.normal(0, 0.3)),
                        "fraud_type": kind,
                    }
                )

    return rows[:fraud_budget]


def generate(cfg: GeneratorConfig) -> pd.DataFrame:
    """Generate one deterministic dataset for the given config."""
    rng = np.random.default_rng(cfg.seed)
    world = _World(cfg, rng)

    legit = _generate_legit(cfg, rng, world)
    fraud_budget = cfg.n_transactions - len(legit["t"])
    fraud_rows = _episode_rows(cfg, rng, world, fraud_budget)

    t = np.concatenate([legit["t"], np.array([r["t"] for r in fraud_rows])])
    t = np.round(t)  # second precision keeps batch and stream timestamps exactly aligned
    card_idx = np.concatenate(
        [legit["card_idx"], np.array([r["card_idx"] for r in fraud_rows], dtype=np.int64)]
    )
    user_idx = np.concatenate(
        [legit["user_idx"], np.array([r["user_idx"] for r in fraud_rows], dtype=np.int64)]
    )
    merchant_idx = np.concatenate(
        [legit["merchant_idx"], np.array([r["merchant_idx"] for r in fraud_rows], dtype=np.int64)]
    )
    cat_idx = np.concatenate(
        [legit["cat_idx"], np.array([r["cat_idx"] for r in fraud_rows], dtype=np.int64)]
    )
    amount = np.concatenate([legit["amount"], np.array([r["amount"] for r in fraud_rows])])
    lat = np.concatenate([legit["lat"], np.array([r["lat"] for r in fraud_rows])])
    lon = np.concatenate([legit["lon"], np.array([r["lon"] for r in fraud_rows])])
    is_fraud = np.concatenate([legit["is_fraud"], np.ones(len(fraud_rows), dtype=np.int64)])
    fraud_type = np.concatenate(
        [legit["fraud_type"], np.array([r["fraud_type"] for r in fraud_rows], dtype=object)]
    )

    df = pd.DataFrame(
        {
            "event_time": cfg.start + pd.to_timedelta(t, unit="s"),
            "card_id": pd.Series(card_idx).map(lambda i: f"c{i:05d}"),
            "user_id": pd.Series(user_idx).map(lambda i: f"u{i:05d}"),
            "merchant_id": pd.Series(merchant_idx).map(lambda i: f"m{i:05d}"),
            "merchant_category": pd.Series(cat_idx).map(lambda i: CATEGORIES[i].name),
            "mcc": pd.Series(cat_idx).map(lambda i: CATEGORIES[i].mcc).astype(np.int64),
            "amount": amount,
            "lat": lat,
            "lon": lon,
            "is_fraud": is_fraud,
            "fraud_type": fraud_type.astype(str),
        }
    )
    df = df.sort_values("event_time", kind="stable").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"tx{i:09d}" for i in range(len(df))])
    return df[COLUMNS]


def stream(df: pd.DataFrame) -> Iterator[dict[str, Any]]:
    """Yield transactions one by one in event-time order, as JSON-friendly dicts."""
    for row in df.itertuples(index=False):
        record = row._asdict()
        record["event_time"] = record["event_time"].isoformat()
        yield record


def main() -> None:
    parser = argparse.ArgumentParser(description="Write train/reference/current datasets.")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--n-transactions", type=int, default=200_000)
    parser.add_argument("--n-eval", type=int, default=None, help="rows in reference/current")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_eval = args.n_eval or max(args.n_transactions // 5, 10_000)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("train.parquet", GeneratorConfig(n_transactions=args.n_transactions, seed=args.seed)),
        ("reference.parquet", GeneratorConfig(n_transactions=n_eval, seed=args.seed + 1)),
        ("current.parquet", GeneratorConfig(n_transactions=n_eval, seed=args.seed + 2, drift=True)),
    ]
    for name, cfg in jobs:
        df = generate(cfg)
        path = args.out_dir / name
        df.to_parquet(path, index=False)
        print(
            f"{path}  rows={len(df)}  fraud_rate={df['is_fraud'].mean():.4f}  "
            f"span={df['event_time'].min()} .. {df['event_time'].max()}"
        )


if __name__ == "__main__":
    main()

"""Kafka scoring consumer: ``transactions`` topic in, ``fraud-scores`` topic out.

Consumes raw transaction events, maintains per-card rolling state with
``CardTracker`` (same feature definitions as the batch pipeline), scores each
event with the committed XGBoost model and produces a compact score record.

Needs a running broker - use ``docker compose up``. The module itself imports
without aiokafka installed; the dependency is only pulled in when ``run`` starts.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import numpy as np
import xgboost as xgb

from fraud_ml.config import Settings, get_settings
from fraud_ml.features.build import FEATURE_COLUMNS
from fraud_ml.features.online import CardTracker

log = logging.getLogger("fraud_ml.consumer")


class StreamScorer:
    """Stateful scorer shared by the consumer loop; no Kafka dependency, easy to test."""

    def __init__(self, settings: Settings) -> None:
        self.booster = xgb.Booster()
        self.booster.load_model(str(settings.model_path))
        self.threshold = settings.fraud_threshold
        self.tracker = CardTracker()

    def score(self, txn: dict[str, Any]) -> dict[str, Any]:
        vector = np.array([self.tracker.vector(txn)])
        matrix = xgb.DMatrix(vector, feature_names=FEATURE_COLUMNS)
        probability = float(self.booster.predict(matrix)[0])
        return {
            "transaction_id": txn["transaction_id"],
            "card_id": txn["card_id"],
            "event_time": txn["event_time"],
            "amount": txn["amount"],
            "fraud_probability": round(probability, 6),
            "decision": "review" if probability >= self.threshold else "allow",
            # ground-truth label passed through for demo comparison only
            "label": txn.get("is_fraud"),
        }


async def run(settings: Settings) -> None:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

    consumer = AIOKafkaConsumer(
        settings.transactions_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.consumer_group,
        auto_offset_reset="earliest",
        value_deserializer=lambda raw: json.loads(raw.decode("utf-8")),
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda obj: json.dumps(obj).encode("utf-8"),
    )

    scorer = StreamScorer(settings)
    await consumer.start()
    await producer.start()
    log.info("consuming %s -> producing %s", settings.transactions_topic, settings.scores_topic)

    scored = 0
    flagged = 0
    try:
        async for message in consumer:
            record = scorer.score(message.value)
            await producer.send_and_wait(
                settings.scores_topic, record, key=record["card_id"].encode("utf-8")
            )
            scored += 1
            flagged += record["decision"] == "review"
            if scored % 1000 == 0:
                log.info("scored=%d flagged=%d (%.2f%%)", scored, flagged, 100 * flagged / scored)
    finally:
        await consumer.stop()
        await producer.stop()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(run(get_settings()))


if __name__ == "__main__":
    main()

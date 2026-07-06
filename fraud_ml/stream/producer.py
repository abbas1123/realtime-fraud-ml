"""Demo producer: generates a synthetic dataset and replays it into Kafka.

Streams transactions in event-time order at a configurable rate so the scoring
consumer has something realistic to chew on. Runs as the ``producer`` service in
docker compose.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from fraud_ml.config import Settings, get_settings
from fraud_ml.data.generator import GeneratorConfig, generate, stream

log = logging.getLogger("fraud_ml.producer")


async def run(settings: Settings, n_transactions: int, rate: float, seed: int) -> None:
    from aiokafka import AIOKafkaProducer

    cfg = GeneratorConfig(n_transactions=n_transactions, n_users=500, n_merchants=200, seed=seed)
    df = generate(cfg)
    log.info("generated %d transactions (fraud rate %.3f)", len(df), df["is_fraud"].mean())

    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda obj: json.dumps(obj).encode("utf-8"),
    )
    await producer.start()
    delay = 1.0 / rate if rate > 0 else 0.0
    sent = 0
    try:
        for record in stream(df):
            await producer.send_and_wait(
                settings.transactions_topic, record, key=record["card_id"].encode("utf-8")
            )
            sent += 1
            if sent % 1000 == 0:
                log.info("sent %d/%d", sent, len(df))
            if delay:
                await asyncio.sleep(delay)
    finally:
        await producer.stop()
        log.info("done, sent %d transactions", sent)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Replay generated transactions into Kafka.")
    parser.add_argument("--n-transactions", type=int, default=20_000)
    parser.add_argument("--rate", type=float, default=50.0, help="events per second")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    asyncio.run(run(get_settings(), args.n_transactions, args.rate, args.seed))


if __name__ == "__main__":
    main()

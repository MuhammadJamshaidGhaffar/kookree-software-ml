#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frame‑consumer for Kookree assignment.

Features
--------
• Subscribes to a Kafka / Redpanda topic.
• Sends each JPEG frame to the gRPC inference server.
• Logs label + latency for every frame.
• Prints FPS / latency summary every <window> seconds.
• Exposes Prometheus metrics on :9100.

Example
-------
$ python consumer.py --bootstrap localhost:9092 --grpc localhost:50051
"""

from __future__ import annotations
import time
import logging
import argparse
from statistics import mean
from datetime import datetime
from pathlib import Path

import grpc
from kafka import KafkaConsumer
from prometheus_client import start_http_server, Counter, Gauge


# Add proto directory to sys.path for generated gRPC imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "proto"))

import image_infer_pb2
import image_infer_pb2_grpc


# ──────────────────────────────────────────────────────────────────────────
# Prometheus metrics
# ──────────────────────────────────────────────────────────────────────────
FRAME_COUNT = Counter("streaming_consumer_frames_total", "Frames processed")
INFER_FAILS = Counter("streaming_consumer_failures_total", "Inference failures")
FPS_GAUGE   = Gauge("streaming_consumer_fps", "Frames per second")

# ──────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────
LOG_DIR: Path = Path(__file__).parent / "logs" / "consumer"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log: logging.Logger = logging.getLogger("consumer")
log.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[*] %(message)s"))

timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file: Path = LOG_DIR / f"{timestamp}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))

log.addHandler(console_handler)
log.addHandler(file_handler)


# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    """CLI entry‑point."""
    parser = argparse.ArgumentParser(description="Kafka → gRPC inference consumer")
    parser.add_argument("--bootstrap", default="localhost:9092",
                        help="Kafka/Redpanda bootstrap servers")
    parser.add_argument("--topic", default="frames",
                        help="Topic name to consume")
    parser.add_argument("--grpc", default="localhost:50051",
                        help="Inference gRPC address host:port")
    parser.add_argument("--window", type=int, default=30,
                        help="Seconds between performance summaries")
    args = parser.parse_args()

    consumer: KafkaConsumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.bootstrap,
        auto_offset_reset="latest",
        group_id="frame_consumers",
        value_deserializer=lambda v: v  # raw bytes
    )

    channel: grpc.Channel = grpc.insecure_channel(args.grpc)
    stub: image_infer_pb2_grpc.InferenceStub = image_infer_pb2_grpc.InferenceStub(channel)

    # Expose Prometheus at :9100
    start_http_server(9100)
    log.info("[+] Prometheus consumer metrics on :9100")
    log.info("[+] Consumer connected -> %s | gRPC -> %s", args.bootstrap, args.grpc)

    latencies: list[float] = []
    frame_count: int = 0
    window_start: float = time.time()

    try:
        for msg in consumer:
            img_bytes: bytes = msg.value
            start_time: float = time.time()

            try:
                response = stub.ClassifyImage(
                    image_infer_pb2.ImageRequest(image=img_bytes))
                label: str = response.label
            except grpc.RpcError as e:
                label = "error"
                INFER_FAILS.inc()
                log.error("[!] gRPC error: %s", e)

            end_time: float = time.time()
            latency_ms: float = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            frame_count += 1

            FPS_GAUGE.set(1.0 / (end_time - start_time))
            FRAME_COUNT.inc()

            log.info("[+] Prediction: %-20s | Latency: %.2f ms", label, latency_ms)

            # periodic summary
            now: float = time.time()
            if now - window_start >= args.window and latencies:
                avg_lat = mean(latencies)
                fps = frame_count / (now - window_start)
                log.info("[*] === %ds SUMMARY ===", args.window)
                log.info("[+] Frames processed: %d", frame_count)
                log.info("[+] Avg latency: %.2f ms", avg_lat)
                log.info("[+] Throughput: %.2f FPS", fps)
                latencies.clear()
                frame_count = 0
                window_start = now

    except KeyboardInterrupt:
        log.info("[!] Interrupted by user")

    finally:
        consumer.close()
        channel.close()
        log.info("[+] Consumer shutdown complete")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Frame-consumer for Kookree assignment.

Features
--------
• Subscribes to a Kafka / Redpanda topic for JPEG frames.
• Sends each JPEG frame to the gRPC inference server.
• Logs label and latency for every frame.
• Prints FPS / latency summary every <window> seconds.
• Exposes Prometheus metrics on :9100 for monitoring.

Example
-------
    $ python consumer.py --bootstrap localhost:9092 --grpc localhost:50051

Arguments:
    --bootstrap  Redpanda/Kafka bootstrap server
    --topic      Topic name to consume
    --grpc       Inference gRPC address host:port
    --window     Seconds between performance summaries
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


timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ──────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────
def setup_logger() -> logging.Logger:
    """Configure and return a logger for the consumer."""
    log_dir = Path(__file__).parent / "logs" / "consumer"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("consumer")
    logger.setLevel(logging.INFO)
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[*] %(message)s"))
    # File handler
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


# ──────────────────────────────────────────────────────────────────────────

def process_messages(consumer: KafkaConsumer, stub: image_infer_pb2_grpc.InferenceStub, log: logging.Logger, window: int = 30) -> None:
    """Process messages from Kafka, send to gRPC, and log results and performance."""
    latencies: list[float] = []
    frame_count: int = 0
    window_start: float = time.time()
    for msg in consumer:
        img_bytes: bytes = msg.value
        start_time: float = time.time()
        try:
            response = stub.ClassifyImage(image_infer_pb2.ImageRequest(image=img_bytes))
            label: str = response.label
            inference_latency = response.latency
        except grpc.RpcError as e:
            label = "error"
            inference_latency = None
            INFER_FAILS.inc()
            log.error("[!] gRPC error: %s", e)
        end_time: float = time.time()
        rtt_latency_ms: float = (end_time - start_time) * 1000
        if inference_latency is not None:
            latencies.append(inference_latency * 1000)  # convert s to ms
            log.info("[+] Prediction: %-20s | Inference Latency: %.2f ms | RTT: %.2f ms", label, inference_latency * 1000, rtt_latency_ms)
        else:
            latencies.append(rtt_latency_ms)
            log.info("[+] Prediction: %-20s | RTT Latency: %.2f ms", label, rtt_latency_ms)
        frame_count += 1
        FRAME_COUNT.inc()
        # periodic summary
        now: float = time.time()
        if now - window_start >= window and latencies:
            avg_lat = mean(latencies)
            fps = frame_count / (now - window_start)
            log.info("[*] === %ds SUMMARY ===", window)
            log.info("[+] Frames processed: %d", frame_count)
            log.info("[+] Avg latency: %.2f ms", avg_lat)
            log.info("[+] Throughput: %.2f FPS", fps)
            latencies.clear()
            frame_count = 0
            window_start = now

def main() -> None:
    """
    CLI entry-point for the frame consumer. Subscribes to Kafka, sends frames to gRPC,
    logs predictions and latency, and exposes Prometheus metrics.
    """
    parser = argparse.ArgumentParser(description="Kafka → gRPC inference consumer")
    parser.add_argument("--bootstrap", default="localhost:9092",
                        help="Redpanda bootstrap server")
    parser.add_argument("--topic", default="frames",
                        help="Topic name to consume")
    parser.add_argument("--grpc", default="localhost:50051",
                        help="Inference gRPC address host:port")
    parser.add_argument("--window", type=int, default=30,
                        help="Seconds between performance summaries")
    args = parser.parse_args()

    log = setup_logger()
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
    try:
        process_messages(consumer, stub, log, window=args.window)
    except KeyboardInterrupt:
        log.info("[!] Interrupted by user")
    finally:
        consumer.close()
        channel.close()
        log.info("[+] Consumer shutdown complete")


if __name__ == "__main__":
    main()

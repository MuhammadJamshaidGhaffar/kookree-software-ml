#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Frame-producer for Kookree assignment.

Features
--------
• Reads webcam **or** video file as input source.
• Encodes each frame to JPEG format.
• Publishes raw JPEG bytes to a Kafka / Redpanda topic.
• Optional FPS throttling for rate control.
• Logs to console and to `logs/producer/<timestamp>.log`.

Example
-------
    $ python producer.py --source 0 --fps 15 \
        --bootstrap localhost:9092 --topic frames

Arguments:
    --source     Video file path or camera index (default 0)
    --fps        Optional frame rate limit (e.g. 30)
    --bootstrap  Redpanda/Kafka bootstrap servers
    --topic      Topic name to publish to
"""

from __future__ import annotations
import cv2
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from kafka import KafkaProducer


timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ──────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────
def setup_logger() -> logging.Logger:
    """Configure and return a logger for the producer."""
    log_dir = Path(__file__).parent / "logs" / "producer"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("producer")
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
# Main entry
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    CLI entry‑point for the frame producer. Opens the video/camera source, encodes frames,
    sends them to Kafka/Redpanda, and logs activity. Supports FPS throttling and live preview.
    """
    parser = argparse.ArgumentParser(description="Video / camera frame producer")
    parser.add_argument("--source", default="0",
                        help="Video file path or camera index (default 0)")
    parser.add_argument("--fps", type=float,
                        help="Optional frame‑rate limit (e.g. 30)")
    parser.add_argument("--bootstrap", default="localhost:9092",
                        help="Redpanda bootstrap servers")
    parser.add_argument("--topic", default="frames",
                        help="Topic name to publish to")
    args = parser.parse_args()

    log = setup_logger()

    # Open video/camera source
    src = int(args.source) if args.source.isdigit() else args.source
    cap: cv2.VideoCapture = cv2.VideoCapture(src)
    if not cap.isOpened():
        log.error("[!] Cannot open source %s", args.source)
        return

    # Initialize Kafka producer
    producer: KafkaProducer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        linger_ms=5,
        value_serializer=lambda v: v  # keep bytes as‑is
    )

    log.info("[+] Producer started -> %s [%s]", args.topic, args.bootstrap)

    frame_interval: float | None = 1.0 / args.fps if args.fps else None
    last_sent: float = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.info("[!] End of stream or read error; exiting")
                break

            # Show live webcam video (press 'q' to exit)
            cv2.imshow("Live Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                log.info("[!] 'q' pressed, exiting")
                break

            # Encode frame as JPEG
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                log.warning("[!] JPEG encode failed; skipping frame")
                continue

            # Send JPEG bytes to Kafka/Redpanda
            producer.send(args.topic, buf.tobytes())
            log.info("[+] Frame sent (%d bytes)", len(buf))

            # FPS throttling (if enabled)
            if frame_interval:
                now: float = time.time()
                time.sleep(max(0.0, frame_interval - (now - last_sent)))
                last_sent = time.time()

    except KeyboardInterrupt:
        log.info("[!] Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        producer.flush()
        producer.close()
        log.info("[+] Producer shutdown complete")


if __name__ == "__main__":
    main()

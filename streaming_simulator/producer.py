#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frame‑producer for Kookree assignment.

Features
--------
• Reads webcam **or** video file.
• Encodes each frame to JPEG.
• Publishes raw JPEG bytes to a Kafka / Redpanda topic.
• Optional FPS throttling.
• Logs to console and to `logs/producer/<timestamp>.log`.

Example
-------
$ python producer.py --source 0 --fps 15 \
    --bootstrap localhost:9092 --topic frames
"""

from __future__ import annotations
import cv2
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from kafka import KafkaProducer


# ──────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────
LOG_DIR: Path = Path(__file__).parent / "logs" / "producer"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log: logging.Logger = logging.getLogger("producer")
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
# Main entry
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    """CLI entry‑point."""
    parser = argparse.ArgumentParser(description="Video / camera frame producer")
    parser.add_argument("--source", default="0",
                        help="Video file path or camera index (default 0)")
    parser.add_argument("--fps", type=float,
                        help="Optional frame‑rate limit (e.g. 30)")
    parser.add_argument("--bootstrap", default="localhost:9092",
                        help="Kafka/Redpanda bootstrap servers")
    parser.add_argument("--topic", default="frames",
                        help="Topic name to publish to")
    args = parser.parse_args()

    # open capture
    src = int(args.source) if args.source.isdigit() else args.source
    cap: cv2.VideoCapture = cv2.VideoCapture(src)
    if not cap.isOpened():
        log.error("[!] Cannot open source %s", args.source)
        return

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

            # Show live webcam video
            cv2.imshow("Live Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                log.info("[!] 'q' pressed, exiting")
                break

            ok, buf = cv2.imencode(".jpg", frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                log.warning("[!] JPEG encode failed; skipping frame")
                continue

            producer.send(args.topic, buf.tobytes())
            log.info("[+] Frame sent (%d bytes)", len(buf))

            # FPS throttling
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

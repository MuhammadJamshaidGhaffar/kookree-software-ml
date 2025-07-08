#!/usr/bin/env python3

"""
Run both producer and consumer with CLI args and live logs.

This script launches both the streaming producer and consumer as subprocesses,
passing through relevant CLI arguments, and displays their logs in real time.

Usage:
    python run_streaming.py --source 0 --fps 10 --bootstrap localhost:9092 --topic frames --grpc localhost:50051 --window 30

Arguments:
    --source     Video file path or camera index for producer
    --fps        Optional frame rate limit for producer
    --bootstrap  Kafka/Redpanda bootstrap servers
    --topic      Kafka topic name
    --grpc       gRPC inference server address
    --window     Summary window (seconds) for consumer
"""

import subprocess
import os
import sys
import signal
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
producer_path = os.path.join(SCRIPT_DIR, "producer.py")
consumer_path = os.path.join(SCRIPT_DIR, "consumer.py")


def main():
    """
    Parse CLI arguments, launch producer and consumer subprocesses, and stream their logs.
    Handles graceful shutdown on Ctrl+C.
    """
    parser = argparse.ArgumentParser(description="Run both producer and consumer")
    parser.add_argument("--source", default="0", help="Video file path or camera index")
    parser.add_argument("--fps", type=float, help="Optional frame rate limit for producer")
    parser.add_argument("--bootstrap", default="localhost:9092", help="Kafka/Redpanda bootstrap servers")
    parser.add_argument("--topic", default="frames", help="Kafka topic name")
    parser.add_argument("--grpc", default="localhost:50051", help="gRPC inference server address")
    parser.add_argument("--window", type=int, default=30, help="Summary window (seconds)")

    args = parser.parse_args()

    # Build CLI arguments for producer subprocess
    producer_args = [
        "--source", args.source,
        "--bootstrap", args.bootstrap,
        "--topic", args.topic,
    ]
    if args.fps:
        producer_args += ["--fps", str(args.fps)]

    # Build CLI arguments for consumer subprocess
    consumer_args = [
        "--bootstrap", args.bootstrap,
        "--topic", args.topic,
        "--grpc", args.grpc,
        "--window", str(args.window)
    ]

    print("[*] Starting producer and consumer...")
    print(f"[*] Producer args: {' '.join(producer_args)}")
    print(f"[*] Consumer args: {' '.join(consumer_args)}")

    # Launch producer subprocess
    producer_proc = subprocess.Popen(
        [sys.executable, producer_path] + producer_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    # Launch consumer subprocess
    consumer_proc = subprocess.Popen(
        [sys.executable, consumer_path] + consumer_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    def stream_logs(proc, label):
        """Stream and print logs from a subprocess with a label prefix."""
        for line in proc.stdout:
            print(f"[{label}] {line.decode().strip()}")

    try:
        # Start threads to stream logs from both subprocesses
        import threading
        threading.Thread(target=stream_logs, args=(producer_proc, "PRODUCER"), daemon=True).start()
        threading.Thread(target=stream_logs, args=(consumer_proc, "CONSUMER"), daemon=True).start()

        # Keep main thread alive while subprocesses run
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C received. Stopping processes...")
        # Gracefully terminate both subprocesses
        for p in [producer_proc, consumer_proc]:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()

    print("[+] Shutdown complete.")


if __name__ == "__main__":
    main()

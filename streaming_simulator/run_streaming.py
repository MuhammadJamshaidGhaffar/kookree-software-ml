#!/usr/bin/env python3
"""
Run both producer and consumer with CLI args and live logs.
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
    parser = argparse.ArgumentParser(description="Run both producer and consumer")
    parser.add_argument("--source", default="0", help="Video file path or camera index")
    parser.add_argument("--fps", type=float, help="Optional frame rate limit for producer")
    parser.add_argument("--bootstrap", default="localhost:9092", help="Kafka/Redpanda bootstrap servers")
    parser.add_argument("--topic", default="frames", help="Kafka topic name")
    parser.add_argument("--grpc", default="localhost:50051", help="gRPC inference server address")
    parser.add_argument("--window", type=int, default=30, help="Summary window (seconds)")

    args = parser.parse_args()

    # Build producer args
    producer_args = [
        "--source", args.source,
        "--bootstrap", args.bootstrap,
        "--topic", args.topic,
    ]
    if args.fps:
        producer_args += ["--fps", str(args.fps)]

    # Build consumer args
    consumer_args = [
        "--bootstrap", args.bootstrap,
        "--topic", args.topic,
        "--grpc", args.grpc,
        "--window", str(args.window)
    ]

    print("[*] Starting producer and consumer...")
    print(f"[*] Producer args: {' '.join(producer_args)}")
    print(f"[*] Consumer args: {' '.join(consumer_args)}")

    producer_proc = subprocess.Popen(
        [sys.executable, producer_path] + producer_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    consumer_proc = subprocess.Popen(
        [sys.executable, consumer_path] + consumer_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    def stream_logs(proc, label):
        for line in proc.stdout:
            print(f"[{label}] {line.decode().strip()}")

    try:
        import threading
        threading.Thread(target=stream_logs, args=(producer_proc, "PRODUCER"), daemon=True).start()
        threading.Thread(target=stream_logs, args=(consumer_proc, "CONSUMER"), daemon=True).start()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[!] Ctrl+C received. Stopping processes...")
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

#!/usr/bin/env python3

"""
Load test for gRPC inference endpoint.
Sends parallel image requests and reports latency/FPS.

Usage:
    python load_test_grpc.py --grpc localhost:50051 --image path/to/image.jpg --requests 100 --concurrency 10

Arguments:
    --grpc        gRPC server address (host:port)
    --image       Optional image path (if not provided, uses random image)
    --requests    Number of total requests to send
    --concurrency Number of concurrent threads
"""

import time
import argparse
import grpc
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..","streaming_simulator" ,"proto"))

import image_infer_pb2, image_infer_pb2_grpc
import cv2


def read_image_bytes(path=None):
    """
    Reads an image from the given path and encodes it as JPEG bytes.
    If no path is provided, generates a random 224x224 RGB image.
    Returns:
        bytes: JPEG-encoded image bytes
    """
    if path:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image from {path}")
        _, buf = cv2.imencode(".jpg", image)
        return buf.tobytes()
    else:
        # Generate a random image (224x224 RGB)
        random_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", random_img)
        return buf.tobytes()


def send_request(stub, image_bytes):
    """
    Sends a single gRPC inference request and measures the round-trip latency.
    Args:
        stub: gRPC stub for the Inference service
        image_bytes: JPEG-encoded image bytes
    Returns:
        (latency_ms, label): Tuple of latency in ms and predicted label (or error)
    """
    try:
        response = stub.ClassifyImage(image_infer_pb2.ImageRequest(image=image_bytes))
        print(f"[+] Received response: {response.label} (latency: {response.latency:.2f} ms)")
        return response.latency, response.label
    except Exception as e:
        return None, f"[!] ERROR: {str(e)}"


def run_test(stub, image_bytes, num_requests, concurrency):
    """
    Runs the load test by sending parallel requests to the gRPC server.
    Args:
        stub: gRPC stub for the Inference service
        image_bytes: JPEG-encoded image bytes to send
        num_requests: Total number of requests to send
        concurrency: Number of concurrent threads
    """
    latencies = []
    results = []
    lock = threading.Lock()

    def worker():
        # Worker function for each thread
        latency, result = send_request(stub, image_bytes)
        with lock:
            latencies.append(latency)
            results.append(result)

    t_start = time.time()

    # Launch threads to send requests in parallel
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(num_requests)]
        for f in as_completed(futures):
            pass  # Wait for all to complete

    t_end = time.time()
    duration = t_end - t_start
    successful = [l for l in latencies if l is not None]

    # Print summary statistics
    print("\n[*] ==== Load Test Summary ====")
    print(f"[+] Total Requests:    {num_requests}")
    print(f"[+] Concurrency:       {concurrency}")
    print(f"[+] Duration:          {duration:.2f} sec")
    print(f"[+] Successful:        {len(successful)}")
    if successful:
        print(f"[+] Avg Latency:       {np.mean(successful):.2f} ms")
        print(f"[+] Max Latency:       {np.max(successful):.2f} ms")
        print(f"[+] Min Latency:       {np.min(successful):.2f} ms")
        print(f"[+] Throughput:        {len(successful) / duration:.2f} req/sec")

    failed = [r for r in results if isinstance(r, str) and r.startswith("[!]")]
    if failed:
        print(f"[!] Failures: {len(failed)}")
        for f in failed[:3]:  # show only a few
            print("    " + f)

    if successful and isinstance(results[0], str) is False:
        print(f"[+] Sample Label:      {results[0]}")


def main():
    """
    CLI entry point for the load test script. Parses arguments, loads image, and runs the test.
    """
    parser = argparse.ArgumentParser(description="gRPC Inference Load Tester")
    parser.add_argument("--grpc", default="localhost:50051", help="gRPC server host:port")
    parser.add_argument("--image", help="Optional image path (else random)")
    parser.add_argument("--requests", type=int, default=100, help="Number of total requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent threads")

    args = parser.parse_args()

    # Load image (from file or random)
    image_bytes = read_image_bytes(args.image)

    # Create gRPC channel and stub
    channel = grpc.insecure_channel(args.grpc)
    stub = image_infer_pb2_grpc.InferenceStub(channel)

    # Run the load test
    run_test(stub, image_bytes, args.requests, args.concurrency)

if __name__ == "__main__":
    main()

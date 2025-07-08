#!/usr/bin/env python3
"""
Load test for gRPC inference endpoint.
Sends parallel image requests and reports latency/FPS.
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
    t0 = time.time()
    try:
        response = stub.ClassifyImage(image_infer_pb2.ImageRequest(image=image_bytes))
        latency_ms = (time.time() - t0) * 1000
        return latency_ms, response.label
    except Exception as e:
        return None, f"[!] ERROR: {str(e)}"

def run_test(stub, image_bytes, num_requests, concurrency):
    latencies = []
    results = []
    lock = threading.Lock()

    def worker():
        latency, result = send_request(stub, image_bytes)
        with lock:
            latencies.append(latency)
            results.append(result)

    t_start = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(num_requests)]
        for f in as_completed(futures):
            pass

    t_end = time.time()
    duration = t_end - t_start
    successful = [l for l in latencies if l is not None]

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
    parser = argparse.ArgumentParser(description="gRPC Inference Load Tester")
    parser.add_argument("--grpc", default="localhost:50051", help="gRPC server host:port")
    parser.add_argument("--image", help="Optional image path (else random)")
    parser.add_argument("--requests", type=int, default=100, help="Number of total requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent threads")

    args = parser.parse_args()

    # Load image
    image_bytes = read_image_bytes(args.image)

    # gRPC
    channel = grpc.insecure_channel(args.grpc)
    stub = image_infer_pb2_grpc.InferenceStub(channel)

    run_test(stub, image_bytes, args.requests, args.concurrency)

if __name__ == "__main__":
    main()

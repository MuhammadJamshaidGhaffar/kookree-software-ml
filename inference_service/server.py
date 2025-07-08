#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gRPC ONNX Inference Microservice

Features
--------
• /healthz  – HTTP readiness probe (port 8080)
• /metrics  – Prometheus metrics (port 8000)
• gRPC      – ClassifyImage on port 50051
• Prometheus histogram for latency (industry standard)
• Logs to console and file

Usage:
    python server.py
"""

import os
import time
import threading
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.request

import cv2
import grpc
import numpy as np
import onnxruntime as ort
from concurrent import futures

from prometheus_client import start_http_server, Counter, Histogram

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'proto'))
from proto import image_infer_pb2, image_infer_pb2_grpc

# ──────────────────────────────────────────────────────────────────────────────
# Prometheus Metrics
# ──────────────────────────────────────────────────────────────────────────────
REQUEST_COUNT   = Counter(  "inference_requests_total",
                            "Total gRPC inference requests")
REQUEST_ERRORS  = Counter(  "inference_request_errors_total",
                            "Total inference errors")
# Histogram buckets: 0.05s, 0.1s, 0.2s, 0.5s, 1s, 2s (50ms, 100ms, 200ms, 500ms, 1s, 2s)
INFERENCE_LATENCY_HISTOGRAM = Histogram(
    "inference_latency_seconds",
    "Inference request latency in seconds",
    buckets=[0.05, 0.1, 0.2, 0.5, 1, 2]
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
log = logging.getLogger("inference")
log.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[*] %(message)s"))

file_handler = logging.FileHandler("inference_service.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))

log.addHandler(console_handler)
log.addHandler(file_handler)

# ──────────────────────────────────────────────────────────────────────────────
# ImageNet Class Names
# ──────────────────────────────────────────────────────────────────────────────
CLASS_NAMES_URL  = ("https://raw.githubusercontent.com/pytorch/hub/"
                    "refs/heads/master/imagenet_classes.txt")
CLASS_NAMES_PATH = "model/imagenet_classes.txt"

def download_class_names() -> None:
    if not os.path.exists(CLASS_NAMES_PATH):
        try:
            log.info("[*] Downloading ImageNet class names …")
            urllib.request.urlretrieve(CLASS_NAMES_URL, CLASS_NAMES_PATH)
            log.info("[+] Saved to %s", CLASS_NAMES_PATH)
        except Exception as e:
            log.warning("[!] Could not download class names: %s", e)

def load_class_names():
    download_class_names()
    try:
        with open(CLASS_NAMES_PATH, "r") as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        log.warning("[!] Failed to load class names: %s", e)
        return [str(i) for i in range(1000)]

CLASS_NAMES = load_class_names()

# ──────────────────────────────────────────────────────────────────────────────
# HTTP Readiness Probe (/healthz)
# ──────────────────────────────────────────────────────────────────────────────
def start_http_health_server() -> None:
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):                                # noqa N802
            if self.path == "/healthz":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
            else:
                self.send_response(404)
                self.end_headers()
        def log_message(self, *_):                       # silence default
            pass

    server = HTTPServer(("0.0.0.0", 8080), HealthHandler)
    threading.Thread(target=server.serve_forever,
                     daemon=True).start()
    log.info("[+] HTTP /healthz endpoint on :8080")

# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────
_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
_STD  = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

def preprocess(img_bytes: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD          # normalize
    img = img.transpose(2, 0, 1)        # HWC → CHW
    return np.expand_dims(img, 0).astype(np.float32)  # add batch dim

# ──────────────────────────────────────────────────────────────────────────────
# gRPC Servicer
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# gRPC Inference Servicer
# ──────────────────────────────────────────────────────────────────────────────
class InferenceServicer(image_infer_pb2_grpc.InferenceServicer):
    def __init__(self):
        """Initialize ONNX session and providers."""
        prefer_gpu = os.getenv("USE_GPU", "false").lower() == "true"
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if prefer_gpu else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession("model/resnet18.onnx", providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        log.info("[+] ONNX providers: %s", self.session.get_providers())

    def ClassifyImage(self, request, context):
        """
        Handle gRPC image classification request.
        Records latency in Prometheus histogram and returns label + latency.
        """
        REQUEST_COUNT.inc()
        start_time = time.time()
        try:
            x = preprocess(request.image)
            logits = self.session.run(None, {self.input_name: x})[0]
            idx = int(np.argmax(logits))
            label = CLASS_NAMES[idx]
            latency = time.time() - start_time
            INFERENCE_LATENCY_HISTOGRAM.observe(latency)
            log.info("[+] Predicted: %s | Inference latency: %.4f s", label, latency)
            return image_infer_pb2.ImageResponse(label=label, latency=latency)
        except Exception as e:
            REQUEST_ERRORS.inc()
            log.error("[!] Inference error: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Inference failed")
            return image_infer_pb2.ImageResponse(label="error", latency=0.0)
            



# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
def serve() -> None:
    log.info("[*] Starting inference microservice")

    # readiness probe
    start_http_health_server()

    # Prometheus metrics
    start_http_server(8000)
    log.info("[+] Prometheus metrics on :8000/metrics")

    # gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(
                         max_workers=os.cpu_count() or 2))
    image_infer_pb2_grpc.add_InferenceServicer_to_server(
        InferenceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    log.info("[+] gRPC server on :50051")
    server.wait_for_termination()

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    serve()

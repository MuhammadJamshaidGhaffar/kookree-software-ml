# Kookree Real‑Time Image Classification Pipeline

A step‑by‑step guide to spin up a full real‑time image‑classification stack using PyTorch (ONNX), Docker, gRPC, and Redpanda.

---

## 📦 Step 1 – Clone & Install Host Dependencies

```bash
git clone https://github.com/YOUR_USER/kookree-pipeline.git
cd kookree-pipeline
# Install Python deps for producer / consumer & load test
pip install -r requirements.txt
```

---

## 🐳 Step 2 – Build and Start Core Services

```bash
# Build inference image & launch everything
docker compose build           # multi‑stage build
docker compose up -d                             # inference, redpanda, prometheus, grafana
```

Services started:

| Service           | Purpose                                    | Ports             |
| ----------------- | ------------------------------------------ | ----------------- |
| inference_service | gRPC 50051 · /healthz 8080 · /metrics 8000 | 50051, 8080, 8000 |
| redpanda          | Kafka‑compatible broker                    | 9092, 9644        |
| prometheus        | Metrics storage                            | 9090              |
| grafana           | Dashboard UI                               | 3000              |

Verify readiness:

```bash
curl http://localhost:8080/healthz   # → OK
```

---

## 🎥 Step 3 – Start Streaming Simulator

Launch producer (webcam 0, 15 FPS) **and** consumer in one command:

```bash
python streaming_simulator/run_streaming.py \
    --source 0 \
    --fps 15 \
    --bootstrap localhost:9092 \
    --grpc localhost:50051 \
    --window 30
```

After 30 s you’ll see an FPS / latency summary in the terminal and in `logs/`.

---

## 🧪 Step 4 – Load‑test the gRPC Endpoint (optional)

```bash
python tests/load_test_grpc.py \
    --grpc localhost:50051 \
    --requests 200 \
    --concurrency 20
```

Outputs total throughput, avg latency, and sample label.

---

## 📈 Monitoring (optional)

### Prometheus

- Inference metrics: <http://localhost:8000/metrics>
- Consumer metrics: <http://localhost:9100/metrics>

Prometheus UI: <http://localhost:9090>

### Grafana Dashboard

1. Open <http://localhost:3000> (admin / admin)
2. **Dashboards → Import → Upload JSON**
3. Select `monitoring/grafana/dashboards/kookree_dashboard.json`
4. Click **Import** to visualise latency, FPS, errors, etc.

---

## 📂 Log Files

```
logs/
├── producer/YYYY-MM-DD_HH-MM-SS.log
├── consumer/YYYY-MM-DD_HH-MM-SS.log
└── inference_service.log
```

Each run creates fresh timestamped logs for traceability.

---

## 🔄 Rebuild Tips

Enable caching of Torch download:

```dockerfile
RUN --mount=type=cache,target=/root/.cache \
    python inference_service/model/export_to_onnx.py
```

Re‑build:

```bash
docker compose build --no-cache   # if needed
```

---

## ❓ Troubleshooting

| Symptom               | Fix                                         |
| --------------------- | ------------------------------------------- |
| Redpanda not ready    | `docker compose logs redpanda`              |
| Camera not accessible | Run `producer.py` on host, not in container |
| gRPC errors           | Check `inference_service.log`               |

---

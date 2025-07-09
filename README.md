# Kookree Real‑Time Image Classification Pipeline

![image](https://github.com/user-attachments/assets/398fe4a0-fbba-41be-ad7e-db38b7f4ae3d)

A step‑by‑step guide to spin up a full real‑time image‑classification stack using PyTorch (ONNX), Docker, gRPC, and Redpanda.

---

## 📦 Step 1 – Clone & Install Host Dependencies

````bash
# Clone the repository
git clone https://github.com/MuhammadJamshaidGhaffar/kookree-software-ml.git
cd kookree-pipeline

# (Optional but recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate

# Install Python dependencies for producer / consumer & load testing
pip install -r requirements.txt
````

---

## 🐳 Step 2 – Build and Start Core Services

```bash
# Build inference image & launch everything
docker compose build
docker compose up -d  # inference, redpanda, prometheus, grafana
```

For running on GPU you've to set env variable in inference_service of docker-compose.yml
```bash
environment:
  - USE_GPU=true
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

## 🧾 Step 3 – Generate Python gRPC Stubs

Before running the streaming simulator, generate Python gRPC client files for `image_infer.proto`:

```bash
chmod +x gen_proto.sh
./gen_proto.sh
```

This creates `streaming_simulator/proto/` with:

```
proto/
├── __init__.py
├── image_infer_pb2.py
└── image_infer_pb2_grpc.py
```

> ⚠️ This step is required for gRPC client code to import `image_infer_pb2` and `image_infer_pb2_grpc` properly.

---

## 🎥 Step 4 – Start Streaming Simulator

Launch producer (webcam 0, 15 FPS) **and** consumer in one command:

```bash
python streaming_simulator/run_streaming.py
```

or alternatively if you want to change default parameters

```bash
python streaming_simulator/run_streaming.py     --source 0     --fps 15     --bootstrap localhost:9092     --grpc localhost:50051     --window 30
```

It will open your webcam and start streaming frames to redpanda.
After 30 s you’ll see an FPS / latency summary in the terminal and in `logs/`.

---

## 🧪 Step 5 – Load‑test the gRPC Endpoint (optional)

```bash
python tests/load_test_grpc.py     --grpc localhost:50051     --requests 200     --concurrency 20
```

Outputs total throughput, avg latency, and sample label.

---

## 📈Step 6 – Monitoring (optional)

### Prometheus

- Inference metrics: <http://localhost:8000/metrics>
- Consumer metrics: <http://localhost:9100/metrics>
- Prometheus UI: <http://localhost:9090>

### Grafana Dashboard

1. Open Grafana: <http://localhost:3000>  
   _(Login with `admin` / `admin`, then change password if prompted)_

2. 👉 **First, add a data source**:

   - Go to **⚙️ Settings → Data Sources**
   - Click **"Add data source"**
   - Choose **"Prometheus"**
   - Set URL to: `http://prometheus:9090`
   - Click **"Save & test"** — it should connect successfully.

3. Now, **import the dashboard**:
   - Go to **Dashboards → Import → Upload JSON**
   - Select: `monitoring/grafana/dashboards/kookree_dashboard.json`
   - Choose the **Prometheus** data source you just added
   - Click **Import**

You should now see latency, FPS, error rates, and other metrics live in Grafana.

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

Re‑build:

```bash
docker compose build
```

---

## ❓ Troubleshooting

| Symptom               | Fix                                         |
| --------------------- | ------------------------------------------- |
| Redpanda not ready    | `docker compose logs redpanda`              |
| Camera not accessible | Run `producer.py` on host, not in container |
| gRPC errors           | Check `inference_service.log`               |

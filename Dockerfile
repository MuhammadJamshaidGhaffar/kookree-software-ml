#######################################################################
# Stage 1 – export ONNX + compile proto + download classes
#######################################################################
FROM cnstark/pytorch:2.3.0-py3.10.15-ubuntu22.04 AS exporter
WORKDIR /workspace

# Install curl and other tools
RUN apt-get update && apt-get install -y curl

# Install only needed extras
RUN pip install --no-cache-dir grpcio-tools onnx

COPY inference_service /workspace/inference_service

RUN --mount=type=cache,target=/root/.cache \
    python inference_service/model/export_to_onnx.py
    


# Download ImageNet classes
RUN curl -sSL https://raw.githubusercontent.com/pytorch/hub/refs/heads/master/imagenet_classes.txt \
      -o inference_service/model/imagenet_classes.txt

# Generate gRPC stubs
WORKDIR /workspace/inference_service
RUN mkdir -p proto && \
    python -m grpc_tools.protoc -I. \
    --python_out=proto \
    --grpc_python_out=proto image_infer.proto

RUN touch /workspace/inference_service/proto/__init__.py

# test if the model is exported correctly ising ls
RUN ls -l /workspace/inference_service/model/

RUN test -f /workspace/inference_service/model/resnet18.onnx


#######################################################################
# Stage 2 – install runtime deps only
#######################################################################
FROM python:3.11-slim AS builder
WORKDIR /app

COPY inference_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt   # must contain onnxruntime, prometheus_client, etc.

# Bring model + stubs + server
COPY --from=exporter /workspace/inference_service/model ./model
COPY --from=exporter /workspace/inference_service/proto ./proto
COPY --from=exporter /workspace/inference_service/server.py ./server.py


#######################################################################
# Stage 3 – final runtime image
#######################################################################
FROM python:3.11-slim AS runtime
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

EXPOSE 50051 8080 8000
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 CMD \
  curl -sf http://localhost:8080/healthz || exit 1

CMD ["python", "server.py"]

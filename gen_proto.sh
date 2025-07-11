#!/usr/bin/env bash
#
# Regenerate Python gRPC stubs for image_infer.proto.
# Creates `streaming_simulator/proto/` with:
#   ├─ __init__.py
#   ├─ image_infer_pb2.py
#   └─ image_infer_pb2_grpc.py
#
# Usage:
#   chmod +x gen_proto.sh
#   ./gen_proto.sh
#
# NOTE: If you are using a Python virtual environment (venv),
#       make sure to activate it before running this script:
#           source <venv_dir>/bin/activate   # on Linux/Mac
#           <venv_dir>\Scripts\activate     # on Windows
#       Otherwise, grpc_tools may not be found.
#
PROTO_FILE="inference_service/image_infer.proto"
OUT_DIR="streaming_simulator/proto"

mkdir -p "$OUT_DIR"
touch "$OUT_DIR/__init__.py"

python -m grpc_tools.protoc -I inference_service \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_FILE"

echo "[+] Protobuf files generated in $OUT_DIR/"
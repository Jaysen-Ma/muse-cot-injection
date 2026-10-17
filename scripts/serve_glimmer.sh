#!/usr/bin/env bash
# Serve the target reasoning model (Muse Glimmer 30B) with vLLM on 127.0.0.1:${PORT}.
# Flags mirror the configuration the results were produced with.
#   MODEL_HOST_DIR  (required) host directory with the model weights
#   LORA_DIR        (optional) defence adapter dir; served as model name "glimmer-defense"
#   JUDGE_CONTAINER (optional) name of the judge server container that shares the GPU
set -euo pipefail

MODEL_HOST_DIR="${MODEL_HOST_DIR:?set MODEL_HOST_DIR to the Muse Glimmer weights directory}"
LORA_DIR="${LORA_DIR:-}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.28.0}"
NAME="${CONTAINER_NAME:-muse-glimmer-serve}"
PORT="${PORT:-8001}"
JUDGE_CONTAINER="${JUDGE_CONTAINER:-gemma4-comp-serve}"
GPU_UTIL="${GPU_UTIL:-0.75}"

if [[ ! -d "$MODEL_HOST_DIR" ]]; then
  echo "Model directory not found: $MODEL_HOST_DIR" >&2
  exit 1
fi
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "Container $NAME already exists. Stop it first: docker stop $NAME" >&2
  exit 1
fi
if docker ps --format '{{.Names}}' | grep -qx "$JUDGE_CONTAINER"; then
  echo "WARN: $JUDGE_CONTAINER is running and shares GPU memory; lower GPU_UTIL if startup fails." >&2
fi

DOCKER_ARGS=(
  --rm -d
  --entrypoint vllm
  --name "$NAME"
  --gpus all
  --ipc=host
  -p "127.0.0.1:${PORT}:8000"
  -v "${MODEL_HOST_DIR}:/model:ro"
)
VLLM_ARGS=(
  serve /model
  --host 0.0.0.0
  --port 8000
  --served-model-name muse-glimmer
  --tensor-parallel-size 1
  --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len 8192
  --enable-auto-tool-choice
  --tool-call-parser muse_glimmer
  --reasoning-parser muse_glimmer
  --generation-config auto
)
if [[ -n "$LORA_DIR" ]]; then
  DOCKER_ARGS+=(-v "${LORA_DIR}:/adapter:ro")
  VLLM_ARGS+=(--enable-lora --max-lora-rank 16 --lora-modules glimmer-defense=/adapter)
fi

docker run "${DOCKER_ARGS[@]}" "$IMAGE" "${VLLM_ARGS[@]}"
echo "Started $NAME on 127.0.0.1:${PORT}. Follow: docker logs -f $NAME"

# Reasoning-channel injection (scripts/run_attack.py) does not go through the chat API:
# it renders the chat template via /tokenize + /detokenize and appends
#   " to=self<|message|>{COT}<|eom|><|start|>assistant"
# then calls /v1/completions (see cot_injection/atem.py).

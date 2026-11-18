#!/usr/bin/env bash
# Start the selected VLM in Docker/SGLang, caption the source image by default,
# stage from the host conda environment, then remove only this temporary server.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${1:-${ROOT_DIR}/output/cityscapes_gt_smoke_01}"
MODEL_PATH="${MODEL_PATH:-}"
I2T_MODEL_FAMILY="${I2T_MODEL_FAMILY:-qwen3_vl}"
I2T_CAPTION_INPUT_MODE="${I2T_CAPTION_INPUT_MODE:-source_only}"
case "$I2T_CAPTION_INPUT_MODE" in
    source_only|source_and_map) ;;
    *) printf 'Unsupported I2T_CAPTION_INPUT_MODE: %s\n' "$I2T_CAPTION_INPUT_MODE" >&2; exit 2 ;;
esac
case "$I2T_MODEL_FAMILY" in
    qwen3_vl)
        I2T_DEFAULT_DOCKER_IMAGE="lmsysorg/sglang:v0.5.6.post2"
        I2T_DEFAULT_IMAGE_FORMAT="legacy_base64"
        ;;
    gemma4)
        I2T_DEFAULT_DOCKER_IMAGE="lmsysorg/sglang:v0.5.21"
        I2T_DEFAULT_IMAGE_FORMAT="data_url"
        ;;
    *) printf 'Unsupported I2T_MODEL_FAMILY: %s\n' "$I2T_MODEL_FAMILY" >&2; exit 2 ;;
esac
DOCKER_IMAGE="${DOCKER_IMAGE:-$I2T_DEFAULT_DOCKER_IMAGE}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MODEL_PORT="${MODEL_PORT:-30000}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-900}"
CHECK_INTERVAL="${CHECK_INTERVAL:-10}"
I2T_ENV="${I2T_ENV:-}"
CONDA_SH="${CONDA_SH:-}"
QWEN_IMAGE_FORMAT="${QWEN_IMAGE_FORMAT:-$I2T_DEFAULT_IMAGE_FORMAT}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
MAX_RETRIES="${MAX_RETRIES:-3}"
I2T_PROMPT_FILE="${I2T_PROMPT_FILE:-}"
API_EXTRA_BODY_JSON="${API_EXTRA_BODY_JSON:-}"
[ -n "$API_EXTRA_BODY_JSON" ] || API_EXTRA_BODY_JSON='{}'
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"

MODEL_MOUNT_SOURCE="$(dirname "$MODEL_PATH")"
CONTAINER_MODEL_PATH="/models/$(basename "$MODEL_PATH")"
MODEL_CONTAINER_NAME="cityscapes-${I2T_MODEL_FAMILY}-smoke-$(date +%Y%m%d-%H%M%S)-$$"
SERVER_LOG="${SERVER_LOG:-${RUN_DIR}/sglang_server.log}"
CLIENT_LOG="${CLIENT_LOG:-${RUN_DIR}/caption_client.log}"
SERVICE_NAME_FILE="${SERVICE_NAME_FILE:-${RUN_DIR}/service_container_name.txt}"

log() {
    printf '[cityscapes-smoke] %s\n' "$*"
}

fail() {
    printf '[cityscapes-smoke] ERROR: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    local status=$?
    local containers
    trap - EXIT INT TERM
    if ! containers="$(docker ps -a --format '{{.Names}}' 2>&1)"; then
        log "ERROR: cannot inspect Docker containers during SGLang cleanup: $containers"
        exit 1
    fi
    if printf '%s\n' "$containers" | grep -Fxq "$MODEL_CONTAINER_NAME"; then
        docker logs "$MODEL_CONTAINER_NAME" >"$SERVER_LOG" 2>&1 || true
        log "Stopping temporary SGLang container: ${MODEL_CONTAINER_NAME}"
        if ! docker stop "$MODEL_CONTAINER_NAME" >/dev/null 2>&1; then
            log "ERROR: docker stop failed for ${MODEL_CONTAINER_NAME}"
            status=1
        fi
        if ! docker rm "$MODEL_CONTAINER_NAME" >/dev/null 2>&1; then
            log "ERROR: docker rm failed for ${MODEL_CONTAINER_NAME}"
            status=1
        fi
        if ! containers="$(docker ps -a --format '{{.Names}}' 2>&1)"; then
            log "ERROR: cannot verify Docker cleanup: $containers"
            status=1
        elif printf '%s\n' "$containers" | grep -Fxq "$MODEL_CONTAINER_NAME"; then
            log "ERROR: SGLang container still exists: ${MODEL_CONTAINER_NAME}"
            status=1
        fi
    fi
    exit "$status"
}

wait_for_server() {
    local elapsed=0
    while [ "$elapsed" -lt "$SERVER_TIMEOUT" ]; do
        if ! docker ps --format '{{.Names}}' | grep -Fxq "$MODEL_CONTAINER_NAME"; then
            docker logs "$MODEL_CONTAINER_NAME" >"$SERVER_LOG" 2>&1 || true
            fail "SGLang container exited early. See ${SERVER_LOG}"
        fi
        if curl -fsS "http://127.0.0.1:${MODEL_PORT}/health" >/dev/null 2>&1; then
            log "SGLang is ready after ${elapsed}s"
            return
        fi
        sleep "$CHECK_INTERVAL"
        elapsed=$((elapsed + CHECK_INTERVAL))
        log "Waiting for SGLang: ${elapsed}/${SERVER_TIMEOUT}s"
    done
    docker logs "$MODEL_CONTAINER_NAME" >"$SERVER_LOG" 2>&1 || true
    fail "SGLang was not ready within ${SERVER_TIMEOUT}s. See ${SERVER_LOG}"
}

[ "$#" -le 1 ] || fail "Usage: bash scripts/run_cityscapes_gt_caption_sglang.sh [RUN_DIR]"
[ -f "${ROOT_DIR}/scripts/cityscapes_gt_smoke.py" ] || fail "Missing smoke script under ${ROOT_DIR}"
[ -f "${RUN_DIR}/manifest.jsonl" ] || fail "Missing prepared manifest: ${RUN_DIR}/manifest.jsonl"
[ ! -e "${RUN_DIR}/captions.jsonl" ] || fail "Will not overwrite ${RUN_DIR}/captions.jsonl"
[ -d "$MODEL_PATH" ] || fail "Missing model directory: ${MODEL_PATH}"
[ "$I2T_MODEL_FAMILY" != gemma4 ] || [ "$QWEN_IMAGE_FORMAT" = data_url ] || \
    fail "Gemma 4 requires QWEN_IMAGE_FORMAT=data_url"
[ -z "$I2T_ENV" ] || [ -f "$CONDA_SH" ] || fail "Set CONDA_SH when I2T_ENV is specified"
command -v docker >/dev/null 2>&1 || fail "docker is unavailable"
command -v curl >/dev/null 2>&1 || fail "curl is unavailable"

if ss -ltn 2>/dev/null | grep -q ":${MODEL_PORT}[[:space:]]"; then
    fail "Port ${MODEL_PORT} is already occupied; set MODEL_PORT to a free port"
fi

python - "$MODEL_PATH" "$I2T_MODEL_FAMILY" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
family = sys.argv[2]
for name in ("config.json", "configuration.json"):
    path = model_dir / name
    if not path.is_file():
        continue
    config = json.loads(path.read_text(encoding="utf-8"))
    model_type = str(config.get("model_type", "")).lower()
    if family == "gemma4":
        if model_type != "gemma4" or not isinstance(config.get("vision_config"), dict):
            raise SystemExit("Selected checkpoint is not a vision-capable Gemma 4 model")
    has_vision = (
        "vision_config" in config
        or "vision" in config
        or "vl" in model_type
        or "vision" in model_type
    )
    print(f"{name}: model_type={config.get('model_type', '<missing>')}")
    if has_vision:
        raise SystemExit(0)
raise SystemExit("No vision configuration found in the selected checkpoint")
PY

if [ -n "$I2T_ENV" ]; then
    set +u
    source "$CONDA_SH"
    conda activate "$I2T_ENV"
    set -u
    log "Activated conda environment: ${I2T_ENV}"
fi

python - "$ROOT_DIR" "$I2T_CAPTION_INPUT_MODE" "$I2T_PROMPT_FILE" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]) / "scripts"))
from cityscapes_gt_smoke import load_caption_prompt

load_caption_prompt(Path(sys.argv[3]) if sys.argv[3] else None, sys.argv[2])
PY

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

SERVER_CMD=(
    docker run -d
    --name "$MODEL_CONTAINER_NAME"
    --gpus "\"device=${GPU_DEVICES}\""
    --shm-size 64g
    -p "${MODEL_PORT}:${MODEL_PORT}"
    --ipc=host
    -v "${MODEL_MOUNT_SOURCE}:/models"
    "$DOCKER_IMAGE"
    python3 -m sglang.launch_server
    --model-path "$CONTAINER_MODEL_PATH"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --host 0.0.0.0
    --port "$MODEL_PORT"
    --trust-remote-code
)
if [ -n "$SERVER_EXTRA_ARGS" ]; then
    read -r -a extra_server_args <<< "$SERVER_EXTRA_ARGS"
    SERVER_CMD+=("${extra_server_args[@]}")
fi

log "Starting ${I2T_MODEL_FAMILY} ${CONTAINER_MODEL_PATH} with ${DOCKER_IMAGE} on GPU(s) ${GPU_DEVICES}; image format=${QWEN_IMAGE_FORMAT}; caption input=${I2T_CAPTION_INPUT_MODE}"
printf '%s\n' "$MODEL_CONTAINER_NAME" > "$SERVICE_NAME_FILE"
"${SERVER_CMD[@]}" >/dev/null
wait_for_server

caption_prompt_args=()
if [ -n "$I2T_PROMPT_FILE" ]; then
    [ -f "$I2T_PROMPT_FILE" ] || fail "Prompt file not found: $I2T_PROMPT_FILE"
    caption_prompt_args+=(--prompt-file "$I2T_PROMPT_FILE")
fi
python -u "${ROOT_DIR}/scripts/cityscapes_gt_smoke.py" caption \
    --run-dir "$RUN_DIR" \
    --llm-api "http://127.0.0.1:${MODEL_PORT}/v1/chat/completions" \
    --api-model-name "$CONTAINER_MODEL_PATH" \
    --i2t-model-family "$I2T_MODEL_FAMILY" \
    --caption-input-mode "$I2T_CAPTION_INPUT_MODE" \
    --qwen-image-format "$QWEN_IMAGE_FORMAT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --max-retries "$MAX_RETRIES" \
    --api-extra-body-json "$API_EXTRA_BODY_JSON" \
    "${caption_prompt_args[@]}" 2>&1 | tee "$CLIENT_LOG"

log "Caption smoke stage completed: ${RUN_DIR}/captions.jsonl"
log "Client log: ${CLIENT_LOG}"

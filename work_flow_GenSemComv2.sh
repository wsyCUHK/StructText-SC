#!/usr/bin/env bash
# GenSemCom V2: Cityscapes segmentation -> caption -> channel -> reconstruction.
set -euo pipefail
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATASET_ROOT="${DATASET_ROOT:-}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/output/$(date +%Y%m%d_%H%M%S)}"
SPLIT="${SPLIT:-val}"
LIMIT="${LIMIT:-2}"
WIDTH="${WIDTH:-2048}"
HEIGHT="${HEIGHT:-1024}"
MASK_SOURCE="${MASK_SOURCE:-drn}"
I2T_MODEL_FAMILY="${I2T_MODEL_FAMILY:-qwen3_vl}"
I2T_CAPTION_INPUT_MODE="${I2T_CAPTION_INPUT_MODE:-source_only}"
MODEL_PATH="${MODEL_PATH:-}"
I2T_PROMPT_FILE="${I2T_PROMPT_FILE:-}"
IMAGE_BACKEND="${IMAGE_BACKEND:-sdcpp_server}"
IMAGE_MODEL="${IMAGE_MODEL:-}"
CHANNEL_MODE="${CHANNEL_MODE:-packet_awgn}"
CODEBOOK_CORPUS="${CODEBOOK_CORPUS:-}"
GPU_DEVICES="${GPU_DEVICES:-0}"
IMAGE_WORKERS="${IMAGE_WORKERS:-1}"
BLIP_WORKERS="${BLIP_WORKERS:-1}"
BLIP_DEVICE="${BLIP_DEVICE:-cuda}"
BLIP_MAX_NEW_TOKENS="${BLIP_MAX_NEW_TOKENS:-40}"
BLIP_NUM_BEAMS="${BLIP_NUM_BEAMS:-3}"
BLIP_CPU_THREADS="${BLIP_CPU_THREADS:-4}"
DRN_REPO_ROOT="${DRN_REPO_ROOT:-}"
DRN_CHECKPOINT="${DRN_CHECKPOINT:-}"
DRN_DEVICE="${DRN_DEVICE:-auto}"
DRN_MULTI_SCALE="${DRN_MULTI_SCALE:-0}"
SWINJSCC_ROOT="${SWINJSCC_ROOT:-}"
SWINJSCC_CHECKPOINT="${SWINJSCC_CHECKPOINT:-}"
SWINJSCC_GPU="${SWINJSCC_GPU:-${GPU_DEVICES%%,*}}"
JSCC_SNR_DB="${JSCC_SNR_DB:-10}"
JSCC_SEED="${JSCC_SEED:-42}"
JSCC_MAP_VARIANT="${JSCC_MAP_VARIANT:-raw}"
PACKET_LOSS_RATE="${PACKET_LOSS_RATE:-0.1}"
PACKET_K="${PACKET_K:-4}"
PACKET_P="${PACKET_P:-4}"
PACKET_PAYLOAD_SIZE="${PACKET_PAYLOAD_SIZE:-128}"
PACKET_CHANNEL_SEED="${PACKET_CHANNEL_SEED:-42}"
PACKET_INTERLEAVER_SEED="${PACKET_INTERLEAVER_SEED:-2025}"
HUFFMAN_PSEUDOCOUNT="${HUFFMAN_PSEUDOCOUNT:-1}"
TEXT_TOKENIZER_MODE="${TEXT_TOKENIZER_MODE:-legacy}"
MAP_SNR_DB="${MAP_SNR_DB:-10}"
MAP_AWGN_SEED="${MAP_AWGN_SEED:-42}"
SEED="${SEED:-42}"
STEPS="${STEPS:-40}"
FUSION_MASK_PROMPT_POLICY="${FUSION_MASK_PROMPT_POLICY:-conditional}"
IMAGE_PLACEMENT="${IMAGE_PLACEMENT:-cpu_offload}"
SDCPP_SERVER_BIN="${SDCPP_SERVER_BIN:-}"
SDCPP_DIFFUSION_MODEL="${SDCPP_DIFFUSION_MODEL:-}"
SDCPP_VAE="${SDCPP_VAE:-}"
SDCPP_LLM="${SDCPP_LLM:-}"
SDCPP_LLM_VISION="${SDCPP_LLM_VISION:-}"
SDCPP_PRECISION_PROFILE="${SDCPP_PRECISION_PROFILE:-bf16}"
SDCPP_PORT="${SDCPP_PORT:-12345}"
SDCPP_AUTO_FIT="${SDCPP_AUTO_FIT:-on}"
SDCPP_BACKEND_SPEC="${SDCPP_BACKEND_SPEC:-diffusion=cuda0,te=cuda0,vae=cuda0}"
SDCPP_MAX_VRAM="${SDCPP_MAX_VRAM:-}"
SDCPP_CFG_SCALE="${SDCPP_CFG_SCALE:-1.0}"
SDCPP_DIFFUSION_FA="${SDCPP_DIFFUSION_FA:-1}"
SDCPP_VERBOSE="${SDCPP_VERBOSE:-1}"
SDCPP_OFFLOAD_TO_CPU="${SDCPP_OFFLOAD_TO_CPU:-0}"
SDCPP_STARTUP_TIMEOUT="${SDCPP_STARTUP_TIMEOUT:-900}"
SDCPP_CAPABILITIES_TIMEOUT="${SDCPP_CAPABILITIES_TIMEOUT:-30}"
SDCPP_JOB_TIMEOUT="${SDCPP_JOB_TIMEOUT:-1200}"
CONDA_SH="${CONDA_SH:-}"
I2T_ENV="${I2T_ENV:-}"
BLIP_ENV="${BLIP_ENV:-$I2T_ENV}"
DRN_ENV="${DRN_ENV:-$I2T_ENV}"
PACKET_ENV="${PACKET_ENV:-$I2T_ENV}"
SWINJSCC_ENV="${SWINJSCC_ENV:-$I2T_ENV}"
IMAGE_ENV="${IMAGE_ENV:-$I2T_ENV}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

case "$CHANNEL_MODE" in packet_awgn|packet_swinjscc|oracle) ;; *) echo "未知 CHANNEL_MODE" >&2; exit 2;; esac
case "$MASK_SOURCE" in drn|gt) ;; *) echo "未知 MASK_SOURCE" >&2; exit 2;; esac
case "$I2T_MODEL_FAMILY" in qwen3_vl|gemma4|blip) ;; *) echo "未知 I2T_MODEL_FAMILY" >&2; exit 2;; esac
case "$I2T_CAPTION_INPUT_MODE" in source_only|source_and_map) ;; *) echo "未知 I2T_CAPTION_INPUT_MODE" >&2; exit 2;; esac
[[ "$I2T_MODEL_FAMILY" != blip || "$I2T_CAPTION_INPUT_MODE" == source_only ]] || { echo "BLIP supports source_only caption input only" >&2; exit 2; }
case "$IMAGE_BACKEND" in sdcpp_server|diffusers) ;; *) echo "未知 IMAGE_BACKEND" >&2; exit 2;; esac
[[ "$SPLIT" != test || "$MASK_SOURCE" != gt ]] || { echo "test 没有公开 GT 标签" >&2; exit 2; }
[[ -n "$DATASET_ROOT" && -d "$DATASET_ROOT/leftImg8bit/$SPLIT" ]] || { echo "请设置 DATASET_ROOT 为 Cityscapes 数据集根目录" >&2; exit 2; }
[[ -n "$MODEL_PATH" && -d "$MODEL_PATH" ]] || { echo "请设置 MODEL_PATH 为图生文模型目录" >&2; exit 2; }
[[ -n "$CODEBOOK_CORPUS" && -s "$CODEBOOK_CORPUS" && -f "$CODEBOOK_CORPUS.json" ]] || { echo "请设置 CODEBOOK_CORPUS，并提供其 .json 元数据" >&2; exit 2; }
if [[ "$MASK_SOURCE" == drn ]]; then
    [[ -n "$DRN_REPO_ROOT" && -d "$DRN_REPO_ROOT" && -n "$DRN_CHECKPOINT" && -s "$DRN_CHECKPOINT" ]] || { echo "请设置 DRN_REPO_ROOT 和 DRN_CHECKPOINT" >&2; exit 2; }
fi
if [[ "$CHANNEL_MODE" == packet_swinjscc ]]; then
    [[ -n "$SWINJSCC_ROOT" && -d "$SWINJSCC_ROOT" && -n "$SWINJSCC_CHECKPOINT" && -s "$SWINJSCC_CHECKPOINT" ]] || { echo "请设置 SWINJSCC_ROOT 和 SWINJSCC_CHECKPOINT" >&2; exit 2; }
fi
if [[ "$IMAGE_BACKEND" == diffusers ]]; then
    [[ -n "$IMAGE_MODEL" && -d "$IMAGE_MODEL" ]] || { echo "请设置 IMAGE_MODEL 为 Diffusers 模型目录" >&2; exit 2; }
else
    [[ -n "$SDCPP_SERVER_BIN" && -x "$SDCPP_SERVER_BIN" && -s "$SDCPP_DIFFUSION_MODEL" && -s "$SDCPP_VAE" && -s "$SDCPP_LLM" ]] || { echo "请设置 SDCPP_SERVER_BIN、SDCPP_DIFFUSION_MODEL、SDCPP_VAE 和 SDCPP_LLM" >&2; exit 2; }
fi
[[ ! -e "$RUN_DIR" ]] || { echo "拒绝覆盖已有运行目录：$RUN_DIR" >&2; exit 2; }
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR" "$RUN_DIR/logs" "$RUN_DIR/config"
LOG_FILE="$RUN_DIR/logs/workflow.log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'status=$?; echo "工作流结束：status=$status，目录=$RUN_DIR"' EXIT
run_step() { echo "[$(date '+%F %T')] $1"; shift; "$@"; }
activate_env() {
    if [[ -n "$1" ]]; then
        [[ -f "$CONDA_SH" ]] || { echo "设置 CONDA_SH 指向 conda.sh" >&2; return 2; }
        set +u; source "$CONDA_SH"; conda activate "$1"; set -u
    fi
}
export ROOT_DIR MODEL_PATH I2T_MODEL_FAMILY I2T_CAPTION_INPUT_MODE I2T_ENV CONDA_SH GPU_DEVICES TENSOR_PARALLEL_SIZE
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
export SERVER_LOG="$RUN_DIR/logs/sglang_server.log" CLIENT_LOG="$RUN_DIR/logs/caption_client.log"
export SERVICE_NAME_FILE="$RUN_DIR/logs/service_container_name.txt"
if [[ -n "$I2T_PROMPT_FILE" ]]; then
    [[ "$I2T_MODEL_FAMILY" != blip ]] || { echo "BLIP 不使用 prompt 文件" >&2; exit 2; }
    cp "$I2T_PROMPT_FILE" "$RUN_DIR/config/i2t_prompt.json"
    I2T_PROMPT_FILE="$RUN_DIR/config/i2t_prompt.json"
fi
export I2T_PROMPT_FILE
for name in ROOT_DIR DATASET_ROOT RUN_DIR SPLIT LIMIT WIDTH HEIGHT MASK_SOURCE I2T_MODEL_FAMILY I2T_CAPTION_INPUT_MODE MODEL_PATH I2T_PROMPT_FILE IMAGE_BACKEND IMAGE_MODEL IMAGE_PLACEMENT CHANNEL_MODE CODEBOOK_CORPUS GPU_DEVICES TENSOR_PARALLEL_SIZE IMAGE_WORKERS BLIP_WORKERS BLIP_DEVICE BLIP_MAX_NEW_TOKENS BLIP_NUM_BEAMS BLIP_CPU_THREADS DRN_REPO_ROOT DRN_CHECKPOINT DRN_DEVICE DRN_MULTI_SCALE SWINJSCC_ROOT SWINJSCC_CHECKPOINT SWINJSCC_GPU JSCC_SNR_DB JSCC_SEED JSCC_MAP_VARIANT PACKET_LOSS_RATE PACKET_K PACKET_P PACKET_PAYLOAD_SIZE PACKET_CHANNEL_SEED PACKET_INTERLEAVER_SEED HUFFMAN_PSEUDOCOUNT TEXT_TOKENIZER_MODE MAP_SNR_DB MAP_AWGN_SEED SEED STEPS FUSION_MASK_PROMPT_POLICY SDCPP_SERVER_BIN SDCPP_DIFFUSION_MODEL SDCPP_VAE SDCPP_LLM SDCPP_LLM_VISION SDCPP_PRECISION_PROFILE SDCPP_PORT SDCPP_CFG_SCALE SDCPP_AUTO_FIT SDCPP_BACKEND_SPEC SDCPP_MAX_VRAM SDCPP_DIFFUSION_FA SDCPP_VERBOSE SDCPP_OFFLOAD_TO_CPU SDCPP_STARTUP_TIMEOUT SDCPP_CAPABILITIES_TIMEOUT SDCPP_JOB_TIMEOUT; do
    printf '%s=%q\n' "$name" "${!name}" >> "$RUN_DIR/config/run_config.env"
done
sha256sum "$CODEBOOK_CORPUS" > "$RUN_DIR/config/codebook_sha256.txt"
cp "$0" "$RUN_DIR/config/workflow.sh"
DATA_RUN_DIR="$RUN_DIR/data"
TRANSPORT_DIR="$DATA_RUN_DIR/transport_$CHANNEL_MODE"
GENERATED_DIR="$DATA_RUN_DIR/generated_$CHANNEL_MODE"

activate_env "$I2T_ENV"
run_step "准备 Cityscapes 图片和分割图清单" python "$ROOT_DIR/scripts/cityscapes_gt_smoke.py" prepare \
    --dataset-root "$DATASET_ROOT" --run-dir "$DATA_RUN_DIR" --split "$SPLIT" --limit "$LIMIT" \
    --width "$WIDTH" --height "$HEIGHT" --mask-source "$MASK_SOURCE" --require-native-size
if [[ "$MASK_SOURCE" == drn ]]; then
    activate_env "$DRN_ENV"
    drn_args=()
    [[ "$DRN_MULTI_SCALE" != 1 ]] || drn_args+=(--multi-scale)
    run_step "DRN 发送端语义分割" python "$ROOT_DIR/scripts/cityscapes_v2_sender_drn.py" \
        --run-dir "$DATA_RUN_DIR" --repo-root "$DRN_REPO_ROOT" --checkpoint "$DRN_CHECKPOINT" --device "$DRN_DEVICE" "${drn_args[@]}"
fi
if [[ "$I2T_MODEL_FAMILY" == blip ]]; then
    activate_env "$BLIP_ENV"
    run_step "BLIP 图生文" python "$ROOT_DIR/scripts/cityscapes_v2_blip_caption.py" \
        --run-dir "$DATA_RUN_DIR" --model "$MODEL_PATH" --device "$BLIP_DEVICE" --workers "$BLIP_WORKERS" \
        --gpu-devices "$GPU_DEVICES" --max-new-tokens "$BLIP_MAX_NEW_TOKENS" --num-beams "$BLIP_NUM_BEAMS" --cpu-threads "$BLIP_CPU_THREADS"
else
    activate_env "$I2T_ENV"
    run_step "启动 SGLang 图生文服务" bash "$ROOT_DIR/scripts/run_cityscapes_gt_caption_sglang.sh" "$DATA_RUN_DIR"
fi

activate_env "$PACKET_ENV"
if [[ "$CHANNEL_MODE" == oracle ]]; then
    run_step "无噪声对照信道" python "$ROOT_DIR/scripts/cityscapes_v2_transport.py" --run-dir "$DATA_RUN_DIR" \
        --output-dir "$TRANSPORT_DIR" --codebook-corpus "$CODEBOOK_CORPUS" \
        --pseudocount "$HUFFMAN_PSEUDOCOUNT" --require-train-metadata
else
    run_step "准备文本信道" python "$ROOT_DIR/scripts/cityscapes_v2_packet_awgn.py" prepare \
        --run-dir "$DATA_RUN_DIR" --output-dir "$TRANSPORT_DIR" --codebook-corpus "$CODEBOOK_CORPUS"
    run_step "Huffman、RS 和丢包信道" python "$ROOT_DIR/packet_loss_channel_work_flow.py" \
        --input_file "$TRANSPORT_DIR/sender_lines.txt" --output_file "$TRANSPORT_DIR/recovered_lines.txt" \
        --summary_file "$TRANSPORT_DIR/packet_summary.json" --codebook_corpus "$CODEBOOK_CORPUS" \
        --loss_model iid --loss_rate "$PACKET_LOSS_RATE" --channel_seed "$PACKET_CHANNEL_SEED" \
        --interleaver_seed "$PACKET_INTERLEAVER_SEED" --k "$PACKET_K" --p "$PACKET_P" \
        --payload_size "$PACKET_PAYLOAD_SIZE" --huffman_pseudocount "$HUFFMAN_PSEUDOCOUNT" --tokenizer_mode "$TEXT_TOKENIZER_MODE"
    if [[ "$CHANNEL_MODE" == packet_swinjscc ]]; then
        activate_env "$SWINJSCC_ENV"
        run_step "SwinJSCC 分割图信道" env CUDA_VISIBLE_DEVICES="$SWINJSCC_GPU" python "$ROOT_DIR/scripts/cityscapes_v2_swinjscc.py" \
            --checkpoint "$SWINJSCC_CHECKPOINT" --repo-root "$SWINJSCC_ROOT" --run-dir "$DATA_RUN_DIR" \
            --output-dir "$TRANSPORT_DIR" --snr-db "$JSCC_SNR_DB" --seed "$JSCC_SEED" --receiver-map-variant "$JSCC_MAP_VARIANT"
        activate_env "$PACKET_ENV"
        run_step "合并接收端文本与分割图" python "$ROOT_DIR/scripts/cityscapes_v2_packet_awgn.py" finalize \
            --run-dir "$DATA_RUN_DIR" --output-dir "$TRANSPORT_DIR" --map-snr-db "$JSCC_SNR_DB" --map-seed "$JSCC_SEED" --map-stats-jsonl "$TRANSPORT_DIR/map_stats.jsonl"
    else
        run_step "像素 AWGN 分割图信道" python "$ROOT_DIR/scripts/cityscapes_v2_packet_awgn.py" finalize \
            --run-dir "$DATA_RUN_DIR" --output-dir "$TRANSPORT_DIR" --map-snr-db "$MAP_SNR_DB" --map-seed "$MAP_AWGN_SEED"
    fi
fi

activate_env "$IMAGE_ENV"
if [[ "$IMAGE_BACKEND" == diffusers ]]; then
    run_step "Diffusers 生成重建图" python "$ROOT_DIR/scripts/cityscapes_gt_smoke.py" generate \
        --run-dir "$DATA_RUN_DIR" --model "$IMAGE_MODEL" --receiver-jsonl "$TRANSPORT_DIR/receiver.jsonl" \
        --output-dir "$GENERATED_DIR" --seed "$SEED" --steps "$STEPS" --placement "$IMAGE_PLACEMENT" \
        --fusion-mask-prompt-policy "$FUSION_MASK_PROMPT_POLICY" --workers "$IMAGE_WORKERS" --gpu-devices "$GPU_DEVICES"
else
    sdcpp_args=()
    [[ -z "$SDCPP_MAX_VRAM" ]] || sdcpp_args+=(--max-vram "$SDCPP_MAX_VRAM")
    [[ -z "$SDCPP_LLM_VISION" ]] || sdcpp_args+=(--llm-vision "$SDCPP_LLM_VISION")
    [[ "$SDCPP_DIFFUSION_FA" != 1 ]] || sdcpp_args+=(--diffusion-fa)
    [[ "$SDCPP_VERBOSE" != 1 ]] || sdcpp_args+=(--verbose)
    [[ "$SDCPP_OFFLOAD_TO_CPU" != 1 ]] || sdcpp_args+=(--offload-to-cpu)
    sdcpp_command=(python "$ROOT_DIR/scripts/cityscapes_v2_sdcpp_generate.py" \
        --server-bin "$SDCPP_SERVER_BIN" --diffusion-model "$SDCPP_DIFFUSION_MODEL" --vae "$SDCPP_VAE" --llm "$SDCPP_LLM" \
        --precision-profile "$SDCPP_PRECISION_PROFILE" --gpu-device "${GPU_DEVICES%%,*}" --gpu-devices "$GPU_DEVICES" \
        --workers "$IMAGE_WORKERS" --port "$SDCPP_PORT" --auto-fit "$SDCPP_AUTO_FIT" --backend-spec "$SDCPP_BACKEND_SPEC" \
        --seed "$SEED" --steps "$STEPS" --cfg-scale "$SDCPP_CFG_SCALE" --startup-timeout "$SDCPP_STARTUP_TIMEOUT" \
        --capabilities-timeout "$SDCPP_CAPABILITIES_TIMEOUT" --job-timeout "$SDCPP_JOB_TIMEOUT" \
        --fusion-mask-prompt-policy "$FUSION_MASK_PROMPT_POLICY" "${sdcpp_args[@]}")
    run_step "sd-server 生成重建图" "${sdcpp_command[@]}" \
        --run-dir "$DATA_RUN_DIR" --receiver-jsonl "$TRANSPORT_DIR/receiver.jsonl" --output-dir "$GENERATED_DIR"
fi
run_step "整理输入、信道及生成产物" python "$ROOT_DIR/scripts/organize_cityscapes_v2_outputs.py" \
    --run-dir "$RUN_DIR" --data-run-dir "$DATA_RUN_DIR" --transport-dir "$TRANSPORT_DIR" --generated-dir "$GENERATED_DIR"
echo "生成和产物整理完成：$RUN_DIR"

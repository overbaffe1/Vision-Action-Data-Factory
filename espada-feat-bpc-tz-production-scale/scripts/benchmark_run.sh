#!/usr/bin/env bash
# Runs the BPC pipeline on a test image set while sampling GPU utilization
# in the background, then prints a timing/throughput summary.
#
# Usage:
#   scripts/benchmark_run.sh <input_dir> <output_dir> [config_path]
#
# Example:
#   scripts/benchmark_run.sh /home/jovyan/ESpaDa-Data-Pipeline/frames \
#       /home/jovyan/bpc_smoke10k_out configs/bpc.smoke10k.yaml

set -euo pipefail

INPUT_DIR="${1:?Usage: benchmark_run.sh <input_dir> <output_dir> [config_path]}"
OUTPUT_DIR="${2:?Usage: benchmark_run.sh <input_dir> <output_dir> [config_path]}"
CONFIG_PATH="${3:-configs/bpc.smoke10k.yaml}"
GPU_LOG="${OUTPUT_DIR}/logs/gpu_util.csv"
NVSMI_INTERVAL="${NVSMI_INTERVAL:-2}"  # seconds between samples

mkdir -p "${OUTPUT_DIR}/logs"

echo "=== BPC benchmark run ==="
echo "input:  ${INPUT_DIR}"
echo "output: ${OUTPUT_DIR}"
echo "config: ${CONFIG_PATH}"
echo "gpu log: ${GPU_LOG} (every ${NVSMI_INTERVAL}s)"
echo

# --- Background GPU sampler ---
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "timestamp,gpu_index,util_gpu_pct,util_mem_pct,mem_used_mib,mem_total_mib,power_w" > "${GPU_LOG}"
    (
        while true; do
            ts="$(date -u +%Y-%m-%dT%H:%M:%S)"
            nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
                --format=csv,noheader,nounits \
            | while IFS=',' read -r idx ugpu umem mused mtotal pwr; do
                echo "${ts},${idx// /},${ugpu// /},${umem// /},${mused// /},${mtotal// /},${pwr// /}" >> "${GPU_LOG}"
            done
            sleep "${NVSMI_INTERVAL}"
        done
    ) &
    SAMPLER_PID=$!
    echo "GPU sampler started (pid ${SAMPLER_PID})."
else
    echo "WARNING: nvidia-smi not found — skipping GPU utilization sampling."
    SAMPLER_PID=""
fi

cleanup() {
    if [ -n "${SAMPLER_PID}" ] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
        kill "${SAMPLER_PID}" 2>/dev/null || true
        wait "${SAMPLER_PID}" 2>/dev/null || true
        echo "GPU sampler stopped."
    fi
}
trap cleanup EXIT

# --- Run pipeline, timed ---
T_START=$(date +%s)

python espada_bpc.py run \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    --config "${CONFIG_PATH}" \
    --gpus all \
    --log-level INFO

T_END=$(date +%s)
ELAPSED=$((T_END - T_START))

cleanup
trap - EXIT

echo
echo "=== Pipeline finished in ${ELAPSED}s ==="
echo

# --- Summary from state.jsonl / status ---
python espada_bpc.py status --run "${OUTPUT_DIR}"

echo
python scripts/summarize_benchmark.py --run "${OUTPUT_DIR}" --elapsed-sec "${ELAPSED}"

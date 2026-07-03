#!/usr/bin/env bash
set -euo pipefail

# Launch 8 single-card vllm-ascend servers for the target model, one per NPU,
# on ports 30000-30007. Used by 02_prepare_data.sh (answer regeneration).
# Run this in its own terminal (or tmux) and keep it running during step 2 of
# data preparation; stop it before the target-cache step, which needs all NPUs.
#
# SGLang also runs on Ascend and exposes the same OpenAI-compatible /v1 API;
# swap the launch command below if you prefer it (see scripts/data/README.md).

model_path=${model_path:-Qwen/Qwen3-4B}
num_workers=${num_workers:-8}
start_port=${start_port:-30000}
max_model_len=${max_model_len:-8192}
log_dir=${log_dir:-logs/vllm_qwen3_4b}

mkdir -p "${log_dir}"
pids=()

cleanup() {
    echo "Stopping ${#pids[@]} vllm workers..."
    kill "${pids[@]}" 2> /dev/null || true
    wait
}
trap cleanup EXIT INT TERM

for ((worker_id = 0; worker_id < num_workers; worker_id++)); do
    port=$((start_port + worker_id))
    ASCEND_RT_VISIBLE_DEVICES=${worker_id} \
        vllm serve "${model_path}" \
        --port "${port}" \
        --max-model-len "${max_model_len}" \
        --gpu-memory-utilization 0.9 \
        > "${log_dir}/worker_${worker_id}.log" 2>&1 &
    pids+=($!)
    echo "worker ${worker_id}: NPU ${worker_id}, port ${port}, log ${log_dir}/worker_${worker_id}.log"
done

echo "Waiting for workers (watch the logs; first startup downloads the model)..."
wait

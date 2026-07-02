#!/usr/bin/env bash
set -euo pipefail

# Data preparation for DSpark training on 8x Ascend 910B, target Qwen/Qwen3-4B.
# Mirrors scripts/data/prepare_data.sh with two changes for NPU + limited disk:
#   - subsamples the training set (num_samples, default 50k of ~1.4M) so the
#     target cache fits on a normal disk (~1.5 TB instead of ~38 TB),
#   - drives Ascend devices via DEEPSPEC_DEVICE/ASCEND_RT_VISIBLE_DEVICES.
#
# Stage 2 (answer regeneration) needs the target servers from
# 01_launch_target_server.sh running. Stop them before stage 3, which
# occupies all NPUs itself.
#
# `stages` selects which stages run (default all). Common cases:
#   stages=3  train_data_path=/path/to/yours.jsonl   # own JSONL with usable
#             # answers: build the cache directly (cache size scales with the
#             # file; subsample with `head -n N` first)
#   stages=23 train_split_path=/path/to/prompts.jsonl # own prompts: regenerate
#             # answers with the target model (on-policy), then build the cache

stages=${stages:-123}
model_path=${model_path:-Qwen/Qwen3-4B}
config_path=${config_path:-config/dspark/dspark_qwen3_4b.py}
num_samples=${num_samples:-50000}

# Stage-3 knobs. data_max_length overrides the config's data.max_length for
# the cache (shorter sequences = smaller cache + less training memory; the
# training sequence length is fixed by the cache, not by train-time opts).
# cache_local_batch_size: lower it on 32GB cards (target forward memory).
data_max_length=${data_max_length:-}
cache_local_batch_size=${cache_local_batch_size:-16}

train_split_path=${train_split_path:-train_datasets/perfectblend_train.jsonl}
train_data_path=${train_data_path:-train_datasets/qwen3_4b/perfectblend_train_regen.jsonl}
cache_dir=${cache_dir:-${HOME}/.cache/deepspec/qwen3_4b_target_cache}

server_host=${server_host:-127.0.0.1}
num_workers=${num_workers:-8}
start_port=${start_port:-30000}

export DEEPSPEC_DEVICE=npu
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

server_addresses=()
for ((worker_id = 0; worker_id < num_workers; worker_id++)); do
    server_addresses+=("${server_host}:$((start_port + worker_id))")
done

if [[ "${stages}" == *1* ]]; then
echo "Stage 1/3: download and split mlabonne/open-perfectblend"
python scripts/data/download_and_split.py \
    --dataset-name mlabonne/open-perfectblend \
    --test-size 0.05 \
    --train-output-path "${train_split_path}" \
    --test-output-dir eval_datasets \
    --skip-existing
fi

if [[ "${stages}" == *2* ]]; then
echo "Stage 2/3: regenerate answers with ${model_path} (${num_samples} samples)"
echo "  (requires the servers from example/01_launch_target_server.sh)"
python scripts/data/generate_train_data.py \
    --model "${model_path}" \
    --server-address "${server_addresses[@]}" \
    --concurrency 32 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0 \
    --max-tokens 4096 \
    --disable-thinking \
    --num-samples "${num_samples}" \
    --resume \
    --input-file-path "${train_split_path}" \
    --output-file-path "${train_data_path}"
fi

if [[ "${stages}" == *3* ]]; then
echo "Stage 3/3: build target cache under ${cache_dir}"
echo "  Input: ${train_data_path}"
echo "  Stop the vllm servers first - this stage runs the target model on all NPUs."
echo "  Rough disk usage: ~30 KB per token (~1.5 TB at 50k samples)."
echo "  Note: the output dir must be new/empty (the script refuses to overwrite)."
cache_cmd=(
    python scripts/data/prepare_target_cache.py
    --config "${config_path}"
    --train-data-path "${train_data_path}"
    --output-dir "${cache_dir}"
    --local-batch-size "${cache_local_batch_size}"
)
if [[ -n "${data_max_length}" ]]; then
    cache_cmd+=(--opts "data.max_length=${data_max_length}")
fi
"${cache_cmd[@]}"

echo "Done. Target cache: ${cache_dir}"
fi

#!/usr/bin/env bash
# Qwen3-32B DSpark target-cache (hidden states) generation.
#
# !!! PER-CARD MEMORY WARNING !!!
# prepare_target_cache.py loads a FULL copy of the target on EACH visible NPU
# (data-parallel, no tensor parallelism). Qwen3-32B is ~61 GB in bf16, so each
# card must hold the weights PLUS forward activations. This does NOT fit 32 GB
# (910B4) or 64 GB cards. It needs cards large enough for the full model, or a
# device_map/TP change to shard the target across cards (not yet in DeepSpec —
# ask if your 910C cards are 64 GB and this OOMs; the fix is localized to
# prepare_target_cache.py). Node count does not help: this is a per-card limit.
#
# Multi-node only speeds up (more data-parallel shards); memory per card is
# unchanged. Run the SAME command on every node with matching NNODES /
# distinct NODE_RANK.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common_env.sh"

config_path=${config_path:-config/dspark/dspark_qwen3_32b.py}
train_data_path=${train_data_path:-/opt/w00958190/DeepSpec/data/train.jsonl}
cache_dir=${cache_dir:-/opt/w00958190/DeepSpec/data/hidden_qwen3_32b}
data_max_length=${data_max_length:-4096}
# 32B stores 5 layers x 5120 hidden per token -> ~61 KB/token on disk (vs ~49 KB
# for 8B). Keep the target-forward batch small.
cache_local_batch_size=${cache_local_batch_size:-4}

mkdir -p "${cache_dir}"

python scripts/data/prepare_target_cache.py \
    --config "${config_path}" \
    --train-data-path "${train_data_path}" \
    --output-dir "${cache_dir}" \
    --local-batch-size "${cache_local_batch_size}" \
    --opts "data.max_length=${data_max_length}"

echo "[tige] Qwen3-32B target cache done: ${cache_dir}"

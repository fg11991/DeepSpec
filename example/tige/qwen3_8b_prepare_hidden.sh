#!/usr/bin/env bash
# Qwen3-8B DSpark target-cache (hidden states) generation.
# Multi-node capable: each node loads a full Qwen3-8B (bf16 ~16 GB, fits a
# 32 GB card) and processes its 1/(NNODES*8) shard of the data in parallel.
# Run the SAME command on every node with matching NNODES / distinct NODE_RANK.
#
# Single node (8 NPU):
#   bash example/tige/qwen3_8b_prepare_hidden.sh
# 2 nodes:
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0-ip> bash .../qwen3_8b_prepare_hidden.sh   # node 0
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0-ip> bash .../qwen3_8b_prepare_hidden.sh   # node 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common_env.sh"

config_path=${config_path:-config/dspark/dspark_qwen3_8b.py}
train_data_path=${train_data_path:-/opt/w00958190/DeepSpec/data/train.jsonl}
cache_dir=${cache_dir:-/opt/w00958190/DeepSpec/data/hidden_qwen3_8b}
data_max_length=${data_max_length:-4096}
cache_local_batch_size=${cache_local_batch_size:-8}   # 32 GB card: keep small

mkdir -p "${cache_dir}"

python scripts/data/prepare_target_cache.py \
    --config "${config_path}" \
    --train-data-path "${train_data_path}" \
    --output-dir "${cache_dir}" \
    --local-batch-size "${cache_local_batch_size}" \
    --opts "data.max_length=${data_max_length}"

echo "[tige] Qwen3-8B target cache done: ${cache_dir}"

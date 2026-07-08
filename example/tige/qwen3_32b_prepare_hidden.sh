#!/usr/bin/env bash
# Qwen3-32B DSpark target-cache (hidden states) generation.
#
# Qwen3-32B is ~64 GB in bf16 and does not fit one 32/64GB card. With
# tp_size>1, prepare_target_cache.py runs ONE worker per node and shards the
# target across tp_size cards via device_map (accelerate), so it fits. Nodes
# stay data-parallel (each node builds 1/NNODES of the cache).
#
# Set ASCEND_RT_VISIBLE_DEVICES to exactly tp_size cards on each node. Default
# below: single node, all 8 cards hold one sharded target.
#   Multi-node: NNODES=<N> NODE_RANK=<i> MASTER_ADDR=<node0-ip> bash ...
#   Throughput is one model-parallel stream per node (no per-card data
#   parallelism), so this is slower per node than the 8B data-parallel path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common_env.sh"

config_path=${config_path:-config/dspark/dspark_qwen3_32b.py}
train_data_path=${train_data_path:-/opt/w00958190/DeepSpec/data/train.jsonl}
cache_dir=${cache_dir:-/opt/w00958190/DeepSpec/data/hidden_qwen3_32b}
data_max_length=${data_max_length:-4096}
# tp_size: how many cards hold one target replica (>= enough to fit ~64GB bf16;
# 8 x 64GB is plenty, and even 2 x 64GB fits). Keep it = the number of visible
# cards on the node unless you want multiple replicas per node.
tp_size=${tp_size:-8}
cache_local_batch_size=${cache_local_batch_size:-4}

mkdir -p "${cache_dir}"

python scripts/data/prepare_target_cache.py \
    --config "${config_path}" \
    --train-data-path "${train_data_path}" \
    --output-dir "${cache_dir}" \
    --local-batch-size "${cache_local_batch_size}" \
    --tp-size "${tp_size}" \
    --opts "data.max_length=${data_max_length}"

echo "[tige] Qwen3-32B target cache done: ${cache_dir}"

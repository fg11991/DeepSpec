#!/usr/bin/env bash
# Qwen3-8B DSpark draft training.
# The target model is NOT resident during training (only the ~1B draft +
# frozen embed/lm_head), so FSDP full_shard + fsdp_auto_wrap fits comfortably
# on 32 GB cards. Multi-node scales data-parallel workers; run the SAME command
# on every node with matching NNODES / distinct NODE_RANK.
#
# Single node (8 NPU):
#   bash example/tige/qwen3_8b_train.sh
# 8 nodes (64 workers):
#   NNODES=8 NODE_RANK=<i> MASTER_ADDR=<node0-ip> bash .../qwen3_8b_train.sh   # on node i

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common_env.sh"

config_path=${config_path:-config/dspark/dspark_qwen3_8b.py}
cache_dir=${cache_dir:-/opt/w00958190/DeepSpec/data/hidden_qwen3_8b}
exp_name=${exp_name:-dspark_qwen3_8b}

num_anchors=${num_anchors:-256}
local_batch_size=${local_batch_size:-1}
global_batch_size=${global_batch_size:-512}
checkpointing_steps=${checkpointing_steps:-500}

python train.py \
    --config "${config_path}" \
    --opts "data.target_cache_path=${cache_dir}" \
    --opts "train.torch_compile=False" \
    --opts "train.sharding_strategy=full_shard" \
    --opts "train.fsdp_auto_wrap=True" \
    --opts "train.gradient_checkpointing=True" \
    --opts "train.local_batch_size=${local_batch_size}" \
    --opts "train.global_batch_size=${global_batch_size}" \
    --opts "model.num_anchors=${num_anchors}" \
    --opts "logging.checkpointing_steps=${checkpointing_steps}" \
    --opts "exp_name=${exp_name}"

echo "[tige] Qwen3-8B training launched (exp_name=${exp_name})"

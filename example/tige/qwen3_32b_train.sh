#!/usr/bin/env bash
# Qwen3-32B DSpark draft training.
# The 32B target is NOT resident during training. What lives on-card is the
# draft (5 decoder layers at 32B width: hidden 5120 / intermediate 25600,
# ~2.5B trainable) plus the frozen embed/lm_head (~1.6B). With FSDP full_shard
# + fsdp_auto_wrap those shard across ranks, so per-card memory stays modest and
# this trains even on 32 GB cards. Multi-node strongly recommended for the 32B
# draft to raise throughput and shard degree. Run the SAME command on every
# node with matching NNODES / distinct NODE_RANK.
#
# 8 nodes (64 workers):
#   NNODES=8 NODE_RANK=<i> MASTER_ADDR=<node0-ip> bash .../qwen3_32b_train.sh   # on node i

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common_env.sh"

config_path=${config_path:-config/dspark/dspark_qwen3_32b.py}
cache_dir=${cache_dir:-/opt/w00958190/DeepSpec/data/hidden_qwen3_32b}
exp_name=${exp_name:-dspark_qwen3_32b}

# 32B draft is heavier per anchor than 8B; start conservative and raise if the
# card has headroom.
num_anchors=${num_anchors:-128}
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

echo "[tige] Qwen3-32B training launched (exp_name=${exp_name})"

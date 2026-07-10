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

# Launcher: set launcher=torchrun to launch one process per rank via torchrun
# (recommended for multi-node - matches how SpecForge runs on this platform and
# binds each rank's NPU/NIC cleanly). Default is DeepSpec's built-in
# spawn-per-node launcher. train.py auto-detects torchrun via TORCHELASTIC_RUN_ID.
launcher=${launcher:-python}

# Sharding strategy. On >1 node, default to hybrid_shard (HSDP): shard the model
# WITHIN each node (8 NPUs over fast HCCS) and REPLICATE across nodes, so the
# heavy per-micro-batch parameter all-gather never crosses the slow inter-node
# fabric - only the small draft-gradient all-reduce does, once per optimizer
# step. Plain full_shard (ZeRO-3) shards across ALL ranks, so every all-gather
# traverses inter-node links and gets dramatically slower as nodes grow (we
# measured 29 s/it on 2 nodes vs 180 s/it on 4 nodes with full_shard). On a
# single node there is no inter-node fabric, so full_shard is fine and shards
# deepest. Override with sharding_strategy=... (e.g. hybrid_shard_zero2 for
# less intra-node re-gather at higher memory).
#
# hybrid_shard shards WITHIN each node by default (8 NPUs). The 32B draft's
# optimizer state may not fit sharded over just 8 cards -> OOM. If so, widen
# the shard group to span multiple nodes via DEEPSPEC_HSDP_SHARD_SIZE (ranks,
# a whole number of nodes): e.g. 16 = 2 nodes halves per-card sharded state
# while the all-gather still only spans those 2 nodes, not the whole job.
#   DEEPSPEC_HSDP_SHARD_SIZE=16 NNODES=4 NODE_RANK=<i> MASTER_ADDR=<ip> \
#       launcher=torchrun bash .../qwen3_32b_train.sh
if [[ "${NNODES}" -gt 1 ]]; then
    sharding_strategy=${sharding_strategy:-hybrid_shard}
else
    sharding_strategy=${sharding_strategy:-full_shard}
fi
export DEEPSPEC_HSDP_SHARD_SIZE=${DEEPSPEC_HSDP_SHARD_SIZE:-}

train_opts=(
    --config "${config_path}"
    --opts "data.target_cache_path=${cache_dir}"
    --opts "train.torch_compile=False"
    --opts "train.sharding_strategy=${sharding_strategy}"
    --opts "train.fsdp_auto_wrap=True"
    --opts "train.gradient_checkpointing=True"
    --opts "train.local_batch_size=${local_batch_size}"
    --opts "train.global_batch_size=${global_batch_size}"
    --opts "model.num_anchors=${num_anchors}"
    --opts "logging.checkpointing_steps=${checkpointing_steps}"
    --opts "exp_name=${exp_name}"
)

if [[ "${launcher}" == "torchrun" ]]; then
    torchrun \
        --nnodes "${NNODES}" \
        --node_rank "${NODE_RANK}" \
        --master_addr "${MASTER_ADDR}" \
        --master_port "${MASTER_PORT}" \
        --nproc_per_node 8 \
        train.py "${train_opts[@]}"
else
    python train.py "${train_opts[@]}"
fi

echo "[tige] Qwen3-32B training launched (exp_name=${exp_name}, launcher=${launcher})"

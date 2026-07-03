#!/usr/bin/env bash
set -euo pipefail

# DSpark draft-model training on 8x Ascend 910B/910C, target Qwen/Qwen3-8B.
# train.py spawns one worker per visible NPU (RANK/WORLD_SIZE are
# node_rank/node_count, so WORLD_SIZE=1 is a single-node run).
#
# torch_compile is disabled: the config default (True) targets CUDA inductor,
# which is not reliable through torch_npu; eager + SDPA is the validated NPU
# path. Checkpoints land in $DEEPSPEC_CKPT_DIR/deepspec/<exp_name>/step_*
# (default ~/checkpoints); tensorboard in $DEEPSPEC_TB_DIR/... Set those env
# vars to redirect both onto a data disk.

config_path=${config_path:-config/dspark/dspark_qwen3_8b.py}
cache_dir=${cache_dir:-${HOME}/.cache/deepspec/qwen3_8b_target_cache}
exp_name=${exp_name:-dspark_qwen3_8b_npu}

num_anchors=${num_anchors:-256}
local_batch_size=${local_batch_size:-1}
global_batch_size=${global_batch_size:-512}
checkpointing_steps=${checkpointing_steps:-500}

export DEEPSPEC_DEVICE=npu
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
# Reduce allocator fragmentation on NPU.
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}

# Memory controls for 32GB cards:
#   fsdp_auto_wrap=True       - per-layer FSDP units; removes the whole-model
#                               backward gradient buffer (fixes backward OOM).
#   sharding_strategy=full_shard - shard params/grads/optimizer across ranks.
#   gradient_checkpointing=True  - recompute activations in backward.
# On 64GB cards you can drop gradient_checkpointing and raise num_anchors.
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

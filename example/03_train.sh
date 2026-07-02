#!/usr/bin/env bash
set -euo pipefail

# DSpark draft-model training on 8x Ascend 910B, target Qwen/Qwen3-4B.
# train.py spawns one worker per visible NPU (RANK/WORLD_SIZE are
# node_rank/node_count, so WORLD_SIZE=1 is a single-node run).
#
# torch_compile is disabled: the config default (True) targets CUDA inductor,
# which is not reliable through torch_npu; eager + SDPA is the validated
# NPU path from upstream PR #9.
#
# Checkpoints land in $DEEPSPEC_CKPT_DIR/deepspec/<exp_name>/step_* (default
# ~/checkpoints); tensorboard in $DEEPSPEC_TB_DIR/... (default ~/tensorboard).
# Set the env vars to redirect both onto a data disk.

config_path=${config_path:-config/dspark/dspark_qwen3_4b.py}
cache_dir=${cache_dir:-${HOME}/.cache/deepspec/qwen3_4b_target_cache}
exp_name=${exp_name:-dspark_block7_qwen3_4b_npu}

# With 50k samples and global_batch_size 512, one epoch is ~100 steps; the
# config default of 3000 would never checkpoint mid-run, so lower it here.
checkpointing_steps=${checkpointing_steps:-500}
local_batch_size=${local_batch_size:-1}

export DEEPSPEC_DEVICE=npu
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

python train.py \
    --config "${config_path}" \
    --opts "data.target_cache_path=${cache_dir}" \
    --opts "train.torch_compile=False" \
    --opts "train.local_batch_size=${local_batch_size}" \
    --opts "logging.checkpointing_steps=${checkpointing_steps}" \
    --opts "exp_name=${exp_name}"

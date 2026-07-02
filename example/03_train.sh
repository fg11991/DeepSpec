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

config_path=${config_path:-config/dspark/dspark_qwen3_8b.py}
cache_dir=${cache_dir:-/opt/w00958190/DeepSpec/0702_test/hidden}
exp_name=${exp_name:-dspark_block7_qwen3_8b_npu}

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
export DEEPSPEC_CKPT_DIR=/opt/w00958190/DeepSpec/0702_test/output
export DEEPSPEC_TB_DIR=/opt/w00958190/DeepSpec/0702_test/tensorboard
# Reduce allocator fragmentation (log showed 24.0 GiB reserved vs 19.4 GiB
# allocated - a ~4.7 GiB fragmentation gap).
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
global_batch_size=${global_batch_size:-32}

# fsdp_auto_wrap: wrap each draft layer / embed / lm_head as its own FSDP
# unit so backward reduce-scatters gradients layer by layer instead of
# allocating one ~4.4 GiB whole-model gradient buffer (the OOM in
# log/oom_log.log). gradient_checkpointing: recompute layer activations in
# backward instead of keeping them alive.
python train.py \
    --config "${config_path}" \
    --opts "data.target_cache_path=${cache_dir}" \
    --opts "train.torch_compile=False" \
    --opts "train.local_batch_size=${local_batch_size}" \
    --opts "train.global_batch_size=${global_batch_size}" \
    --opts "logging.checkpointing_steps=${checkpointing_steps}" \
    --opts "model.num_anchors=64" \
    --opts "train.sharding_strategy=full_shard" \
    --opts "train.gradient_checkpointing=True" \
    --opts "data.max_length=512" \
    --opts "exp_name=${exp_name}"


    # --opts "train.fsdp_auto_wrap=True" \
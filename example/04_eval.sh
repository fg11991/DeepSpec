#!/usr/bin/env bash
set -euo pipefail

# Evaluate speculative-decoding acceptance of the trained DSpark draft
# against Qwen/Qwen3-4B on the benchmarks in eval_datasets/
# (gsm8k, math500, humaneval, mt-bench, ...).
#
# Key numbers in the output: acceptance rate / average acceptance length
# per benchmark. The released deepseek-ai/dspark_qwen3_4b_block7 checkpoint
# is the reference point for a fully-trained draft.

target_name_or_path=${target_name_or_path:-Qwen/Qwen3-4B}
draft_name_or_path=${draft_name_or_path:-${HOME}/checkpoints/deepspec/dspark_block7_qwen3_4b_npu/step_latest}

export DEEPSPEC_DEVICE=npu
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29501}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

python eval.py \
    --target_name_or_path "${target_name_or_path}" \
    --draft_name_or_path "${draft_name_or_path}"

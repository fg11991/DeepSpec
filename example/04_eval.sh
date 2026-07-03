#!/usr/bin/env bash
set -euo pipefail

# Evaluate speculative-decoding acceptance of the trained DSpark draft against
# the target on the benchmarks in eval_datasets/ (gsm8k, math500, humaneval,
# mt-bench, ...). Output: acceptance rate / average acceptance length per set.
#
# The run prints per-sample progress on rank 0 while generating; the only long
# silent gap is between "loading weights" and the first [eval] line. Smoke-test
# the pipeline with one small dataset and short generations:
#   tasks=gsm8k max_samples=8 max_new_tokens=64 bash example/04_eval.sh

target_name_or_path=${target_name_or_path:-Qwen/Qwen3-8B}
draft_name_or_path=${draft_name_or_path:-${DEEPSPEC_CKPT_DIR:-${HOME}/checkpoints}/deepspec/dspark_qwen3_8b_npu/step_latest}

# Which benchmarks to run (comma-separated; empty = all 9). Smallest is aime25.
tasks=${tasks:-}
# Cap samples per dataset (empty = each task's built-in default).
max_samples=${max_samples:-}
max_new_tokens=${max_new_tokens:-2048}

export DEEPSPEC_DEVICE=npu
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29501}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

eval_cmd=(
    python eval.py
    --target_name_or_path "${target_name_or_path}"
    --draft_name_or_path "${draft_name_or_path}"
    --max-new-tokens "${max_new_tokens}"
)
[[ -n "${tasks}" ]] && eval_cmd+=(--tasks "${tasks}")
[[ -n "${max_samples}" ]] && eval_cmd+=(--max-samples "${max_samples}")

"${eval_cmd[@]}"

#!/usr/bin/env bash
set -euo pipefail

# Evaluate speculative-decoding acceptance of the trained DSpark draft
# against Qwen/Qwen3-4B on the benchmarks in eval_datasets/
# (gsm8k, math500, humaneval, mt-bench, ...).
#
# Key numbers in the output: acceptance rate / average acceptance length
# per benchmark. The released deepseek-ai/dspark_qwen3_4b_block7 checkpoint
# is the reference point for a fully-trained draft.
#
# The run prints per-sample progress on rank 0 while generating. There is a
# long silent gap only between "loading weights" and the first [eval] line
# (dataset load + first-sample kernel build); after that you get one line per
# finished sample. To just smoke-test the pipeline, run one small dataset with
# short generations:
#   tasks=gsm8k max_new_tokens=64 max_samples=8 bash example/04_eval.sh

target_name_or_path=${target_name_or_path:-/opt/foundation_model/Qwen3-8B}
draft_name_or_path=${draft_name_or_path:-/opt/w00958190/DeepSpec/0702_test/output/deepspec/dspark_block7_qwen3_8b_npu_4096_256anchors/step_latest}

# Which benchmarks to run (comma-separated; empty = all 9). Smallest is aime25.
tasks=${tasks:-}
# Cap samples per dataset (empty = each task's built-in default). Small = fast.
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
if [[ -n "${tasks}" ]]; then
    eval_cmd+=(--tasks "${tasks}")
fi
if [[ -n "${max_samples}" ]]; then
    eval_cmd+=(--max-samples "${max_samples}")
fi
"${eval_cmd[@]}"

#!/usr/bin/env bash
# Qwen3-8B DSpark acceptance evaluation. Single node, 8 NPU.
# Eval loads the FULL target + draft on every rank (no tensor parallelism), so
# node count does not reduce per-card memory; the benchmarks are small (<=500
# samples/task) and single-node 8-card is plenty. Qwen3-8B (~16 GB) fits 32 GB.
#
#   bash example/tige/qwen3_8b_eval.sh
# Smoke test one small dataset with short generations:
#   tasks=gsm8k max_samples=8 max_new_tokens=64 bash example/tige/qwen3_8b_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Eval is single-node; pin to one node regardless of any inherited NNODES.
# Set MASTER_PORT before sourcing so its default (avoid clashing with a running
# train on 29510) takes effect.
export NNODES=1 NODE_RANK=0
export MASTER_PORT=${MASTER_PORT:-29511}
source "${SCRIPT_DIR}/_common_env.sh"

target_name_or_path=${target_name_or_path:-Qwen/Qwen3-8B}
draft_name_or_path=${draft_name_or_path:-${DEEPSPEC_CKPT_DIR}/deepspec/dspark_qwen3_8b/step_latest}
tasks=${tasks:-}
max_samples=${max_samples:-}
max_new_tokens=${max_new_tokens:-2048}

eval_cmd=(
    python eval.py
    --target_name_or_path "${target_name_or_path}"
    --draft_name_or_path "${draft_name_or_path}"
    --max-new-tokens "${max_new_tokens}"
)
[[ -n "${tasks}" ]] && eval_cmd+=(--tasks "${tasks}")
[[ -n "${max_samples}" ]] && eval_cmd+=(--max-samples "${max_samples}")

"${eval_cmd[@]}"

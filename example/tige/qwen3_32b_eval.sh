#!/usr/bin/env bash
# Qwen3-32B DSpark acceptance evaluation. Single node, 8 NPU.
#
# !!! PER-CARD MEMORY WARNING !!!
# Like prepare_hidden, eval loads a FULL target (+ draft) on EACH rank, no
# tensor parallelism. Qwen3-32B (~61 GB bf16) plus the draft and generation KV
# cache will not fit 32 GB or 64 GB cards. Eval needs cards large enough for the
# full target, or a device_map/TP change (localized to the DSpark evaluator's
# build_models). Multi-node does not reduce per-card memory, so eval stays
# single-node here.
#
#   bash example/tige/qwen3_32b_eval.sh
# Smoke test:
#   tasks=aime25 max_samples=8 max_new_tokens=64 bash example/tige/qwen3_32b_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NNODES=1 NODE_RANK=0
export MASTER_PORT=${MASTER_PORT:-29512}
source "${SCRIPT_DIR}/_common_env.sh"

target_name_or_path=${target_name_or_path:-Qwen/Qwen3-32B}
draft_name_or_path=${draft_name_or_path:-${DEEPSPEC_CKPT_DIR}/deepspec/dspark_qwen3_32b/step_latest}
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

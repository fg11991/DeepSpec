#!/usr/bin/env bash
set -euo pipefail

# Environment setup for DSpark training on a single node with 8x Ascend 910B.
#
# Prerequisites (install these first, they are machine-specific):
#   1. Ascend driver + firmware (check with: npu-smi info)
#   2. CANN toolkit + kernels (this script sources its env below)
# See https://www.hiascend.com/developer for CANN downloads. torch_npu 2.9.x
# requires a matching CANN 8.x — check the compatibility table in the
# torch_npu release notes: https://github.com/Ascend/pytorch

# Source CANN environment.
ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit}
if [[ -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
else
    echo "WARNING: ${ASCEND_TOOLKIT_HOME}/set_env.sh not found; install CANN first." >&2
fi

# Repo dependencies. requirements.txt pins torch==2.9.1 (CPU/CUDA wheel is
# fine as the base; torch_npu plugs in as a backend).
python -m pip install -r requirements.txt

# torch_npu must match the torch minor version (2.9.x).
python -m pip install 'torch_npu==2.9.1'

# Inference engine for the data-regeneration step (01/02 scripts).
# vllm-ascend is the community plugin for running vLLM on Ascend NPU:
#   https://github.com/vllm-project/vllm-ascend
# Pin compatible vllm/vllm-ascend versions per its README if the latest pair
# conflicts with torch 2.9.1.
python -m pip install vllm vllm-ascend

# Sanity check: NPU visible and usable.
python - << 'EOF'
import torch
import torch_npu  # noqa: F401

assert torch.npu.is_available(), "torch.npu is not available - check CANN/driver/torch_npu install"
print(f"NPU devices: {torch.npu.device_count()}")
print(f"Device 0: {torch.npu.get_device_name(0)}")
EOF

echo "Environment ready."

#!/usr/bin/env bash
set -euo pipefail

# Environment setup for DSpark training on a single node with 8x Ascend 910B/910C.
#
# Recommended: run inside the official vllm-ascend image, which ships CANN,
# torch/torch_npu 2.9.0.post1, and vllm pre-installed and matched:
#
#   quay.io/ascend/vllm-ascend:v0.18.0
#
# (See example/README.md for the docker run command.) Inside that image this
# script only installs the remaining repo dependencies. On a bare host you
# need Ascend driver + firmware (npu-smi info) and CANN 8.5.x first; see
# https://www.hiascend.com/developer and the torch_npu compatibility table
# at https://github.com/Ascend/pytorch

# Source CANN environment when present (bare host; images do this already).
ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit}
if [[ -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
fi

if python -c 'import torch_npu' 2> /dev/null; then
    # torch/torch_npu already provided (e.g. vllm-ascend image). Install repo
    # deps but keep the image's NPU-adapted torch: the repo pin (2.9.1) and
    # the image build (2.9.0.post1) are both torch 2.9.x.
    echo "torch_npu detected; keeping existing torch/torch_npu."
    grep -v '^torch==' requirements.txt | python -m pip install -r /dev/stdin
else
    # Bare host: install repo deps, then the matching torch_npu 2.9.x build.
    python -m pip install -r requirements.txt
    python -m pip install 'torch_npu>=2.9.0,<2.10'
    # Inference engine for the data-regeneration step (01/02 scripts):
    #   https://github.com/vllm-project/vllm-ascend
    # Pin a vllm/vllm-ascend pair compatible with torch 2.9.x per its docs.
    python -m pip install vllm vllm-ascend
fi

# Sanity check: NPU visible and usable.
python - << 'EOF'
import torch
import torch_npu  # noqa: F401

assert torch.npu.is_available(), "torch.npu is not available - check CANN/driver/torch_npu install"
print(f"NPU devices: {torch.npu.device_count()}")
print(f"Device 0: {torch.npu.get_device_name(0)}")
EOF

echo "Environment ready."

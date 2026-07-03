#!/usr/bin/env bash
# Shared platform environment for DSpark on the Ascend training platform.
# Sourced by every tige/*.sh script. Edit paths here once.
#
# DeepSpec's launcher is NOT torchrun. train.py / eval.py /
# prepare_target_cache.py each run ONCE PER NODE and internally spawn one
# worker per visible NPU. The multi-node contract is:
#   RANK        = this node's index          (0 .. NNODES-1)
#   WORLD_SIZE  = number of NODES             (NOT total ranks)
#   MASTER_ADDR = rank-0 node IP, same on all nodes
#   MASTER_PORT = same on all nodes
# Global rank = RANK * npus_per_node + local_rank (see utils/distributed.py).
# So on an 8-node x 8-NPU job you launch the same command on all 8 nodes with
# RANK=0..7 and WORLD_SIZE=8; total data-parallel workers = 64.

set -euo pipefail

# ---- Repo root (edit to your platform checkout) ----
export REPO_ROOT=${REPO_ROOT:-/opt/w00958190/DeepSpec/DeepSpec}
cd "${REPO_ROOT}"

# ---- Ascend / HCCL runtime (values mirrored from the platform dflash job) ----
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-2400}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1200}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export DEEPSPEC_DEVICE=npu

# ---- Devices visible on each node ----
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# ---- Multi-node wiring. Set NNODES/NODE_RANK/MASTER_ADDR per node on launch.
#      Single node: leave defaults (NNODES=1, NODE_RANK=0). ----
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29510}

# Map platform-style names onto what DeepSpec's init_dist reads.
export RANK=${NODE_RANK}
export WORLD_SIZE=${NNODES}

# ---- Output locations (land on a shared data disk, not the container) ----
export DEEPSPEC_CKPT_DIR=${DEEPSPEC_CKPT_DIR:-/opt/w00958190/DeepSpec/output/checkpoints}
export DEEPSPEC_TB_DIR=${DEEPSPEC_TB_DIR:-/opt/w00958190/DeepSpec/output/tensorboard}

echo "[tige] REPO_ROOT=${REPO_ROOT} NNODES=${NNODES} NODE_RANK=${NODE_RANK} MASTER=${MASTER_ADDR}:${MASTER_PORT} DEV=${ASCEND_RT_VISIBLE_DEVICES}"

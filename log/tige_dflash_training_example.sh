#!/bin/bash
export HCCL_CONNECT_TIMEOUT=2400
export IS_ON_TRAIN_PLATFORM=True
export HCCL_EXEC_TIMEOUT=1200
export CUSTOM=True
export PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/bin:/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/tools/ccec_compiler/bin:/usr/local/Ascend/cann-8.5.0/tools/show_kernel_debug_data:/usr/local/Ascend/cann-8.5.0/tools/msobjdump:/usr/local/Ascend/cann-8.5.0/bin:/usr/local/Ascend/cann-8.5.0/tools/ccec_compiler/bin:/usr/local/Ascend/cann-8.5.0/tools/profiler/bin:/usr/local/Ascend/cann-8.5.0//tools/ascend_system_advisor/asys:/usr/local/python3.11.14/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/naie/.local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/examples:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/tests/atbopstest:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64/plugin:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/Ascend/cann-8.5.0/tools/aml/lib64:/usr/local/Ascend/cann-8.5.0/tools/aml/lib64/plugin:/usr/local/Ascend/cann-8.5.0/lib64:/usr/local/Ascend/cann-8.5.0/lib64/plugin/opskernel:/usr/local/Ascend/cann-8.5.0/lib64/plugin/nnengine:/usr/local/Ascend/cann-8.5.0/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:/usr/local/python3.11.14/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/cann-8.5.0/python/site-packages:/usr/local/Ascend/cann-8.5.0/opp/built-in/op_impl/ai_core/tbe:/usr/local/python3.11.14/lib:$PYTHONPATH
export PYTHON3_HOME=/usr/local/python3.11.14

ROOT_DIR=/sglang-workspace/SpecForge/
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/specforge_cache/compiled_kernels

TP_SIZE=1
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-96}
export OMP_NUM_THREADS=100
export PYTHONPATH=/sglang-workspace/SpecForge/:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_CONNECT_TIMEOUT=2400

export NNODES=${NNODES:-8}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-29510}
export NPROC_PER_NODE=${NPUS_PER_NODE:-8}

echo "===================================MASTER_ADDR:===================================="
echo $MASTER_ADDR
echo $NNODES
echo $MASTER_PORT

export bs=1
export lr=5e-5
export num_epochs=20
export wramup_ratio=0.08
export max_length=8192
export TRAIN_NAME=qwen3
export TRAIN_TAG="bs${bs}-lr${lr}-max_len${max_length}-total_epoch${num_epochs}-${TRAIN_NAME:-temp}"
export ulysses_size=2

# git -C /sglang-workspace/SpecForge pull https://y00913247:f66GpsbXn2sTGL8YownysSdz@codehub-dg-g.huawei.com/y00913247/SpecForge-Ascend.git
# git -C /sglang-workspace/SpecForge pull https://z00830407:kJ7DAeFLftJ-Z4aqs1SHK5zg@codehub-dg-g.huawei.com/AICC/AICC_Efficient_Tools/SpecForge-Ascend.git

REPO_DIR=/sglang-workspace/SpecForge
REPO_URL="https://z00927893:spMRLQdJrhvHtmnaw5VZgSor@codehub-dg-g.huawei.com/w00958190/SpecForge-Ascend.git"
BRANCH=dflash-dev2

rm -f "$REPO_DIR/pyrightconfig.json"

git -C "$REPO_DIR" fetch "$REPO_URL" "$BRANCH"
git -C "$REPO_DIR" checkout -B "$BRANCH" FETCH_HEAD

echo "================ Git Info: SpecForge ================"
git -C /sglang-workspace/SpecForge branch --show-current
git -C /sglang-workspace/SpecForge log -1 --oneline
git -C /sglang-workspace/SpecForge log -1 --format="commit=%H%nbranch=%D%nauthor=%an%ndate=%ad%nsubject=%s"
git -C /sglang-workspace/SpecForge status --short
echo "====================================================="

CACHE_DIR=/dpc/hot/z00927893/4_qwen3_6/specforge_train/cache
TRAIN_DIR=/dpc/hot/z00927893/4_qwen3_6/specforge_train/train

sudo mkdir -p "$CACHE_DIR"
sudo mkdir -p "$TRAIN_DIR"

sudo chmod 755 /dpc/hot/z00927893
sudo chmod 755 /dpc/hot/z00927893/3_qwen3_5_test_project
sudo chmod 644 /dpc/hot/z00927893/3_qwen3_5_test_project/qwen3.5-35b-a3b-eagle3.json
sudo chmod -R a+rwX "$CACHE_DIR"
sudo chmod -R a+rwX "$TRAIN_DIR"

torchrun \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --nproc_per_node $NPROC_PER_NODE \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path /dpc/hot/model/y00830025/Qwen3-32B \
    --draft-config-path /dpc/hot/z00927893/5_dflash/qwen3_test/configs/qwen3-32b-dflash-correct.json \
    --train-hidden-states-path /dpc/hot/z00927893/5_dflash/qwen3_test/hidden_energy \
    --build-dataset-num-proc 64 \
    --output-dir /dpc/hot/z00927893/5_dflash/qwen3_test/train_energy_20epoch \
    --num-epochs $num_epochs \
    --batch-size $bs \
    --target-model-backend sglang \
    --learning-rate $lr \
    --max-length $max_length \
    --chat-template qwen3_no_system_prompt \
    --tp-size $TP_SIZE \
    --attention-backend usp \
    --sp-ulysses-size 2 \
    --log-interval 1 \
    --embedding-key "model.embed_tokens.weight" \
    --cache-dir /dpc/hot/z00927893/5_dflash/qwen3_test/cache \
    --sglang-mem-fraction-static 0.001 \
    --report-to tensorboard \
    --warmup-ratio $wramup_ratio \
    --ckpt-dir /dpc/hot/w00958190/qwen3-32b-dflash-en-zh \
  # --profile \
  # --profile-start-step 2 \
  # --profile-active-steps 30 \
  # --profile-output-dir /dpc/hot/z00927893/5_dflash/qwen3_test/qwen3_ultrachat_small_profiling


# torchrun \
#     --nnodes $NNODES \
#     --node_rank $NODE_RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT \
#     --nproc_per_node $NPROC_PER_NODE \
#     $ROOT_DIR/scripts/train_dflash.py \
#     --target-model-path /dpc/hot/model/y00830025/Qwen3-32B \
#     --draft-config-path /dpc/hot/z00927893/5_dflash/qwen3_test/configs/qwen3-32b-dflash-correct.json \
#     --train-hidden-states-path /dpc/hot/z00927893/5_dflash/qwen3_test/300k_hidden \
#     --build-dataset-num-proc 64 \
#     --output-dir /dpc/hot/z00927893/5_dflash/qwen3_test/train_200k_epoch10 \
#     --num-epochs $num_epochs \
#     --batch-size $bs \
#     --target-model-backend sglang \
#     --learning-rate $lr \
#     --max-length $max_length \
#     --chat-template qwen3_no_system_promp \
#     --tp-size $TP_SIZE \
#     --attention-backend fa \
#     --log-interval 1 \
#     --embedding-key "model.embed_tokens.weight" \
#     --cache-dir /dpc/hot/z00927893/5_dflash/qwen3_test/caches/0527_qwen3_32b_newconfig \
#     --sglang-mem-fraction-static 0.001 \
#     --report-to tensorboard \
    
    
    
    


# DeepSpec NPU 适配工作记录

fork：`fg11991/DeepSpec`，分支 `npu-support`（基于 `deepseek-ai/DeepSpec` main）。
目标：在昇腾 NPU（910B/910C，含 32GB 的 910B4）上跑通 DSpark draft 模型训练全链路。
记录截至 2026-07-02。

---

## 一、合入的上游 PR

### PR #9「Add Ascend NPU Support While Preserving GPU Compatibility」（commit f80e14a）

没有直接用 PR 分支（它落后 main 几个提交），而是从最新 main 切出 `npu-support`
后将 PR squash 合入，因此同时包含 main 的修复和 NPU 适配。合并时修复了一处
requirements.txt 的 prettytable 重复（PR 加 3.18.0、main 后来加 3.17.0，保留 main 的）。

PR 核心内容与动机：

| 改动 | 为什么 | 作用 |
| --- | --- | --- |
| 新增 `deepspec/utils/device.py` 设备抽象层 | 全仓库硬编码 torch.cuda | 运行时探测 NPU/CUDA，`DEEPSPEC_DEVICE=npu\|cuda` 可强制；HCCL/NCCL 自动选择 |
| attention 稠密 mask 回退（`modeling/dspark/common.py`） | **torch_npu 不支持 FlexAttention BlockMask**（NPU 适配的核心障碍） | NPU 上用 SDPA + 稠密布尔 mask 等价实现 DSpark 的块状 mask；GPU 仍走 FlexAttention。是正确性基线，未做性能优化（研究切入点） |
| `TRAIN_ATTN_IMPLEMENTATION` 按设备选 sdpa/flex_attention | 同上 | qwen3/gemma4 config 自动切换 |
| trainer/分布式/预取/checkpoint 去 CUDA 硬编码 | HCCL 后端、torch.npu stream、RNG 状态等接口差异 | FSDP device mesh、`CUDAPrefetcher`、ckpt RNG（`torch_accelerator_rng`）双后端可用 |
| Eagle3 trainer 懒加载 | Eagle3 依赖 flex_attention，NPU 上 import 即挂 | 不训 Eagle3 时不再受影响 |
| train.sh 支持 `ASCEND_RT_VISIBLE_DEVICES` | CUDA_VISIBLE_DEVICES 无效 | 脚本层设备可见性 |

PR 作者验证环境：910B2（64GB）/910C，Qwen3-8B target，10k 样本 217GB cache，
**num_anchors=256** 下 3 epoch 无 OOM。注意他也是降了 anchors 才跑过的。

---

## 二、本 fork 自己做的适配

### 1. example/ 端到端脚本（34c2558 起，多次迭代）

为什么：上游只有面向 CUDA 的零散脚本和 README，没有 NPU 单机可复现流程。

- `00_setup_env.sh`：容器内补依赖（检测到 torch_npu 就跳过 torch 安装，避免
  requirements.txt 的 `torch==2.9.1` 覆盖镜像里的 NPU 适配版；torch_npu 2.9.x
  的适配版本是 2.9.0.post 系，不存在 2.9.1）。
- `01_launch_target_server.sh`：8 个单卡 vllm-ascend 服务（上游示例用 SGLang，
  任何 OpenAI 兼容引擎均可；换 vllm 是为了一个镜像全覆盖）。
- `02_prepare_data.sh`：三阶段数据准备。`stages` 选择跑哪几步（自有 JSONL 只跑
  stage 3）；`num_samples` 子采样（全量 cache ~38TB）；`data_max_length` 传给
  cache 构建（**序列长度烙在 cache 里，训练时改 data.max_length 无效**）；
  `cache_local_batch_size`（32GB 卡建 8B cache 时降到 4）。
- `03_train.sh` / `04_eval.sh`：训练与接受率评估。

推荐镜像：`quay.io/ascend/vllm-ascend:v0.18.0`（CANN 8.5.1 + torch/torch_npu
2.9.0.post1 + vllm，910B/910C 通用；也是 V4-Flash 官方部署教程对应版本）。
v0.19.1rc1 等价可用；v0.20.2rc1 是 torch 2.10 + CANN 9.0，需宿主机驱动够新。

### 2. 输出目录环境变量（4fe1e31）

为什么：checkpoint/tensorboard 写死 `~/checkpoints`、`~/tensorboard`，容器里
`~`=/root 不挂载即丢；且 `--opts "logging.checkpoint_dir=..."` 无效（finalize_cfg
在 opts 之后执行并重算目录）。

作用：全部 12 个 config 支持 `DEEPSPEC_CKPT_DIR` / `DEEPSPEC_TB_DIR`，从源头改路径。

### 3. 梯度检查点接线（a28b84d）

为什么：32GB 卡（910B4）上 attention 处 OOM。draft 层本就继承
`GradientCheckpointingLayer`，但上游没有任何地方 enable。

作用：`--opts "train.gradient_checkpointing=True"` 开启（非重入式，兼容 FSDP）。
反向重算激活，省掉逐层 attention 反向保存，代价 ~20-30% 步时。**显存够时可关。**

### 4. FSDP 逐层包装（7b6659d，backward OOM 的根治）

为什么：backward 恒定报 `Tried to allocate 4.42 GiB`——上游 FSDP 无
auto_wrap_policy，整个模型是一个 flat 单元，反向须一次性分配覆盖全部 2.37B 参数
（含 1.25B **冻结** embed/lm_head）的梯度缓冲 = 4.42 GiB bf16。该值只由参数结构
决定，num_anchors/max_length/full_shard 都动不了它。

作用：`--opts "train.fsdp_auto_wrap=True"` 把每个 draft 层、embedding、lm_head
各包成独立 FSDP 单元。梯度逐层 reduce-scatter（峰值 ~0.4GB）；全冻结单元不再
分配梯度；参数逐层 gather/释放。**32GB 卡能不能跑就靠它，开销 <10%，建议常开。**

速度调优组合：`shard_grad_op` + `fsdp_auto_wrap=True` + 关 gradient_checkpointing
——shard_grad_op 参数从前向保到反向（不重复 gather，通信量同单 flat 单元），是
速度/显存的甜点位。

### 5. 续训布局指纹校验（df83692）

为什么：切换 fsdp_auto_wrap 后同 exp_name 重跑，自动续训把 A 布局存的优化器
分片灌进 B 布局，爆出难懂的 `aclnnInplaceCopy ... cannot broadcast`。

作用：checkpoint 记录 `sharding_strategy`+`fsdp_auto_wrap` 指纹，续训不匹配时
直接给出可操作的报错。规则：**training_state.rank\*.pt 绑死
world_size+sharding_strategy+fsdp_auto_wrap 三元组**；step_N 里的模型 safetensors
布局无关，可随意加载。

### 6. 文档

- `docs/CODE_GUIDE_zh.md`：代码阅读指南（模块一 prepare hidden / 模块二 training，
  含仓库结构、关键行号、建议阅读顺序）。
- `example/README.md`：镜像、docker run（含输出目录挂载）、磁盘预算、OOM 处置
  阶梯、数据格式要点。

---

## 三、显存问题速查（8×32GB 910B4 实测结论）

| 症状 | 原因 | 解法 |
| --- | --- | --- |
| backward 恒定 OOM ~4.42GiB | 单 flat 单元整体梯度缓冲（参数结构决定，调激活参数无效） | `train.fsdp_auto_wrap=True` |
| attention 处 OOM | NPU SDPA 稠密 mask 路径逐层保存反向激活 | `train.gradient_checkpointing=True`；根源上重建短序列 cache |
| 改 `data.max_length` 无效 | 序列长度在 cache 构建时固定 | 重建 cache 时传 `data_max_length` |
| 全词表激活大 | `[1, anchors, 7, 151936]` 系列张量（与 target 大小无关） | `model.num_anchors=128/256`（PR 作者 64GB 卡用 256） |
| reserved >> allocated | 分配器碎片 | `export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` |
| 续训报 aclnnInplaceCopy broadcast | 分片布局变了还在自动续训 | 换 exp_name 或删 step_latest（已加校验拦截） |

其他事实：优化器/梯度切分 `shard_grad_op` 与 `full_shard` 在本仓库（单 flat 单元
时）等价；训练时 target 模型不在卡上（CPU 拷 embed/lm_head 即删）；cache 生成
阶段每卡加载完整 target，64GB 卡的 target 天花板 ~14B bf16（32B 需改 device_map
跨卡切分）。

---

## 四、推荐使用顺序

```bash
# 0. 容器（输出目录挂载或用 DEEPSPEC_CKPT_DIR 重定向，cache 放数据盘）
docker run ... quay.io/ascend/vllm-ascend:v0.18.0 bash
cd /workspace/DeepSpec && bash example/00_setup_env.sh
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# 1.（可选，仅 stage 2 需要）起 target 推理服务
bash example/01_launch_target_server.sh          # 终端 A

# 2. 数据 → cache
#    自有 JSONL（含可用答案）：只跑 stage 3
stages=3 train_data_path=/path/yours.jsonl cache_dir=/data/cache_512 \
    data_max_length=512 cache_local_batch_size=4 bash example/02_prepare_data.sh
#    自有 prompt（答案交给 target 重写，正式训练推荐——on-policy）：stages=23
#    全流程（下载 perfectblend 起）：stages=123

# 3. 训练（32GB 卡推荐组合；64GB 卡可去掉 gradient_checkpointing、
#    anchors 提到 256）
DEEPSPEC_CKPT_DIR=/data/ckpt bash example/03_train.sh
#    脚本内关键 opts：
#      train.fsdp_auto_wrap=True        # 常开
#      train.sharding_strategy=shard_grad_op   # 或 full_shard（本仓库等价）
#      train.gradient_checkpointing=True # 显存紧才开（约 -25% 速度）
#      model.num_anchors=64~256          # 按卡显存
#    冒烟：--opts "train.max_train_steps=50"，盯 loss 下降 + accept_rate 上升

# 4. 评估接受率（对照官方 deepseek-ai/dspark_qwen3_8b_block7）
draft_name_or_path=$DEEPSPEC_CKPT_DIR/deepspec/<exp_name>/step_latest \
    bash example/04_eval.sh
```

注意事项：换 anchors/布局等做对照实验时**换 exp_name**（避免自动续训）；
cache 输出目录必须为空目录；正式训练前确认 `manifest.json` 里的
max_length/target_layer_ids/num_samples 符合预期。

---

## 五、提交组织（npu-support-clean 分支）

以 main 为基，按适配主题整理成独立提交：

1. Ascend NPU 支持：设备抽象、HCCL、入口适配（合自上游 PR #9）
2. Ascend NPU 支持：SDPA 稠密 mask attention 回退
3. 训练：NPU device mesh、显存控制（梯度检查点 / 逐层 FSDP）、续训布局校验、可配置输出目录
4. Eval：逐样本进度输出 + 数据集/样本选择
5. Qwen3-32B DSpark config
6. example/ NPU 训练流程脚本
7. example/tige/ 多节点平台脚本
8. 文档（代码阅读指南 + 本工作记录）

后续方向：NPU 原生块状稀疏 attention（FlexAttention 替代，训练侧性能）、
vllm-ascend DSpark 推理适配（RFC vllm-project/vllm-ascend#11163，尚无代码）、
更大 target 的 cache 生成跨卡切分（device_map）。

# DeepSpec 代码阅读指南（DSpark 训练链路）

面向想读懂 DSpark 训练代码的人，按数据流分两个模块讲：**模块一：prepare hidden（文本 → target cache）**，**模块二：training（cache → draft 模型）**。每一节给出建议的阅读顺序和关键行号。

全链路一张图：

```
JSONL 对话文本
   │  scripts/data/prepare_target_cache.py（跑 target 模型前向，hook 抓 hidden）
   ▼
target cache（二进制分片：input_ids + loss_mask + 多层 hidden states）
   │  train.py → deepspec/trainer/*（draft 模型只读 cache，不再跑 target）
   ▼
draft checkpoint（~/checkpoints/deepspec/<exp_name>/step_*）
   │  eval.py → deepspec/eval/dspark/*
   ▼
接受率 / 平均接受长度指标
```

理解整个设计只需要记住一件事：**训练时 target 模型完全不在场**。draft 模型需要的所有 target 信息（中间层 hidden、最后一层 hidden）都在数据准备阶段一次性算好落盘，训练阶段只做 draft 小模型的前向反向——这就是 cache 为什么大、训练为什么快。

## 仓库结构

```
DeepSpec/
├── train.py                     # 训练入口：读配置，按可见加速卡数 spawn worker
├── eval.py                      # 评估入口：接受率测量（不是 serving）
├── config/
│   ├── dspark/                  # DSpark 各 target 的训练配置（qwen3_4b/8b/14b, gemma4_12b）
│   ├── dflash/                  # DFlash 配置
│   └── eagle3/                  # Eagle3 配置
├── scripts/
│   ├── data/
│   │   ├── download_and_split.py      # HF 数据集 → JSONL（ShareGPT 格式归一化）
│   │   ├── generate_train_data.py     # 用 target 模型重写 assistant 回答（on-policy）
│   │   ├── launch_sglang_server.sh    # 起 target 推理服务（供上一步调用）
│   │   ├── prepare_target_cache.py    # ★ 模块一主脚本：跑 target 前向，写 cache
│   │   └── prepare_data.sh            # 以上三步的串联 wrapper
│   ├── train/train.sh
│   └── eval/eval.sh
├── deepspec/
│   ├── data/
│   │   ├── jsonl_dataset.py     # mmap 读 JSONL（行偏移索引，带磁盘缓存）
│   │   ├── parser.py            # ★ 对话 → token + loss_mask（chat 模板、assistant 正则）
│   │   ├── target_cache_dataset.py    # ★ cache 的写入器/读取器/collator（全链路最重的文件）
│   │   └── cuda_prefetcher.py   # 侧 stream H2D 预取（NPU 下自动用 torch.npu stream）
│   ├── modeling/
│   │   ├── dspark/
│   │   │   ├── common.py        # ★ anchor 采样、DSpark attention mask、noise embed
│   │   │   ├── loss.py          # ★ CE + L1 + confidence 三项损失
│   │   │   ├── markov_head.py   # markov/gated/rnn 头：给 draft logits 加 prev-token 偏置
│   │   │   ├── qwen3/           # Qwen3 系 draft 模型（config.py + modeling.py）
│   │   │   └── gemma4/          # Gemma4 系 draft 模型
│   │   └── eagle3/              # Eagle3 实现（独立，依赖 flex_attention，NPU 下懒加载）
│   ├── trainer/
│   │   ├── base_trainer.py      # ★ 训练生命周期：FSDP、梯度累积、checkpoint、schedule
│   │   ├── dspark_trainer.py    # ★ DSpark 的 run_batch（很短，先读它）
│   │   ├── eagle3_trainer.py
│   │   └── ckpt_manager.py      # checkpoint 保存/恢复（含 RNG 状态、step_latest 软链）
│   ├── eval/                    # 评估器：真实自回归验证，测接受长度
│   └── utils/
│       ├── device.py            # NPU/CUDA 抽象层（HCCL vs NCCL、DEEPSPEC_DEVICE 开关）
│       ├── distributed.py       # init_dist、分布式 sampler
│       ├── optim.py             # BF16Optimizer（bf16 参数 + fp32 主副本）
│       ├── config.py            # python 文件当配置 + --opts 点路径覆盖
│       └── metrics.py           # 训练指标聚合（分布式 all-reduce）
├── example/                     # 8x910B/910C 上的端到端脚本（本 fork 添加）
└── eval_datasets/               # gsm8k/math500/humaneval 等评估集
```

标 ★ 的是核心文件，两个模块加起来真正需要精读的只有 6 个文件。

---

## 模块一：prepare hidden（文本 → target cache）

**主脚本：`scripts/data/prepare_target_cache.py`（~400 行）。建议从 `main()`（约 211 行）开始顺着读，它把模块一的所有部件串在一起。**

### 1.1 输入：JSONL 怎么变成 token 和 loss_mask

数据流：`JsonLineDataset` → `ConversationCollator` → `GeneralParser.parse`。

- `deepspec/data/jsonl_dataset.py` — `JsonLineDataset` 用 mmap 按行随机访问 JSONL，第一次打开时扫一遍文件建行偏移索引并缓存到 `~/.cache/deepspec/jsonlindex-*.pkl`（所以第二次启动快很多）。每条记录就是一个 `{"id", "conversations"}` 的 dict。
- `deepspec/data/parser.py` — 精读 `GeneralParser.parse`（68 行）：
  1. 用注册的 chat 模板（`TEMPLATE_REGISTRY`，"qwen"/"gemma4" 两个，32–51 行）把 conversations 渲染成完整对话文本；
  2. tokenize（截断到 `max_length=4096`）；
  3. **loss_mask 的来历**：用正则在渲染后的文本里匹配每个 assistant 轮（`<|im_start|>assistant\n` 到 `<|im_end|>`），再通过「前缀重新 tokenize」把字符位置换算成 token 位置（123–138 行），assistant 内容对应的 token 标 1，其余标 0。DSpark 只在 loss_mask=1 的位置上学习。
- `target_cache_dataset.py` 里的 `ConversationCollator`（824 行）把上面的输出攒成 batch，并丢掉 assistant token 少于 `min_loss_tokens=14` 的样本。

### 1.2 核心：hidden 是怎么抓的

精读 `run_target_forward_with_hooks`（`prepare_target_cache.py:85`）：

- 在 target 模型 `backbone.layers[i]`（i ∈ 配置的 `target_layer_ids`，qwen3-4b 是 `[1,9,17,25,33]`）上注册 **forward hook**，正常跑一次前向，hook 把每层输出偷出来；`-1` 表示抓 embedding 层输出；
- 前向结束后把 5 层 hidden 沿最后一维 **concat** 成 `target_hidden_states`（形状 `[seq, 5*hidden]`），`last_hidden_state` 单独存为 `target_last_hidden_states`；
- 注意加载的是 `AutoModel`（不带 lm_head 的 backbone，260 行附近），bf16 + SDPA，纯推理。

每个样本落盘 5 样东西（`main()` 循环里的 `writer.write_sample`，317 行）：
`input_ids`、`attention_mask`、`loss_mask`（都截到真实 seq_len）+ 上面两个 hidden 张量。**这两个 hidden 张量就是磁盘开销的全部来源**（每 token 约 `(5+1) × hidden_size × 2` 字节）。

### 1.3 输出：cache 的磁盘格式

都在 `deepspec/data/target_cache_dataset.py`：

- **写入**：`AsyncTargetCacheWriter`（410 行）后台线程 + 队列，不阻塞 NPU 前向；实际写盘的 `LocalTargetCacheWriter`（305 行）把样本序列化成裸字节流追加进分片文件，单片默认最大 64GB（`--max-shard-bytes`）。多卡各写各的 `_tmp/rank_k/` 目录（样本按 `compute_local_sample_range` 均分到卡，231 行）。
- **收尾**（`main()` 尾部）：rank 0 把各 rank 的分片统一重命名（`rename_local_target_cache_shards`），生成两份元数据：
  - `index.bin` — 定长记录的全局索引，每条记录记录该样本在哪个分片、什么偏移、各张量多长（`pack_index_record`，77 行）；
  - `manifest.json` — 层号、hidden size、chat 模板、target 模型名、git sha 等（`build_target_cache_manifest`，580 行）。训练启动时 `validate_train_cache`（203 行）会拿它和 draft 配置对账，**配置对不上直接 assert**——改了 `target_layer_ids` 就必须重做 cache，原因在这。
- **读取**：`CacheDataset`（615 行）训练时用，mmap 索引 + 分片按需打开（LRU 上限 4 个），`__getitem__`（753 行）按偏移零拷贝切出张量。`CacheCollator`（859 行）把变长样本 pad 成 batch。

**模块一建议阅读顺序**：`prepare_target_cache.py:main()` 通读 → `parser.py:GeneralParser.parse` 精读 → `run_target_forward_with_hooks` 精读 → `target_cache_dataset.py` 挑 `write_sample_bytes`/`pack_index_record`/`CacheDataset.__getitem__` 三处看格式怎么对上。

---

## 模块二：training（cache → draft 模型）

**建议入口：`deepspec/trainer/dspark_trainer.py`（只有 48 行），它是理解训练的地图——`run_batch` 四行代码就是每步训练的全部：模型前向 → 算 loss。剩下的问题只有两个：模型前向里发生了什么，loss 是什么。**

### 2.1 外层骨架（谁在什么时候调用谁）

- `train.py` — 读配置（python 文件 + `--opts` 点路径覆盖，见 `utils/config.py`），`torch.multiprocessing.spawn` 按可见卡数起 worker，每个 worker 构造 trainer 然后 `.train()`。
- `config/dspark/dspark_qwen3_4b.py` — 所有超参的唯一来源，四个 dict（model/train/logging/data）。读模型代码前先把 `model` 段的字段过一遍：`block_size=7`（每个 anchor 并行预测 7 个 token）、`num_anchors=512`（每条样本采 512 个训练点）、`num_draft_layers=5`、`target_layer_ids`（必须和 cache 一致）。
- `deepspec/trainer/base_trainer.py` — 生命周期全在 `__init__`（153 行）和 `train()`（351 行）：
  - `build_models`（246 行）：**draft 模型的 embedding 和 lm_head 直接拷贝自 target checkpoint 并冻结**（264–275 行，target 模型在 CPU 上加载、拷完即删）。所以 draft 学的只有中间 5 层 + 各个头。
  - FSDP 包装（默认 `no_shard`，即等价 DDP）、可选 `torch.compile`（180 行，NPU 上建议关）。
  - `train()` 循环：`CUDAPrefetcher` 侧 stream 预取 → 梯度累积（`no_sync` 跳过通信，373 行）→ `clip_grad_norm_` → `BF16Optimizer.step()`（bf16 参数、fp32 主权重，见 `utils/optim.py`）→ 按步数落 checkpoint。
  - 断点续训：`ckpt_manager.py` 存了 optimizer、scheduler、RNG 状态和 `next_micro_step`，`StatelessResumableDistributedSampler` 能从任意样本偏移继续，保证重启后数据顺序不变。

### 2.2 DSpark 前向（全仓库最核心的 150 行）

精读 `deepspec/modeling/dspark/qwen3/modeling.py` 的 `forward`（388 行起），配合 `common.py`。按执行顺序：

1. **采 anchor**（`common.py:sample_anchor_positions`，150 行）：从 loss_mask=1 的位置里采 `num_anchors=512` 个位置。每个 anchor 的语义是「假设生成到这里，让 draft 一口气猜后面 block_size=7 个 token」。样本有效位置不足时用 dummy anchor 占位，由 `block_keep_mask` 屏蔽。
2. **构造 draft 输入**（`create_noise_embed`，291 行）：draft 侧的输入不是真实 token，而是 `mask_token` 的 embedding（每个 anchor 一段 7 个），位置编码用 `create_position_ids` 给到 anchor+1..anchor+7——**这是「半自回归」的本体：7 个位置一次并行前向，而不是逐 token**。
3. **attention mask**（`common.py:create_dspark_attention_mask`，78 行）：拼接后的 KV 序列是 `[context(seq_len) | draft(512*7)]`。规则只有两条：draft 位置能看到 **anchor 之前的 context**（因果）+ **自己 block 内的位置**；不同 block 互相不可见（所以 512 个 anchor 能塞进同一次前向互不干扰）。GPU 走 FlexAttention BlockMask；`attn_implementation != "flex_attention"` 时（NPU）走 87–110 行的稠密布尔 mask 等价实现。
4. **backbone**（`_forward_backbone`，361 行）：先 `self.fc` 把 cache 里 5 层 concat 的 hidden（5×2560 维）投影回 2560 维 + RMSNorm——**cache 在这里被消费**。然后过 5 层 `Qwen3DSparkDecoderLayer`。看它的 attention（`Qwen3DSparkAttention.forward`，87 行）：**Q 只来自 draft 位置（q_len=3584），K/V 是「target hidden 投影出的 context K/V」和「draft 自身 K/V」的拼接**。这就是为什么训练不需要 target 模型——context 侧的 K/V 由 cache 里的 target 特征直接算出。计算量也因此只有 draft 位置的 3584 个 query，而不是全序列。
5. **标签对齐**（forward 中段，约 434–470 行）：第 i 个 anchor 的第 j 个 draft 位置的标签 = `input_ids[anchor_i + j + 1]`（gather 出 `target_ids`）；同时从 `target_last_hidden_states` gather 出 target 模型在相同位置的 hidden，过冻结 lm_head 得到 `aligned_target_logits`（蒸馏用）。`eval_mask` 标记哪些槽位有效（不越界、且是 loss_mask=1 的连续前缀）。
6. **markov head**（`markov_head.py`）：低秩（rank=256）的 prev-token 修正项，训练时 `apply_block_logits` 用 teacher-forcing 的前一 token 给 `draft_logits` 加偏置，弥补「7 个位置并行猜、彼此看不到对方猜了什么」的先天缺陷。推理时的逐步版本是 `apply_step_logits`/`sample_block_tokens`。
7. **confidence head**（forward 尾部，505–517 行）：一个小 MLP（`common.py:AcceptRatePredictor`），输入 draft hidden（可拼 markov 的 prev-token embedding），输出每个位置「会被 target 接受」的概率——推理时调度器靠它决定验证多长，是论文里动态验证长度的基础。

返回值 `DSparkForwardOutput`（`common.py:12`）的 docstring 把所有张量形状写得很清楚，读 forward 前先看它。

### 2.3 Loss（`deepspec/modeling/dspark/loss.py`）

入口 `compute_dspark_loss`（255 行），三项加权（权重来自 config）：

| 项 | 含义 | config 权重 |
| --- | --- | --- |
| CE | draft_logits vs target_ids 的交叉熵 | `ce_loss_alpha=0.1` |
| L1 | draft_logits vs aligned_target_logits 的 L1 蒸馏 | `l1_loss_alpha=0.9` |
| confidence | confidence_pred vs「draft 是否猜对」的 BCE | `confidence_head_alpha=1.0` |

两个细节：block 内位置 j 有 `loss_decay_gamma=4.0` 的衰减权重（`_build_loss_weight_mask`，25 行）——越靠后的位置越难猜、权重越低；分母做了跨卡 all-reduce（`_all_reduce_loss_denominators`），保证梯度累积和多卡下损失归一化一致。训练日志里的 `accept_rate` 指标来自 `_compute_accept_rate_3d`（60 行），冒烟测试时盯的就是它。

### 2.4 NPU 路径在哪里生效

三处，都已在本 fork 合入（上游 PR #9）：

- `utils/device.py` — 运行时探测 `torch.npu`，决定 `device_type()`/HCCL；`DEEPSPEC_DEVICE=npu|cuda` 可强制。
- `modeling/dspark/*/config.py` — `TRAIN_ATTN_IMPLEMENTATION = "sdpa" if is_npu_available() else "flex_attention"`，联动上面 2.2 第 3 步的稠密 mask 分支。
- `data/cuda_prefetcher.py`、`trainer/base_trainer.py`、`utils/distributed.py` — stream/FSDP mesh/进程组按设备选实现。

**模块二建议阅读顺序**：`dspark_trainer.py` 全文 → `config/dspark/dspark_qwen3_4b.py` 全文 → `common.py` 的 `DSparkForwardOutput` docstring → `qwen3/modeling.py:forward` 逐行（对照 2.2 的七步）→ `Qwen3DSparkAttention.forward` → `loss.py:compute_dspark_loss` → 最后回头扫 `base_trainer.py` 的 `__init__`/`train()`。

---

## 上手验证：边跑边读

读代码时配一个小数据跑起来最有效（参考 `example/`）：

```bash
# 1000 条样本的 cache（~30GB），50 步训练
num_samples=1000 bash example/02_prepare_data.sh
python train.py --config config/dspark/dspark_qwen3_4b.py \
    --opts "data.target_cache_path=${HOME}/.cache/deepspec/qwen3_4b_target_cache" \
    --opts "train.torch_compile=False" \
    --opts "train.max_train_steps=50" \
    --opts "exp_name=smoke_test"
```

跑起来后在 `run_batch` 里断点/打印 `batch` 各键的形状，对照 2.2 的形状符号（`bsz × num_anchors × block_size`），模块二的所有抽象立刻具体化。tensorboard 在 `~/tensorboard/deepspec/<exp_name>`，主要看 `ce_loss` 和 `accept_rate`。

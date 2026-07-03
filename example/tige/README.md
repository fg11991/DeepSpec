# tige 平台脚本（Qwen3-8B / Qwen3-32B DSpark）

复制到昇腾训练平台直接跑的脚本。共 6 个训练脚本 + 1 个公共环境。

```
_common_env.sh              # 公共环境（所有脚本 source 它，路径改这里）
qwen3_8b_prepare_hidden.sh  # 8B  生成 target cache（多节点可用）
qwen3_8b_train.sh           # 8B  训练 draft（多节点可用）
qwen3_8b_eval.sh            # 8B  评估接受率（单节点 8 卡）
qwen3_32b_prepare_hidden.sh # 32B 生成 target cache（多节点可用，见显存警告）
qwen3_32b_train.sh          # 32B 训练 draft（多节点可用）
qwen3_32b_eval.sh           # 32B 评估接受率（单节点 8 卡，见显存警告）
```

## 多节点怎么跑（关键）

DeepSpec **不用 torchrun**，用自己的启动器：`train.py`/`eval.py`/
`prepare_target_cache.py` 每个节点各跑一次，内部自动 spawn 每卡一个 worker。
多节点约定（[utils/distributed.py](../../deepspec/utils/distributed.py) `init_dist`）：

| 变量 | 含义 |
| --- | --- |
| `RANK` | 本节点编号 0..NNODES-1（脚本里由 `NODE_RANK` 映射） |
| `WORLD_SIZE` | **节点数**，不是总卡数（由 `NNODES` 映射） |
| `MASTER_ADDR` / `MASTER_PORT` | rank0 节点 IP / 端口，所有节点一致 |

全局 rank = `NODE_RANK * 8 + local_rank`。8 节点 × 8 卡 = 64 个数据并行 worker。
**在每个节点上跑同一条命令**，只改 `NODE_RANK`：

```bash
# 单节点（默认）
bash example/tige/qwen3_8b_train.sh

# 8 节点：在第 i 个节点上执行（i = 0..7）
NNODES=8 NODE_RANK=<i> MASTER_ADDR=<node0-ip> bash example/tige/qwen3_8b_train.sh
```

## 三个阶段的多节点支持

| 阶段 | 多节点 | 说明 |
| --- | --- | --- |
| prepare_hidden | ✅ | 数据并行，每节点各扫 1/(NNODES×8) 的样本 |
| train | ✅ | 数据并行 + FSDP，节点越多 shard 越充分 |
| eval | 技术上支持，但脚本设为单节点 8 卡 | 基准集小（≤500/任务），单节点足够 |

## Qwen3-32B 的显存硬约束（务必先读）

- **训练（32b_train）没问题**：target 不驻留在卡上，只有 draft + 冻结 embed/lm_head，
  经 `full_shard` + `fsdp_auto_wrap` 分片后 32GB 卡也能跑。
- **prepare_hidden 和 eval 是瓶颈**：这两步每张卡都要加载**完整的 32B target**
  （bf16 ~61GB，无张量并行），32GB / 64GB 卡都装不下。多节点**不能**缓解这一点
  （是每卡显存问题，不是节点数问题）。要么用能装下整份 32B 的大卡，要么需要给
  `prepare_target_cache.py` / DSpark evaluator 加 device_map/TP 切分——**这个改动
  目前仓库没有**。如果你的 910C 是 64GB 且这两步 OOM，告诉我，我可以加上（改动集
  中在这两个文件）。

## 8B 推荐顺序（先跑通）

```bash
# 改 _common_env.sh 里的 REPO_ROOT / 输出目录；准备好 train.jsonl
bash example/tige/qwen3_8b_prepare_hidden.sh          # 或多节点
bash example/tige/qwen3_8b_train.sh                   # 或多节点
tasks=gsm8k max_samples=8 max_new_tokens=64 bash example/tige/qwen3_8b_eval.sh   # 冒烟
bash example/tige/qwen3_8b_eval.sh                    # 完整评估
```

数据格式、cache 大小、OOM 处置等见上级 [example/README.md](../README.md) 和
[docs/CODE_GUIDE_zh.md](../../docs/CODE_GUIDE_zh.md)。

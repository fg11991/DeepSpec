# 移植 USP（Ulysses 序列并行）到 DSpark 的提示

未来若要给 DSpark 训练加序列并行（对齐 SpecForge，或支持超长 context），这份提示
记录了参照来源、可复用部分、以及 DSpark 特有的难点。**目前（seq≤4096）不需要**，
仅在 context 推到 8192+ 且 attention 激活 OOM，或要与 SpecForge 对齐时才做。

## 为什么需要（纠正一个常见误解）

USP shard 的是 **draft 自身的序列维激活**，和 target 在不在场无关——SpecForge 的
offline 训练同样用 USP。它把 q/k/v 投影、MLP、norm 这些随序列长度增长的激活按
`sp_size` 摊到多卡，并在 attention 时用 all-to-all 换成「全序列、部分头」布局，从而
能训单卡放不下的长序列。

## 参照来源（SpecForge）

- `specforge/specforge/distributed.py`
  - `init_distributed(tp_size, sp_ulysses_size, sp_ring_size)`：建 draft_dp × sp 的
    进程组网格；`get_sp_ulysses_group()` / `get_sp_ring_group()`。
  - `SeqAllToAll4D`（autograd Function，all-to-all，scatter_idx=2 头维、gather_idx=1
    序列维）——这是 Ulysses 的核心通信原语。
- `specforge/specforge/modeling/draft/dflash.py` → `Qwen3DFlashUSPAttention`
  - forward 里对 `q / k_ctx / k_noise / v_ctx / v_noise` 各做一次 `_ulysses_scatter`
    （= SeqAllToAll4D），attention 后再 `SeqAllToAll4D` gather 回来。
  - **关键**：DSpark 的 `Qwen3DSparkAttention`（`deepspec/modeling/dspark/qwen3/
    modeling.py`）与它**结构完全同构**（同样的 q + k_ctx/k_noise + v_ctx/v_noise +
    RoPE），所以这段 scatter/gather 可以近乎照抄。
- 参照测试：`tests/test_modeling/test_draft/test_dflash_usp_parity.py`、
  `test_dflash_mask.py`（USP 下 mask 正确性的对照）。

## 移植步骤（大致）

1. 把 `distributed.py` 的 SP 进程组搭建 + `SeqAllToAll4D` 搬进 DeepSpec（新建
   `deepspec/utils/sequence_parallel.py` 之类）。
2. 在 `Qwen3DSparkAttention.forward` 里，RoPE 之后、attention 之前，对 q/k_ctx/
   k_noise/v_ctx/v_noise 做 `_ulysses_scatter`；attention 之后对 output 做 gather。
   加一个 `attention_backend=="usp"` 开关，NPU 上底层 attention 仍走 SDPA。
3. FSDP 的 device_mesh 要与 SP group 组合（draft_dp × sp），改 `_build_fsdp_kwargs`
   和 `init_dist`（新增 `sp_ulysses_size`）。
4. 数据加载/采样按序列分片（每卡拿 seq/sp 的位置）。

## DSpark 特有的难点（不能照抄的部分）

1. **块状 attention mask 的序列分片是最麻烦的**。DSpark 用 `create_dspark_attention_
   mask`（anchor/block 稠密 mask，NPU 上还是稠密布尔版）。Ulysses 把序列切开后，
   mask 必须跟着切分/重建，且要处理 anchor/block 跨分片边界。SpecForge 为此专门有
   parity 测试——DSpark 要自己做同样的 mask 分片适配并加对照测试。
2. **DSpark 的 q 是 anchor 子采样的**（`num_anchors×block_size`，非全序列），而
   DFlash 的 q 基本是全序列。所以 DSpark 真正值得 shard 的是 **context 侧**
   （k_ctx/v_ctx = 全 seq_len）和那份 `[seq, 5×hidden]` 的 target hidden 输入；
   query 侧本就小。这意味着 DSpark 对 USP 的需求比 DFlash 低，实现时可考虑只对
   context 序列维做并行。
3. **NPU**：SpecForge 另有 `fa`（`Qwen3DFlashNPUFlashAttention`）后端，USP 是独立
   后端。NPU 上跑 USP = HCCL all-to-all + 每卡 SDPA，通信可行但需自测数值一致性。

## 判断要不要做

需求排序：query 维被 num_anchors 卡住 → 序列相关大头是 context K/V 和 target hidden
输入 → seq=4096 下不大，现有 `fsdp_auto_wrap + gradient_checkpointing` 够用。只有
① context ≥8192 且 attention 激活 OOM（NPU 稠密 mask 尤其吃这个，USP 比 gradient_
checkpointing 重算更划算），或 ② 对齐 SpecForge 实验设置，才值得投入。

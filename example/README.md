# DSpark 训练示例：单机 8 卡 Ascend 910B

在一台 8×910B 的机器上，从零跑通 DSpark draft 模型的「数据准备 → 训练 → 评估」全流程，并说明训练完成后如何在推理引擎里拉起带投机解码的模型。

## 为什么从 Qwen3-4B 开始

- 这是仓库自带的最小 target 配置（`config/dspark/dspark_qwen3_4b.py`），也是上游 NPU 适配 PR 在 910B 上端到端验证过的链路；
- DeepSeek-V4 这类旗舰 MoE 模型单机 8×910B（约 512GB 显存）既放不下也训不动——DSpark 的官方 V4 draft 权重 DeepSeek 已经放出（见下文「拉起 DeepSeek 模型」），自己训练时用小 target 验证方法论即可；
- 4B target 时 draft 模型只有 5 层，8 卡 `no_shard` FSDP 足够，单卡 64GB 显存无压力。

跑通 4B 之后，换 `config/dspark/dspark_qwen3_8b.py` / `dspark_qwen3_14b.py` 只需改 `config_path` 和 `model_path` 两个变量。

## 推荐镜像

推荐直接用 vllm-ascend 官方镜像，训练、数据再生成、之后的推理部署共用一个环境：

```text
quay.io/ascend/vllm-ascend:v0.18.0
```

自带 CANN 8.5.1 + torch/torch_npu 2.9.0.post1 + vllm，与本仓库锁的 torch 2.9.x 同系。
910B（A2）和 910C（A3）用同一镜像。启动示例：

```bash
docker run -it --name deepspec-train \
    --device /dev/davinci0 --device /dev/davinci1 \
    --device /dev/davinci2 --device /dev/davinci3 \
    --device /dev/davinci4 --device /dev/davinci5 \
    --device /dev/davinci6 --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v <DeepSpec路径>:/workspace/DeepSpec \
    -v <大容量数据盘>:/data \
    -v <大容量数据盘>/checkpoints:/root/checkpoints \
    -v <大容量数据盘>/tensorboard:/root/tensorboard \
    quay.io/ascend/vllm-ascend:v0.18.0 bash
```

进容器后跑 `bash example/00_setup_env.sh` 补齐仓库依赖（脚本会自动保留镜像里
NPU 适配版的 torch，不会被 requirements.txt 覆盖）。target cache 记得放数据盘：
`cache_dir=/data/deepspec/qwen3_4b_target_cache`。

> **训练输出必须挂载出来**：train.py 把 checkpoint 写到 `~/checkpoints/deepspec/<exp_name>/step_*`
> （`step_latest` 软链指向最新，训练重启会自动从它续训）、tensorboard 写到
> `~/tensorboard/deepspec/<exp_name>`。容器里 `~` 是 `/root`，上面两行 `-v` 就是为了
> 让它们落在宿主机数据盘上——漏挂的话容器删除即丢。路径前缀是 config 里的
> `BASE_CKPT_DIR`/`BASE_TB_DIR` 常量，`--opts` 改不了（`finalize_cfg` 在 opts 之后执行
> 且会重算目录），要改位置直接改 config 或按上面挂载。

## 流程

```bash
# 0. 环境：镜像内补齐依赖（裸机则安装 torch_npu + vllm-ascend）
bash example/00_setup_env.sh

# 1. 【终端 A】起 8 个单卡 vllm 服务，供第 2 步再生成答案
bash example/01_launch_target_server.sh

# 2. 【终端 B】数据准备三步：下载切分 → 再生成答案 → 构建 target cache
#    默认子采样 5 万条（全量 ~38TB 缓存，5 万条约 1.5TB，按磁盘调 num_samples）
#    第 3 阶段开始前 Ctrl-C 停掉终端 A 的服务（要占全部 NPU）
bash example/02_prepare_data.sh

# 3. 训练（8 卡，bf16，torch_compile 关闭，走 SDPA 稠密 mask 路径）
bash example/03_train.sh

# 4. 评估接受率（对照官方 deepseek-ai/dspark_qwen3_4b_block7 的水平）
bash example/04_eval.sh
```

所有脚本的关键参数都可以用环境变量覆盖，例如：

```bash
num_samples=20000 bash example/02_prepare_data.sh
exp_name=my_run local_batch_size=2 bash example/03_train.sh
```

### 磁盘预算

Target cache 存每 token 的多层 hidden states，是最大的开销，与样本数线性相关：

| 训练样本数 | cache 大小（约） |
| --- | --- |
| 全量 ~1.4M | 38 TB |
| 50,000（默认） | ~1.5 TB |
| 20,000 | ~600 GB |

进一步压缩可减少配置里的 `model.target_layer_ids`（少存几层，cache 等比例变小）。

### NPU 相关注意事项

- `DEEPSPEC_DEVICE=npu` 强制走 NPU 路径（自动检测通常也够，显式设置更保险）；设备可见性用 `ASCEND_RT_VISIBLE_DEVICES`；
- NPU 上 attention 自动回退为 **SDPA + 稠密布尔 mask**（torch_npu 不支持 FlexAttention BlockMask），是正确性基线，训练吞吐低于 GPU 上的 FlexAttention，属预期；
- `torch_compile` 在训练脚本里显式关闭（config 默认值面向 CUDA inductor）；
- 分布式后端自动选 HCCL，无需手工配置。

## 训练完之后：拉起 DeepSeek 模型做投机推理

先说清楚现状（2026 年 7 月）：**DSpark 的推理侧支持还没有进任何主流引擎的主线**。这个仓库的 `eval.py` 只做接受率测量，不是 serving 引擎。当前的实际选项：

1. **昇腾上等 vllm-ascend 的 DSpark 适配落地**：进行中的 RFC 见
   [vllm-project/vllm-ascend#11163](https://github.com/vllm-project/vllm-ascend/issues/11163)。
   落地后的形态大概率是 `vllm serve <target> --speculative-config` 挂 DSpark draft，
   与现有 EAGLE 系用法一致。
2. **DeepSeek 官方 V4 DSpark 权重**：draft 模块已放在 HF
   （[DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark) /
   [DeepSeek-V4-Pro-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark)），
   但注意 V4 级别的 target 模型单机 8×910B 放不下，多机部署等引擎支持后再考虑。
3. **今天就想在 910B 上跑投机解码 serving**：用你训好的 target + draft 组合走已支持的算法。
   vllm-ascend 已支持 EAGLE 系投机解码，例如用本仓库训的 Eagle3 draft：

   ```bash
   vllm serve Qwen/Qwen3-4B \
       --speculative-config '{"method": "eagle3", "model": "<eagle3_draft_path>", "num_speculative_tokens": 4}'
   ```

   DSpark draft 想真正 serve，需要等 RFC 或自己在 vllm-ascend 里实现 proposer——
   这本身就是一个有价值的研究/贡献方向。

不带投机解码、直接拉起 DeepSeek 系模型（如蒸馏版）做对照基线，vllm-ascend 开箱即用：

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --tensor-parallel-size 2
```

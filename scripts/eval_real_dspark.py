#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DSpark 轻量离线接受率自检 —— 单进程、多卡(device_map)、真实 draft→verify 循环。

针对 **混合注意力 target(Qwen3.6-27B / qwen3_5, 含 linear attention)** 适配:
DeepSpec 原版 verify 循环每轮 crop target KV cache, 对 linear attention 的 recurrent state
没法回滚 -> IndexError。这里保留 **真实的 draft→verify 语义**(target 当场验证 draft 提出的
block, rejection sampling 定接受前缀), 只把 "crop 回滚 KV" 换成 "每轮用已提交序列重建 target cache",
从而绕开混合模型的 crop 问题。贪心/采样都正确。

默认贪心(temperature=0), 逐位置接受率可直接对比训练的 teacher_agreement(0.89->0.64)和 vllm。

前提: 能 import deepspec(把脚本放进 DeepSpec 目录里跑, 或设 PYTHONPATH)。

用法:
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
    python dspark_lite_eval.py \
        --target /opt/foundation_model/Qwen3.6-27B \
        --draft  /opt/.../step40000 --show-output
"""
import argparse
from types import SimpleNamespace

import torch

try:
    import torch_npu  # noqa: F401
    _HAS_NPU = True
except ImportError:
    _HAS_NPU = False

# ==== 按你的 deepspec/fork 路径改 ====
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from deepspec.eval.dspark.evaluator import Qwen3DSparkEvaluator
from deepspec.eval.base_evaluator import (
    resolve_stop_token_ids, assert_no_final_target_layer,
    verify_draft_tokens, has_stop_token,
)
from deepspec.modeling.dspark.common import extract_context_feature
from deepspec.utils.sampling import logits_to_probs, sample_from_probs
from deepspec.data.parser import encode_chat_messages


DEFAULT_PROMPTS = [
    "Write a Python function to compute the nth Fibonacci number iteratively.",
    "Solve step by step: what is the sum of all prime numbers below 50?",
    "用 Python 实现快速排序, 并简单解释。",
    "Explain what a hash table is in two sentences.",
]


class LiteDSparkEval(Qwen3DSparkEvaluator):
    """跳过分布式初始化 + 适配你的 config + target 用 device_map 分片。"""

    def __init__(self, *, device, target, draft, temperature, max_new_tokens,
                 confidence_threshold, seed, target_device_map="auto",
                 target_max_memory=None, num_anchors=512, draft_config_overrides=None):
        self.args = SimpleNamespace(
            target_name_or_path=target, draft_name_or_path=draft,
            temperature=temperature, max_new_tokens=max_new_tokens,
            confidence_threshold=confidence_threshold,
            tensorboard_dir=None, step=None, seed=seed,
        )
        self.device = device
        self.target_device_map = target_device_map
        self.target_max_memory = target_max_memory
        self.num_anchors = num_anchors
        self.draft_config_overrides = draft_config_overrides or {}
        self.target_model, self.draft_model, self.tokenizer = self.build_models()
        self.confidence_head_recorder = None
        self.metrics_rows = []

    def _adapt_draft_config(self):
        import json, os
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(self.args.draft_name_or_path)
        with open(os.path.join(self.args.draft_name_or_path, "config.json")) as f:
            raw = json.load(f)
        dfc = raw.get("dflash_config", {}) or {}

        def ensure(name, value):
            if value is None:
                return
            if not hasattr(cfg, name) or getattr(cfg, name) is None:
                setattr(cfg, name, value)
                print(f"  [config适配] 补 {name} = {value}")

        ensure("target_layer_ids", dfc.get("target_layer_ids", raw.get("target_layer_ids")))
        ensure("mask_token_id", dfc.get("mask_token_id", raw.get("mask_token_id")))
        # SpecForge writes draft_vocab_size at the config top level; without it
        # the draft would be built full-vocabulary and the checkpoint's pruned
        # lm_head / markov_w2 would not fit.
        ensure("draft_vocab_size", raw.get("draft_vocab_size"))
        ensure("num_anchors", self.num_anchors)
        for k, v in self.draft_config_overrides.items():
            setattr(cfg, k, v)
            print(f"  [config覆盖] {k} = {v}")
        return cfg

    def build_models(self):
        target_model = AutoModelForCausalLM.from_pretrained(
            self.args.target_name_or_path, dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
            device_map=self.target_device_map, max_memory=self.target_max_memory,
        ).eval()

        print("加载 draft, 适配 config ...")
        draft_config = self._adapt_draft_config()
        draft_model, loading_info = self.draft_model_cls.from_pretrained(
            self.args.draft_name_or_path, config=draft_config, dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION, output_loading_info=True,
        )
        missing = sorted(loading_info.get("missing_keys", []) or [])
        unexpected = sorted(loading_info.get("unexpected_keys", []) or [])
        print(f"  [权重加载] missing={len(missing)}  unexpected={len(unexpected)}")
        TIED = {"embed_tokens.weight", "lm_head.weight"}
        real_missing = [k for k in missing if k not in TIED]
        if missing:
            print(f"  missing: {missing}")
        if unexpected:
            print(f"  unexpected: {unexpected}")
        if real_missing or unexpected:
            print("  !! 除 embed/lm_head 外还有权重对不上, eval 数可能不可信。")

        draft_model = draft_model.to(self.device).eval()
        with torch.no_grad():
            tgt_embed = target_model.get_input_embeddings().weight
            tgt_head = target_model.get_output_embeddings().weight
            assert draft_model.embed_tokens.weight.shape == tgt_embed.shape
            draft_model.embed_tokens.weight.copy_(tgt_embed.to(draft_model.embed_tokens.weight.device))
            head_weight = tgt_head
            if getattr(draft_model, "use_draft_vocab", False):
                # A pruned draft borrows only the rows t2d keeps, so that
                # generation proposes exactly the tokens training supervised.
                assert bool(draft_model.t2d.any()), (
                    "draft prunes the vocabulary but the checkpoint carried no "
                    "t2d/d2t; export it from a run trained with draft_vocab_size."
                )
                mask = draft_model.t2d.to(device=tgt_head.device, dtype=torch.bool)
                head_weight = tgt_head[mask]
                print(
                    f"  [词表裁剪] draft_vocab={draft_model.draft_vocab_size} / "
                    f"target_vocab={draft_model.vocab_size} "
                    f"({draft_model.vocab_size / draft_model.draft_vocab_size:.2f}x)"
                )
            assert draft_model.lm_head.weight.shape == head_weight.shape
            draft_model.lm_head.weight.copy_(head_weight.to(draft_model.lm_head.weight.device))
        print("  [embed/lm_head] 已从 target 拷入 draft")

        assert_no_final_target_layer(target_model, draft_model.target_layer_ids)
        tokenizer = AutoTokenizer.from_pretrained(self.args.target_name_or_path)
        return target_model, draft_model, tokenizer


@torch.inference_mode()
def real_spec_decode(ev, input_ids, max_new_tokens, stop_token_ids, save_first_logits=None):
    """真实的 draft->verify 投机解码循环, 用【每轮重建 target cache】替代 crop, 兼容混合 target。

    若 save_first_logits 给了路径, 则保存【第一个生成 token 起手、第一次 draft 出的那个 block】的
    logits(base + Markov 修正后)、采样 token、起手 token, 以及该 block 的 target verify 概率。
    """
    device = input_ids.device
    bs = ev.max_proposal_tokens
    temp = float(ev.args.temperature)
    num_input = input_ids.shape[1]
    max_length = num_input + int(max_new_tokens)

    buf = torch.empty((1, max_length + bs + 1), dtype=torch.long, device=device)
    position_ids = torch.arange(buf.shape[1], device=device).unsqueeze(0)
    buf[:, :num_input] = input_ids

    # prefill: 采样第一个 token(target 自己确定的起手)
    pout = ev.target_model(
        input_ids=input_ids, position_ids=position_ids[:, :num_input],
        past_key_values=None, use_cache=True, output_hidden_states=True,
    )
    buf[:, num_input:num_input + 1] = sample_from_probs(logits_to_probs(pout.logits[:, -1:, :], temp))
    start = num_input

    # —— 捕获第一次 draft 的 logits + hidden(只抓第一轮): 临时接管 draft 的方法 + 挂 hook ——
    captured = {}
    hooked = False
    _fc_handle = _hn_handle = None
    if save_first_logits is not None:
        _orig_compute = ev.draft_model.compute_logits
        _orig_sample = ev.draft_model.sample_draft_tokens

        def _cap_compute(hidden_states):
            out = _orig_compute(hidden_states)
            if "base_logits" not in captured:
                captured["base_logits"] = out.detach().float().cpu()   # (1, bs, vocab) backbone+lm_head
                # compute_logits 的输入就是 backbone 最后一层输出(过 lm_head 前的 hidden)
                captured["block_hidden_pre_lmhead"] = hidden_states.detach().float().cpu()  # (1, bs, 5120)
            return out

        def _cap_sample(base_logits, *, first_prev_token_ids, temperature=0.0, hidden_states=None):
            sampled, corrected = _orig_sample(
                base_logits, first_prev_token_ids=first_prev_token_ids,
                temperature=temperature, hidden_states=hidden_states)
            if "corrected_logits" not in captured:
                captured["corrected_logits"] = corrected.detach().float().cpu()  # 过 Markov 后
                captured["sampled_tokens"] = sampled.detach().cpu()
                captured["first_prev_token_id"] = first_prev_token_ids.detach().cpu()
            return sampled, corrected

        # forward hook 抓 fc / hidden_norm 的输入输出(不改 deepspec 代码)
        def _fc_hook(mod, inp, out):
            if "context_feature" not in captured:
                # fc 的输入 = extract_context_feature 输出 = 那几层拼接 = context feature(过 fc 前)
                captured["context_feature"] = inp[0].detach().float().cpu()   # (1, ctx, 25600)
                captured["fc_out"] = out.detach().float().cpu()               # (1, ctx, 5120)

        def _hn_hook(mod, inp, out):
            if "context_feature_projected" not in captured:
                # hidden_norm(fc(...)) = 真正喂给 5 层 attention 的 context feature
                captured["context_feature_projected"] = out.detach().float().cpu()  # (1, ctx, 5120)

        ev.draft_model.compute_logits = _cap_compute
        ev.draft_model.sample_draft_tokens = _cap_sample
        _fc_handle = ev.draft_model.fc.register_forward_hook(_fc_hook)
        _hn_handle = ev.draft_model.hidden_norm.register_forward_hook(_hn_hook)
        hooked = True

    prop_lengths, acc_draft_lengths, acc_lengths = [], [], []
    if has_stop_token(buf[:, num_input:num_input + 1], stop_token_ids):
        if hooked:
            del ev.draft_model.compute_logits, ev.draft_model.sample_draft_tokens
            if _fc_handle is not None: _fc_handle.remove()
            if _hn_handle is not None: _hn_handle.remove()
        return SimpleNamespace(output_ids=buf[:, :num_input + 1], num_input_tokens=num_input,
                               proposal_lengths=[], accepted_draft_lengths=[], acceptance_lengths=[])

    round_idx = 0
    while start < max_length:
        # (A) 重建 target cache([0..start-1]) + 取 draft 上下文特征
        pout = ev.target_model(
            input_ids=buf[:, :start], position_ids=position_ids[:, :start],
            past_key_values=None, use_cache=True, output_hidden_states=True,
        )
        target_cache = pout.past_key_values
        context = SimpleNamespace(
            past_key_values_draft=DynamicCache(),
            target_hidden_states=extract_context_feature(
                pout.hidden_states, ev.draft_model.target_layer_ids),
        )
        # (B) draft 提 block(第一轮会触发上面的捕获)
        proposal = ev._propose(
            context=context, output_ids=buf, position_ids=position_ids,
            start=start, stop_token_ids=stop_token_ids,
        )
        # (C) target 当场 verify(追加到 cache, 不 crop)
        verification = verify_draft_tokens(
            target_model=ev.target_model, proposal=proposal, position_ids=position_ids,
            start=start, past_key_values_target=target_cache, temperature=temp,
            max_proposal_tokens=bs, current_token_ids=buf[:, start:start + 1],
            stop_token_ids=stop_token_ids,
        )

        # 第一轮: 补齐并落盘第一次 draft 的 logits + hidden
        if hooked and round_idx == 0 and "base_logits" in captured:
            captured["seed_token_id"] = buf[:, start:start + 1].detach().cpu()   # 起手的第一个生成 token
            captured["target_verify_probs"] = verification.target_probs.detach().float().cpu()  # (1, bs+1, vocab)
            captured["verify_input_ids"] = proposal.verify_input_ids.detach().cpu()
            # 喂进 backbone 的 aux 特征(指定 target 层)
            tl = list(ev.draft_model.target_layer_ids)
            captured["target_layer_ids"] = tl
            # context_feature / fc_out / context_feature_projected 已由 fc/hidden_norm 的 hook 存好。
            # 这里再单独存那几层各自的 raw hidden, 方便逐层和 vllm 比, {layer_id: (1, ctx_len, 5120)}
            captured["aux_hidden_per_layer"] = {
                int(l): pout.hidden_states[l].detach().float().cpu() for l in tl
            }
            captured["temperature"] = temp
            captured["block_size"] = bs
            captured["start"] = int(start)
            torch.save(captured, save_first_logits)
            print(f"  [已保存] 第一次 draft 的 logits + hidden -> {save_first_logits}")
            del ev.draft_model.compute_logits, ev.draft_model.sample_draft_tokens
            if _fc_handle is not None: _fc_handle.remove()
            if _hn_handle is not None: _hn_handle.remove()
            hooked = False

        # (D) 提交 + 推进
        a = int(verification.accepted_draft_tokens)
        prop_lengths.append(int(verification.effective_proposal_length))
        acc_draft_lengths.append(a)
        buf[:, start:start + a + 1] = proposal.verify_input_ids[:, :a + 1]

        if verification.terminated_by_stop_token:
            acc_lengths.append(a)
            start += a
            break
        buf[:, start + a + 1] = verification.next_token
        new_tokens = buf[:, start + 1:start + a + 2]
        acc_lengths.append(a + 1)
        start += a + 1
        round_idx += 1
        if has_stop_token(new_tokens, stop_token_ids):
            break

    if hooked:   # 兜底: 万一没进循环也恢复
        del ev.draft_model.compute_logits, ev.draft_model.sample_draft_tokens
        if _fc_handle is not None: _fc_handle.remove()
        if _hn_handle is not None: _hn_handle.remove()

    return SimpleNamespace(
        output_ids=buf[:, :min(start + 1, buf.shape[1])], num_input_tokens=num_input,
        proposal_lengths=prop_lengths, accepted_draft_lengths=acc_draft_lengths,
        acceptance_lengths=acc_lengths,
    )


@torch.inference_mode()
def run(ev, prompts, show_output, max_new_tokens, save_first_logits=None):
    bs = ev.max_proposal_tokens
    stop_ids = resolve_stop_token_ids(ev.target_model, ev.tokenizer)
    proposals_at_pos = [0] * bs
    accepted_at_pos = [0] * bs
    n_rounds = 0
    accepted_sum = 0

    for i, prompt in enumerate(prompts):
        prompt_ids = encode_chat_messages(
            ev.tokenizer, [{"role": "user", "content": prompt}],
            add_generation_prompt=True, enable_thinking=False,
        ).to(ev.device)

        # 只在第一道题保存第一次 draft 的 logits
        resp = real_spec_decode(
            ev, prompt_ids, max_new_tokens, stop_ids,
            save_first_logits=(save_first_logits if i == 0 else None),
        )

        this = []
        for al, pl, adl in zip(resp.acceptance_lengths, resp.proposal_lengths, resp.accepted_draft_lengths):
            n_rounds += 1
            accepted_sum += int(al)
            this.append(int(adl))
            for p in range(bs):
                if pl > p:
                    proposals_at_pos[p] += 1
                if adl > p:
                    accepted_at_pos[p] += 1

        avg = sum(this) / len(this) if this else 0.0
        print(f"[{i+1}/{len(prompts)}] verify轮数={len(this):>3}  平均接受draft数={avg:.2f}  | {prompt[:38]}")
        if show_output:
            txt = ev.tokenizer.decode(resp.output_ids[0, resp.num_input_tokens:], skip_special_tokens=True)
            print("    └─ 输出:", txt[:160].replace("\n", " ") + ("..." if len(txt) > 160 else ""))

    print("\n" + "=" * 66)
    if n_rounds == 0:
        print("没有产生任何 verify 轮(都被 stop token 立即终止?)")
        return
    print(f"总 verify 轮数        : {n_rounds}")
    print(f"平均 accept_len (含+1): {accepted_sum / n_rounds:.3f}   (block_size={bs})")
    print("\n逐位置条件接受率 (accept_rate@pos), 直接对比 vllm 的那条:")
    for p in range(bs):
        d = proposals_at_pos[p]
        r = (accepted_at_pos[p] / d) if d else float("nan")
        bar = "#" * int((r if r == r else 0) * 40)
        print(f"  pos {p}: {r:.4f}  {bar}")
    print("=" * 66)
    print("""
判读:
  - 逐位置接近训练的 0.89->0.64 且 accept_len 明显 >1 -> drafter 正常,
    vllm 的 20.7% 是 vllm-ascend 接线问题(Markov/confidence head 没 wire、fc 维度、采样温度)。
  - 这里也断崖 -> drafter 或数据侧问题, 和 vllm 无关。
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--prompt", action="append", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--confidence-threshold", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=980406)
    ap.add_argument("--device", default=None)
    ap.add_argument("--target-device-map", default="auto")
    ap.add_argument("--target-max-memory", default=None,
                    help="每卡上限, 如 '0:20GiB,1:28GiB,2:28GiB,3:28GiB'")
    ap.add_argument("--num-anchors", type=int, default=512)
    ap.add_argument("--draft-config-override", action="append", default=None)
    ap.add_argument("--save-first-logits", default=None,
                    help="给路径就保存第一个生成 token 起手、第一次 draft 出的 block 的 logits(.pt); 不给就不存")
    ap.add_argument("--show-output", action="store_true")
    args = ap.parse_args()

    max_memory = None
    if args.target_max_memory:
        max_memory = {}
        for item in args.target_max_memory.split(","):
            k, v = item.split(":")
            max_memory[int(k.strip())] = v.strip()

    overrides = {}
    if args.draft_config_override:
        import json as _json
        for item in args.draft_config_override:
            k, v = item.split("=", 1)
            try:
                v = _json.loads(v)
            except Exception:
                pass
            overrides[k.strip()] = v

    if args.device is None:
        args.device = "npu:0" if _HAS_NPU else ("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"device={args.device}  temperature={args.temperature}  max_new_tokens={args.max_new_tokens}")
    ev = LiteDSparkEval(
        device=args.device, target=args.target, draft=args.draft,
        temperature=args.temperature, max_new_tokens=args.max_new_tokens,
        confidence_threshold=args.confidence_threshold, seed=args.seed,
        target_device_map=args.target_device_map, target_max_memory=max_memory,
        num_anchors=args.num_anchors, draft_config_overrides=overrides,
    )
    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS
    run(ev, prompts, args.show_output, args.max_new_tokens, save_first_logits=args.save_first_logits)


if __name__ == "__main__":
    main()
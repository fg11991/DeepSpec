from __future__ import annotations
import argparse
import json
import torch
from transformers import AutoConfig
from deepspec.eval.dspark import Gemma4DSparkEvaluator, Qwen3DSparkEvaluator
from deepspec.eval.eagle3 import Gemma4Eagle3Evaluator, Qwen3Eagle3Evaluator
from deepspec.utils import CustomJSONEncoder, device_count

EVALUATORS = {
    "Qwen3DSparkModel": Qwen3DSparkEvaluator,
    "Gemma4DSparkModel": Gemma4DSparkEvaluator,
    "Qwen3Eagle3Model": Qwen3Eagle3Evaluator,
    "Gemma4Eagle3Model": Gemma4Eagle3Evaluator,
    "Eagle3DraftModel": Qwen3Eagle3Evaluator,
}

TASKS = [
    ("gsm8k", 500),
    ("math500", 500),
    ("aime25",30),
    ("humaneval", 164),
    ("mbpp", 256),
    ("livecodebench", 500),
    ("mt-bench", 80),
    ("alpaca", 500),
    ("arena-hard-v2", 500),
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name_or_path", type=str, required=True)
    parser.add_argument("--draft_name_or_path",type=str,required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=("Confidence-head early-stop threshold. Confidence calibration metrics are collected only when this is 0.0."),
    )
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--step", type=int, default=None,help=("step for tensorboard logging"),)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of eval datasets to run (e.g. 'gsm8k' or "
            "'gsm8k,humaneval'). Default: all. Available: "
            + ", ".join(name for name, _ in TASKS)
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap samples per dataset (overrides the per-task default). Use a small value to smoke-test.",
    )
    args = parser.parse_args()
    tasks = list(TASKS)
    if args.tasks is not None:
        wanted = [name.strip() for name in args.tasks.split(",") if name.strip()]
        available = {name for name, _ in TASKS}
        unknown = [name for name in wanted if name not in available]
        assert not unknown, f"Unknown --tasks {unknown}; available: {sorted(available)}"
        tasks = [(name, cap) for name, cap in tasks if name in wanted]
    if args.max_samples is not None:
        tasks = [(name, int(args.max_samples)) for name, _ in tasks]
    args.tasks = tasks
    return args


def main(local_rank: int, args):
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)
    evaluator_cls = EVALUATORS[draft_config.architectures[0]]
    evaluator = evaluator_cls(local_rank, args)
    evaluator.evaluate()
    evaluator.clean_up()

if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(
        main,
        args=(args,),
        nprocs=device_count(),
    )

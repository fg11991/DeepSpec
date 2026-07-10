import os
import warnings

# Silence noisy third-party warnings before torch / torch_npu import so spawned
# workers inherit it too: torch_npu file-owner/permission mismatches, CANN owner
# checks, FSDP state_dict deprecation, the distributed device_id notice, and the
# c10 "Driver Version ... is invalid" C++ warning.
os.environ.setdefault("PYTHONWARNINGS", "ignore")       # inherited by spawned workers
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "error")   # quiet c10 [W...] warnings
warnings.filterwarnings("ignore")

import argparse
import json
import torch
from deepspec.utils import (
    CustomJSONEncoder,
    device_count,
    get_git_diff,
    load_config,
    parse_opts_to_config,
    seed_all,
    get_git_sha,
)

os.environ['USE_TORCH']='true'
os.environ['WANDB_DISABLED']='true'
os.environ['TOKENIZERS_PARALLELISM']='false'
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--opts", action="append", default=[])
    args = parser.parse_args()
    config = parse_opts_to_config(args.opts, load_config(args.config))
    config._origin_config_path = os.path.abspath(args.config)
    config._origin_opts = list(args.opts)
    return config


def main(local_rank):
    args = parse_args()
    seed_all(int(args.seed))
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    trainer = args.train.trainer_cls(local_rank, args)
    trainer.train()
    trainer.clean_up()


if __name__ == "__main__":
    if os.path.exists(".git"):
        print(f"git status:", "\n\n".join(get_git_sha(detail_info=True)))
        print("git diff:", get_git_diff())
    # Detect torchrun via TORCHELASTIC_RUN_ID, NOT LOCAL_RANK: some platforms
    # export LOCAL_RANK as the visible-device list (e.g. "0,1,2,3,4,5,6,7")
    # even for the built-in spawn launcher, so keying on LOCAL_RANK would send
    # the spawn path into int("0,1,...") and crash. torchrun always sets
    # TORCHELASTIC_RUN_ID and a clean integer LOCAL_RANK per process.
    if "TORCHELASTIC_RUN_ID" in os.environ:
        # Launched by torchrun: this process already IS one rank; don't spawn.
        main(int(os.environ["LOCAL_RANK"]))
    else:
        # Built-in launcher: spawn one worker per visible device on this node.
        torch.multiprocessing.spawn(main, nprocs=device_count())

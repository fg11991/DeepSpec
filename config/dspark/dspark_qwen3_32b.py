import os
from deepspec.trainer import Qwen3DSparkTrainer
BASE_TB_DIR = os.path.expanduser(os.environ.get("DEEPSPEC_TB_DIR", "~/tensorboard"))
BASE_CKPT_DIR = os.path.expanduser(os.environ.get("DEEPSPEC_CKPT_DIR", "~/checkpoints"))
project_name = "deepspec"
exp_name = "dspark_block7_qwen3_32b"
seed = 42

model = dict(
    target_model_name_or_path="Qwen/Qwen3-32B",
    block_size=7,
    num_draft_layers=5,
    # Qwen3-32B has 64 decoder layers (vs 36 for Qwen3-8B). These 5 ids are
    # evenly spaced (step 15) across the depth and deliberately exclude the
    # final layer (63): eval forbids it (base_evaluator.assert_no_final_target_layer)
    # because the last layer's hidden state is captured separately as
    # target_last_hidden_states. Tune if needed, but the cache and draft
    # checkpoint must be rebuilt on any change.
    target_layer_ids=[1, 16, 31, 46, 61],
    mask_token_id=151669,
    num_anchors=512,

    ## markov head
    markov_rank=256,
    markov_head_type='vanilla',

    ## confidence head
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    ## loss
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    trainer_cls=Qwen3DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    # HSDP shard-group size in RANKS (only used when sharding_strategy is a
    # hybrid_shard variant). None = one node. Widen to span multiple nodes
    # (e.g. 16 = 2 nodes) when the draft won't fit sharded over a single node.
    # Env DEEPSPEC_HSDP_SHARD_SIZE overrides this.
    hsdp_shard_size=None,
    torch_compile=True,
    gradient_checkpointing=False,
    fsdp_auto_wrap=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
)

data = dict(
    target_cache_path=None,
    chat_template="qwen",
    max_length=4096,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name=str(cfg['project_name'])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg

    return cfg

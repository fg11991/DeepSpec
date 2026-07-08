from .config import build_draft_config
try:
    from .modeling import Gemma4DSparkModel
except ImportError:
    # modeling.py imports transformers.models.gemma4 and flex_attention, which
    # may be unavailable (older transformers, torch_npu). Keep the package
    # importable so the config submodule and the Qwen3 path still work; only
    # actually training Gemma4 needs the model class.
    Gemma4DSparkModel = None

__all__ = [
    "Gemma4DSparkModel",
    "build_draft_config",
]

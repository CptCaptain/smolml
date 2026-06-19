"""Model interface, registry, and baseline.

Importing this package registers the built-in models (their ``@register_model``
decorators run on import), so ``list_models()`` is populated.
"""

from smolml.models.context_mixing import ContextMixing, ContextMixingConfig
from smolml.models.fast_weight import FastWeightConfig, FastWeightMemory
from smolml.models.gated_mix import GatedMix, GatedMixConfig
from smolml.models.pc_refine import PCRefine, PCRefineConfig
from smolml.models.registry import (
    LanguageModel,
    build_model,
    get_model,
    list_models,
    register_model,
)
from smolml.models.transformer import Transformer, TransformerConfig
from smolml.models.warm_mix import WarmMix

__all__ = [
    "FastWeightConfig",
    "FastWeightMemory",
    "GatedMix",
    "GatedMixConfig",
    "LanguageModel",
    "ContextMixing",
    "ContextMixingConfig",
    "PCRefine",
    "PCRefineConfig",
    "Transformer",
    "TransformerConfig",
    "WarmMix",
    "build_model",
    "get_model",
    "list_models",
    "register_model",
]

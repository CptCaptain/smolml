"""Model interface, registry, and baseline.

Importing this package registers the built-in models (their ``@register_model``
decorators run on import), so ``list_models()`` is populated.
"""

from smolml.models.chemotaxis_min import ChemoMinConfig, ChemotaxisMin
from smolml.models.context_mixing import ContextMixing, ContextMixingConfig
from smolml.models.delta_mix import DeltaMix, DeltaMixConfig
from smolml.models.fast_weight import FastWeightConfig, FastWeightMemory
from smolml.models.gated_mix import GatedMix, GatedMixConfig
from smolml.models.hashed_mix import HashedMix, HashedMixConfig
from smolml.models.pc_refine import PCRefine, PCRefineConfig
from smolml.models.registry import (
    LanguageModel,
    build_model,
    get_model,
    list_models,
    register_model,
)
from smolml.models.reservoir import (
    Reservoir,
    ReservoirConfig,
    ReservoirPlastic,
    ReservoirPlasticConfig,
)
from smolml.models.transformer import Transformer, TransformerConfig
from smolml.models.warm_mix import WarmMix

__all__ = [
    "FastWeightConfig",
    "FastWeightMemory",
    "GatedMix",
    "GatedMixConfig",
    "HashedMix",
    "HashedMixConfig",
    "LanguageModel",
    "ContextMixing",
    "ChemoMinConfig",
    "ChemotaxisMin",
    "ContextMixingConfig",
    "DeltaMix",
    "DeltaMixConfig",
    "PCRefine",
    "Reservoir",
    "ReservoirConfig",
    "ReservoirPlastic",
    "ReservoirPlasticConfig",
    "PCRefineConfig",
    "Transformer",
    "TransformerConfig",
    "WarmMix",
    "build_model",
    "get_model",
    "list_models",
    "register_model",
]

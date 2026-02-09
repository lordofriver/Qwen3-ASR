# coding=utf-8
# Compatibility layer for transformers 4.46.x (Python 3.8 support)
"""
This module provides compatibility shims for features introduced in transformers 4.47+
to allow Qwen3-ASR to work with transformers 4.46.x on Python 3.8.
"""
from typing import Any, Dict
import torch
from torch import nn


# ============================================================================
# GradientCheckpointingLayer compatibility
# ============================================================================
try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    # transformers 4.46 doesn't have GradientCheckpointingLayer
    # Use a simple wrapper around nn.Module
    class GradientCheckpointingLayer(nn.Module):
        """Fallback for GradientCheckpointingLayer in transformers < 4.47"""
        pass


# ============================================================================
# RoPE utilities compatibility
# ============================================================================
try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
except ImportError:
    # Fallback for transformers 4.46
    from transformers.modeling_utils import ROPE_INIT_FUNCTIONS
    
    def dynamic_rope_update(func):
        """Fallback decorator for dynamic_rope_update"""
        return func


# ============================================================================
# Kwargs utilities compatibility
# ============================================================================
try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
except ImportError:
    # Fallback: use Dict[str, Any]
    FlashAttentionKwargs = Dict[str, Any]

try:
    from transformers.utils.generic import TransformersKwargs
except ImportError:
    # Fallback: use Dict[str, Any]
    TransformersKwargs = Dict[str, Any]


# ============================================================================
# Deprecation utilities compatibility
# ============================================================================
try:
    from transformers.utils.deprecation import deprecate_kwarg
except ImportError:
    # Fallback: no-op decorator
    def deprecate_kwarg(old_name: str, new_name: str = None, version: str = None):
        """Fallback decorator for deprecate_kwarg"""
        def decorator(func):
            return func
        return decorator


# ============================================================================
# check_model_inputs compatibility
# ============================================================================
try:
    from transformers.utils.generic import check_model_inputs
except ImportError:
    # Fallback: no-op decorator
    def check_model_inputs(func):
        """Fallback decorator for check_model_inputs"""
        return func

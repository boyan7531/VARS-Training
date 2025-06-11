"""
Multi-task Multi-view Video Model Package

This package provides modular implementations of multi-task, multi-view video models
for sports action classification using ResNet3D or MViTv2 architectures.
"""

from .resnet3d_model import MultiTaskMultiViewResNet3D, ResNet3DBackbone
from .unified_model import MultiTaskMultiViewMViT, create_unified_model
from .config import ModelConfig
from .loader import ModelLoader, ModelLoadingError
from .embedding_manager import EmbeddingManager
from .view_aggregator import ViewAggregator
from .validator import InputValidator, ValidationError

__version__ = "1.0.0"
__author__ = "VARS Team"

__all__ = [
    # Main model classes
    "MultiTaskMultiViewResNet3D",
    "MultiTaskMultiViewMViT", 
    "ResNet3DBackbone",
    # Unified interface
    "create_unified_model",
    # Configuration and utilities
    "ModelConfig",
    "ModelLoader",
    "ModelLoadingError",
    # Components
    "EmbeddingManager",
    "ViewAggregator",
    "InputValidator",
    "ValidationError",
] 
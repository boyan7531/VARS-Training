"""
Multi-task Multi-view Video Model Package

This package provides modular implementations of multi-task, multi-view video models
for sports action classification using ResNet3D architecture.
"""

from .resnet3d_model import MultiTaskMultiViewResNet3D, ResNet3DBackbone
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
    "ResNet3DBackbone",
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
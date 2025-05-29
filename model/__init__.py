"""
Multi-task Multi-view MViT Model Package

This package provides a modular, maintainable implementation of a multi-task,
multi-view video transformer model for sports action classification.
"""

from .model import MultiTaskMultiViewMViT, create_multitask_multiview_mvit
from .config import ModelConfig
from .loader import ModelLoader, ModelLoadingError
from .embedding_manager import EmbeddingManager
from .view_aggregator import ViewAggregator
from .validator import InputValidator, ValidationError

__version__ = "1.0.0"
__author__ = "VARS Team"

__all__ = [
    # Main model class
    "MultiTaskMultiViewMViT",
    "create_multitask_multiview_mvit",  # Legacy compatibility
    
    # Configuration
    "ModelConfig",
    
    # Components
    "ModelLoader",
    "EmbeddingManager", 
    "ViewAggregator",
    "InputValidator",
    
    # Exceptions
    "ModelLoadingError",
    "ValidationError",
] 
"""
Freezing strategies and utilities for Multi-Task Multi-View ResNet3D training.

This module provides various parameter freezing strategies ranging from simple
fixed approaches to advanced gradient-guided and multi-metric strategies.
"""

from .base_utils import (
    calculate_class_weights,
    freeze_backbone,
    unfreeze_backbone_gradually,
    setup_discriminative_optimizer,
    get_phase_info,
    log_trainable_parameters
)

from .smart_manager import SmartFreezingManager

from .gradient_guided_manager import GradientGuidedFreezingManager

from .advanced_manager import AdvancedFreezingManager

from .model_setup import (
    create_model,
    setup_freezing_strategy
)

__all__ = [
    # Base utilities
    'calculate_class_weights',
    'freeze_backbone',
    'unfreeze_backbone_gradually',
    'setup_discriminative_optimizer',
    'get_phase_info',
    'log_trainable_parameters',
    
    # Freezing managers
    'SmartFreezingManager',
    'GradientGuidedFreezingManager',
    'AdvancedFreezingManager',
    
    # Model setup
    'create_model',
    'setup_freezing_strategy'
] 
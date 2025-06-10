"""
Model utilities, freezing strategies, and parameter management for Multi-Task Multi-View ResNet3D training.

This module now serves as a compatibility layer that imports from the new modular freezing package.
The original functionality has been split into multiple focused modules within the freezing package.
"""

# Import all freezing functionality from the new modular structure
from .freezing import (
    # Base utilities
    calculate_class_weights,
    freeze_backbone,
    unfreeze_backbone_gradually,
    setup_discriminative_optimizer,
    get_phase_info,
    log_trainable_parameters,
    
    # Freezing managers
    SmartFreezingManager,
    GradientGuidedFreezingManager,
    AdvancedFreezingManager,
    
    # Model setup
    create_model,
    setup_freezing_strategy
)

# Re-export everything for backward compatibility
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
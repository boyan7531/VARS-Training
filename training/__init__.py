"""
Training package for Multi-Task Multi-View ResNet3D.

This package provides a modular training system with the following components:
- config: Configuration and argument parsing
- data: Data loading, augmentation, and dataset utilities  
- model_utils: Model creation, freezing strategies, and parameter management
- training_utils: Training loops, validation, and loss functions
- checkpoint_utils: Checkpoint saving/loading and logging utilities
- train: Main training script that orchestrates everything
"""

from .config import parse_args, log_configuration_summary
from .data import create_datasets, create_dataloaders, create_gpu_augmentation
from .model_utils import create_model, setup_freezing_strategy, SmartFreezingManager
from .training_utils import train_one_epoch, validate_one_epoch, EarlyStopping
from .checkpoint_utils import save_checkpoint, load_checkpoint, save_training_history

__version__ = "1.0.0"
__author__ = "VARS Training Team"

__all__ = [
    # Configuration
    'parse_args',
    'log_configuration_summary',
    
    # Data utilities  
    'create_datasets',
    'create_dataloaders', 
    'create_gpu_augmentation',
    
    # Model utilities
    'create_model',
    'setup_freezing_strategy',
    'SmartFreezingManager',
    
    # Training utilities
    'train_one_epoch',
    'validate_one_epoch', 
    'EarlyStopping',
    
    # Checkpoint utilities
    'save_checkpoint',
    'load_checkpoint',
    'save_training_history',
] 
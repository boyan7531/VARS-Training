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
from .model_utils import create_model, setup_freezing_strategy, SmartFreezingManager, AdvancedFreezingManager
from .training_utils import train_one_epoch, validate_one_epoch, EarlyStopping
from .checkpoint_utils import save_checkpoint, load_checkpoint, save_training_history

# Import optimization and error recovery if available
try:
    from .data_optimization import DataLoadingProfiler, IntelligentBatchSizer, create_optimized_dataloader
    from .error_recovery import OOMRecoveryManager, ConfigValidator, RobustTrainingWrapper
    _OPTIMIZATION_AVAILABLE = True
except ImportError:
    _OPTIMIZATION_AVAILABLE = False

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
    'AdvancedFreezingManager',
    
    # Training utilities
    'train_one_epoch',
    'validate_one_epoch', 
    'EarlyStopping',
    
    # Checkpoint utilities
    'save_checkpoint',
    'load_checkpoint',
    'save_training_history',
]

# Add optimization and error recovery to exports if available
if _OPTIMIZATION_AVAILABLE:
    __all__.extend([
        # Data optimization
        'DataLoadingProfiler',
        'IntelligentBatchSizer', 
        'create_optimized_dataloader',
        
        # Error recovery
        'OOMRecoveryManager',
        'ConfigValidator',
        'RobustTrainingWrapper',
    ]) 
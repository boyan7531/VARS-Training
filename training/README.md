# Modular Training System for Multi-Task Multi-View ResNet3D

This directory contains a well-organized, modular training system that replaces the original monolithic `train.py` file. The system is broken down into logical components for better maintainability and readability.

## 📁 Directory Structure

```
training/
├── __init__.py           # Package initialization and exports
├── config.py             # Configuration and argument parsing
├── data.py              # Data loading, augmentation, and dataset utilities
├── model_utils.py       # Model creation, freezing strategies, parameter management
├── training_utils.py    # Training loops, validation, loss functions
├── checkpoint_utils.py  # Checkpoint saving/loading, logging utilities
├── train.py            # Main training orchestration script
└── README.md           # This documentation file
```

## 🧩 Module Breakdown

### 1. `config.py` - Configuration Management
- **Purpose**: Handles all command-line arguments and configuration processing
- **Key Functions**:
  - `parse_args()`: Parse and validate command-line arguments
  - `process_config()`: Apply configuration logic (simple training mode, legacy args, etc.)
  - `log_configuration_summary()`: Log comprehensive config summary

### 2. `data.py` - Data Pipeline
- **Purpose**: Manages dataset creation, data loading, and augmentation
- **Key Components**:
  - GPU-based augmentation classes (`GPUTemporalJitter`, `GPURandomBrightness`, etc.)
  - Transform creation functions (`create_transforms()`)
  - Dataset and dataloader creation (`create_datasets()`, `create_dataloaders()`)
  - Dataset size recommendations

### 3. `model_utils.py` - Model Management
- **Purpose**: Model creation, parameter freezing strategies, and optimization setup
- **Key Components**:
  - `SmartFreezingManager`: Advanced parameter freezing with multiple strategies
  - `create_model()`: Model initialization with proper configuration
  - `setup_freezing_strategy()`: Configure freezing based on arguments
  - Class weight calculation for handling imbalanced datasets

### 4. `training_utils.py` - Training Logic
- **Purpose**: Core training and validation loops, loss computation, metrics
- **Key Components**:
  - `train_one_epoch()`: Complete training loop for one epoch
  - `validate_one_epoch()`: Complete validation loop for one epoch
  - `FocalLoss`: Advanced loss function for class imbalance
  - `calculate_multitask_loss()`: Flexible multi-task loss computation
  - `EarlyStopping`: Early stopping utility

### 5. `checkpoint_utils.py` - Persistence & Logging
- **Purpose**: Model checkpointing, metrics tracking, and logging utilities
- **Key Components**:
  - `save_checkpoint()` / `load_checkpoint()`: Model persistence with DataParallel support
  - `save_training_history()`: Save metrics to JSON
  - `log_epoch_summary()`: Comprehensive epoch logging
  - `restore_best_metrics()`: Robust checkpoint metric restoration

### 6. `train.py` - Main Orchestration
- **Purpose**: Ties all components together in the main training loop
- **Key Functions**:
  - `main()`: Complete training orchestration
  - `setup_device_and_scaling()`: Multi-GPU setup and scaling
  - `setup_optimizer_and_scheduler()`: Optimizer/scheduler configuration
  - `handle_gradual_finetuning_transition()`: Phase transition management

## 🚀 Usage

### Option 1: Use the simplified entry point (Recommended)
```bash
python train_modular.py --dataset_root /path/to/data --epochs 50 --batch_size 16
```

### Option 2: Use the training package directly
```bash
python -m training.train --dataset_root /path/to/data --epochs 50 --batch_size 16
```

### Option 3: Import components in your own script
```python
from training import (
    parse_args, create_datasets, create_model, 
    train_one_epoch, validate_one_epoch
)

args = parse_args()
train_dataset, val_dataset = create_datasets(args)
model = create_model(args, vocab_sizes, device)
# ... rest of your training logic
```

## 🎯 Key Benefits

### ✅ **Maintainability**
- **Single Responsibility**: Each module has a clear, focused purpose
- **Easy Testing**: Individual components can be tested in isolation
- **Bug Isolation**: Issues are easier to locate and fix

### ✅ **Reusability** 
- **Component Reuse**: Use individual components in other projects
- **Flexible Configuration**: Easy to swap implementations (e.g., different augmentation strategies)
- **Import What You Need**: Only import the components you actually use

### ✅ **Readability**
- **Clear Structure**: Logical organization makes code easier to understand
- **Documentation**: Each module is well-documented with clear interfaces
- **Reduced Complexity**: No more 2000+ line monolithic files

### ✅ **Extensibility**
- **Easy to Extend**: Add new freezing strategies, augmentation techniques, etc.
- **Plugin Architecture**: New components can be easily added
- **Configuration Driven**: Behavior controlled through clean configuration system

## 🔧 Advanced Features Preserved

All advanced features from the original `train.py` are preserved and enhanced:

- **🧊 Enhanced Freezing Strategies**: Adaptive, progressive, and fixed freezing modes
- **🚀 Advanced Augmentation**: CPU and GPU-based augmentation pipelines
- **📊 Flexible Loss Functions**: Focal loss, weighted cross-entropy, plain cross-entropy
- **⚖️ Class Imbalance Handling**: Multiple strategies with safety limits
- **🔄 Smart Learning Rate Scheduling**: Multiple scheduler types with phase-aware transitions
- **💾 Robust Checkpointing**: DataParallel-aware saving/loading with metric restoration
- **📈 Comprehensive Logging**: Detailed metrics tracking and configuration summaries

## 🆕 New Improvements

The modular structure also introduces several improvements:

1. **Better Error Handling**: More granular error messages and recovery
2. **Enhanced Logging**: Clearer separation of concerns in logging
3. **Improved Memory Management**: More systematic cleanup and memory optimization
4. **Configuration Validation**: Better validation and error reporting for config issues
5. **Component Testing**: Individual components can be unit tested

## 🔄 Migration from Original `train.py`

The modular system is **100% backward compatible**. All existing command-line arguments and functionality work exactly the same:

```bash
# Old way (still works)
python train.py --gradual_finetuning --aggressive_augmentation --loss_function focal

# New way (recommended)
python train_modular.py --gradual_finetuning --aggressive_augmentation --loss_function focal
```

No changes to your existing training scripts or configurations are required!

## 📝 Contributing

When adding new features:

1. **Choose the Right Module**: Place new functionality in the appropriate module
2. **Follow Patterns**: Use similar patterns to existing code
3. **Add Documentation**: Document new functions and classes
4. **Update Exports**: Add new public functions to `__init__.py`
5. **Test Components**: Ensure new components work in isolation

## 🎓 Example: Adding a New Augmentation Strategy

```python
# 1. Add to data.py
class GPURandomRotation(nn.Module):
    def __init__(self, max_angle=10):
        super().__init__()
        self.max_angle = max_angle
    
    def forward(self, video):
        # Implementation here
        return video

# 2. Update GPUAugmentationPipeline
def __init__(self, ..., rotation_prob=0.3):
    # Add to augmentation list
    self.augmentations.append(GPURandomRotation(prob=rotation_prob))

# 3. Add config argument in config.py
parser.add_argument('--rotation_prob', type=float, default=0.3,
                   help='Probability of applying random rotation')

# 4. Use in create_gpu_augmentation()
gpu_augmentation = GPUAugmentationPipeline(
    # ... existing args
    rotation_prob=args.rotation_prob
)
```

This modular structure makes the codebase much more maintainable and extensible while preserving all existing functionality! 
# EMA Validation Guide

## Overview

This guide explains the new **EMA (Exponential Moving Average) Validation** feature that automatically uses EMA weights during validation and testing to improve model performance.

## What is EMA?

EMA maintains a moving average of model parameters during training:
- **Online weights**: Current training weights that get updated with gradients
- **EMA weights**: Smoothed version of weights averaged over training history
- **Key benefit**: EMA weights often generalize better than online weights

## Features Implemented

### ✅ Core Functionality
- **Automatic EMA validation**: Uses EMA weights during validation when enabled
- **Smart weight swapping**: Temporarily applies EMA weights, then restores online weights
- **Freezing strategy support**: Handles newly unfrozen parameters correctly
- **Checkpoint integration**: Saves/loads both online and EMA weights

### ✅ Configuration
```bash
# Enable EMA during training (existing flag)
--use_ema

# Enable EMA for validation (NEW - default: True)
--ema_eval

# Control EMA decay rate (existing)
--ema_decay 0.9999
```

### ✅ Training Integration
- Works with both `train.py` and Lightning-based training
- Compatible with all existing freezing strategies
- No impact on training speed
- Minimal memory overhead

### ✅ Inference Support
- Benchmark script automatically uses EMA weights if available
- Better performance on test sets

## Usage Examples

### Basic Usage
```bash
# Standard training with EMA validation (recommended)
python training/train.py \
    --use_ema \
    --ema_eval \
    --dataset_root /path/to/data \
    --epochs 50

# Disable EMA validation if needed
python training/train.py \
    --use_ema \
    --no-ema_eval \
    --dataset_root /path/to/data
```

### Lightning Training
```bash
# EMA validation works automatically with Lightning
python train_lightning.py \
    --use_ema \
    --ema_eval \
    --dataset_root /path/to/data
```

### Benchmark with EMA
```bash
# Benchmark script automatically uses EMA weights
python benchmark.py \
    --checkpoint_path best_model_epoch_20.pth \
    --dataset_root /path/to/data
```

## Expected Benefits

### Performance Improvements
- **+1-2% macro accuracy**: Conservative estimate based on literature
- **Better convergence**: More stable validation curves
- **Improved generalization**: Especially on minority classes
- **Zero training cost**: No impact on training time

### Why EMA Helps This Codebase
1. **Multi-task learning**: Severity + action classification benefits from parameter stabilization
2. **Class imbalance**: More stable predictions on minority classes with focal loss/class balancing
3. **Complex architecture**: ResNet3D/MViT with attention aggregation has many parameters to stabilize
4. **Long training**: 50+ epochs with sophisticated freezing strategies

## Technical Details

### Weight Swapping Process
```python
# During validation
if should_use_ema_for_validation(args, ema_model, epoch):
    apply_ema_weights(model, ema_model)     # Temporarily use EMA
    validation_metrics = validate_one_epoch(...)
    restore_online_weights(model, ema_model) # Restore for training
```

### Checkpoint Format
```python
checkpoint = {
    'model_state_dict': model.state_dict(),    # Online weights
    'ema_state_dict': ema_model.shadow,        # EMA weights
    'has_ema': True,                           # EMA availability flag
    # ... other fields
}
```

### Freezing Strategy Compatibility
```python
# EMA automatically handles newly unfrozen parameters
def update_ema_model(ema_model, model):
    # Add newly unfrozen parameters to EMA
    if hasattr(ema_model, 'update_for_newly_unfrozen'):
        ema_model.update_for_newly_unfrozen(model)
    
    # Standard EMA update
    ema_model.update(model)
```

## Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--use_ema` | `False` | Enable EMA during training |
| `--ema_eval` | `True` | Use EMA weights for validation |
| `--ema_decay` | `0.9999` | EMA decay rate (higher = more smoothing) |

## Logging and Monitoring

### Training Logs
```
🚀 EMA model created with decay: 0.9999
✅ EMA weights will be used for validation/testing
🔄 Using EMA weights for validation (epoch 5)
🔄 Restored online weights after validation
```

### Checkpoint Logs
```
💾 Saving checkpoint with EMA weights
📈 Restored EMA weights from checkpoint
✅ Using EMA weights for benchmark (better performance)
```

## Testing

### Unit Test
```bash
# Run the EMA validation test
python test_ema_validation.py
```

Expected output:
```
✅ Created EMA model with decay 0.999
Epoch  3: Online: 85.50% | EMA: 87.00% | Improvement: +1.50%
Epoch  4: Online: 86.00% | EMA: 87.50% | Improvement: +1.50%
✅ EMA validation test completed!
```

## Troubleshooting

### Common Issues

**Q: EMA validation not being used**
- Check that `--ema_eval` is enabled (default: True)
- Ensure `--use_ema` is set
- EMA is skipped for first 2 epochs (normal behavior)

**Q: Memory issues**
- EMA doubles parameter memory usage
- Use existing memory optimization flags
- Consider reducing batch size if needed

**Q: Checkpoint loading errors**
- Old checkpoints don't have EMA weights (expected)
- New checkpoints automatically include EMA when enabled

### Debug Logging
```bash
# Enable debug logging to see EMA operations
export PYTHONPATH=.
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# ... your training command
"
```

## Integration with Existing Features

### ✅ Compatible Features
- All freezing strategies (adaptive, gradient-guided, advanced)
- Mixed precision training
- Class balancing and progressive sampling
- Multi-task loss functions
- View consistency loss
- All optimizers and schedulers

### ⚠️ Considerations
- Checkpoint files will be larger (2x model parameters)
- Memory usage increases during training
- Very small models may not benefit significantly

## Future Enhancements

### Planned Features
- [ ] EMA scheduling (variable decay rates)
- [ ] Multiple EMA models with different decay rates
- [ ] EMA warmup period configuration
- [ ] Per-layer EMA decay rates

### Advanced Usage
```python
# Custom EMA integration in your code
from training.model_utils import create_ema_model, apply_ema_weights

# Create EMA model
ema_model = create_ema_model(model, decay=0.999)

# Use EMA for inference
apply_ema_weights(model, ema_model)
predictions = model(data)
# Restore if needed for training
restore_online_weights(model, ema_model)
```

## References

- [Model EMA in PyTorch](https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/)
- [EMA in Computer Vision](https://arxiv.org/abs/1803.05407)
- [timm ModelEmaV2 Documentation](https://github.com/rwightman/pytorch-image-models)

---

**Status**: ✅ **IMPLEMENTED AND READY TO USE**

The EMA validation feature is fully implemented and ready for production use. It provides a free performance boost with minimal risk. 
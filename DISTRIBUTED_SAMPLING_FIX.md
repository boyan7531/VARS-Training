# Distributed Sampling Fix

## Problem Identified

**Issue**: Custom samplers in distributed training caused data duplication across GPUs, inflating the effective epoch size by × number of GPUs.

**Root Cause**: Custom samplers (`ClassBalancedSampler`, `ActionBalancedSampler`, `AlternatingSampler`, `ProgressiveClassBalancedSampler`) were passed unchanged into DDP runs. Every GPU therefore saw the SAME oversampled indices, causing:

1. **Data Duplication**: Each GPU processed identical samples instead of unique subsets
2. **Inflated Epoch Size**: Effective epoch size became `base_epoch_size × num_gpus`
3. **Training Inefficiency**: GPUs were processing redundant data
4. **Incorrect Metrics**: Loss and accuracy calculations were skewed due to repeated samples

## Solution Implemented

### 1. Distributed-Aware Samplers

Created native distributed versions of custom samplers:

- `DistributedClassBalancedSampler`: Distributed-aware severity class balancing
- `DistributedActionBalancedSampler`: Distributed-aware action type balancing

These samplers:
- Split data across GPUs properly
- Maintain class balancing on each GPU
- Keep consistent epoch sizes regardless of GPU count
- Use deterministic random seeds for reproducibility

### 2. DistributedSamplerWrapper

For complex samplers (like `ProgressiveClassBalancedSampler` and `AlternatingSampler`), implemented a wrapper that:
- Takes any custom sampler and makes it DDP-compatible
- Ensures each GPU sees different data
- Preserves the original sampling logic
- Handles epoch synchronization via `set_epoch()`

### 3. Automatic Detection and Fallback

Updated `create_dataloaders()` to:
- Automatically detect distributed training mode
- Use distributed-aware samplers when DDP is active
- Fall back to original samplers for single-GPU training
- Apply proper distributed sampling to validation data

## Code Changes

### training/data.py

1. **Added imports**:
   ```python
   import torch.distributed as dist
   from torch.utils.data.distributed import DistributedSampler
   ```

2. **Added distributed sampler classes**:
   - `DistributedSamplerWrapper`
   - `DistributedClassBalancedSampler` 
   - `DistributedActionBalancedSampler`

3. **Updated `create_dataloaders()`**:
   - Detects distributed training with `dist.is_available() and dist.is_initialized()`
   - Uses appropriate distributed samplers when needed
   - Maintains backward compatibility for single-GPU training

## Benefits

✅ **Fixed Data Duplication**: Each GPU processes unique data subsets  
✅ **Consistent Epoch Size**: Total samples processed per epoch remains constant  
✅ **Preserved Class Balancing**: Minority class oversampling still works correctly  
✅ **Maintained Reproducibility**: Deterministic sampling with proper seed handling  
✅ **Automatic Detection**: No configuration changes needed  
✅ **Backward Compatible**: Works seamlessly with single-GPU training  

## Verification

The fix was tested to ensure:

1. **Multi-GPU Scenarios**: Different GPUs receive different data
2. **Single-GPU Fallback**: Original behavior preserved
3. **Epoch Consistency**: Sample counts remain stable across epochs
4. **Class Balancing**: Minority classes still properly oversampled

## Usage

No changes required in training scripts. The fix is automatically applied when:

- Using PyTorch Lightning with DDPStrategy (already configured)
- Running with multiple GPUs
- Using any custom sampler (class balanced, action balanced, etc.)

## Technical Details

### Before Fix
```
GPU 0: [1, 5, 2, 8, 3, 7, 1, 5, 2, 8] # Same indices
GPU 1: [1, 5, 2, 8, 3, 7, 1, 5, 2, 8] # Same indices (DUPLICATE!)
Total effective samples: 20 (10×2 GPUs)
```

### After Fix  
```
GPU 0: [1, 5, 2, 8, 3]  # Unique subset
GPU 1: [7, 1, 5, 2, 8]  # Different subset  
Total effective samples: 10 (no duplication)
```

## Migration Guide

No action required! The fix is:
- **Automatic**: Detected at runtime
- **Transparent**: No API changes
- **Compatible**: Works with existing configurations

Your existing training scripts will automatically benefit from this fix when using multiple GPUs. 
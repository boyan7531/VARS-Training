# NaN Origin Trace Plan Implementation

This document describes the comprehensive NaN detection system implemented to identify exactly where NaN values first appear in the training pipeline.

## Overview

The implementation follows a systematic approach to catch NaN values at the **first transition** where they appear, providing unambiguous identification of the culprit component. All probes are designed to be lightweight and will cause early termination with clear error messages.

## Implementation Details

### 1. Raw Video Loading Detection (`dataset.py`)

**Location**: `SoccerNetMVFoulDataset._get_video_clip()`

```python
# [NaN-origin] Step 1: Check raw frames right after tensor creation
if torch.isnan(clip).any():
    logger.error(f"[NaN-origin] raw frames NaN – video {video_path_str}")
    raise RuntimeError("NaN in raw video")
```

**Purpose**: Detects if NaN values come from corrupted video files on disk.

### 2. CPU Transformation Pipeline Detection (`dataset.py`)

**Location**: `SoccerNetMVFoulDataset.__getitem__()`

```python
# [NaN-origin] Step 2: Identify which CPU-side augmentation inserts NaNs
for t_idx, t in enumerate(self.transform.transforms):
    clip_before = clip.clone()
    clip = t(clip)
    if torch.isnan(clip).any() and not torch.isnan(clip_before).any():
        logger.error(f"[NaN-origin] CPU transform {t_idx}:{t.__class__.__name__}")
        raise RuntimeError("NaN introduced")
```

**Purpose**: Identifies which specific CPU augmentation transform introduces NaN values.

### 3. SeverityAwareAugmentation Detection (`dataset.py`)

**Location**: `SeverityAwareAugmentation.forward()`

```python
# [NaN-origin] Step 3: Check SeverityAwareAugmentation separately
clip_before = clip.clone()
# ... apply augmentation ...
if torch.isnan(clip_result).any() and not torch.isnan(clip_before).any():
    logger.error(f"[NaN-origin] SeverityAwareAugmentation augmentation pipeline for severity {self.severity_label}")
    raise RuntimeError("NaN introduced by severity-aware augmentation")
```

**Purpose**: Specifically monitors severity-aware augmentation that can overflow due to aggressive brightness/contrast mixing.

### 4. Collate Function Detection (`dataset.py`)

**Location**: `variable_views_collate_fn()`

```python
# [NaN-origin] Step 4: Verify collate & padding step
if torch.isnan(padded_clips).any():
    logger.error(f"[NaN-origin] collate_fn padding step")
    raise RuntimeError("NaN introduced in collate_fn padding step")
```

**Purpose**: Catches NaN values introduced during batch collation and padding operations.

### 5. GPU Augmentation Detection (`training/data.py`)

**Location**: Multiple GPU augmentation classes

```python
# [NaN-origin] Step 5: GPU augmentations before/after pattern
for aug_idx, aug in enumerate(self.augmentations):
    video_before = video.clone()
    video = aug(video)
    if torch.isnan(video).any() and not torch.isnan(video_before).any():
        logger.error(f"[NaN-origin] GPU augmentation {aug_idx}:{aug.__class__.__name__}")
        raise RuntimeError("NaN introduced by GPU augmentation")
```

**Purpose**: Monitors GPU-based augmentations including Kornia transformations.

### 6. Lightning Module Safety Net (`training/lightning_module.py`)

**Location**: `MultiTaskVideoLightningModule.training_step()`

```python
# [NaN-origin] Step 6: Safety net at Lightning level
if torch.isnan(batch["clips"]).any():
    self.log("batch_has_nan", True)
    logger.error(f"[NaN-origin] NaN reached the model in batch {batch_idx}")
    raise RuntimeError("NaN reached the model")
```

**Purpose**: Final safety check that stops training immediately if NaN values leak past the dataloader.

## Usage Instructions

### 1. Standard Training with NaN Detection

Run training normally. If NaN values are detected, the training will stop with a clear error message indicating the exact location:

```bash
python train_lightning.py --config conf/config.yaml
```

### 2. Focused Debugging

Use the debug script for single-sample testing:

```bash
python debug_nan_trace.py --sample-idx 0 --dataset-path mvfouls
```

### 3. Single-Process Debugging

For easier debugging, use `num_workers=0`:

```bash
python train_lightning.py --config conf/config.yaml --num-workers 0
```

## Error Message Format

All NaN detection points use the consistent format:

```
[NaN-origin] <location_description>
```

Examples:
- `[NaN-origin] raw frames NaN – video /path/to/video`
- `[NaN-origin] CPU transform 3:RandomBrightnessContrast`
- `[NaN-origin] SeverityAwareAugmentation augmentation pipeline for severity 4`
- `[NaN-origin] collate_fn padding step`
- `[NaN-origin] GPU augmentation 2:GPURandomNoise`
- `[NaN-origin] NaN reached the model`

## Debugging Workflow

1. **Enable single-process mode**: Set `num_workers=0` to avoid multiprocessing complexity
2. **Run until crash**: When it crashes, read the last `[NaN-origin]` message
3. **Identify the culprit**: The message tells you the precise transform or step
4. **Fix the component**: Add clamping, re-normalization, or fix the identified component
5. **Repeat**: Continue until all detection points pass
6. **Re-enable multiprocessing**: Set `num_workers` back to desired value

## Key Features

- **Early termination**: Stops at the first NaN occurrence rather than continuing
- **Lightweight**: Minimal performance impact, only adds clone() and isnan() checks
- **Comprehensive**: Covers the entire pipeline from disk to model input
- **Unambiguous**: Each check point has a unique identifier
- **Non-intrusive**: Doesn't change model logic, only adds monitoring

## Files Modified

- `dataset.py`: Added logging import and NaN detection in multiple locations
- `training/data.py`: Added NaN detection in GPU augmentation pipelines
- `training/lightning_module.py`: Added safety net in training step
- `debug_nan_trace.py`: New debug script for single-sample testing
- `NAN_TRACE_IMPLEMENTATION.md`: This documentation

## Expected Outcomes

When you run the system with this implementation:

1. **If no NaNs**: Training proceeds normally with no additional output
2. **If NaNs present**: Training stops immediately with a clear error message indicating:
   - Exact location where NaN first appeared
   - Specific transform or component responsible
   - Context information (video path, batch index, etc.)

This enables rapid identification and fixing of NaN sources without guesswork or extensive debugging sessions. 
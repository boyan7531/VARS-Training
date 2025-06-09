# Multi-Task Multi-View ResNet3D Training Arguments

This document explains all available command-line arguments for training the Multi-Task Multi-View ResNet3D model for foul recognition.

## Quick Start

### Recommended Training Command
```bash
python train_modular.py \
  --dataset_root . \
  --epochs 50 \
  --batch_size 16 \
  --lr 2e-4 \
  --backbone_name r2plus1d_18 \
  --freezing_strategy advanced \
  --loss_function focal \
  --use_class_balanced_sampler \
  --progressive_class_balancing \
  --aggressive_augmentation \
  --gpu_augmentation \
  --severity_aware_augmentation \
  --mixed_precision \
  --scheduler cosine \
  --label_smoothing 0.1 \
  --num_workers 12 \
  --save_dir checkpoints
```

### Test Command (for verification)
```bash
python train_modular.py \
  --dataset_root . \
  --test_run \
  --test_batches 3 \
  --batch_size 2 \
  --epochs 2 \
  --num_workers 0 \
  --frames_per_clip 8 \
  --img_height 112 \
  --img_width 200 \
  --emergency_unfreeze_epoch 1 \
  --freezing_strategy advanced \
  --save_dir test_checkpoints
```

---

## Basic Training Configuration

### Core Settings
- **`--dataset_root`** (str, default="")  
  Root directory containing the mvfouls folder. **Required argument.**

- **`--epochs`** (int, default=50)  
  Number of training epochs. Typical range: 30-100 depending on dataset size.

- **`--batch_size`** (int, default=8)  
  Batch size for training and validation. Adjust based on GPU memory (4-16 typical).

- **`--lr`** (float, default=2e-4)  
  Learning rate. 2e-4 works well for most cases. For larger datasets, try 1e-4.

- **`--weight_decay`** (float, default=1e-4)  
  Weight decay for AdamW optimizer. Helps prevent overfitting.

- **`--seed`** (int, default=42)  
  Random seed for reproducibility. Keep consistent across experiments.

### Data Loading
- **`--num_workers`** (int, default=4)  
  Number of workers for DataLoader. Set to 0 for debugging, 4-8 for performance.

- **`--worker_timeout`** (int, default=0)  
  Timeout for DataLoader workers in seconds. 0 = no timeout.

- **`--prefetch_factor`** (int, default=2)  
  Number of batches loaded in advance by each worker. Higher = more memory usage.

- **`--pin_memory`** (flag)  
  Pin memory for DataLoader. Speeds up CPU->GPU transfers.

---

## Model Configuration

### Architecture
- **`--backbone_name`** (str, default='r2plus1d_18')  
  ResNet3D backbone variant. Choices: resnet3d_18, mc3_18, r2plus1d_18, resnet3d_50.  
  **r2plus1d_18 recommended** for best accuracy/speed balance.

- **`--frames_per_clip`** (int, default=16)  
  Number of frames per video clip. 16 frames captures ~1 second at 15 FPS.

- **`--target_fps`** (int, default=15)  
  Target FPS for clips. 15 FPS balances temporal resolution with processing speed.

### Video Processing
- **`--start_frame`** (int, default=67)  
  Start frame index for foul-centered extraction (8 frames before foul at frame 75).

- **`--end_frame`** (int, default=82)  
  End frame index for foul-centered extraction (7 frames after foul at frame 75).

- **`--img_height`** (int, default=224)  
  Target image height. 224 is standard for most vision models.

- **`--img_width`** (int, default=398)  
  Target image width. 398 matches original VARS paper aspect ratio.

- **`--max_views`** (int, default=None)  
  Optional limit on max views per action. None = use all available views.

- **`--attention_aggregation`** (flag, default=True)  
  Use attention mechanism for view aggregation instead of simple averaging.

---

## Training Optimization

### Optimizer Settings
- **`--optimizer`** (str, default='adamw')  
  Optimizer type. Choices: adamw, sgd, adam. AdamW recommended for stability.

- **`--momentum`** (float, default=0.9)  
  SGD momentum. Only used when optimizer=sgd.

- **`--mixed_precision`** (flag)  
  Use mixed precision training. **Recommended** for faster training and lower memory.

- **`--gradient_clip_norm`** (float, default=1.0)  
  Gradient clipping norm. Prevents exploding gradients.

- **`--early_stopping_patience`** (int, default=None)  
  Stop training after N epochs without improvement. Auto-set based on training mode.

---

## Learning Rate Scheduling

### Scheduler Types
- **`--scheduler`** (str, default='cosine')  
  Learning rate scheduler. Choices: cosine, onecycle, step, exponential, reduce_on_plateau, none.  
  **cosine recommended** for smooth convergence.

- **`--scheduler_warmup_epochs`** (int, default=5)  
  Number of warmup epochs for scheduler. Gradual learning rate increase at start.

### Scheduler Parameters
- **`--step_size`** (int, default=10)  
  Step size for StepLR scheduler. LR reduced every N epochs.

- **`--gamma`** (float, default=0.5)  
  Decay factor for StepLR/ExponentialLR. LR *= gamma every step.

- **`--plateau_patience`** (int, default=5)  
  Patience for ReduceLROnPlateau. Wait N epochs before reducing LR.

- **`--min_lr`** (float, default=1e-6)  
  Minimum learning rate. Prevents LR from becoming too small.

---

## Multi-Task Loss Configuration

### Task Weights
- **`--main_task_weights`** (float list, default=[3.0, 3.0])  
  Loss weights for [severity, action_type] tasks. Higher = more importance.

- **`--auxiliary_task_weight`** (float, default=0.5)  
  Loss weight for auxiliary tasks (currently not implemented).

- **`--label_smoothing`** (float, default=0.1)  
  Label smoothing factor. **0.1 recommended** for small datasets to prevent overfitting.

### Loss Function Types
- **`--loss_function`** (str, default='weighted')  
  Loss function type:
  - **focal**: FocalLoss (good for class imbalance)
  - **weighted**: Weighted CrossEntropyLoss (balanced approach)
  - **plain**: Standard CrossEntropyLoss (no class balancing)

- **`--focal_gamma`** (float, default=2.0)  
  Focal Loss gamma parameter. Higher = more focus on hard examples.

---

## Class Imbalance Handling

### Class Weighting
- **`--class_weighting_strategy`** (str, default='balanced_capped')  
  Strategy for calculating class weights:
  - **balanced_capped**: Balanced weights with maximum ratio cap (recommended)
  - **sqrt**: Square root of inverse frequency
  - **log**: Logarithmic weighting
  - **effective_number**: Effective number of samples
  - **none**: No class weighting

- **`--max_weight_ratio`** (float, default=10.0)  
  Maximum ratio between highest and lowest class weight. Prevents training instability.

### Sampling Strategies
- **`--use_class_balanced_sampler`** (flag, default=True)  
  Use class-balanced sampler to oversample minority classes. **Recommended.**

- **`--oversample_factor`** (float, default=4.0)  
  Factor by which to oversample minority classes. Higher = more aggressive balancing.

### Progressive Balancing
- **`--progressive_class_balancing`** (flag, default=False)  
  Enable progressive class-balanced sampling. Gradually increases minority representation.

- **`--progressive_start_factor`** (float, default=1.5)  
  Starting balancing factor for progressive sampling.

- **`--progressive_end_factor`** (float, default=3.0)  
  Ending balancing factor for progressive sampling.

- **`--progressive_epochs`** (int, default=15)  
  Number of epochs for progressive sampling transition.

---

## Freezing Strategies

### Emergency Unfreezing
- **`--emergency_unfreeze_epoch`** (int, default=4)  
  Epoch to force unfreeze if no layers unfrozen yet. Prevents getting stuck.

- **`--min_unfreeze_layers`** (int, default=1)  
  Minimum number of backbone layers to unfreeze.

- **`--emergency_unfreeze_gradual`** (flag, default=True)  
  Use gradual emergency unfreezing (sub-blocks instead of full layers).

- **`--validation_plateau_patience`** (int, default=2)  
  Epochs to wait before treating validation performance as plateaued.

### Freezing Strategy Types
- **`--freezing_strategy`** (str, default='fixed')  
  Strategy for parameter freezing:
  - **none**: No freezing, train all parameters
  - **fixed**: Timed phases (works with gradual_finetuning)
  - **adaptive**: Unfreeze based on validation plateau
  - **progressive**: Gradual unfreezing over time
  - **gradient_guided**: Unfreeze based on gradient importance
  - **advanced**: Most sophisticated strategy with rollback (recommended)

### Adaptive Freezing
- **`--adaptive_patience`** (int, default=3)  
  Epochs to wait before unfreezing next layer in adaptive mode.

- **`--adaptive_min_improvement`** (float, default=0.001)  
  Minimum validation improvement to reset patience counter.

### Gradient-Guided Freezing
- **`--importance_threshold`** (float, default=0.01)  
  Minimum gradient importance threshold for unfreezing layers.

- **`--warmup_epochs`** (int, default=3)  
  Number of epochs to warmup newly unfrozen layers with reduced LR.

- **`--unfreeze_patience`** (int, default=2)  
  Minimum epochs between layer unfreezing operations.

- **`--max_layers_per_step`** (int, default=1)  
  Maximum number of layers to unfreeze in a single step.

- **`--sampling_epochs`** (int, default=2)  
  Number of epochs to sample gradients before first unfreezing decision.

### Advanced Freezing
- **`--base_importance_threshold`** (float, default=0.002)  
  Base importance threshold for advanced freezing (adapts dynamically).

- **`--performance_threshold`** (float, default=0.001)  
  Minimum performance improvement threshold for unfreezing decisions.

- **`--rollback_patience`** (int, default=2)  
  Epochs to wait before performing rollback in advanced freezing.

- **`--gradient_momentum`** (float, default=0.9)  
  Momentum factor for smoothing gradient importance.

- **`--analysis_window`** (int, default=2)  
  Number of epochs to analyze before making unfreezing decisions.

- **`--enable_rollback`** (flag, default=True)  
  Enable performance-based rollback in advanced freezing.

- **`--enable_dependency_analysis`** (flag, default=True)  
  Enable layer dependency analysis in advanced freezing.

---

## Gradual Fine-Tuning (Legacy/Fixed Mode)

### Phase Configuration
- **`--gradual_finetuning`** (flag, default=True)  
  Enable gradual fine-tuning with discriminative learning rates.

- **`--phase1_epochs`** (int, default=8)  
  Number of epochs for Phase 1 (head-only training).

- **`--phase2_epochs`** (int, default=15)  
  Number of epochs for Phase 2 (gradual unfreezing).

### Learning Rates
- **`--head_lr`** (float, default=1e-3)  
  Learning rate for classification heads in Phase 1.

- **`--backbone_lr`** (float, default=1e-5)  
  Learning rate for unfrozen backbone layers in Phase 2.

- **`--unfreeze_blocks`** (int, default=3)  
  Number of final residual blocks to unfreeze in Phase 2.

### Phase 2 Adjustments
- **`--phase2_backbone_lr_scale_factor`** (float, default=1.0)  
  Scale factor for backbone LR in Phase 2.

- **`--phase2_head_lr_ratio`** (float, default=5.0)  
  Ratio for Phase 2 head LR relative to Phase 2 backbone LR.

### Phase 1 Plateau Settings
- **`--phase1_plateau_patience`** (int, default=3)  
  Patience for ReduceLROnPlateau scheduler in Phase 1.

- **`--phase1_plateau_factor`** (float, default=0.2)  
  Factor for ReduceLROnPlateau scheduler in Phase 1.

---

## Augmentation Configuration

### Augmentation Strength
- **`--augmentation_strength`** (str, default='aggressive')  
  Overall augmentation strength:
  - **none**: No augmentation
  - **mild**: Light augmentation
  - **moderate**: Standard augmentation
  - **aggressive**: Strong augmentation (recommended for small datasets)
  - **extreme**: Maximum augmentation (for very small datasets)

### Legacy Augmentation Flags
- **`--aggressive_augmentation`** (flag, default=True)  
  Enable aggressive augmentation pipeline. Use `--augmentation_strength` instead.

- **`--extreme_augmentation`** (flag, default=False)  
  Enable EXTREME augmentation. Use `--augmentation_strength extreme` instead.

### Specific Augmentation Parameters
- **`--temporal_jitter_strength`** (int, default=3)  
  Max temporal jitter in frames. Higher = more temporal variation.

- **`--dropout_prob`** (float, default=0.2)  
  Frame dropout probability. Higher = more aggressive temporal dropout.

- **`--spatial_crop_strength`** (float, default=0.7)  
  Minimum crop scale. Lower = more aggressive spatial crops.

- **`--color_aug_strength`** (float, default=0.3)  
  Color augmentation strength. Higher = more color variation.

- **`--noise_strength`** (float, default=0.06)  
  Maximum Gaussian noise standard deviation.

### GPU Augmentation
- **`--gpu_augmentation`** (flag, default=False)  
  Use GPU-based augmentation instead of CPU. **Recommended** for dual GPU setups.

- **`--severity_aware_augmentation`** (flag, default=False)  
  Use severity-aware augmentation with class-specific strengths. **Recommended.**

---

## Debugging & Flexibility Options

### Training Modes
- **`--simple_training`** (flag, default=False)  
  Enable simple training mode: disables class balancing, augmentation, uses plain loss.

- **`--disable_class_balancing`** (flag, default=False)  
  Disable all class balancing (both weights and sampler).

- **`--disable_augmentation`** (flag, default=False)  
  Disable all augmentation (both in-dataset and in-model).

- **`--disable_in_model_augmentation`** (flag, default=False)  
  Disable only GPU-based augmentation applied inside model forward pass.

---

## Checkpointing and Logging

### Save/Resume
- **`--save_dir`** (str, default='checkpoints')  
  Directory to save model checkpoints.

- **`--resume`** (str, default=None)  
  Path to checkpoint to resume from.

- **`--resume_best_acc`** (float, default=None)  
  Manually override best validation accuracy when resuming.

### Memory Management
- **`--memory_cleanup_interval`** (int, default=20)  
  Interval (batches) for memory cleanup. 0 or negative to disable.

---

## Data Loading Optimization

- **`--enable_data_optimization`** (flag, default=True)  
  Enable data loading optimization features.

- **`--disable_data_optimization`** (flag, default=False)  
  Disable data loading optimization features.

---

## Error Recovery

### OOM Recovery
- **`--enable_oom_recovery`** (flag, default=True)  
  Enable automatic Out-of-Memory recovery.

- **`--oom_reduction_factor`** (float, default=0.75)  
  Factor to reduce batch size on OOM.

- **`--min_batch_size`** (int, default=1)  
  Minimum batch size for OOM recovery.

### Configuration Validation
- **`--enable_config_validation`** (flag, default=True)  
  Enable configuration validation.

- **`--strict_config_validation`** (flag, default=False)  
  Treat config warnings as errors.

---

## Testing and Development

### Test Mode
- **`--test_run`** (flag)  
  Perform a quick test run (1 epoch, few batches, no saving).

- **`--test_batches`** (int, default=2)  
  Number of batches to run in test mode.

- **`--force_batch_size`** (flag)  
  Force specified batch size even with multi-GPU.

---

## Advanced Options

### Additional Features
- **`--multi_scale`** (flag)  
  Enable multi-scale training for better accuracy.

- **`--dropout_rate`** (float, default=0.1)  
  Dropout rate for regularization.

- **`--lr_warmup`** (flag)  
  Enable learning rate warmup.

- **`--discriminative_lr`** (flag, default=False)  
  Enable discriminative learning rates for different layers.

---

## Legacy Arguments (Deprecated)

- **`--use_focal_loss`** (flag) → Use `--loss_function focal`
- **`--use_class_weighted_loss`** (flag) → Use `--loss_function weighted`

---

## Training Recommendations by Dataset Size

### Very Small Dataset (<100 samples)
```bash
--augmentation_strength extreme \
--progressive_class_balancing \
--progressive_end_factor 6.0 \
--label_smoothing 0.15 \
--loss_function focal \
--severity_aware_augmentation
```

### Small Dataset (100-500 samples)
```bash
--augmentation_strength aggressive \
--progressive_class_balancing \
--progressive_end_factor 4.0 \
--label_smoothing 0.1 \
--loss_function focal
```

### Medium Dataset (500-1000 samples)
```bash
--augmentation_strength aggressive \
--use_class_balanced_sampler \
--oversample_factor 3.0 \
--label_smoothing 0.1
```

### Large Dataset (>1000 samples)
```bash
--augmentation_strength moderate \
--use_class_balanced_sampler \
--oversample_factor 2.0 \
--label_smoothing 0.05
```

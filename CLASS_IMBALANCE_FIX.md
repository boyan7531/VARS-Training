# Class Imbalance Fix Implementation

## Problem Identified

The model was severely biased towards predicting severity 5 (75.1% of predictions) and action "High leg" due to **double imbalance correction**:

1. **Progressive Oversampling**: Boosted minority classes (severity 4&5) by 3x 
2. **In-Model Augmentation**: Applied 6-8x stronger augmentation to same minority classes
3. **Hardcoded Majority Class**: Sampler assumed severity "1.0" was majority, but actual majority is severity 1 (1402 samples)

### Actual Class Distribution
```
Severity: {1: 1402, 3: 687, 2: 403, 0: 353, 4: 44, 5: 27}
Action:   {8: 1264, 9: 448, 6: 361, 1: 383, 4: 178, 5: 103, 7: 88, 3: 52, 2: 28, 0: 11}
```

## Fixes Applied

### 1. Dynamic Majority Class Detection
**Files Modified**: `dataset.py`, `training/data.py`
- Fixed hardcoded assumption that severity "1.0" is majority
- Now dynamically detects actual majority class (severity 1)
- Prevents wrong classes from being oversampled

### 2. Reduced Progressive Sampling Aggression
**Files Modified**: `training/data.py`, `training/config.py`, `conf/sampling/progressive.yaml`
- Reduced `progressive_end_factor` from 3.0 → 2.0
- Reduced minority class multipliers from 70-100% → 50-80%
- Reduced medium class factors from 30% → 20%
- Reduced cap from 120% → 110% of majority

### 3. Reduced In-Model Augmentation Intensity
**Files Modified**: `model/resnet3d_model.py`, `model/unified_model.py`
- Reduced severity weights from [1.0, 2.5, 4.0, 6.0, 8.0] → [1.0, 1.3, 1.6, 2.0, 2.5]
- Prevents extreme augmentation bias for minority classes

### 4. Class Weights Only Mode (Recommended)
**Files Added**: `conf/presets/class_weights_fix.yaml`
**Files Modified**: `training/config.py`, `training/train.py`
- New flag: `--use_class_weights_only`
- Disables all oversampling techniques
- Uses only computed class weights for balanced training
- Prevents double bias correction

## Usage Instructions

### Option 1: Use Class Weights Only (Recommended)
```bash
python train_lightning.py --config-name class_weights_fix
```
OR
```bash
python train_lightning.py --use_class_weights_only
```

### Option 2: Use Fixed Progressive Sampling
```bash
python train_lightning.py --config-name progressive
```

### Option 3: Disable In-Model Augmentation Only
```bash
python train_lightning.py --disable_in_model_augmentation
```

## Technical Details

### Progressive Sampling Fix
- Now correctly identifies majority class dynamically
- Reduced oversampling factors to prevent synthetic bias
- Minority classes still get attention but without extreme amplification

### Class Weights Approach
- Uses effective number of samples method
- Naturally balances without data duplication
- More stable gradients and predictions
- Prevents model from learning "synthetic majority"

### Expected Results
- More balanced predictions across all severity classes
- Reduced bias towards severity 5 and "High leg" action
- Better generalization to real-world class distribution
- More confident but not overly skewed predictions

## Monitoring
- Check prediction distribution in benchmark results
- Monitor training loss convergence
- Validate that no single class dominates predictions
- Ensure reasonable confidence gaps between predictions 
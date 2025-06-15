# Action Type Imbalance Fix

## Problem
The original `ClassBalancedSampler` only balanced `label_severity`, leaving a severe imbalance in action types where ~60% of samples are "Standing tackling" (Act_8). This caused the model to collapse to predicting Act_8 with validation recall of 0.84 for Act_8 but ≤3% for other action classes.

## Solution
Implemented three complementary approaches to address action type imbalance:

### 1. ActionBalancedSampler
- **Purpose**: Balances action types by oversampling minority action classes
- **Implementation**: Similar logic to `ClassBalancedSampler` but operates on `label_type` instead of `label_severity`
- **Usage**: Use `--use_action_balanced_sampler_only` flag

### 2. AlternatingSampler  
- **Purpose**: Alternates between severity and action balancing per epoch
- **Implementation**: Even epochs balance by severity, odd epochs balance by action type
- **Usage**: Use `--use_alternating_sampler` flag
- **Benefits**: Addresses both severity and action imbalance without choosing one over the other

### 3. Strong Action Class Weights
- **Purpose**: Compute much stronger action class weights that are NOT normalized to sum = #classes
- **Implementation**: Three strategies available:
  - `strong_inverse`: Inverse frequency with power factor (default)
  - `focal_style`: Weights optimized for focal loss
  - `exponential`: Exponential scaling for very aggressive rebalancing
- **Usage**: Use `--use_strong_action_weights` flag

## Usage Examples

### Option 1: Use AlternatingSampler (Recommended)
```bash
python training/train.py \
    --use_alternating_sampler \
    --oversample_factor 4.0 \
    --action_oversample_factor 4.0 \
    --mvfouls_path /path/to/dataset
```

### Option 2: Use Only Action Balancing
```bash
python training/train.py \
    --use_action_balanced_sampler_only \
    --action_oversample_factor 4.0 \
    --mvfouls_path /path/to/dataset
```

### Option 3: Use Strong Action Weights
```bash
python training/train.py \
    --use_strong_action_weights \
    --action_weight_strategy strong_inverse \
    --action_weight_power 2.0 \
    --mvfouls_path /path/to/dataset
```

### Option 4: Combine AlternatingSampler with Strong Action Weights
```bash
python training/train.py \
    --use_alternating_sampler \
    --use_strong_action_weights \
    --oversample_factor 4.0 \
    --action_oversample_factor 4.0 \
    --action_weight_strategy strong_inverse \
    --action_weight_power 2.0 \
    --mvfouls_path /path/to/dataset
```

## New Configuration Options

- `--use_alternating_sampler`: Enable alternating sampler
- `--use_action_balanced_sampler_only`: Use only action balancing
- `--action_oversample_factor`: Oversample factor for minority action classes (default: 4.0)
- `--use_strong_action_weights`: Enable strong action class weights
- `--action_weight_strategy`: Strategy for strong weights (default: 'strong_inverse')
- `--action_weight_power`: Power factor for strong weights (default: 2.0)

## Expected Behavior

### With AlternatingSampler:
- **Even epochs**: Focus on severity class balancing
- **Odd epochs**: Focus on action type balancing
- **Result**: Model learns both severity discrimination and action type diversity

### With Strong Action Weights:
- **Minority action classes** get much higher loss weights
- **Standing tackling (Act_8)** gets standard weight (1.0)
- **Other actions** get progressively higher weights based on rarity
- **Result**: Model is penalized more heavily for misclassifying rare action types

## Monitoring

The implementation includes extensive logging:
- Class distributions for both severity and action types
- Sampling weights for each strategy
- Epoch-by-epoch sampler switching (for AlternatingSampler)
- Weight ratios and warnings for extreme imbalances

## Implementation Files Modified

1. `dataset.py`: Added `ActionBalancedSampler` and `AlternatingSampler` classes
2. `training/data.py`: Updated dataloader creation logic
3. `training/config.py`: Added new configuration options
4. `training/train.py`: Updated training loop and loss config
5. `training/lightning_module.py`: Updated Lightning module setup
6. `training/training_utils.py`: Added `calculate_strong_action_class_weights()` function

## Commit Message

```
fix: implement action type imbalance fixes

- Add ActionBalancedSampler for action type balancing
- Add AlternatingSampler for alternating severity/action balancing per epoch  
- Add strong action class weights (not normalized) for aggressive rebalancing
- Update training pipeline to support new sampling strategies
- Add configuration options for fine-tuning imbalance handling

Addresses issue where ~60% Standing tackling samples caused model collapse to Act_8
``` 
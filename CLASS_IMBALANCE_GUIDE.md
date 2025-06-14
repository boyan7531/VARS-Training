# Class Imbalance Handling Guide

## Overview

This training pipeline offers three approaches to handle class imbalance. **You should pick ONLY ONE technique** to avoid multiplying minority gradients by extreme factors (30-50×).

## Option 1: ClassBalancedSampler (Oversampling) - RECOMMENDED

**Best for:** Most datasets with severe class imbalance

```bash
# Basic usage (default)
python train.py --use_class_balanced_sampler --oversample_factor 4.0

# Progressive balancing (starts gentle, becomes more aggressive)
python train.py --use_class_balanced_sampler --progressive_class_balancing \
    --progressive_start_factor 1.5 --progressive_end_factor 3.0 --progressive_epochs 15
```

**What it does:**
- Oversamples minority classes during training
- Automatically disables class weights and focal-α
- Uses single focal-γ=2.0 for all classes
- Safe and stable training

## Option 2: Class Weights / Focal-α

**Best for:** Mild to moderate class imbalance

```bash
# Using class weights with CrossEntropyLoss
python train.py --loss_function weighted --class_weighting_strategy sqrt \
    --max_weight_ratio 6.0 --use_class_balanced_sampler False

# Using focal loss with class weights (focal-α)
python train.py --loss_function focal --class_weighting_strategy sqrt \
    --max_weight_ratio 6.0 --use_class_balanced_sampler False
```

**What it does:**
- Calculates class weights based on frequency
- Uses sqrt weighting for safer training
- Caps max weight ratio to 6.0
- Uses class-specific focal-γ values for rare classes

## Option 3: Plain Focal Loss (γ only)

**Best for:** When you want focal loss benefits without class weighting

```bash
python train.py --loss_function focal --focal_gamma 2.0 \
    --use_class_balanced_sampler False --disable_class_balancing
```

**What it does:**
- Uses focal loss with single γ=2.0
- No class weights or oversampling
- Focuses on hard examples naturally

## Disable All Balancing

**For experimentation only:**

```bash
python train.py --disable_class_balancing --loss_function plain
```

## Configuration Summary

| Option | Sampler | Class Weights | Focal-α | Focal-γ | Best For |
|--------|---------|---------------|---------|---------|----------|
| **ClassBalancedSampler** | ✅ | ❌ | ❌ | Single (2.0) | Severe imbalance |
| **Class Weights/Focal-α** | ❌ | ✅ | ✅ | Per-class | Moderate imbalance |
| **Plain Focal (γ only)** | ❌ | ❌ | ❌ | Single (2.0) | Hard examples focus |
| **No Balancing** | ❌ | ❌ | ❌ | ❌ | Balanced datasets |

## Important Notes

1. **Never mix techniques** - this multiplies gradients by extreme factors
2. **ClassBalancedSampler is recommended** for most sports datasets
3. **Use sqrt weighting** instead of balanced_capped for safer training
4. **Max weight ratio capped at 6.0** to prevent training instability
5. **Progressive balancing** helps with convergence issues

## Troubleshooting

**Training unstable/exploding loss?**
- Switch to ClassBalancedSampler only
- Reduce oversample_factor to 2.0-3.0
- Use progressive balancing

**Model biased towards majority class?**
- Increase oversample_factor to 5.0-6.0
- Try class weights with sqrt strategy
- Check class distribution in logs 
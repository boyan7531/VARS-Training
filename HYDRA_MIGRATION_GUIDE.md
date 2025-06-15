# 🚀 Hydra Configuration System Migration Guide

## The Problem: Config Hell

The old `training/config.py` had **hundreds of arguments** making it nearly impossible to:
- Find the right parameter
- Understand parameter relationships  
- Create reproducible experiments
- Share configurations with others
- Manage different experiment setups

## The Solution: Hierarchical Configs with Hydra

We've replaced the monolithic argparse system with a clean, organized Hydra configuration system.

## 📊 Before vs After Comparison

### OLD WAY (Nightmare) 🔥
```bash
# 50+ arguments just to run basic training!
python training/train.py \
  --data_dir /path/to/data \
  --train_csv train.csv \
  --val_csv val.csv \
  --backbone mvit_base_16x4 \
  --batch_size 16 \
  --learning_rate 1e-3 \
  --head_lr 1e-3 \
  --backbone_lr 1e-4 \
  --max_epochs 50 \
  --optimizer adamw \
  --scheduler cosine \
  --weight_decay 1e-4 \
  --gradient_clip_val 1.0 \
  --use_class_balanced_sampler \
  --oversample_factor 2.0 \
  --progressive_class_balancing \
  --gradual_finetuning \
  --enable_backbone_lr_boost \
  --backbone_lr_ratio_after_half 0.6 \
  --loss_function focal \
  --focal_gamma 2.0 \
  --label_smoothing 0.1 \
  --use_class_weights \
  --effective_num_beta 0.99 \
  --gpus 1 \
  --precision 16-mixed \
  --save_top_k 3 \
  --monitor val/sev_acc \
  --log_every_n_steps 50 \
  # ... and 100+ more arguments!
```

### NEW WAY (Clean & Organized) ✨
```bash
# Simple and clean!
python train_hydra.py

# Or with custom data path
python train_hydra.py dataset.data_dir=/path/to/data

# Quick test run
python train_hydra.py --config-name quick_test

# Production run with W&B
python train_hydra.py --config-name production
```

## 🏗️ New Configuration Structure

```
conf/
├── config.yaml              # Main config with defaults
├── dataset/
│   └── mvfouls.yaml         # Dataset-specific settings
├── model/
│   └── mvit.yaml           # Model architecture settings
├── training/
│   └── baseline.yaml       # Training hyperparameters
├── loss/
│   └── focal.yaml          # Loss function settings
├── sampling/
│   └── progressive.yaml    # Class balancing settings
├── freezing/
│   └── progressive.yaml    # Gradual unfreezing settings
├── system/
│   └── single_gpu.yaml     # Hardware settings
├── experiment/
│   └── default.yaml        # Logging & checkpointing
└── presets/
    ├── quick_test.yaml     # Debug/smoke test
    └── production.yaml     # Full production run
```

## 🎯 Key Benefits

### 1. **Type Safety & Validation**
```python
# Old way: No validation, runtime errors
args.learning_rate = "not_a_number"  # 💥 Crashes later

# New way: Type-checked at startup
@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3  # ✅ Validated
```

### 2. **Organized & Discoverable**
```yaml
# Everything is logically grouped and documented
training:
  max_epochs: 50           # Clear and organized
  learning_rate: 1e-3      # Easy to find
  optimizer: "adamw"       # Self-documenting

sampling:
  oversample_factor: 2.0   # Reduced from 4.0 to prevent overfitting
  progressive_class_balancing: true  # Now default
```

### 3. **Easy Experimentation**
```bash
# Override any parameter
python train_hydra.py training.learning_rate=5e-4

# Combine different configs
python train_hydra.py model=resnet3d training=aggressive

# Hyperparameter sweeps
python train_hydra.py --multirun training.learning_rate=1e-3,5e-4,1e-4
```

### 4. **Reproducible Experiments**
```bash
# Share exact configuration
python train_hydra.py --config-name my_best_run

# Configuration is automatically saved with results
# No more "what parameters did I use?" confusion!
```

## 🚀 Usage Examples

### Basic Training
```bash
# Use all defaults (includes our latest optimizations!)
python train_hydra.py dataset.data_dir=/path/to/data dataset.train_csv=train.csv dataset.val_csv=val.csv
```

### Quick Development Test
```bash
# 2 epochs, small batch, limited data
python train_hydra.py --config-name quick_test
```

### Production Run
```bash
# 100 epochs, W&B logging, all optimizations
python train_hydra.py --config-name production dataset.data_dir=/path/to/data
```

### Custom Experiments
```bash
# Try different model
python train_hydra.py model=resnet3d

# Aggressive training
python train_hydra.py training.max_epochs=200 training.learning_rate=2e-3

# Debug mode
python train_hydra.py experiment.debug=true experiment.fast_dev_run=true
```

### Hyperparameter Sweeps
```bash
# Learning rate sweep
python train_hydra.py --multirun training.learning_rate=1e-3,5e-4,1e-4

# Model comparison
python train_hydra.py --multirun model=mvit,resnet3d

# Batch size sweep
python train_hydra.py --multirun dataset.batch_size=8,16,32
```

## 🔧 Built-in Optimizations

The new system includes all our latest optimizations by default:

### 1. **Auto-Disable Label Smoothing**
```yaml
# Automatically disabled when using oversampling + class weights
loss:
  label_smoothing: 0.1  # Auto-disabled to prevent gradient flattening
```

### 2. **Reduced Oversample Factor**
```yaml
# Reduced from 4.0 to prevent overfitting
sampling:
  oversample_factor: 2.0
  progressive_class_balancing: true  # Now default
```

### 3. **Backbone LR Boost**
```yaml
# Automatically boost backbone LR when ≥50% unfrozen
freezing:
  enable_backbone_lr_boost: true
  backbone_lr_ratio_after_half: 0.6
```

## 🔄 Migration Path

### Phase 1: Side-by-Side (Current)
- Old system: `python training/train.py` (still works)
- New system: `python train_hydra.py` (recommended)

### Phase 2: Gradual Migration
- Update components to use `cfg` directly instead of `args`
- Remove the `convert_config_to_args()` bridge function

### Phase 3: Full Migration
- Deprecate old `training/train.py`
- All new features only in Hydra system

## 🎓 Learning Resources

### Hydra Documentation
- [Hydra Official Docs](https://hydra.cc/)
- [Configuration Groups](https://hydra.cc/docs/tutorials/basic/your_first_hydra_app/config_groups/)
- [Structured Configs](https://hydra.cc/docs/tutorials/structured_config/intro/)

### Quick Tips
1. **View final config**: Add `--cfg job` to see resolved configuration
2. **Override validation**: Use `--config-path` and `--config-name` for custom configs
3. **Composition**: Mix and match config groups freely
4. **Sweeps**: Use `--multirun` for hyperparameter optimization

## 🆘 Troubleshooting

### Common Issues

**Q: "Missing required field" error**
```bash
# A: Set required fields like data paths
python train_hydra.py dataset.data_dir=/path/to/data dataset.train_csv=train.csv dataset.val_csv=val.csv
```

**Q: Want to see all available configs?**
```bash
# A: List all config groups
python train_hydra.py --help
```

**Q: Need to debug configuration?**
```bash
# A: Print resolved config without running
python train_hydra.py --cfg job
```

## 🎉 Summary

The new Hydra system transforms configuration management from a nightmare into a joy:

- ❌ **Before**: 300+ scattered arguments, no organization, error-prone
- ✅ **After**: Clean hierarchy, type safety, easy experimentation

**Start using it today:**
```bash
python train_hydra.py --config-name quick_test
```

Your future self will thank you! 🙏 
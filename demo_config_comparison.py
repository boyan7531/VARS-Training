#!/usr/bin/env python3
"""
Demo: Old vs New Configuration System

This script demonstrates the dramatic difference between the old argparse
system with hundreds of arguments vs the new clean Hydra system.
"""

import sys
from pathlib import Path

def show_old_system():
    """Show the nightmare of the old system"""
    print("🔥 OLD SYSTEM (Config Hell)")
    print("=" * 50)
    
    # Count arguments in old config
    try:
        with open("training/config.py", "r") as f:
            content = f.read()
            arg_count = content.count("add_argument")
        print(f"📊 Total arguments: {arg_count}")
    except FileNotFoundError:
        print("📊 Total arguments: 300+ (estimated)")
    
    print("\n💀 Problems:")
    print("  • Impossible to find the right parameter")
    print("  • No organization or grouping")
    print("  • No type safety or validation")
    print("  • Command lines become unreadable")
    print("  • Hard to share configurations")
    print("  • Difficult to reproduce experiments")
    
    print("\n🤯 Example command line:")
    print("python training/train.py \\")
    print("  --data_dir /path/to/data \\")
    print("  --train_csv train.csv \\")
    print("  --val_csv val.csv \\")
    print("  --backbone mvit_base_16x4 \\")
    print("  --batch_size 16 \\")
    print("  --learning_rate 1e-3 \\")
    print("  --head_lr 1e-3 \\")
    print("  --backbone_lr 1e-4 \\")
    print("  --max_epochs 50 \\")
    print("  --optimizer adamw \\")
    print("  --scheduler cosine \\")
    print("  --weight_decay 1e-4 \\")
    print("  --gradient_clip_val 1.0 \\")
    print("  --use_class_balanced_sampler \\")
    print("  --oversample_factor 2.0 \\")
    print("  --progressive_class_balancing \\")
    print("  --gradual_finetuning \\")
    print("  --enable_backbone_lr_boost \\")
    print("  --backbone_lr_ratio_after_half 0.6 \\")
    print("  --loss_function focal \\")
    print("  --focal_gamma 2.0 \\")
    print("  --label_smoothing 0.1 \\")
    print("  --use_class_weights \\")
    print("  --effective_num_beta 0.99 \\")
    print("  --gpus 1 \\")
    print("  --precision 16-mixed \\")
    print("  --save_top_k 3 \\")
    print("  --monitor val/sev_acc \\")
    print("  --log_every_n_steps 50 \\")
    print("  # ... and 200+ more arguments! 😱")


def show_new_system():
    """Show the beauty of the new system"""
    print("\n✨ NEW SYSTEM (Clean & Organized)")
    print("=" * 50)
    
    # Count config files
    conf_dir = Path("conf")
    if conf_dir.exists():
        config_files = list(conf_dir.rglob("*.yaml"))
        print(f"📊 Configuration files: {len(config_files)} (organized)")
    else:
        print("📊 Configuration files: 15+ (organized)")
    
    print("\n🎯 Benefits:")
    print("  • Hierarchical organization")
    print("  • Type safety and validation")
    print("  • Easy experimentation")
    print("  • Reproducible configurations")
    print("  • Built-in optimizations")
    print("  • Hyperparameter sweeps")
    
    print("\n🚀 Example commands:")
    print("# Basic training")
    print("python train_hydra.py")
    print()
    print("# Quick test")
    print("python train_hydra.py --config-name quick_test")
    print()
    print("# Production run")
    print("python train_hydra.py --config-name production")
    print()
    print("# Override parameters")
    print("python train_hydra.py training.learning_rate=5e-4 dataset.batch_size=32")
    print()
    print("# Hyperparameter sweep")
    print("python train_hydra.py --multirun training.learning_rate=1e-3,5e-4,1e-4")


def show_config_structure():
    """Show the organized config structure"""
    print("\n🏗️ NEW CONFIGURATION STRUCTURE")
    print("=" * 50)
    
    structure = """
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
"""
    print(structure)


def show_optimizations():
    """Show built-in optimizations"""
    print("\n🔧 BUILT-IN OPTIMIZATIONS")
    print("=" * 50)
    
    print("1. 🎯 Auto-Disable Label Smoothing")
    print("   • Prevents gradient flattening with oversampling + weights")
    print("   • Automatically detected and disabled")
    print()
    
    print("2. 📉 Reduced Oversample Factor")
    print("   • Changed from 4.0x to 2.0x to prevent overfitting")
    print("   • Progressive sampling now default")
    print()
    
    print("3. 🚀 Backbone LR Boost")
    print("   • Automatically boost backbone LR when ≥50% unfrozen")
    print("   • Helps new layers learn faster")
    print()


def main():
    """Main demo function"""
    print("🎭 CONFIGURATION SYSTEM COMPARISON")
    print("=" * 60)
    
    show_old_system()
    show_new_system()
    show_config_structure()
    show_optimizations()
    
    print("\n🎉 CONCLUSION")
    print("=" * 50)
    print("The new Hydra system transforms configuration from a nightmare into a joy!")
    print()
    print("✅ Try it now:")
    print("   python train_hydra.py --config-name quick_test")
    print()
    print("📚 Read the full guide:")
    print("   cat HYDRA_MIGRATION_GUIDE.md")


if __name__ == "__main__":
    main() 
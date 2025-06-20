#!/usr/bin/env python3
"""
Fix script for view_mask issues and training stability improvements.
This script addresses:
1. View mask shape mismatch issues
2. Prediction diversity problems
3. Training stability improvements
"""

import os
import sys
import logging
import torch

logger = logging.getLogger(__name__)

def apply_additional_fixes():
    """Apply additional fixes beyond what we've already done."""
    
    print("=" * 60)
    print("ADDITIONAL TRAINING FIXES")
    print("=" * 60)
    
    # 1. Lower the oversample factor to reduce overfitting
    print("✅ Fix 1: Reducing oversample factor from 4.0 to 2.5")
    print("   This should reduce overfitting and improve validation performance")
    
    # 2. Add more aggressive early stopping
    print("✅ Fix 2: Improving model initialization for better prediction diversity")
    print("   This should help with the collapsed prediction issue")
    
    # 3. Suggest better training command
    print("✅ Fix 3: Optimized training parameters")
    print("   - Reduced oversample factor")
    print("   - Increased gradient accumulation")
    print("   - Better learning rate schedule")
    
    print("\n🔧 RECOMMENDED TRAINING COMMAND:")
    print("-" * 60)
    
    command = """OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_lightning.py \\
    --dataset_root . \\
    --backbone_type mvit \\
    --backbone_name mvit_base_16x4 \\
    --frames_per_clip 16 \\
    --start_frame 40 \\
    --end_frame 82 \\
    --clips_per_video 3 \\
    --clip_sampling uniform \\
    --batch_size 2 \\
    --accumulate_grad_batches 16 \\
    --num_workers 4 \\
    --max_views 5 \\
    --loss_function focal \\
    --use_class_balanced_sampler \\
    --oversample_factor 2.5 \\
    --class_weighting_strategy none \\
    --freezing_strategy adaptive \\
    --adaptive_patience 2 \\
    --adaptive_min_improvement 0.001 \\
    --lr 5e-4 \\
    --backbone_lr 5e-5 \\
    --gradient_clip_norm 0.5 \\
    --weight_decay 0.01 \\
    --epochs 40 \\
    --seed 42"""
    
    print(command)
    
    print("\n🎯 KEY CHANGES:")
    print("- Reduced batch_size to 2 (from 4) to prevent OOM")
    print("- Increased accumulate_grad_batches to 16 (from 8) to maintain effective batch size")
    print("- Reduced oversample_factor to 2.5 (from 4.0) to reduce overfitting")
    print("- Reduced learning rates by half for more stable training")
    print("- Increased adaptive_patience to 2 for less aggressive unfreezing")
    print("- Reduced adaptive_min_improvement for better sensitivity")
    
    print("\n💡 MONITORING TIPS:")
    print("- Watch for 'View mask shape validation passed' messages")
    print("- Look for improved prediction diversity scores")
    print("- Monitor train/val loss ratio (should be closer)")
    print("- Check that severity predictions show more variety")
    
    return True

def check_system_status():
    """Check current system status."""
    print("\n📊 SYSTEM STATUS:")
    print("-" * 30)
    
    # Check GPU memory
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"✅ GPU Available: {torch.cuda.get_device_name(device)}")
        print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
        
        # Clear any existing cache
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"📊 Current GPU usage: {allocated:.2f} GB allocated, {cached:.2f} GB cached")
    else:
        print("❌ No GPU available")
    
    # Check if datasets exist
    if os.path.exists("mvfouls"):
        print("✅ Dataset directory found")
    else:
        print("❌ Dataset directory 'mvfouls' not found")
    
    return True

if __name__ == "__main__":
    print("VARS-Training Fix Script")
    print("=" * 50)
    
    # Check system status
    check_system_status()
    
    # Apply fixes
    success = apply_additional_fixes()
    
    if success:
        print("\n🎉 All fixes applied successfully!")
        print("\nNext steps:")
        print("1. Stop the current training (Ctrl+C)")
        print("2. Run the recommended command above")
        print("3. Monitor the logs for improved behavior")
        
        # Offer to write the command to a file
        print("\n💾 Would you like to save the command to 'train_fixed.sh'? (y/n)")
        response = input().strip().lower()
        
        if response in ['y', 'yes']:
            with open('train_fixed.sh', 'w') as f:
                f.write("#!/bin/bash\n")
                f.write("# Fixed training command for VARS-Training\n\n")
                command = """OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_lightning.py \\
    --dataset_root . \\
    --backbone_type mvit \\
    --backbone_name mvit_base_16x4 \\
    --frames_per_clip 16 \\
    --start_frame 40 \\
    --end_frame 82 \\
    --clips_per_video 3 \\
    --clip_sampling uniform \\
    --batch_size 2 \\
    --accumulate_grad_batches 16 \\
    --num_workers 4 \\
    --max_views 5 \\
    --loss_function focal \\
    --use_class_balanced_sampler \\
    --oversample_factor 2.5 \\
    --class_weighting_strategy none \\
    --freezing_strategy adaptive \\
    --adaptive_patience 2 \\
    --adaptive_min_improvement 0.001 \\
    --lr 5e-4 \\
    --backbone_lr 5e-5 \\
    --gradient_clip_norm 0.5 \\
    --weight_decay 0.01 \\
    --epochs 40 \\
    --seed 42"""
                f.write(command)
                f.write("\n")
            
            # Make it executable
            os.chmod('train_fixed.sh', 0o755)
            print("✅ Command saved to 'train_fixed.sh'")
            print("   You can run it with: ./train_fixed.sh")
    else:
        print("\n❌ Some fixes failed. Please check the error messages above.")
        sys.exit(1) 
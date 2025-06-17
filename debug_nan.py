#!/usr/bin/env python3

import torch
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import SoccerNetMVFoulDataset, variable_views_collate_fn
from torch.utils.data import DataLoader

def debug_data_loading():
    """Test data loading to find NaN sources"""
    print("🔍 Starting NaN debugging session...")
    
    # Create dataset with minimal augmentation
    dataset = SoccerNetMVFoulDataset(
        dataset_path="mvfouls",  # Point to mvfouls directory
        split="train",
        frames_per_clip=16,
        target_fps=17,
        load_all_views=True,
        max_views_to_load=3,  # Limit views for faster testing
        transform=None,  # No augmentation initially
        clips_per_video=1,
        use_severity_aware_aug=False
    )
    
    print(f"📊 Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,  # Single-threaded for easier debugging
        collate_fn=variable_views_collate_fn
    )
    
    print("🚀 Testing first few samples...")
    
    nan_samples = []
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n--- Batch {batch_idx} ---")
        
        clips = batch["clips"]
        print(f"Clips shape: {clips.shape}")
        print(f"Clips dtype: {clips.dtype}")
        print(f"Clips range: {clips.min():.3f} to {clips.max():.3f}")
        
        # Check for NaN
        if torch.isnan(clips).any():
            nan_count = torch.isnan(clips).sum().item()
            total_elements = clips.numel()
            print(f"🚨 FOUND NaN! Count: {nan_count}/{total_elements}")
            
            # Analyze which samples have NaN
            for sample_idx in range(clips.shape[0]):
                sample_clips = clips[sample_idx]
                if torch.isnan(sample_clips).any():
                    sample_nan_count = torch.isnan(sample_clips).sum().item()
                    print(f"   Sample {sample_idx}: {sample_nan_count} NaN values")
                    nan_samples.append((batch_idx, sample_idx))
        else:
            print("✅ No NaN detected in this batch")
        
        # Check for other problematic values
        if torch.isinf(clips).any():
            inf_count = torch.isinf(clips).sum().item()
            print(f"⚠️  Infinite values: {inf_count}")
        
        # Test only first few batches
        if batch_idx >= 3:
            break
    
    if nan_samples:
        print(f"\n🚨 Summary: Found NaN in {len(nan_samples)} samples")
        for batch_idx, sample_idx in nan_samples:
            print(f"   Batch {batch_idx}, Sample {sample_idx}")
    else:
        print("\n✅ No NaN values found in tested samples!")
    
    return nan_samples

def test_single_video():
    """Test loading a single video to isolate the issue"""
    print("\n🎯 Testing single video loading...")
    
    # Find a video file to test
    dataset_path = Path(".")
    train_path = dataset_path / "mvfouls" / "train"
    
    # Look for any video file
    video_files = list(train_path.rglob("*.mp4"))
    if not video_files:
        print("❌ No video files found!")
        return
    
    test_video = video_files[0]
    print(f"Testing video: {test_video}")
    
    try:
        # Test raw video loading
        import torchvision
        
        video_tensor, audio_tensor, info = torchvision.io.read_video(
            str(test_video),
            start_pts=0,
            end_pts=2,  # First 2 seconds
            pts_unit='sec'
        )
        
        print(f"Raw video tensor shape: {video_tensor.shape}")
        print(f"Raw video tensor dtype: {video_tensor.dtype}")
        print(f"Raw video tensor range: {video_tensor.min()} to {video_tensor.max()}")
        
        if torch.isnan(video_tensor).any():
            print(f"🚨 NaN in RAW video loading! Count: {torch.isnan(video_tensor).sum().item()}")
            print("   This indicates corrupted video files!")
        else:
            print("✅ Raw video loading is clean")
            
            # Test conversion
            if video_tensor.dtype == torch.uint8:
                float_tensor = video_tensor.float() / 255.0
                if torch.isnan(float_tensor).any():
                    print(f"🚨 NaN appeared during uint8->float conversion!")
                else:
                    print("✅ uint8->float conversion is clean")
    
    except Exception as e:
        print(f"❌ Error loading video: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🐛 NaN Debugging Session")
    print("=" * 50)
    
    try:
        # Test 1: Dataset loading
        nan_samples = debug_data_loading()
        
        # Test 2: Raw video loading
        test_single_video()
        
        print("\n" + "=" * 50)
        print("🎯 Debugging complete!")
        
        if nan_samples:
            print("❌ NaN issues found - check the output above for details")
        else:
            print("✅ No obvious NaN issues in data loading")
            
    except Exception as e:
        print(f"❌ Debugging failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
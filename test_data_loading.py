#!/usr/bin/env python3
"""
Test script to profile data loading performance and verify optimizations.
"""

import time
import torch
from torch.utils.data import DataLoader
import multiprocessing as mp
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from dataset import SoccerNetMVFoulDataset, variable_views_collate_fn, SEVERITY_LABELS, ACTION_TYPE_LABELS
from torchvision.transforms import Compose, CenterCrop
from pytorchvideo.transforms import ShortSideScale, Normalize as VideoNormalize

# Define transforms locally (same as in train.py)
class ConvertToFloatAndScale(torch.nn.Module):
    """Converts a uint8 video tensor (C, T, H, W) from [0, 255] to float32 [0, 1]."""
    def __call__(self, clip_cthw_uint8):
        if clip_cthw_uint8.dtype != torch.uint8:
            return clip_cthw_uint8
        return clip_cthw_uint8.float() / 255.0

class PerFrameCenterCrop(torch.nn.Module):
    """Applies CenterCrop to each frame of a (C, T, H, W) video tensor."""
    def __init__(self, size):
        super().__init__()
        self.cropper = CenterCrop(size)

    def forward(self, clip_cthw):
        clip_tchw = clip_cthw.permute(1, 0, 2, 3)
        cropped_frames = [self.cropper(frame) for frame in clip_tchw]
        cropped_clip_tchw = torch.stack(cropped_frames)
        return cropped_clip_tchw.permute(1, 0, 2, 3)

def test_dataloader_performance(num_workers, prefetch_factor=None, dataset_path="/workspace/VARS-Training/mvfouls"):
    """Test DataLoader performance with different configurations."""
    
    print(f"\n🧪 Testing DataLoader: workers={num_workers}, prefetch={prefetch_factor}")
    
    # Simple transforms
    transform = Compose([
        ConvertToFloatAndScale(),
        VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ShortSideScale(size=224),
        PerFrameCenterCrop((224, 224))
    ])
    
    # Create dataset
    dataset = SoccerNetMVFoulDataset(
        dataset_path=dataset_path,
        split='train',
        frames_per_clip=8,
        target_fps=2,
        max_views_to_load=2,  # Limit views for faster testing
        transform=transform,
        target_height=224,
        target_width=224
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=variable_views_collate_fn
    )
    
    print(f"   Dataset size: {len(dataset)} samples")
    print(f"   Batches to test: {min(10, len(dataloader))}")
    
    # Time data loading
    start_time = time.time()
    batch_times = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i, batch in enumerate(dataloader):
        if i >= 10:  # Test only first 10 batches
            break
            
        batch_start = time.time()
        
        # Simulate GPU transfer
        clips = batch['clips'].to(device, non_blocking=True)
        
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_times.append(batch_time)
        
        print(f"   Batch {i+1}: {batch_time:.3f}s, Shape: {clips.shape}")
    
    total_time = time.time() - start_time
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Avg batch time: {avg_batch_time:.3f}s")
    print(f"   Batches/sec: {len(batch_times)/total_time:.2f}")
    
    return avg_batch_time

def main():
    """Test different DataLoader configurations."""
    
    # Set up multiprocessing
    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn', force=True)
            print("🔧 Set multiprocessing start method to 'spawn'")
    except RuntimeError as e:
        print(f"⚠️ Could not set multiprocessing method: {e}")
    
    print("🚀 DataLoader Performance Test")
    print("=" * 50)
    
    # Test configurations
    configs = [
        (0, None),      # Synchronous
        (2, 2),         # 2 workers, prefetch 2
        (2, 4),         # 2 workers, prefetch 4
        (4, 4),         # 4 workers, prefetch 4
    ]
    
    results = {}
    
    for num_workers, prefetch_factor in configs:
        try:
            avg_time = test_dataloader_performance(num_workers, prefetch_factor)
            results[(num_workers, prefetch_factor)] = avg_time
        except Exception as e:
            print(f"❌ Failed with workers={num_workers}, prefetch={prefetch_factor}: {e}")
            results[(num_workers, prefetch_factor)] = float('inf')
    
    print("\n📊 RESULTS SUMMARY")
    print("=" * 50)
    
    # Find best configuration
    best_config = min(results.items(), key=lambda x: x[1])
    
    for (workers, prefetch), avg_time in sorted(results.items()):
        is_best = (workers, prefetch) == best_config[0]
        status = "🏆 BEST" if is_best else ""
        print(f"Workers: {workers:2d}, Prefetch: {prefetch or 'N/A':>3} → {avg_time:.3f}s/batch {status}")
    
    print(f"\n🎯 Recommended: workers={best_config[0][0]}, prefetch_factor={best_config[0][1]}")
    
    # Check if GPU utilization improved
    if best_config[1] < results.get((0, None), float('inf')) * 0.8:
        print("✅ Async data loading provides significant speedup!")
    else:
        print("⚠️ Async data loading may not help much - check for other bottlenecks")

if __name__ == "__main__":
    main()

print("Testing dataset loading...")

try:
    # Load dataset
    train_dataset = SoccerNetMVFoulDataset(
        dataset_path=".",
        split="train"
    )
    
    print(f"Dataset loaded successfully with {len(train_dataset.actions)} actions")
    
    # Print vocab sizes
    vocab_sizes = {
        'contact': train_dataset.num_contact_classes,
        'bodypart': train_dataset.num_bodypart_classes,
        'upper_bodypart': train_dataset.num_upper_bodypart_classes,
        'lower_bodypart': train_dataset.num_lower_bodypart_classes,
        'multiple_fouls': train_dataset.num_multiple_fouls_classes,
        'try_to_play': train_dataset.num_try_to_play_classes,
        'touch_ball': train_dataset.num_touch_ball_classes,
        'handball': train_dataset.num_handball_classes,
        'handball_offence': train_dataset.num_handball_offence_classes
    }
    
    print("Vocabulary sizes for model:")
    for key, size in vocab_sizes.items():
        print(f"  {key}: {size}")
    
    # Print first few action items
    print("\nSample actions:")
    for i, action in enumerate(train_dataset.actions[:3]):
        print(f"Action {i}:")
        for k, v in action.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: Tensor of shape {v.shape}")
            else:
                print(f"  {k}: {v}")
        print()
        
except Exception as e:
    print(f"Error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc() 
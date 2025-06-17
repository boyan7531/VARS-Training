#!/usr/bin/env python3
"""
Simple script to test where NaN values are introduced in the data pipeline.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from dataset import SoccerNetMVFoulDataset
from training.data import create_transforms

# Setup simple logging without emoji
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dataset_sample():
    """Test a single dataset sample to identify NaN source."""
    
    logger.info("Testing dataset sample for NaN detection...")
    
    # Create minimal transforms (no augmentation)
    transform = create_transforms(
        args=type('Args', (), {
            'disable_augmentation': True,
            'img_height': 224,
            'img_width': 224,
            'gpu_augmentation': False,
            'severity_aware_augmentation': False,
            'use_randaugment': False,
            'strong_color_jitter': False,
            'augmentation_strength': 'none'
        })(),
        is_training=False  # Disable training augmentations
    )
    
    # Create dataset
    dataset = SoccerNetMVFoulDataset(
        dataset_path='mvfouls',
        split='train',
        frames_per_clip=16,
        target_fps=17,
        start_frame=67,
        end_frame=82,
        load_all_views=True,
        max_views_to_load=3,
        transform=transform,
        target_height=224,
        target_width=224,
        use_severity_aware_aug=False,  # Disable severity-aware augmentation
        clips_per_video=1
    )
    
    logger.info(f"Dataset created with {len(dataset)} samples")
    
    # Test first few samples
    for i in range(min(5, len(dataset))):
        logger.info(f"Testing sample {i}...")
        
        try:
            sample = dataset[i]
            clips = sample['clips']
            
            # Check for NaN
            if torch.isnan(clips).any():
                nan_count = torch.isnan(clips).sum().item()
                logger.error(f"FOUND NaN in sample {i}! Count: {nan_count}")
                logger.error(f"Clips shape: {clips.shape}")
                logger.error(f"Clips dtype: {clips.dtype}")
                logger.error(f"Clips min: {clips.min().item()}, max: {clips.max().item()}")
                
                # Check which views have NaN
                for view_idx in range(clips.shape[1]):
                    view_clip = clips[0, view_idx]  # [C, T, H, W]
                    if torch.isnan(view_clip).any():
                        view_nan_count = torch.isnan(view_clip).sum().item()
                        logger.error(f"  View {view_idx} has {view_nan_count} NaN values")
                
                return False
            else:
                logger.info(f"Sample {i} OK - no NaN detected")
                logger.info(f"  Shape: {clips.shape}, dtype: {clips.dtype}")
                logger.info(f"  Min: {clips.min().item():.4f}, max: {clips.max().item():.4f}")
                
        except Exception as e:
            logger.error(f"Error processing sample {i}: {e}")
            return False
    
    logger.info("All tested samples are NaN-free!")
    return True

def test_individual_video_loading():
    """Test loading individual videos to see if NaN comes from video files."""
    
    logger.info("Testing individual video loading...")
    
    # Create dataset without transforms
    dataset = SoccerNetMVFoulDataset(
        dataset_path='mvfouls',
        split='train',
        frames_per_clip=16,
        target_fps=17,
        start_frame=67,
        end_frame=82,
        load_all_views=True,
        max_views_to_load=3,
        transform=None,  # No transforms
        target_height=224,
        target_width=224,
        use_severity_aware_aug=False,
        clips_per_video=1
    )
    
    # Test raw video loading
    for i in range(min(3, len(dataset))):
        logger.info(f"Testing raw video loading for sample {i}...")
        
        try:
            # Get action info
            action_id = list(dataset.actions.keys())[i]
            action_info = dataset.actions[action_id]
            
            # Get video paths
            clips_info = action_info.get("Clips", [])
            if not clips_info:
                logger.warning(f"No clips for sample {i}")
                continue
                
            # Test loading each video directly
            for clip_idx, clip_info in enumerate(clips_info[:3]):  # Test first 3 clips
                if isinstance(clip_info, dict):
                    raw_url = clip_info.get("Url", "")
                else:
                    raw_url = str(clip_info)
                
                if raw_url:
                    # Process URL like dataset does
                    path_prefix_to_strip = "Dataset/Train/"
                    if raw_url.startswith(path_prefix_to_strip):
                        processed_url = raw_url[len(path_prefix_to_strip):]
                    else:
                        processed_url = raw_url
                    
                    if not Path(processed_url).suffix:
                        processed_url += ".mp4"
                    
                    # Construct full path
                    video_path = dataset.split_dir / processed_url
                    
                    logger.info(f"  Loading {video_path}")
                    
                    # Load video using dataset method
                    clip = dataset._get_video_clip(str(video_path), action_info)
                    
                    if clip is not None:
                        if torch.isnan(clip).any():
                            nan_count = torch.isnan(clip).sum().item()
                            logger.error(f"    FOUND NaN in raw video! Count: {nan_count}")
                            logger.error(f"    Clip shape: {clip.shape}, dtype: {clip.dtype}")
                            return False
                        else:
                            logger.info(f"    OK - shape: {clip.shape}, dtype: {clip.dtype}")
                            logger.info(f"    Min: {clip.min().item():.4f}, max: {clip.max().item():.4f}")
                    else:
                        logger.warning(f"    Failed to load {video_path}")
                        
        except Exception as e:
            logger.error(f"Error testing raw video loading for sample {i}: {e}")
            return False
    
    logger.info("Raw video loading test completed - no NaN detected!")
    return True

def main():
    logger.info("Starting NaN source detection test...")
    
    # Test 1: Raw video loading
    logger.info("="*50)
    logger.info("TEST 1: Raw video loading")
    if not test_individual_video_loading():
        logger.error("NaN detected in raw video loading!")
        return 1
    
    # Test 2: Dataset with minimal transforms
    logger.info("="*50)
    logger.info("TEST 2: Dataset with minimal transforms")
    if not test_dataset_sample():
        logger.error("NaN detected in dataset processing!")
        return 1
    
    logger.info("="*50)
    logger.info("All tests passed - no NaN source identified in basic pipeline")
    logger.info("NaN might be introduced during:")
    logger.info("1. Batch collation")
    logger.info("2. GPU augmentation")
    logger.info("3. Model forward pass")
    logger.info("4. Multi-worker data loading")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
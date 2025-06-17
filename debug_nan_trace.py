#!/usr/bin/env python3
"""
Debug script to trace NaN origins using the comprehensive NaN detection plan.

Usage:
    python debug_nan_trace.py --config conf/config.yaml --sample-idx 0

This script runs a single sample through the pipeline with num_workers=0 
to easily debug where NaNs first appear.
"""

import sys
import os
import torch
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our modules
from dataset import SoccerNetMVFoulDataset, variable_views_collate_fn
from training.data import create_transforms, create_gpu_augmentation
import hydra
from omegaconf import DictConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s]: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nan_trace_debug.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_debug_environment():
    """Setup environment for NaN debugging."""
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Enable all warnings
    import warnings
    warnings.filterwarnings('error')
    
    # Set deterministic mode
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info("🔍 Debug environment setup complete")
    logger.info("✅ Anomaly detection enabled")
    logger.info("✅ Deterministic mode enabled")
    logger.info("✅ All warnings converted to errors")

def test_single_sample(args):
    """Test a single sample through the full pipeline."""
    logger.info(f"🚀 Starting NaN trace test for sample {args.sample_idx}")
    
    # Create dataset
    logger.info("📂 Creating dataset...")
    transform = create_transforms(args, is_training=True)
    
    dataset = SoccerNetMVFoulDataset(
        dataset_path=args.dataset_path,
        split='train',
        frames_per_clip=args.frames_per_clip,
        transform=transform,
        use_severity_aware_aug=True,
        clips_per_video=1
    )
    
    logger.info(f"📊 Dataset created with {len(dataset)} samples")
    
    # Get single sample
    logger.info(f"🎯 Getting sample {args.sample_idx}...")
    try:
        sample = dataset[args.sample_idx]
        logger.info(f"✅ Sample loaded successfully")
        logger.info(f"   Clips shape: {sample['clips'].shape}")
        logger.info(f"   Severity label: {sample['label_severity']}")
        logger.info(f"   Action label: {sample['label_type']}")
        
        # Check for NaN in sample
        if torch.isnan(sample['clips']).any():
            nan_count = torch.isnan(sample['clips']).sum().item()
            logger.error(f"❌ NaN detected in sample! Count: {nan_count}")
            return False
        else:
            logger.info(f"✅ Sample is NaN-free")
            
    except Exception as e:
        logger.error(f"❌ Error loading sample: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test collate function
    logger.info("🔄 Testing collate function...")
    try:
        batch = variable_views_collate_fn([sample])
        logger.info(f"✅ Collate function successful")
        logger.info(f"   Batch clips shape: {batch['clips'].shape}")
        
        # Check for NaN in batch
        if torch.isnan(batch['clips']).any():
            nan_count = torch.isnan(batch['clips']).sum().item()
            logger.error(f"❌ NaN detected in batch! Count: {nan_count}")
            return False
        else:
            logger.info(f"✅ Batch is NaN-free")
            
    except Exception as e:
        logger.error(f"❌ Error in collate function: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test GPU augmentation if enabled
    if hasattr(args, 'gpu_augmentation') and args.gpu_augmentation:
        logger.info("🎮 Testing GPU augmentation...")
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            gpu_aug = create_gpu_augmentation(args, device)
            
            clips_gpu = batch['clips'].to(device)
            augmented = gpu_aug(clips_gpu)
            
            if torch.isnan(augmented).any():
                nan_count = torch.isnan(augmented).sum().item()
                logger.error(f"❌ NaN detected after GPU augmentation! Count: {nan_count}")
                return False
            else:
                logger.info(f"✅ GPU augmentation is NaN-free")
                
        except Exception as e:
            logger.error(f"❌ Error in GPU augmentation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    logger.info("🎉 All tests passed! No NaNs detected in the pipeline.")
    return True

def main():
    parser = argparse.ArgumentParser(description='Debug NaN origins in the data pipeline')
    parser.add_argument('--config', type=str, default='conf/config.yaml', help='Config file path')
    parser.add_argument('--sample-idx', type=int, default=0, help='Sample index to test')
    parser.add_argument('--dataset-path', type=str, default='mvfouls', help='Dataset path')
    parser.add_argument('--frames-per-clip', type=int, default=16, help='Frames per clip')
    parser.add_argument('--gpu-augmentation', action='store_true', help='Test GPU augmentation')
    
    args = parser.parse_args()
    
    # Setup debug environment
    setup_debug_environment()
    
    # Run test
    success = test_single_sample(args)
    
    if success:
        logger.info("✅ NaN trace test completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ NaN trace test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 
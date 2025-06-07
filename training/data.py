"""
Data loading, augmentation, and dataset utilities for Multi-Task Multi-View ResNet3D training.

This module handles dataset creation, transforms, data loaders, and augmentation pipelines.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop
import numpy as np
import random
import logging

# Import transforms directly
from pytorchvideo.transforms import ShortSideScale, Normalize as VideoNormalize

# Imports from our other files
from dataset import (
    SoccerNetMVFoulDataset, variable_views_collate_fn,
    TemporalJitter, RandomTemporalReverse, RandomFrameDropout,
    RandomBrightnessContrast, RandomSpatialCrop, RandomHorizontalFlip,
    RandomGaussianNoise, SeverityAwareAugmentation, ClassBalancedSampler,
    RandomRotation, RandomMixup, RandomCutout, RandomTimeWarp
)

logger = logging.getLogger(__name__)


# GPU-based augmentation classes using native PyTorch operations
class GPUTemporalJitter(nn.Module):
    """GPU-based temporal jittering"""
    def __init__(self, max_jitter=3):
        super().__init__()
        self.max_jitter = max_jitter
    
    def forward(self, video):
        if self.training and self.max_jitter > 0:
            # Handle multi-view format: (B, V, C, T, H, W) or single view: (B, C, T, H, W)
            if len(video.shape) == 6:  # Multi-view: (B, V, C, T, H, W)
                B, V, C, T, H, W = video.shape
                jitter = torch.randint(-self.max_jitter, self.max_jitter + 1, (B,), device=video.device)
                
                # Apply temporal shift by rolling frames
                for i, j in enumerate(jitter):
                    if j != 0:
                        video[i] = torch.roll(video[i], j.item(), dims=2)  # Roll along time dimension (index 2)
            else:  # Single view: (B, C, T, H, W)
                B, C, T, H, W = video.shape
                jitter = torch.randint(-self.max_jitter, self.max_jitter + 1, (B,), device=video.device)
                
                # Apply temporal shift by rolling frames
                for i, j in enumerate(jitter):
                    if j != 0:
                        video[i] = torch.roll(video[i], j.item(), dims=1)  # Roll along time dimension (index 1)
        return video


class GPURandomBrightness(nn.Module):
    """GPU-based brightness adjustment"""
    def __init__(self, strength=0.3):
        super().__init__()
        self.strength = strength
    
    def forward(self, video):
        if self.training:
            B = video.shape[0]
            # Handle multi-view format: (B, V, C, T, H, W) or single view: (B, C, T, H, W)
            if len(video.shape) == 6:  # Multi-view
                brightness_factors = 1.0 + (torch.rand(B, 1, 1, 1, 1, 1, device=video.device) - 0.5) * 2 * self.strength
            else:  # Single view
                brightness_factors = 1.0 + (torch.rand(B, 1, 1, 1, 1, device=video.device) - 0.5) * 2 * self.strength
            video = video * brightness_factors
            video = torch.clamp(video, 0, 1)
        return video


class GPURandomContrast(nn.Module):
    """GPU-based contrast adjustment"""
    def __init__(self, strength=0.3):
        super().__init__()
        self.strength = strength
    
    def forward(self, video):
        if self.training:
            B = video.shape[0]
            # Handle multi-view format: (B, V, C, T, H, W) or single view: (B, C, T, H, W)
            if len(video.shape) == 6:  # Multi-view
                contrast_factors = 1.0 + (torch.rand(B, 1, 1, 1, 1, 1, device=video.device) - 0.5) * 2 * self.strength
            else:  # Single view
                contrast_factors = 1.0 + (torch.rand(B, 1, 1, 1, 1, device=video.device) - 0.5) * 2 * self.strength
            # Apply contrast: (x - 0.5) * contrast + 0.5
            video = (video - 0.5) * contrast_factors + 0.5
            video = torch.clamp(video, 0, 1)
        return video


class GPURandomNoise(nn.Module):
    """GPU-based Gaussian noise"""
    def __init__(self, noise_strength=0.06):
        super().__init__()
        self.noise_strength = noise_strength
    
    def forward(self, video):
        if self.training:
            noise = torch.randn_like(video) * self.noise_strength
            video = video + noise
            video = torch.clamp(video, 0, 1)
        return video


class GPURandomHorizontalFlip(nn.Module):
    """GPU-based horizontal flip"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, video):
        if self.training:
            B = video.shape[0]
            flip_mask = torch.rand(B, device=video.device) < self.p
            for i, should_flip in enumerate(flip_mask):
                if should_flip:
                    video[i] = torch.flip(video[i], dims=[-1])  # Flip along width dimension (works for both formats)
        return video


class GPURandomFrameDropout(nn.Module):
    """GPU-based frame dropout"""
    def __init__(self, dropout_prob=0.1):
        super().__init__()
        self.dropout_prob = dropout_prob
    
    def forward(self, video):
        if self.training and self.dropout_prob > 0:
            # Handle multi-view format: (B, V, C, T, H, W) or single view: (B, C, T, H, W)
            if len(video.shape) == 6:  # Multi-view: (B, V, C, T, H, W)
                B, V, C, T, H, W = video.shape
                # Create dropout mask for each frame
                keep_mask = torch.rand(B, 1, 1, T, 1, 1, device=video.device) > self.dropout_prob
                # Ensure at least one frame remains per sample
                for i in range(B):
                    if not keep_mask[i, 0, 0, :, 0, 0].any():
                        keep_mask[i, 0, 0, 0, 0, 0] = True  # Keep first frame
            else:  # Single view: (B, C, T, H, W)
                B, C, T, H, W = video.shape
                # Create dropout mask for each frame
                keep_mask = torch.rand(B, 1, T, 1, 1, device=video.device) > self.dropout_prob
                # Ensure at least one frame remains per sample
                for i in range(B):
                    if not keep_mask[i, 0, :, 0, 0].any():
                        keep_mask[i, 0, 0, 0, 0] = True  # Keep first frame
            
            video = video * keep_mask.float()
        return video


class GPUAugmentationPipeline(nn.Module):
    """Complete GPU-based augmentation pipeline"""
    def __init__(self, 
                 temporal_jitter_strength=3,
                 color_strength=0.3,
                 noise_strength=0.06,
                 dropout_prob=0.1,
                 hflip_prob=0.5,
                 aggressive=True):
        super().__init__()
        
        self.augmentations = nn.ModuleList()
        
        if aggressive:
            # Aggressive augmentation for small datasets
            self.augmentations.extend([
                GPUTemporalJitter(max_jitter=temporal_jitter_strength),
                GPURandomBrightness(strength=color_strength),
                GPURandomContrast(strength=color_strength),
                GPURandomNoise(noise_strength=noise_strength),
                GPURandomHorizontalFlip(p=hflip_prob),
                GPURandomFrameDropout(dropout_prob=dropout_prob),
            ])
        else:
            # Basic augmentation
            self.augmentations.extend([
                GPURandomBrightness(strength=color_strength * 0.5),
                GPURandomHorizontalFlip(p=hflip_prob),
            ])
    
    def forward(self, video):
        for aug in self.augmentations:
            video = aug(video)
        return video


# Define transforms locally instead of importing from test file
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


def worker_init_fn(worker_id):
    """Initialize worker with unique seed for reproducibility."""
    torch.manual_seed(42 + worker_id)
    np.random.seed(42 + worker_id)


def create_transforms(args, is_training=True):
    """Create training or validation transforms based on configuration."""
    
    if args.gpu_augmentation and is_training:
        logger.info("🖥️  Using GPU-based augmentation pipeline for maximum throughput!")
        # Minimal CPU transforms - just basic preprocessing
        transform = Compose([
            ConvertToFloatAndScale(),
            VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ShortSideScale(size=args.img_height),
            PerFrameCenterCrop((args.img_height, args.img_width))
        ])
        
        logger.info(f"   - GPU augmentation mode: {'Aggressive' if args.aggressive_augmentation or args.extreme_augmentation else 'Standard'}")
        logger.info(f"   - Temporal jitter: ±{args.temporal_jitter_strength} frames")
        logger.info(f"   - Color variation: ±{args.color_aug_strength*100:.1f}%")
        logger.info(f"   - Gaussian noise: up to {args.noise_strength:.3f} std")
        logger.info(f"   - Frame dropout: {args.dropout_prob*100:.1f}% probability")
        
        return transform
    
    elif is_training and not args.disable_augmentation:
        # Traditional CPU-based augmentation pipeline
        logger.info("🖥️  Using CPU-based augmentation pipeline")
        
        # Enhanced transforms with AGGRESSIVE/EXTREME augmentation for small dataset
        augmentation_stages = [ConvertToFloatAndScale()]
        
        # STAGE 1: Aggressive temporal augmentations (before normalization)
        augmentation_stages.extend([
            TemporalJitter(max_jitter=args.temporal_jitter_strength),
            RandomTemporalReverse(prob=0.5 if args.extreme_augmentation else (0.4 if args.aggressive_augmentation else 0.2)),
            RandomFrameDropout(
                dropout_prob=args.dropout_prob * (1.5 if args.extreme_augmentation else 1.0),
                max_consecutive=min(4, args.temporal_jitter_strength + 1)
            ),
        ])
        
        # STAGE 1.5: EXTREME temporal augmentations (only in extreme mode)
        if args.extreme_augmentation:
            augmentation_stages.extend([
                RandomTimeWarp(warp_factor=0.3, prob=0.4),  # More aggressive time warping
                RandomMixup(alpha=0.3, prob=0.4),  # Inter-frame mixing
            ])
        
        # STAGE 2: Aggressive spatial augmentations
        augmentation_stages.extend([
            RandomSpatialCrop(
                crop_scale_range=(args.spatial_crop_strength * (0.9 if args.extreme_augmentation else 1.0), 1.0),
                prob=0.9 if args.extreme_augmentation else (0.8 if args.aggressive_augmentation else 0.5)
            ),
            RandomHorizontalFlip(prob=0.7 if args.extreme_augmentation else (0.6 if args.aggressive_augmentation else 0.5)),
        ])
        
        # STAGE 2.5: EXTREME spatial augmentations (only in extreme mode)
        if args.extreme_augmentation:
            augmentation_stages.extend([
                RandomRotation(max_angle=8, prob=0.5),  # Small rotations
                RandomCutout(max_holes=2, max_height=15, max_width=15, prob=0.5),  # Cutout augmentation
            ])
        
        # STAGE 3: Color/intensity augmentations (before normalization)
        augmentation_stages.extend([
            RandomBrightnessContrast(
                brightness_range=args.color_aug_strength * (1.2 if args.extreme_augmentation else 1.0),
                contrast_range=args.color_aug_strength * (1.2 if args.extreme_augmentation else 1.0),
                prob=0.9 if args.extreme_augmentation else (0.8 if args.aggressive_augmentation else 0.5)
            ),
            RandomGaussianNoise(
                std_range=(0.01, args.noise_strength * (1.3 if args.extreme_augmentation else 1.0)),
                prob=0.6 if args.extreme_augmentation else (0.5 if args.aggressive_augmentation else 0.3)
            ),
        ])
        
        # STAGE 4: Standard preprocessing
        augmentation_stages.extend([
            VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ShortSideScale(size=int(args.img_height * (1.5 if args.extreme_augmentation else (1.4 if args.aggressive_augmentation else 1.2)))),
            PerFrameCenterCrop((args.img_height, args.img_width)),
        ])
        
        # Create the final transform
        transform = Compose(augmentation_stages)
        
        # Log augmentation settings
        if args.extreme_augmentation:
            logger.info("🔥 EXTREME AUGMENTATION MODE ENABLED - Maximum data variety for tiny datasets!")
            logger.info(f"   - Temporal jitter: ±{args.temporal_jitter_strength} frames")
            logger.info(f"   - Frame dropout: {args.dropout_prob*1.5*100:.1f}% probability")
            logger.info(f"   - Time warping: 30% probability with 0.3 factor")
            logger.info(f"   - Inter-frame mixup: 40% probability")
            logger.info(f"   - Spatial crops: {args.spatial_crop_strength*0.9}-1.0 scale range")
            logger.info(f"   - Random rotation: ±8° with 50% probability")
            logger.info(f"   - Random cutout: 2 holes, 50% probability")
            logger.info(f"   - Color variation: ±{args.color_aug_strength*1.2*100:.1f}%")
            logger.info(f"   - Gaussian noise: up to {args.noise_strength*1.3:.3f} std")
            logger.info(f"   - Scale factor: {1.5}x for maximum crop variety")
        elif args.aggressive_augmentation:
            logger.info("🚀 AGGRESSIVE AUGMENTATION MODE ENABLED for small dataset!")
            logger.info(f"   - Temporal jitter: ±{args.temporal_jitter_strength} frames")
            logger.info(f"   - Frame dropout: {args.dropout_prob*100:.1f}% probability")
            logger.info(f"   - Spatial crops: {args.spatial_crop_strength}-1.0 scale range")
            logger.info(f"   - Color variation: ±{args.color_aug_strength*100:.1f}%")
            logger.info(f"   - Gaussian noise: up to {args.noise_strength:.3f} std")
        else:
            logger.info("📊 Standard augmentation mode")
        
        return transform
    
    else:
        # Standard validation transforms (NO augmentation for consistent evaluation)
        return Compose([
            ConvertToFloatAndScale(),
            VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ShortSideScale(size=args.img_height),
            PerFrameCenterCrop((args.img_height, args.img_width))
        ])


def create_gpu_augmentation(args, device):
    """Create GPU-based augmentation pipeline if enabled."""
    if not args.gpu_augmentation:
        return None
        
    # GPU augmentation will be applied in the training loop
    gpu_augmentation = GPUAugmentationPipeline(
        temporal_jitter_strength=args.temporal_jitter_strength,
        color_strength=args.color_aug_strength,
        noise_strength=args.noise_strength,
        dropout_prob=args.dropout_prob,
        hflip_prob=0.7 if args.extreme_augmentation else (0.6 if args.aggressive_augmentation else 0.5),
        aggressive=args.aggressive_augmentation or args.extreme_augmentation
    ).to(device)
    
    return gpu_augmentation


def create_datasets(args):
    """Create training and validation datasets."""
    
    # Create transforms
    train_transform = create_transforms(args, is_training=True)
    val_transform = create_transforms(args, is_training=False)
    
    logger.info("Loading datasets...")
    try:
        train_dataset = SoccerNetMVFoulDataset(
            dataset_path=args.mvfouls_path,
            split='train',
            frames_per_clip=args.frames_per_clip,
            target_fps=args.target_fps,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            max_views_to_load=args.max_views,  # None by default = use all views
            transform=train_transform,
            target_height=args.img_height,
            target_width=args.img_width
        )
        
        val_dataset = SoccerNetMVFoulDataset(
            dataset_path=args.mvfouls_path,
            split='valid', 
            frames_per_clip=args.frames_per_clip,
            target_fps=args.target_fps,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            max_views_to_load=args.max_views,  # None by default = use all views
            transform=val_transform,
            target_height=args.img_height,
            target_width=args.img_width
        )
    except FileNotFoundError as e:
        logger.error(f"Error loading dataset: {e}")
        raise
        
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.error("One or both datasets are empty after loading.")
        raise ValueError("Empty datasets detected")

    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def create_dataloaders(args, train_dataset, val_dataset):
    """Create training and validation dataloaders."""
    
    logger.info("Creating data loaders...")
    
    # Setup training loader with optional class balancing
    if args.use_class_balanced_sampler and not args.test_run:
        logger.info("🎯 Using ClassBalancedSampler to address class imbalance!")
        train_sampler = ClassBalancedSampler(
            train_dataset, 
            oversample_factor=args.oversample_factor
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            sampler=train_sampler,  # Use custom sampler
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=False,  # Disable to prevent worker memory accumulation
            prefetch_factor=2 if args.num_workers > 0 else None,  # Reduce prefetch to lower memory pressure
            drop_last=True,  # Better for training stability
            collate_fn=variable_views_collate_fn,
            worker_init_fn=worker_init_fn  # Ensure reproducibility
        )
        
        logger.info(f"   - Oversample factor: {args.oversample_factor}x for minority classes")
        logger.info(f"   - Effective training samples per epoch: {len(train_sampler)}")
        
    else:
        # Standard random sampling
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=False,  # Disable to prevent worker memory accumulation
            prefetch_factor=2 if args.num_workers > 0 else None,  # Reduce prefetch to lower memory pressure
            drop_last=True,  # Better for training stability
            collate_fn=variable_views_collate_fn,
            worker_init_fn=worker_init_fn  # Ensure reproducibility
        )
    
    # Determine number of workers for validation DataLoader
    if args.num_workers == 0:
        val_num_workers = 0
    elif args.num_workers == 1:
        val_num_workers = 1
    elif args.num_workers < 4: # Handles 2 or 3
        val_num_workers = 2
    else: # args.num_workers >= 4
        val_num_workers = args.num_workers // 2
        
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=val_num_workers,
        pin_memory=True,
        persistent_workers=False,  # Disable to prevent worker memory accumulation
        prefetch_factor=2 if val_num_workers > 0 else None,
        collate_fn=variable_views_collate_fn,
        worker_init_fn=worker_init_fn  # Ensure reproducibility
    )
    
    # Log data loading optimization details
    if args.num_workers > 0:
        logger.info(f"🚀 Async data loading enabled:")
        logger.info(f"   - Training workers: {args.num_workers} (prefetch_factor=4)")
        logger.info(f"   - Validation workers: {val_num_workers} (prefetch_factor=2)")
        logger.info(f"   - Pin memory: True (faster CPU->GPU transfers)")
        logger.info(f"   - Persistent workers: False (avoid memory accumulation)")
    else:
        logger.info(f"⚠️  Synchronous data loading (num_workers=0)")
        logger.info(f"   - This may cause GPU starvation and low utilization")
        logger.info(f"   - Consider setting num_workers >= 2 for better performance")
    
    return train_loader, val_loader


def log_dataset_recommendations(train_dataset):
    """Log dataset size recommendations."""
    dataset_size = len(train_dataset)
    
    if dataset_size < 100:
        logger.info("💡 RECOMMENDATION: Your dataset is very small (<100 samples)!")
        logger.info("   Consider using --extreme_augmentation for maximum data variety")
        logger.info("   Also try: --oversample_factor 6.0 --label_smoothing 0.15")
    elif dataset_size < 500:
        logger.info("💡 RECOMMENDATION: Your dataset is small (<500 samples)!")
        logger.info("   Current aggressive augmentation should help")
        logger.info("   Consider: --oversample_factor 4.0 --label_smoothing 0.1")
    elif dataset_size < 1000:
        logger.info("💡 RECOMMENDATION: Medium-sized dataset - current settings should work well")
    else:
        logger.info("💡 Large dataset detected - consider reducing augmentation intensity") 
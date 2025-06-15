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
    RandomRotation, RandomMixup, RandomCutout, RandomTimeWarp,
    VariableLengthAugmentation, MultiScaleTemporalAugmentation
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
                 noise_strength=0.02,
                 dropout_prob=0.05,
                 hflip_prob=0.5,
                 aggressive=False):
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
            # Basic augmentation for medium datasets - mirrors CPU moderate pipeline
            self.augmentations.extend([
                GPURandomBrightness(strength=color_strength * 0.5),  # Reduced color strength
                GPURandomContrast(strength=color_strength * 0.5),   # Reduced color strength
                GPURandomHorizontalFlip(p=hflip_prob),
                # Only include minimal temporal and noise for moderate mode
                GPURandomFrameDropout(dropout_prob=dropout_prob * 0.5),  # Very light dropout
            ])
    
    def forward(self, video):
        for aug in self.augmentations:
            video = aug(video)
        return video


class SeverityAwareGPUAugmentation(nn.Module):
    """
    Class-specific augmentation pipeline that applies different augmentation
    strengths based on the severity class of each sample in the batch.
    
    This approach increases augmentation intensity for minority classes (higher severity)
    while using gentler augmentation for majority classes.
    """
    def __init__(self, 
                 device,
                 temporal_jitter_strength=3,
                 color_strength_base=0.3,
                 noise_strength_base=0.02,
                 dropout_prob_base=0.05,
                 hflip_prob=0.5,
                 severity_multipliers=None):
        super().__init__()
        self.device = device
        
        # Default severity multipliers if not provided
        # Higher severity classes get stronger augmentation
        if severity_multipliers is None:
            self.severity_multipliers = {
                0: 1.0,   # Severity 0 - standard augmentation
                1: 0.8,   # Severity 1 (majority class) - gentler augmentation
                2: 1.1,   # Severity 2 - slightly stronger
                3: 1.2,   # Severity 3 - stronger
                4: 1.5,   # Severity 4 (rare) - much stronger
                5: 1.7    # Severity 5 (very rare) - most aggressive
            }
        else:
            self.severity_multipliers = severity_multipliers
            
        logger.info("Using Severity-Aware GPU Augmentation:")
        for severity, multiplier in self.severity_multipliers.items():
            logger.info(f"  Severity {severity}: {multiplier:.1f}x augmentation strength")
        
        # Base augmentation parameters
        self.temporal_jitter_strength = temporal_jitter_strength
        self.color_strength_base = color_strength_base
        self.noise_strength_base = noise_strength_base
        self.dropout_prob_base = dropout_prob_base
        self.hflip_prob = hflip_prob
        
        # Create augmentation modules for each severity level
        self.augmentation_pipelines = nn.ModuleDict()
        for severity, multiplier in self.severity_multipliers.items():
            self.augmentation_pipelines[str(severity)] = GPUAugmentationPipeline(
                temporal_jitter_strength=int(temporal_jitter_strength * multiplier),
                color_strength=color_strength_base * multiplier,
                noise_strength=noise_strength_base * multiplier,
                dropout_prob=min(dropout_prob_base * multiplier, 0.5),  # Cap at 0.5
                hflip_prob=hflip_prob,
                aggressive=True  # Always use aggressive for this pipeline
            )
    
    def forward(self, video, severity_labels):
        """
        Apply severity-specific augmentation to each sample in the batch.
        
        Args:
            video: Video tensor with shape (B, V, C, T, H, W) or (B, C, T, H, W)
            severity_labels: Tensor of severity labels for each sample in batch
        
        Returns:
            Augmented video tensor with same shape as input
        """
        if not self.training:
            return video
            
        # Create an output tensor of the same shape as input
        B = video.shape[0]
        out = video.clone()
        
        # Apply appropriate augmentation to each sample based on its severity
        for i in range(B):
            severity = severity_labels[i].item()
            severity_key = str(min(max(severity, 0), 5))  # Ensure severity is in [0,5]
            
            if severity_key in self.augmentation_pipelines:
                # Get sample and apply corresponding augmentation
                sample = video[i:i+1]  # Keep batch dimension
                out[i:i+1] = self.augmentation_pipelines[severity_key](sample)
                
        return out


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
        logger.info("Using GPU-based augmentation pipeline for maximum throughput!")
        
        # Determine device for GPU augmentation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create GPU augmentation
        gpu_augmentation = create_gpu_augmentation(args, device)
        
        # Wrapper class so that transform is picklable by DataLoader workers
        class GPUAugTransform(torch.nn.Module):
            """Picklable wrapper to apply a pre-created GPU augmentation pipeline."""
            def __init__(self, aug, device):
                super().__init__()
                self.aug = aug
                self.device = device

            def forward(self, clips):
                # Accept numpy or tensor
                if not isinstance(clips, torch.Tensor):
                    clips = torch.from_numpy(clips)

                # Ensure format (T, C, H, W)
                if clips.dim() == 4 and clips.shape[-1] == 3:  # (T, H, W, C)
                    clips = clips.permute(0, 3, 1, 2)

                clips = clips.to(self.device)
                clips = self.aug(clips.unsqueeze(0)).squeeze(0)
                return clips.cpu().numpy()

        return GPUAugTransform(gpu_augmentation, device)
    
    elif is_training and not args.disable_augmentation:
        logger.info("Using CPU-based augmentation pipeline")
        
        # Traditional CPU-based augmentation pipeline
        
        # Enhanced transforms with AGGRESSIVE/EXTREME augmentation for small dataset
        augmentation_stages = [ConvertToFloatAndScale()]
        
        # STAGE 1: Temporal augmentations (scaled by mode)
        if args.aggressive_augmentation or args.extreme_augmentation:
            # Aggressive temporal augmentations for small datasets
            augmentation_stages.extend([
                TemporalJitter(max_jitter=args.temporal_jitter_strength),
                RandomTemporalReverse(prob=0.5 if args.extreme_augmentation else 0.4),
                RandomFrameDropout(
                    dropout_prob=args.dropout_prob * (1.5 if args.extreme_augmentation else 1.0),
                    max_consecutive=min(4, args.temporal_jitter_strength + 1)
                ),
            ])
        else:
            # Moderate temporal augmentations for medium datasets
            augmentation_stages.extend([
                TemporalJitter(max_jitter=1),  # Light temporal jitter only
                RandomFrameDropout(
                    dropout_prob=args.dropout_prob * 0.5,  # Very light dropout
                    max_consecutive=2
                ),
            ])
        
        # STAGE 1.5: Domain shift reduction augmentations (for aggressive/extreme modes)
        if args.aggressive_augmentation or args.extreme_augmentation:
            augmentation_stages.extend([
                VariableLengthAugmentation(
                    min_frames=10 if args.extreme_augmentation else 12, 
                    max_frames=24 if args.extreme_augmentation else 20, 
                    action_position_variance=0.4 if args.extreme_augmentation else 0.3,
                    prob=0.4 if args.extreme_augmentation else 0.3,
                    target_frames=args.frames_per_clip
                ),
                MultiScaleTemporalAugmentation(
                    scale_factors=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0] if args.extreme_augmentation else [0.75, 1.0, 1.25, 1.5],
                    prob=0.3 if args.extreme_augmentation else 0.2,
                    target_frames=args.frames_per_clip
                )
            ])
        
        # STAGE 1.6: EXTREME temporal augmentations (only in extreme mode)
        if args.extreme_augmentation:
            augmentation_stages.extend([
                RandomTimeWarp(warp_factor=0.3, prob=0.4),  # More aggressive time warping
                RandomMixup(alpha=0.3, prob=0.4),  # Inter-frame mixing
            ])
        
        # STAGE 2: Spatial augmentations (scaled by mode)
        if args.aggressive_augmentation or args.extreme_augmentation:
            # Aggressive spatial augmentations for small datasets
            augmentation_stages.extend([
                RandomSpatialCrop(
                    crop_scale_range=(args.spatial_crop_strength * (0.9 if args.extreme_augmentation else 1.0), 1.0),
                    prob=0.9 if args.extreme_augmentation else 0.8
                ),
                RandomHorizontalFlip(prob=0.7 if args.extreme_augmentation else 0.6),
            ])
        else:
            # Moderate spatial augmentations for medium datasets - basic only
            augmentation_stages.extend([
                RandomHorizontalFlip(prob=0.5),  # Basic horizontal flip only
            ])
        
        # STAGE 2.5: EXTREME spatial augmentations (only in extreme mode)
        if args.extreme_augmentation:
            augmentation_stages.extend([
                RandomRotation(max_angle=8, prob=0.5),  # Small rotations
                RandomCutout(max_holes=2, max_height=15, max_width=15, prob=0.5),  # Cutout augmentation
            ])
        
        # STAGE 3: Color/intensity augmentations (before normalization)
        if args.aggressive_augmentation or args.extreme_augmentation:
            # Aggressive color augmentations for small datasets
            augmentation_stages.extend([
                RandomBrightnessContrast(
                    brightness_range=args.color_aug_strength * (1.2 if args.extreme_augmentation else 1.0),
                    contrast_range=args.color_aug_strength * (1.2 if args.extreme_augmentation else 1.0),
                    prob=0.9 if args.extreme_augmentation else 0.8
                ),
                RandomGaussianNoise(
                    std_range=(0.01, args.noise_strength * (1.3 if args.extreme_augmentation else 1.0)),
                    prob=0.6 if args.extreme_augmentation else 0.5
                ),
            ])
        else:
            # Moderate color augmentations for medium datasets - basic color jitter only
            augmentation_stages.extend([
                RandomBrightnessContrast(
                    brightness_range=args.color_aug_strength * 0.5,  # Reduced strength
                    contrast_range=args.color_aug_strength * 0.5,    # Reduced strength
                    prob=0.5  # Moderate probability
                ),
                # Minimal noise for moderate mode
                RandomGaussianNoise(
                    std_range=(0.005, args.noise_strength * 0.5),
                    prob=0.3
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
            logger.info(f"   - Variable length clips: 10-24 frames (40% prob) - reduces domain shift")
            logger.info(f"   - Multi-scale temporal: 0.5x-2.0x speeds (30% prob) - simulates different FPS")
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
            logger.info(f"   - Variable length clips: 12-20 frames (30% prob) - reduces domain shift")
            logger.info(f"   - Multi-scale temporal: 0.75x-1.5x speeds (20% prob) - simulates different FPS")
            logger.info(f"   - Spatial crops: {args.spatial_crop_strength}-1.0 scale range")
            logger.info(f"   - Color variation: ±{args.color_aug_strength*100:.1f}%")
            logger.info(f"   - Gaussian noise: up to {args.noise_strength:.3f} std")
        else:
            logger.info("MODERATE AUGMENTATION MODE for medium dataset (2916 clips)")
            logger.info("   - Light temporal jitter: ±1 frame only")
            logger.info("   - Minimal frame dropout: 2.5% probability")
            logger.info("   - Basic horizontal flip: 50% probability")
            logger.info(f"   - Reduced color jitter: ±{args.color_aug_strength*100:.1f}%")
            logger.info(f"   - Minimal noise: up to {args.noise_strength:.3f} std")
        
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
        
    # Check if we should use severity-aware augmentation
    if args.severity_aware_augmentation:
        # Severity-aware augmentation with adaptive strength per class
        gpu_augmentation = SeverityAwareGPUAugmentation(
            device=device,
            temporal_jitter_strength=args.temporal_jitter_strength,
            color_strength_base=args.color_aug_strength,
            noise_strength_base=args.noise_strength,
            dropout_prob_base=args.dropout_prob,
            hflip_prob=0.7 if args.extreme_augmentation else (0.6 if args.aggressive_augmentation else 0.5),
            severity_multipliers=None  # Use default multipliers
        )
        logger.info("🔥 Using severity-aware GPU augmentation pipeline!")
        return gpu_augmentation
        
    # Standard GPU augmentation (same for all samples)
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
    """Create training and validation dataloaders with optimization features."""
    
    logger.info("Creating data loaders...")
    
    # Import optimization features if available
    try:
        from .data_optimization import create_optimized_dataloader, DataLoadingProfiler
        use_optimization = getattr(args, 'enable_data_optimization', True)
        logger.info(f"Data loading optimization: {'Enabled' if use_optimization else 'Disabled'}")
    except ImportError:
        logger.warning("Data optimization module not available, using standard DataLoader")
        use_optimization = False
    
    # Setup training loader with optional class balancing
    if args.use_class_balanced_sampler and not args.test_run:
        # Choose between progressive or standard class balancing
        if args.progressive_class_balancing:
            logger.info("🚀 Using ProgressiveClassBalancedSampler for dynamic class balancing!")
            train_sampler = ProgressiveClassBalancedSampler(
                train_dataset,
                oversample_factor_start=args.progressive_start_factor,  # Start with mild oversampling
                oversample_factor_end=args.progressive_end_factor,  # End with full oversampling
                duration_epochs=args.progressive_epochs,  # Duration of progression
                current_epoch=0  # Start at epoch 0
            )
            
            logger.info(f"   - Progressive balancing from {args.progressive_start_factor}x to {args.progressive_end_factor}x")
            logger.info(f"   - Duration: {args.progressive_epochs} epochs")
            logger.info(f"   - Initial samples per epoch: {len(train_sampler)}")
        else:
            logger.info("🎯 Using ClassBalancedSampler to address class imbalance!")
            train_sampler = ClassBalancedSampler(
                train_dataset, 
                oversample_factor=args.oversample_factor
            )
            
            logger.info(f"   - Oversample factor: {args.oversample_factor}x for minority classes")
            logger.info(f"   - Effective training samples per epoch: {len(train_sampler)}")
        
        # Use optimized dataloader if available
        if use_optimization:
            train_loader = create_optimized_dataloader(
                dataset=train_dataset, 
                batch_size=args.batch_size, 
                sampler=train_sampler,  # Use custom sampler
                num_workers=args.num_workers,
                pin_memory=True,
                prefetch_factor=args.prefetch_factor,
                drop_last=True,  # Better for training stability
                collate_fn=variable_views_collate_fn,
                worker_init_fn=worker_init_fn,  # Ensure reproducibility
                enable_optimizations=True
            )
        else:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                sampler=train_sampler,  # Use custom sampler
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True if args.num_workers > 0 else False,  # Only enable if num_workers > 0
                prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                drop_last=True,  # Better for training stability
                collate_fn=variable_views_collate_fn,
                worker_init_fn=worker_init_fn  # Ensure reproducibility
            )
        
    else:
        # Standard random sampling
        if use_optimization:
            train_loader = create_optimized_dataloader(
                dataset=train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=args.num_workers,
                pin_memory=True,
                prefetch_factor=args.prefetch_factor,
                drop_last=True,  # Better for training stability
                collate_fn=variable_views_collate_fn,
                worker_init_fn=worker_init_fn,  # Ensure reproducibility
                enable_optimizations=True
            )
        else:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True if args.num_workers > 0 else False,  # Only enable if num_workers > 0
                prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
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
        
    # Create validation loader (optimization less critical for validation)
    if use_optimization:
        val_loader = create_optimized_dataloader(
            dataset=val_dataset, 
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=val_num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            collate_fn=variable_views_collate_fn,
            enable_optimizations=False  # Disable adaptive features for validation
        )
    else:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=val_num_workers,
            pin_memory=True,
            persistent_workers=True if val_num_workers > 0 else False,  # Only enable if num_workers > 0
            prefetch_factor=args.prefetch_factor if val_num_workers > 0 else None,
            collate_fn=variable_views_collate_fn,
            # worker_init_fn=worker_init_fn  # Ensure reproducibility
        )
    
    # Log data loading optimization details
    if args.num_workers > 0:
        logger.info("Async data loading enabled:")
        logger.info(f"   - Training workers: {args.num_workers} (prefetch_factor={args.prefetch_factor})")
        logger.info(f"   - Validation workers: {val_num_workers} (prefetch_factor={args.prefetch_factor})")
        logger.info(f"   - Pin memory: True (faster CPU->GPU transfers)")
        logger.info(f"   - Persistent workers: True (reduce startup overhead)")
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


class ProgressiveClassBalancedSampler(torch.utils.data.Sampler):
    """
    Progressive balanced sampler that gradually increases the representation
    of minority classes over the course of training.
    
    This addresses class imbalance while avoiding convergence issues by:
    1. Starting with a mild oversampling of minority classes
    2. Progressively increasing minority class representation
    3. Reaching a final balanced distribution by a target epoch
    
    Args:
        dataset: Dataset to sample from
        oversample_factor_start: Initial oversampling factor for minority classes
        oversample_factor_end: Final oversampling factor for minority classes
        duration_epochs: Number of epochs over which to progress from start to end factor
        current_epoch: Current epoch counter (updated externally)
        max_targets_multiplier: Maximum multiplier for minority class representation
    """
    def __init__(
        self, 
        dataset, 
        oversample_factor_start=1.5,
        oversample_factor_end=3.0,
        duration_epochs=15,
        current_epoch=0,
        max_targets_multiplier=3.0
    ):
        self.dataset = dataset
        self.oversample_factor_start = oversample_factor_start
        self.oversample_factor_end = oversample_factor_end
        self.duration_epochs = duration_epochs
        self.current_epoch = current_epoch
        self.max_targets_multiplier = max_targets_multiplier
        
        # Get class labels from dataset
        self.severity_labels = []
        for action in dataset.actions:
            self.severity_labels.append(action["label_severity"])
        
        # Compute the number of samples per severity level
        self.class_counts = {}
        for i in range(6):  # 6 severity levels (0-5)
            self.class_counts[i] = self.severity_labels.count(i)
        
        # Remove empty classes
        self.class_counts = {k: v for k, v in self.class_counts.items() if v > 0}
        
        # Compute the majority class count
        self.majority_count = max(self.class_counts.values())
        
        # Log initial class distribution
        logger.info("Initial class distribution for ProgressiveClassBalancedSampler:")
        for cls, count in sorted(self.class_counts.items()):
            logger.info(f"  Class {cls}: {count} samples")
        
        # Calculate current oversample factor based on epoch
        self._update_targets()
        
    def _update_targets(self):
        """Update target counts based on current epoch."""
        # Calculate the current oversample factor with smoother progression
        progress = min(self.current_epoch / self.duration_epochs, 1.0)
        # Use exponential smoothing for more gradual progression
        smooth_progress = 1 - (1 - progress) ** 2  # Quadratic easing
        current_factor = self.oversample_factor_start + smooth_progress * (self.oversample_factor_end - self.oversample_factor_start)
        
        # Calculate target counts for each class
        self.targets_per_class = {}
        for cls, count in self.class_counts.items():
            minority_ratio = count / self.majority_count
            
            if minority_ratio >= 0.8:
                # Large classes (≥80% of majority): no oversampling
                self.targets_per_class[cls] = count
            elif minority_ratio >= 0.4:
                # Medium classes (40-80% of majority): minimal oversampling
                class_specific_factor = 1.0 + (current_factor - 1.0) * 0.3  # 30% of full factor
                target_count = int(count * class_specific_factor)
                self.targets_per_class[cls] = min(target_count, int(self.majority_count * 1.2))  # Cap at 120% of majority
            else:
                # True minority classes (<40% of majority): progressive oversampling
                # Linear scaling based on rarity
                rarity_factor = (1.0 - minority_ratio) / 0.6  # Scale from 0 to 1 as ratio goes from 0.4 to 0
                class_specific_factor = 1.0 + (current_factor - 1.0) * (0.7 + 0.3 * rarity_factor)  # 70-100% of full factor
                
                # Cap the maximum oversampling
                class_specific_factor = min(class_specific_factor, self.max_targets_multiplier)
                
                # Calculate targets (but ensure at least the original count)
                target_count = max(int(count * class_specific_factor), count)
                self.targets_per_class[cls] = target_count
        
        # Calculate indices to sample
        self.indices_per_class = {cls: [] for cls in self.class_counts.keys()}
        for i, label in enumerate(self.severity_labels):
            if label in self.indices_per_class:
                self.indices_per_class[label].append(i)
        
        self.total_size = sum(self.targets_per_class.values())
    
    def set_epoch(self, epoch):
        """Update the current epoch - must be called at the start of each epoch."""
        self.current_epoch = epoch
        self._update_targets()
        
        # Log current sampling strategy
        if epoch % 5 == 0:  # Log every 5 epochs to avoid spam
            progress = min(self.current_epoch / self.duration_epochs, 1.0)
            current_factor = self.oversample_factor_start + progress * (self.oversample_factor_end - self.oversample_factor_start)
            logger.info(f"Epoch {epoch}: Progressive sampling at {progress*100:.0f}% ({current_factor:.2f}x factor)")
            logger.info("Current target distribution:")
            for cls, target in sorted(self.targets_per_class.items()):
                logger.info(f"  Class {cls}: {self.class_counts[cls]} → {target} samples ({target/self.class_counts[cls]:.1f}x)")
    
    def __iter__(self):
        """Return an iterator over the indices."""
        # Create a list of indices to sample
        indices = []
        for cls, count in self.targets_per_class.items():
            # Oversample class to reach target count
            class_indices = self.indices_per_class[cls]
            samples_needed = self.targets_per_class[cls]
            
            # Full cycles first
            full_cycles = samples_needed // len(class_indices)
            for _ in range(full_cycles):
                indices.extend(class_indices)
            
            # Then add remaining samples randomly
            remaining = samples_needed % len(class_indices)
            if remaining > 0:
                indices.extend(random.sample(class_indices, remaining))
        
        # Shuffle indices
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        """Return the total size of the sampler."""
        return self.total_size 
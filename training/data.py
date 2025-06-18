"""
Data loading, augmentation, and dataset utilities for Multi-Task Multi-View ResNet3D training.

This module handles dataset creation, transforms, data loaders, and augmentation pipelines.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import v2
import numpy as np
import cv2
import random
import logging
from pathlib import Path
from collections import defaultdict, deque
import time
import threading
from typing import Optional

# Distributed training imports
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# Import transforms directly
from pytorchvideo.transforms import ShortSideScale, Normalize as VideoNormalize

# Imports from our other files
from dataset import (
    SoccerNetMVFoulDataset, variable_views_collate_fn,
    TemporalJitter, RandomTemporalReverse, RandomFrameDropout,
    RandomBrightnessContrast, RandomSpatialCrop, RandomHorizontalFlip,
    RandomGaussianNoise, SeverityAwareAugmentation, ClassBalancedSampler,
    ActionBalancedSampler, AlternatingSampler,
    RandomRotation, RandomMixup, RandomCutout, RandomTimeWarp,
    VariableLengthAugmentation, MultiScaleTemporalAugmentation,
    # New advanced augmentations
    MultiScaleCrop, StrongColorJitter, VideoRandAugment, 
    VideoMixUp, VideoCutMix, TemporalMixUp
)

logger = logging.getLogger(__name__)

# Try to import Kornia for advanced GPU augmentation
try:
    import kornia
    import kornia.augmentation as K
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("Kornia not available. Using basic GPU augmentation only.")

# GPU-based augmentation classes using native PyTorch operations
import torch.nn.functional as F

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
            
            # Clamp contrast factors to prevent extreme values that could cause NaN
            contrast_factors = torch.clamp(contrast_factors, 0.1, 3.0)
            
            # Apply contrast: (x - 0.5) * contrast + 0.5
            video = (video - 0.5) * contrast_factors + 0.5
            video = torch.clamp(video, 0, 1)
            
            # Safety check for NaN values
            if torch.isnan(video).any():
                logger.warning("NaN detected in contrast augmentation - using original video")
                video = torch.nan_to_num(video, nan=0.5)
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
        # [NaN-origin] Step 5: GPU augmentations before/after pattern
        for aug_idx, aug in enumerate(self.augmentations):
            video_before = video.clone()
            video = aug(video)
            if torch.isnan(video).any() and not torch.isnan(video_before).any():
                logger.error(f"[NaN-origin] GPU augmentation {aug_idx}:{aug.__class__.__name__}")
                raise RuntimeError("NaN introduced by GPU augmentation")
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


class IdentityTransform(torch.nn.Module):
    """Identity transform that returns input unchanged."""
    def __call__(self, x):
        return x

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
        self.cropper = transforms.CenterCrop(size)

    def forward(self, clip_cthw):
        clip_tchw = clip_cthw.permute(1, 0, 2, 3)
        cropped_frames = [self.cropper(frame) for frame in clip_tchw]
        cropped_clip_tchw = torch.stack(cropped_frames)
        return cropped_clip_tchw.permute(1, 0, 2, 3)


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
        return clips  # Keep tensor on GPU to avoid PCIe round-trip


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
        
        return GPUAugTransform(gpu_augmentation, device)
    
    elif is_training and not args.disable_augmentation:
        logger.info("Using simplified CPU-based augmentation pipeline")
        
        # Simplified augmentation pipeline to avoid PyTorchVideo shape issues
        # Since videos are already processed in dataset.py, we only need basic augmentations
        augmentation_stages = [
            # Identity transform - videos are already processed in dataset.py
            IdentityTransform()
        ]
        
        # Create the final transform
        transform = transforms.Compose(augmentation_stages)
        
        return transform
    
    else:
        # Standard validation transforms (NO augmentation for consistent evaluation)
        # Note: Videos are already converted to float32, resized, and normalized in dataset.py
        # So we can use a minimal transform pipeline to avoid shape issues
        return transforms.Compose([
            # Identity transform - videos are already processed in dataset.py
            IdentityTransform()
        ])


def create_gpu_augmentation(args, device):
    """Create GPU-based augmentation pipeline if enabled."""
    if not args.gpu_augmentation:
        return None
    
    # Check if Kornia is available for advanced augmentation
    use_kornia = KORNIA_AVAILABLE and (getattr(args, 'use_randaugment', False) or 
                                       getattr(args, 'strong_color_jitter', False) or 
                                       getattr(args, 'multi_scale', False))
    
    if use_kornia:
        logger.info("🚀 Using advanced Kornia-based GPU augmentation pipeline!")
        gpu_augmentation = KorniaGPUAugmentationPipeline(
            use_randaugment=getattr(args, 'use_randaugment', False),
            use_strong_color=getattr(args, 'strong_color_jitter', False),
            use_multi_scale=getattr(args, 'multi_scale', False),
            randaugment_n=getattr(args, 'randaugment_n', 2),
            randaugment_m=getattr(args, 'randaugment_m', 10),
            aggressive=args.aggressive_augmentation or args.extreme_augmentation
        ).to(device)
        return gpu_augmentation
        
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
    
    logger.info("Using basic GPU augmentation pipeline")
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
            target_width=args.img_width,
            clips_per_video=getattr(args, 'clips_per_video', 1),
            clip_sampling=getattr(args, 'clip_sampling', 'uniform')
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
            target_width=args.img_width,
            clips_per_video=getattr(args, 'clips_per_video', 1),
            clip_sampling=getattr(args, 'clip_sampling', 'uniform')
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
    
    # Import sampling classes
    from dataset import ClassBalancedSampler, ActionBalancedSampler, AlternatingSampler
    
    # Import optimization features if available 
    try:
        from .data_optimization import create_optimized_dataloader, DataLoadingProfiler
        use_optimization = getattr(args, 'enable_data_optimization', True)
        logger.info(f"Data loading optimization: {'Enabled' if use_optimization else 'Disabled'}")
    except ImportError:
        logger.warning("Data optimization module not available, using standard DataLoader")
        use_optimization = False
    
    # Check if we're in distributed training mode
    is_distributed = dist.is_available() and dist.is_initialized()
    if is_distributed:
        logger.info(f"🌐 Distributed training detected: {dist.get_world_size()} GPUs")
    
    # Setup training loader with optional class balancing
    if args.use_class_balanced_sampler and not args.test_run:
        # Choose between different sampling strategies
        if args.use_alternating_sampler:
            logger.info("🔄 Using AlternatingSampler to address both severity and action imbalance!")
            if is_distributed:
                # For distributed training, wrap the sampler
                base_sampler = AlternatingSampler(
                    train_dataset,
                    severity_oversample_factor=args.oversample_factor,
                    action_oversample_factor=args.action_oversample_factor
                )
                train_sampler = DistributedSamplerWrapper(base_sampler)
                logger.info("   - Using DistributedSamplerWrapper for multi-GPU training")
            else:
                train_sampler = AlternatingSampler(
                    train_dataset,
                    severity_oversample_factor=args.oversample_factor,
                    action_oversample_factor=args.action_oversample_factor
                )
            
            logger.info(f"   - Severity oversample factor: {args.oversample_factor}x")
            logger.info(f"   - Action oversample factor: {args.action_oversample_factor}x")
            logger.info(f"   - Alternates between severity and action balancing per epoch")
            logger.info(f"   - Initial samples per epoch: {len(train_sampler)}")
        elif args.use_action_balanced_sampler_only:
            logger.info("⚖️ Using ActionBalancedSampler to address action type imbalance!")
            if is_distributed:
                train_sampler = DistributedActionBalancedSampler(
                    train_dataset, 
                    oversample_factor=args.action_oversample_factor
                )
                logger.info("   - Using distributed-aware ActionBalancedSampler")
            else:
                train_sampler = ActionBalancedSampler(
                    train_dataset, 
                    oversample_factor=args.action_oversample_factor
                )
            
            logger.info(f"   - Action oversample factor: {args.action_oversample_factor}x for minority action classes")
            logger.info(f"   - Effective training samples per epoch: {len(train_sampler)}")
        elif args.progressive_class_balancing:
            logger.info("🚀 Using ProgressiveClassBalancedSampler for dynamic class balancing!")
            if is_distributed:
                # For progressive sampler, wrap with DistributedSamplerWrapper since it's more complex
                base_sampler = ProgressiveClassBalancedSampler(
                    train_dataset,
                    oversample_factor_start=args.progressive_start_factor,
                    oversample_factor_end=args.progressive_end_factor,
                    duration_epochs=args.progressive_epochs,
                    current_epoch=0
                )
                train_sampler = DistributedSamplerWrapper(base_sampler)
                logger.info("   - Using DistributedSamplerWrapper for multi-GPU training")
            else:
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
            logger.info("🎯 Using ClassBalancedSampler to address severity class imbalance!")
            if is_distributed:
                train_sampler = DistributedClassBalancedSampler(
                    train_dataset, 
                    oversample_factor=args.oversample_factor
                )
                logger.info("   - Using distributed-aware ClassBalancedSampler")
            else:
                train_sampler = ClassBalancedSampler(
                    train_dataset, 
                    oversample_factor=args.oversample_factor
                )
            
            logger.info(f"   - Oversample factor: {args.oversample_factor}x for minority severity classes")
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
    val_sampler = None
    if is_distributed:
        # Use DistributedSampler for validation to ensure each GPU sees different data
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        logger.info("   - Using DistributedSampler for validation data")
    
    if use_optimization:
        val_loader = create_optimized_dataloader(
            dataset=val_dataset, 
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False if val_sampler is not None else False, 
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
            sampler=val_sampler,
            shuffle=False if val_sampler is not None else False, 
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
                class_specific_factor = 1.0 + (current_factor - 1.0) * 0.2  # Reduced from 30% to 20%
                target_count = int(count * class_specific_factor)
                self.targets_per_class[cls] = min(target_count, int(self.majority_count * 1.1))  # Reduced cap from 120% to 110%
            else:
                # True minority classes (<40% of majority): progressive oversampling
                # Linear scaling based on rarity
                rarity_factor = (1.0 - minority_ratio) / 0.6  # Scale from 0 to 1 as ratio goes from 0.4 to 0
                class_specific_factor = 1.0 + (current_factor - 1.0) * (0.5 + 0.3 * rarity_factor)  # Reduced from 70-100% to 50-80%
                
                # Cap the maximum oversampling to prevent extreme imbalance
                class_specific_factor = min(class_specific_factor, self.max_targets_multiplier * 0.7)  # Reduced max multiplier
                
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


class DistributedSamplerWrapper:
    """
    Wrapper that makes custom samplers compatible with distributed training.
    
    This ensures that:
    1. Each GPU sees different data (no duplication)
    2. The effective epoch size remains consistent regardless of number of GPUs
    3. Custom sampling logic (class balancing, etc.) is preserved
    
    Compatible with PyTorch >= 1.12's DistributedSamplerWrapper approach.
    """
    def __init__(self, sampler, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        
    def __iter__(self):
        # Get indices from the underlying sampler
        if hasattr(self.sampler, 'set_epoch'):
            # Set epoch for underlying sampler if it supports it
            self.sampler.set_epoch(self.epoch)
        
        indices = list(self.sampler)
        
        # Add extra samples to make it evenly divisible across GPUs
        indices += indices[:(self.num_replicas - len(indices) % self.num_replicas) % self.num_replicas]
        assert len(indices) % self.num_replicas == 0
        
        # Subsample for this rank
        indices = indices[self.rank:len(indices):self.num_replicas]
        
        if self.shuffle:
            # Shuffle indices in a deterministic way based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_indices = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in random_indices]
            
        return iter(indices)
    
    def __len__(self):
        return len(self.sampler) // self.num_replicas
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)


class DistributedClassBalancedSampler(torch.utils.data.Sampler):
    """
    Distributed-aware version of ClassBalancedSampler.
    
    Each GPU gets a balanced subsample of the data, ensuring:
    1. No data duplication across GPUs
    2. Class balance is maintained on each GPU
    3. Total epoch size is consistent regardless of number of GPUs
    """
    def __init__(self, dataset, oversample_factor=3.0, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available") 
            rank = dist.get_rank()
            
        self.dataset = dataset
        self.oversample_factor = oversample_factor
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        
        # Import from dataset.py
        from dataset import SEVERITY_LABELS
        
        # Count samples per severity class
        self.class_counts = defaultdict(int)
        self.class_indices = defaultdict(list)
        
        for idx, action in enumerate(dataset.actions):
            severity_label = action['label_severity']
            self.class_counts[severity_label] += 1
            self.class_indices[severity_label].append(idx)
        
        # Calculate sampling weights (same logic as original)
        max_count = 1
        if self.class_counts:
            max_count = max(self.class_counts.values())
        
        # Find the actual majority class (class with highest count)
        majority_class = max(self.class_counts, key=self.class_counts.get) if self.class_counts else 1
        
        self.sampling_weights = {}
        for class_id, count in self.class_counts.items():
            if count == 0:
                self.sampling_weights[class_id] = 0
                continue

            # Dynamic majority class detection instead of hardcoded severity "1.0"
            if class_id == majority_class:  # Actual majority class
                self.sampling_weights[class_id] = 1.0
            else:  # Minority classes
                self.sampling_weights[class_id] = min(oversample_factor, max_count / count)
        
        # Calculate total samples per epoch
        self.total_samples = 0
        for class_id, count in self.class_counts.items():
            if class_id in self.sampling_weights:
                self.total_samples += int(count * self.sampling_weights[class_id])
        
        # Calculate per-replica samples
        self.num_samples = self.total_samples // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas
        
        if self.rank == 0:
            logger.info(f"DistributedClassBalancedSampler: {self.total_samples} total samples across {self.num_replicas} GPUs")
            logger.info(f"Per-GPU samples: {self.num_samples}")
            logger.info(f"Class distribution: {dict(self.class_counts)}")
            logger.info(f"Sampling weights: {self.sampling_weights}")
    
    def __iter__(self):
        # Set random seed based on epoch for reproducibility
        g = torch.Generator()
        g.manual_seed(self.epoch)
        
        # Generate all indices using original logic
        all_indices = []
        for class_id, class_indices in self.class_indices.items():
            weight = self.sampling_weights[class_id]
            num_samples = int(len(class_indices) * weight)
            
            if weight > 1.0:
                # Use generator for deterministic sampling
                sampled_indices = [class_indices[torch.randint(len(class_indices), (1,), generator=g).item()] 
                                 for _ in range(num_samples)]
            else:
                sampled_indices = class_indices[:num_samples]
            
            all_indices.extend(sampled_indices)
        
        # Shuffle all indices
        shuffle_indices = torch.randperm(len(all_indices), generator=g).tolist()
        all_indices = [all_indices[i] for i in shuffle_indices]
        
        # Pad to be evenly divisible by num_replicas
        while len(all_indices) < self.total_size:
            all_indices.extend(all_indices[:self.total_size - len(all_indices)])
        
        # Subsample for this rank
        indices = all_indices[self.rank:self.total_size:self.num_replicas]
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedActionBalancedSampler(torch.utils.data.Sampler):
    """
    Distributed-aware version of ActionBalancedSampler.
    """
    def __init__(self, dataset, oversample_factor=3.0, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available") 
            rank = dist.get_rank()
            
        self.dataset = dataset
        self.oversample_factor = oversample_factor
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        
        # Import from dataset.py
        from dataset import ACTION_TYPE_LABELS
        
        # Count samples per action type class
        self.class_counts = defaultdict(int)
        self.class_indices = defaultdict(list)
        
        for idx, action in enumerate(dataset.actions):
            action_label = action['label_type']
            self.class_counts[action_label] += 1
            self.class_indices[action_label].append(idx)
        
        # Calculate sampling weights (same logic as original)
        max_count = 1
        if self.class_counts:
            max_count = max(self.class_counts.values())
        
        self.sampling_weights = {}
        for class_id, count in self.class_counts.items():
            if count == 0:
                self.sampling_weights[class_id] = 0
                continue

            if class_id == ACTION_TYPE_LABELS.get("Standing tackling", 8):  # Majority class
                self.sampling_weights[class_id] = 1.0
            else:  # Minority classes
                self.sampling_weights[class_id] = min(oversample_factor, max_count / count)
        
        # Calculate total samples per epoch
        self.total_samples = 0
        for class_id, count in self.class_counts.items():
            if class_id in self.sampling_weights:
                self.total_samples += int(count * self.sampling_weights[class_id])
        
        # Calculate per-replica samples
        self.num_samples = self.total_samples // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas
        
        if self.rank == 0:
            logger.info(f"DistributedActionBalancedSampler: {self.total_samples} total samples across {self.num_replicas} GPUs")
            logger.info(f"Per-GPU samples: {self.num_samples}")
            logger.info(f"Action class distribution: {dict(self.class_counts)}")
            logger.info(f"Action sampling weights: {self.sampling_weights}")
    
    def __iter__(self):
        # Set random seed based on epoch for reproducibility
        g = torch.Generator()
        g.manual_seed(self.epoch)
        
        # Generate all indices using original logic
        all_indices = []
        for class_id, class_indices in self.class_indices.items():
            weight = self.sampling_weights[class_id]
            num_samples = int(len(class_indices) * weight)
            
            if weight > 1.0:
                # Use generator for deterministic sampling
                sampled_indices = [class_indices[torch.randint(len(class_indices), (1,), generator=g).item()] 
                                 for _ in range(num_samples)]
            else:
                sampled_indices = class_indices[:num_samples]
            
            all_indices.extend(sampled_indices)
        
        # Shuffle all indices
        shuffle_indices = torch.randperm(len(all_indices), generator=g).tolist()
        all_indices = [all_indices[i] for i in shuffle_indices]
        
        # Pad to be evenly divisible by num_replicas
        while len(all_indices) < self.total_size:
            all_indices.extend(all_indices[:self.total_size - len(all_indices)])
        
        # Subsample for this rank
        indices = all_indices[self.rank:self.total_size:self.num_replicas]
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch 


# ================================
# ADVANCED KORNIA-BASED GPU AUGMENTATION
# ================================

class KorniaGPUAugmentationPipeline(nn.Module):
    """Advanced GPU augmentation using Kornia library"""
    def __init__(self, 
                 use_randaugment=False,
                 use_strong_color=False,
                 use_multi_scale=False,
                 randaugment_n=2,
                 randaugment_m=10,
                 aggressive=False):
        super().__init__()
        
        if not KORNIA_AVAILABLE:
            logger.warning("Kornia not available. Falling back to basic GPU augmentation.")
            self.use_kornia = False
            return
        
        self.use_kornia = True
        self.aggressive = aggressive
        
        # Create Kornia augmentation container
        aug_list = []
        
        # Basic spatial augmentations
        if aggressive:
            aug_list.extend([
                K.RandomHorizontalFlip(p=0.6),
                K.RandomRotation(degrees=8, p=0.4),
                K.RandomCrop(size=(224, 224), p=0.5, cropping_mode='resample'),
                K.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=0.7),
            ])
        else:
            aug_list.extend([
                K.RandomHorizontalFlip(p=0.5),
                K.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), p=0.3),
            ])
        
        # Multi-scale cropping
        if use_multi_scale:
            aug_list.append(
                K.RandomResizedCrop(
                    size=(224, 224), 
                    scale=(0.7, 1.0) if aggressive else (0.85, 1.0),
                    p=0.5 if aggressive else 0.3
                )
            )
        
        # Color augmentations
        if use_strong_color:
            aug_list.extend([
                K.ColorJitter(
                    brightness=0.4 if aggressive else 0.2,
                    contrast=0.4 if aggressive else 0.2,
                    saturation=0.4 if aggressive else 0.2,
                    hue=0.1 if aggressive else 0.05,
                    p=0.8 if aggressive else 0.5
                ),
                K.RandomGamma(gamma=(0.8, 1.2), p=0.3),
                K.RandomSolarize(thresholds=0.5, p=0.2 if aggressive else 0.1),
            ])
        else:
            aug_list.extend([
                K.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    p=0.5
                ),
            ])
        
        # Advanced augmentations for aggressive mode
        if aggressive:
            aug_list.extend([
                K.RandomChannelShuffle(p=0.1),
                K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.2),
                K.RandomMotionBlur(kernel_size=3, angle=35, direction=0.5, p=0.15),
                K.RandomGaussianNoise(mean=0., std=0.02, p=0.3),
            ])
        
        # RandAugment
        if use_randaugment:
            aug_list.append(
                K.auto.RandAugment(n=randaugment_n, magnitude=randaugment_m, p=0.6 if aggressive else 0.4)
            )
        
        # Create the sequential container
        self.kornia_aug = K.AugmentationSequential(*aug_list, data_keys=["input"])
    
    def forward(self, video):
        if not self.use_kornia:
            return video
        
        if not self.training:
            return video
        
        # [NaN-origin] Step 5: Check input to KorniaGPUAugmentationPipeline
        if torch.isnan(video).any():
            logger.error(f"[NaN-origin] KorniaGPUAugmentationPipeline input NaN")
            raise RuntimeError("NaN in KorniaGPUAugmentationPipeline input")
        
        # Handle different video tensor formats
        original_shape = video.shape
        
        if video.dim() == 5:  # (B, C, T, H, W)
            B, C, T, H, W = video.shape
            # Reshape to (B*T, C, H, W) for frame-wise augmentation
            video_2d = video.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)
            
            # Apply Kornia augmentations
            video_2d_aug = self.kornia_aug(video_2d)
            
            # Reshape back to (B, C, T, H, W)
            video = video_2d_aug.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
            
        elif video.dim() == 6:  # (B, V, C, T, H, W) - Multi-view
            B, V, C, T, H, W = video.shape
            # Reshape to (B*V*T, C, H, W)
            video_2d = video.permute(0, 1, 3, 2, 4, 5).contiguous().view(B*V*T, C, H, W)
            
            # Apply Kornia augmentations
            video_2d_aug = self.kornia_aug(video_2d)
            
            # Reshape back to (B, V, C, T, H, W)
            video = video_2d_aug.view(B, V, T, C, H, W).permute(0, 1, 3, 2, 4, 5).contiguous()
        
        # [NaN-origin] Step 5: Check output from KorniaGPUAugmentationPipeline
        if torch.isnan(video).any():
            logger.error(f"[NaN-origin] KorniaGPUAugmentationPipeline output NaN")
            raise RuntimeError("NaN in KorniaGPUAugmentationPipeline output")
        
        return video


class GPUVideoMixUp(nn.Module):
    """GPU-accelerated MixUp for video data"""
    def __init__(self, alpha=0.2, prob=0.5):
        super().__init__()
        self.alpha = alpha
        self.prob = prob
    
    def forward(self, video, labels=None):
        if not self.training or torch.rand(1).item() > self.prob:
            return video, labels
        
        batch_size = video.size(0)
        if batch_size < 2:
            return video, labels
        
        # Generate mixing parameter
        lam = torch.from_numpy(np.random.beta(self.alpha, self.alpha, (batch_size, 1, 1, 1, 1))).to(video.device)
        if video.dim() == 6:  # Multi-view
            lam = lam.unsqueeze(1)  # Add view dimension
        
        # Create random permutation
        indices = torch.randperm(batch_size, device=video.device)
        
        # Mix videos
        mixed_video = lam * video + (1 - lam) * video[indices]
        
        if labels is not None:
            lam_1d = lam.squeeze()
            return mixed_video, (labels, labels[indices], lam_1d)
        
        return mixed_video, None


class GPUVideoCutMix(nn.Module):
    """GPU-accelerated CutMix for video data"""
    def __init__(self, alpha=1.0, prob=0.5):
        super().__init__()
        self.alpha = alpha
        self.prob = prob
    
    def forward(self, video, labels=None):
        if not self.training or torch.rand(1).item() > self.prob:
            return video, labels
        
        batch_size = video.size(0)
        if batch_size < 2:
            return video, labels
        
        # Get dimensions
        if video.dim() == 6:  # Multi-view
            B, V, C, T, H, W = video.shape
        else:  # Single view
            B, C, T, H, W = video.shape
        
        # Generate cut parameters
        lam_original = np.random.beta(self.alpha, self.alpha)
        cut_ratio = np.sqrt(1.0 - lam_original)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        
        # Random position
        cx = torch.randint(0, W, (1,)).item()
        cy = torch.randint(0, H, (1,)).item()
        
        bbx1 = max(0, cx - cut_w // 2)
        bby1 = max(0, cy - cut_h // 2)
        bbx2 = min(W, cx + cut_w // 2)
        bby2 = min(H, cy + cut_h // 2)
        
        # Create random permutation
        indices = torch.randperm(batch_size, device=video.device)
        
        # Apply CutMix
        mixed_video = video.clone()
        if video.dim() == 6:  # Multi-view
            mixed_video[:, :, :, :, bby1:bby2, bbx1:bbx2] = video[indices][:, :, :, :, bby1:bby2, bbx1:bbx2]
        else:  # Single view
            mixed_video[:, :, :, bby1:bby2, bbx1:bbx2] = video[indices][:, :, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        if labels is not None:
            return mixed_video, (labels, labels[indices], torch.tensor(lam))
        
        return mixed_video, None 


class SafeVideoNormalize(torch.nn.Module):
    """
    Safe video normalization that prevents NaN values.
    
    This implementation adds epsilon to prevent division by zero
    and clamps extreme values to prevent NaN propagation.
    """
    def __init__(self, mean, std, eps=1e-6):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1, 1))
        self.eps = eps
    
    def forward(self, video):
        """
        Args:
            video: Tensor of shape (C, T, H, W) or (N, C, T, H, W)
        """
        # Ensure video is float
        if video.dtype == torch.uint8:
            video = video.float() / 255.0
        
        # Handle NaN/inf values before normalization
        if torch.isnan(video).any() or torch.isinf(video).any():
            logger.warning("Input video contains NaN/inf values before normalization")
            video = torch.nan_to_num(video, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Clamp to reasonable range to prevent extreme values
        video = torch.clamp(video, 0.0, 1.0)
        
        # Add epsilon to std to prevent division by zero
        safe_std = self.std + self.eps
        
        # Normalize
        normalized = (video - self.mean) / safe_std
        
        # Final safety check
        if torch.isnan(normalized).any() or torch.isinf(normalized).any():
            logger.warning("NaN/inf detected after normalization - applying fallback")
            normalized = torch.nan_to_num(normalized, nan=0.0, posinf=3.0, neginf=-3.0)
        
        # Clamp to reasonable range (approximately ±3 standard deviations)
        normalized = torch.clamp(normalized, -4.0, 4.0)
        
        return normalized 


class SafeVideoNormalize4D(torch.nn.Module):
    """
    Safe video normalization for 4D tensors (C, T, H, W).
    
    This is specifically designed for single video tensors without batch dimension,
    which is the format used in dataset transforms.
    """
    def __init__(self, mean, std, eps=1e-6):
        super().__init__()
        # For 4D tensors: (C, T, H, W) -> mean/std shape should be (C, 1, 1, 1)
        self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(-1, 1, 1, 1))
        self.eps = eps
    
    def forward(self, video):
        """
        Args:
            video: Tensor of shape (C, T, H, W)
        """
        # Ensure video is float
        if video.dtype == torch.uint8:
            video = video.float() / 255.0
        
        # Handle NaN/inf values before normalization
        if torch.isnan(video).any() or torch.isinf(video).any():
            logger.warning("Input video contains NaN/inf values before normalization")
            video = torch.nan_to_num(video, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Clamp to reasonable range to prevent extreme values
        video = torch.clamp(video, 0.0, 1.0)
        
        # Add epsilon to std to prevent division by zero
        safe_std = self.std + self.eps
        
        # Normalize
        normalized = (video - self.mean) / safe_std
        
        # Final safety check
        if torch.isnan(normalized).any() or torch.isinf(normalized).any():
            logger.warning("NaN/inf detected after normalization - applying fallback")
            normalized = torch.nan_to_num(normalized, nan=0.0, posinf=3.0, neginf=-3.0)
        
        # Clamp to reasonable range (approximately ±3 standard deviations)
        normalized = torch.clamp(normalized, -4.0, 4.0)
        
        return normalized 
# train.py
"""
Enhanced Multi-Task Multi-View ResNet3D Training Script with FLEXIBLE CLASS IMBALANCE HANDLING

FLEXIBLE LOSS FUNCTION OPTIONS:
===============================

⚙️  LOSS FUNCTION CONTROL:
Choose your loss function strategy:

1. 🎯 FOCAL LOSS (for severe class imbalance):
   --loss_function focal --focal_gamma 2.0
   - Automatically focuses on hard examples
   - Can use class weights as alpha parameter
   - Good for severe imbalance

2. 📊 WEIGHTED CROSS-ENTROPY (traditional approach):
   --loss_function weighted --class_weighting_strategy balanced_capped
   - Standard CrossEntropyLoss with class weights
   - Multiple weighting strategies available
   - Proven approach

3. 🆓 PLAIN CROSS-ENTROPY (no balancing):
   --loss_function plain
   - Standard CrossEntropyLoss with no modifications
   - Good for debugging and well-balanced datasets
   - Fastest training

DEBUGGING & FLEXIBILITY OPTIONS:
===============================

🔧 DISABLE FEATURES FOR DEBUGGING:
--disable_class_balancing        # Disable all class balancing (sampler + weights)
--disable_augmentation          # Disable all augmentation
--disable_in_model_augmentation # Disable only in-model augmentation
--simple_training               # Disable all advanced features (plain training)

🎛️  FINE-TUNE INDIVIDUAL FEATURES:
--loss_function [focal|weighted|plain]
--class_weighting_strategy [none|balanced_capped|sqrt|log]
--use_class_balanced_sampler [true|false]
--aggressive_augmentation [true|false]

USAGE EXAMPLES:
==============

# Plain training (no class balancing, no augmentation):
python train.py --simple_training

# Focal loss with class balancing:
python train.py --loss_function focal --focal_gamma 2.0

# Weighted loss with conservative settings:
python train.py --loss_function weighted --class_weighting_strategy sqrt

# Debug training (minimal features):
python train.py --loss_function plain --disable_class_balancing --disable_augmentation

# Previous behavior (weighted + balancing):
python train.py --loss_function weighted --class_weighting_strategy balanced_capped

CLASS IMBALANCE STRATEGIES (SAFE OPTIONS):
=========================================

⚠️  PROBLEM: Severe class imbalance can cause training instability with extreme loss weights!

RECOMMENDED SOLUTIONS (from most stable to least):

1. 🥇 FOCAL LOSS (MOST STABLE):
   --loss_function focal --focal_gamma 2.0
   - Automatically focuses on hard examples
   - No extreme weight ratios
   - Very stable training

2. 🥈 CAPPED CLASS WEIGHTS (STABLE):
   --loss_function weighted --class_weighting_strategy balanced_capped --max_weight_ratio 10.0
   - Limits maximum weight ratio (prevents instability)
   - Default: 10x max ratio (recommended)

3. 🥉 GENTLE WEIGHTING (VERY STABLE):
   --loss_function weighted --class_weighting_strategy sqrt  # or 'log' for even gentler
   - Gentler than inverse frequency
   - Much safer than raw inverse weights

4. 🚫 AVOID: Uncapped inverse frequency weighting (can cause 400x weight ratios!)

AUGMENTATION MODES FOR SMALL DATASETS:
=====================================

1. STANDARD MODE (default):
   - Basic augmentations for medium/large datasets
   - Use when you have >1000 training samples

2. AGGRESSIVE MODE (--aggressive_augmentation):
   - Enhanced augmentation pipeline for small datasets
   - Enabled by default, suitable for 200-1000 samples
   - Includes temporal jitter, frame dropout, spatial crops, color variation

3. EXTREME MODE (--extreme_augmentation):
   - Maximum augmentation for very small datasets (<200 samples)
   - Includes all aggressive augmentations PLUS:
     * Time warping and inter-frame mixup
     * Random rotations and cutout
     * More aggressive parameters
   
USAGE EXAMPLES:
==============

# For tiny datasets (<100 samples):
python train.py --extreme_augmentation --oversample_factor 6.0 --label_smoothing 0.15

# For small datasets (100-500 samples):
python train.py --aggressive_augmentation --oversample_factor 4.0 --label_smoothing 0.1

# For medium datasets (500+ samples):
python train.py --aggressive_augmentation false

TUNABLE PARAMETERS:
==================
--temporal_jitter_strength: Frame jitter amount (default: 3)
--dropout_prob: Frame dropout probability (default: 0.2)
--spatial_crop_strength: Min crop scale, lower=more aggressive (default: 0.7)
--color_aug_strength: Color variation strength (default: 0.3)
--noise_strength: Max Gaussian noise std (default: 0.06)
--oversample_factor: Minority class oversampling (default: 4.0)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torchvision.transforms import Compose, CenterCrop
from pathlib import Path
import argparse
import time
import os
import logging
import json
from collections import defaultdict
import numpy as np
import random
from tqdm import tqdm
import torch.nn.functional as F
import multiprocessing as mp

# Imports from our other files
from dataset import (
    SoccerNetMVFoulDataset, variable_views_collate_fn,
    TemporalJitter, RandomTemporalReverse, RandomFrameDropout,
    RandomBrightnessContrast, RandomSpatialCrop, RandomHorizontalFlip,
    RandomGaussianNoise, SeverityAwareAugmentation, ClassBalancedSampler,
    RandomRotation, RandomMixup, RandomCutout, RandomTimeWarp
)
from model import MultiTaskMultiViewResNet3D, ModelConfig

# Import transforms directly
from pytorchvideo.transforms import ShortSideScale, Normalize as VideoNormalize

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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration & Hyperparameters ---
def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-Task Multi-View ResNet3D for Foul Recognition')
    parser.add_argument('--dataset_root', type=str, default="", help='Root directory containing the mvfouls folder')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--backbone_name', type=str, default='r2plus1d_18', 
                        choices=['resnet3d_18', 'mc3_18', 'r2plus1d_18', 'resnet3d_50'], 
                        help="ResNet3D backbone variant (r2plus1d_18 recommended for best accuracy)")
    parser.add_argument('--frames_per_clip', type=int, default=16, help='Number of frames per clip')
    parser.add_argument('--target_fps', type=int, default=15, help='Target FPS for clips')
    parser.add_argument('--start_frame', type=int, default=67, help='Start frame index for foul-centered extraction (8 frames before foul at frame 75)')
    parser.add_argument('--end_frame', type=int, default=82, help='End frame index for foul-centered extraction (7 frames after foul at frame 75)')
    parser.add_argument('--img_height', type=int, default=224, help='Target image height')
    parser.add_argument('--img_width', type=int, default=398, help='Target image width (matches original VARS paper)')
    # Note: ResNet3D supports rectangular inputs unlike MViT
    parser.add_argument('--max_views', type=int, default=None, help='Optional limit on max views per action (default: use all available)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Advanced training options
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--resume_best_acc', type=float, default=None, 
                       help='Manually override best validation accuracy when resuming (useful for checkpoint issues)')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                       help='Patience for early stopping (epochs with no improvement).')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['cosine', 'onecycle', 'step', 'exponential', 'reduce_on_plateau', 'none'], 
                       help='Learning rate scheduler type')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    
    # Additional scheduler parameters
    parser.add_argument('--step_size', type=int, default=10, help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, help='Decay factor for StepLR/ExponentialLR')
    parser.add_argument('--plateau_patience', type=int, default=5, help='Patience for ReduceLROnPlateau')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    
    # Multi-task loss balancing
    parser.add_argument('--main_task_weights', type=float, nargs=2, default=[3.0, 3.0], 
                       help='Loss weights for [severity, action_type] - main tasks')
    parser.add_argument('--auxiliary_task_weight', type=float, default=0.5,
                       help='Loss weight for all auxiliary tasks (currently not implemented)')
    
    parser.add_argument('--attention_aggregation', action='store_true', default=True, help='Use attention for view aggregation')
    
    # Test run arguments
    parser.add_argument('--test_run', action='store_true', help='Perform a quick test run (1 epoch, few batches, no saving)')
    parser.add_argument('--test_batches', type=int, default=2, help='Number of batches to run in test mode')
    
    # New argument for forcing batch size even with multi-GPU
    parser.add_argument('--force_batch_size', action='store_true', 
                       help='Force specified batch size even with multi-GPU (disable auto-scaling)')
    
    # Multi-scale and advanced training options for higher accuracy
    parser.add_argument('--multi_scale', action='store_true', help='Enable multi-scale training for better accuracy')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor (0.1 recommended for small datasets)')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for regularization')
    parser.add_argument('--lr_warmup', action='store_true', help='Enable learning rate warmup')
    
    # === NEW FLEXIBLE LOSS FUNCTION CONTROLS ===
    parser.add_argument('--loss_function', type=str, default='weighted', 
                       choices=['focal', 'weighted', 'plain'],
                       help='Loss function type: focal (FocalLoss), weighted (weighted CrossEntropyLoss), plain (standard CrossEntropyLoss)')
    
    # === DEBUGGING & FLEXIBILITY OPTIONS ===
    parser.add_argument('--simple_training', action='store_true', default=False,
                       help='Enable simple training mode: disables class balancing, augmentation, and uses plain CrossEntropyLoss')
    parser.add_argument('--disable_class_balancing', action='store_true', default=False,
                       help='Disable all class balancing (both class weights and balanced sampler)')
    parser.add_argument('--disable_augmentation', action='store_true', default=False,
                       help='Disable all augmentation (both in-dataset and in-model)')
    
    # ENHANCED AUGMENTATION OPTIONS FOR SMALL DATASETS
    parser.add_argument('--use_class_balanced_sampler', action='store_true', default=True,
                       help='Use class-balanced sampler to oversample minority classes')
    parser.add_argument('--oversample_factor', type=float, default=4.0,
                       help='Factor by which to oversample minority classes (higher = more aggressive)')
    parser.add_argument('--aggressive_augmentation', action='store_true', default=True,
                       help='Enable aggressive augmentation pipeline for small datasets')
    parser.add_argument('--extreme_augmentation', action='store_true', default=False,
                       help='Enable EXTREME augmentation with all techniques (use for very small datasets)')
    parser.add_argument('--temporal_jitter_strength', type=int, default=3,
                       help='Max temporal jitter in frames (higher = more temporal variation)')
    parser.add_argument('--dropout_prob', type=float, default=0.2,
                       help='Frame dropout probability (higher = more aggressive temporal dropout)')
    parser.add_argument('--spatial_crop_strength', type=float, default=0.7,
                       help='Minimum crop scale (lower = more aggressive spatial crops)')
    parser.add_argument('--color_aug_strength', type=float, default=0.3,
                       help='Color augmentation strength (higher = more variation)')
    parser.add_argument('--noise_strength', type=float, default=0.06,
                       help='Maximum Gaussian noise standard deviation')
    
    # Advanced optimization
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd', 'adam'], help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    
    # Gradual fine-tuning options
    parser.add_argument('--gradual_finetuning', action='store_true', default=True, 
                       help='Enable gradual fine-tuning with discriminative learning rates')
    parser.add_argument('--phase1_epochs', type=int, default=8, 
                       help='Number of epochs for Phase 1 (head-only training)')
    parser.add_argument('--phase2_epochs', type=int, default=15, 
                       help='Number of epochs for Phase 2 (gradual unfreezing)')
    parser.add_argument('--head_lr', type=float, default=1e-3, 
                       help='Learning rate for classification heads in Phase 1')
    parser.add_argument('--backbone_lr', type=float, default=1e-5, 
                       help='Learning rate for unfrozen backbone layers in Phase 2')
    parser.add_argument('--unfreeze_blocks', type=int, default=3,  # Changed default from 2 to 3
                       help='Number of final residual blocks to unfreeze in Phase 2')
    
    # New arguments for more granular Phase 2 LR control
    parser.add_argument('--phase2_backbone_lr_scale_factor', type=float, default=1.0,
                       help='Scale factor for args.backbone_lr in Phase 2 (e.g., 2.0 for 2x base backbone_lr)')
    parser.add_argument('--phase2_head_lr_ratio', type=float, default=5.0,
                       help='Ratio for Phase 2 head LR relative to Phase 2 backbone LR (e.g., head_lr = p2_backbone_lr * ratio)')
    
    # New arguments for Phase 1 ReduceLROnPlateau scheduler
    parser.add_argument('--phase1_plateau_patience', type=int, default=3,
                       help='Patience for ReduceLROnPlateau scheduler in Phase 1 (gradual fine-tuning only).')
    parser.add_argument('--phase1_plateau_factor', type=float, default=0.2,
                       help='Factor for ReduceLROnPlateau scheduler in Phase 1 (gradual fine-tuning only).')
    
    # Modified class weighting options (now controlled by loss_function and disable_class_balancing)
    parser.add_argument('--class_weighting_strategy', type=str, default='balanced_capped',
                       choices=['none', 'balanced_capped', 'sqrt', 'log', 'effective_number'],
                       help='Strategy for calculating class weights (balanced_capped recommended for stability)')
    parser.add_argument('--max_weight_ratio', type=float, default=10.0,
                       help='Maximum ratio between highest and lowest class weight (prevents training instability)')
    
    # Focal loss specific parameters
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal Loss gamma parameter (higher = more focus on hard examples)')
    
    # Legacy arguments (for backward compatibility)
    parser.add_argument('--use_focal_loss', action='store_true', default=False,
                       help='DEPRECATED: Use --loss_function focal instead')
    parser.add_argument('--use_class_weighted_loss', action='store_true', default=True,
                       help='DEPRECATED: Use --loss_function weighted instead')
    
    # New argument to control in-model augmentation
    parser.add_argument('--disable_in_model_augmentation', action='store_true', default=False,
                       help='Disable VideoAugmentation applied inside the model forward pass (for bottleneck testing)')
    
    # New argument for memory cleanup interval
    parser.add_argument('--memory_cleanup_interval', type=int, default=20,
                       help='Interval (in batches) for calling memory cleanup. 0 or negative to disable in train/val loops.')
    
    args = parser.parse_args()
    
    # Handle simple training mode
    if args.simple_training:
        logger.info("🔧 SIMPLE TRAINING MODE ENABLED - Disabling advanced features for debugging")
        args.loss_function = 'plain'
        args.disable_class_balancing = True
        args.disable_augmentation = True
        args.disable_in_model_augmentation = True
        args.use_class_balanced_sampler = False
        args.aggressive_augmentation = False
        args.extreme_augmentation = False
        args.gradual_finetuning = False
        args.label_smoothing = 0.0
        logger.info("   - Loss function: plain CrossEntropyLoss")
        logger.info("   - Class balancing: disabled")
        logger.info("   - Augmentation: disabled")
        logger.info("   - Gradual fine-tuning: disabled")
        logger.info("   - Label smoothing: disabled")
    
    # Handle disable flags
    if args.disable_class_balancing:
        args.use_class_balanced_sampler = False
        args.class_weighting_strategy = 'none'
        logger.info("🚫 Class balancing disabled (both sampler and weights)")
    
    if args.disable_augmentation:
        args.aggressive_augmentation = False
        args.extreme_augmentation = False
        args.disable_in_model_augmentation = True
        logger.info("🚫 All augmentation disabled")
    
    # Handle legacy arguments and provide warnings
    if args.use_focal_loss and args.loss_function == 'weighted':
        logger.warning("⚠️  --use_focal_loss is deprecated. Setting --loss_function to 'focal'")
        args.loss_function = 'focal'
    
    if not args.use_class_weighted_loss and args.loss_function == 'weighted':
        logger.warning("⚠️  --use_class_weighted_loss=False detected. Setting --loss_function to 'plain'")
        args.loss_function = 'plain'
    
    # Construct the specific mvfouls path from the root
    if not args.dataset_root:
        raise ValueError("Please provide the --dataset_root argument.")
    args.mvfouls_path = str(Path(args.dataset_root) / "mvfouls")
    
    # Adjust total epochs for gradual fine-tuning
    if args.gradual_finetuning:
        args.total_epochs = args.epochs  # Keep user's original setting
        args.epochs = args.phase1_epochs + args.phase2_epochs  # Set actual training epochs
        logger.info(f"[GRADUAL] Gradual fine-tuning enabled: Phase 1={args.phase1_epochs} epochs, Phase 2={args.phase2_epochs} epochs")
    
    # Set default early stopping patience based on gradual fine-tuning
    if args.early_stopping_patience is None:
        if args.gradual_finetuning:
            args.early_stopping_patience = args.phase1_epochs + args.phase2_epochs // 2 # e.g., 8 + 15//2 = 8+7 = 15
            logger.info(f"💡 Default early stopping patience set to {args.early_stopping_patience} for gradual fine-tuning.")
        else:
            args.early_stopping_patience = 10 # Default for standard training
            logger.info(f"💡 Default early stopping patience set to {args.early_stopping_patience} for standard training.")
    else:
        logger.info(f"💡 User-defined early stopping patience: {args.early_stopping_patience}")
    
    return args

def set_seed(seed):
    """Sets seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For reproducibility in CuDNN operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_accuracy(outputs, labels):
    """Calculates accuracy for a single task."""
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total if total > 0 else 0

def calculate_f1_score(outputs, labels, num_classes):
    """Calculate F1 score for multi-class classification."""
    from sklearn.metrics import f1_score
    _, predicted = torch.max(outputs.data, 1)
    return f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted', zero_division=0)

def calculate_class_weights(dataset, num_classes, device, weighting_strategy='balanced_capped', max_weight_ratio=10.0):
    """
    Calculate class weights for loss balancing with multiple safe strategies.
    
    Args:
        dataset: Training dataset
        num_classes: Number of classes
        device: Device to put weights on
        weighting_strategy: Strategy for calculating weights
            - 'none': No weighting (all weights = 1.0)
            - 'balanced_capped': Inverse frequency with capped ratios (RECOMMENDED)
            - 'sqrt': Square root of inverse frequency (gentler)
            - 'log': Logarithmic weighting (very gentle)
            - 'effective_number': Uses effective number of samples
        max_weight_ratio: Maximum ratio between highest and lowest weight
    
    Returns:
        class_weights tensor
    """
    class_counts = torch.zeros(num_classes)
    
    for action in dataset.actions:
        severity_label = action['label_severity']
        if 0 <= severity_label < num_classes:
            class_counts[severity_label] += 1
    
    # Avoid division by zero
    class_counts[class_counts == 0] = 1
    
    if weighting_strategy == 'none':
        class_weights = torch.ones(num_classes)
        logger.info("Using no class weighting (all weights = 1.0)")
        
    elif weighting_strategy == 'balanced_capped':
        # Inverse frequency with capping - MUCH SAFER!
        total_samples = class_counts.sum()
        class_weights = total_samples / (num_classes * class_counts)
        
        # Cap the weights to prevent extreme ratios
        max_weight = class_weights.min() * max_weight_ratio
        class_weights = torch.clamp(class_weights, max=max_weight)
        
        # Normalize so minimum weight is 1.0
        class_weights = class_weights / class_weights.min()
        
        logger.info(f"Balanced capped weighting (max ratio: {max_weight_ratio})")
        
    elif weighting_strategy == 'sqrt':
        # Square root weighting - gentler than inverse frequency
        total_samples = class_counts.sum()
        class_weights = torch.sqrt(total_samples / (num_classes * class_counts))
        class_weights = class_weights / class_weights.min()
        
        logger.info("Square root weighting (gentler)")
        
    elif weighting_strategy == 'log':
        # Logarithmic weighting - very gentle
        total_samples = class_counts.sum()
        raw_weights = total_samples / (num_classes * class_counts)
        class_weights = torch.log(raw_weights + 1)  # +1 to avoid log(0)
        class_weights = class_weights / class_weights.min()
        
        logger.info("Logarithmic weighting (very gentle)")
        
    elif weighting_strategy == 'effective_number':
        # Effective number of samples - sophisticated approach
        beta = 0.999  # Hyperparameter
        effective_num = 1.0 - torch.pow(beta, class_counts.float())
        class_weights = (1.0 - beta) / effective_num
        class_weights = class_weights / class_weights.min()
        
        logger.info(f"Effective number weighting (beta={beta})")
        
    else:
        raise ValueError(f"Unknown weighting strategy: {weighting_strategy}")
    
    # Log class distribution and weights
    logger.info("Class distribution and weights:")
    for i in range(num_classes):
        if class_counts[i] > 0:
            logger.info(f"  Class {i}: {int(class_counts[i])} samples → Weight: {class_weights[i]:.2f}")
    
    max_ratio = (class_weights.max() / class_weights.min()).item()
    logger.info(f"Weight ratio (max/min): {max_ratio:.1f}")
    
    if max_ratio > 50:
        logger.warning(f"⚠️  Large weight ratio ({max_ratio:.1f}) may cause training instability!")
        logger.warning("   Consider using 'sqrt' or 'log' weighting strategy")
    
    return class_weights.to(device)

def calculate_multitask_loss(sev_logits, act_logits, batch_data, main_weights, aux_weight=0.5, 
                           label_smoothing=0.0, severity_class_weights=None, loss_function='weighted', focal_gamma=2.0):
    """
    Calculate weighted multi-task loss with flexible loss function selection.
    
    Args:
        sev_logits: Severity classification logits
        act_logits: Action type classification logits  
        batch_data: Batch data containing labels
        main_weights: [severity_weight, action_weight] for main tasks
        aux_weight: Weight for auxiliary tasks (currently not implemented)
        label_smoothing: Label smoothing factor for regularization
        severity_class_weights: Optional class weights for severity loss
        loss_function: 'focal', 'weighted', or 'plain'
        focal_gamma: Focal Loss gamma parameter
    
    Returns:
        total_loss, loss_sev, loss_act
    """
    if loss_function == 'focal':
        # Use Focal Loss - often more stable for class imbalance
        focal_criterion = FocalLoss(
            alpha=severity_class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing
        )
        loss_sev = focal_criterion(sev_logits, batch_data["label_severity"]) * main_weights[0]
        
        # Use standard CrossEntropyLoss for action type (usually more balanced)
        loss_act = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(act_logits, batch_data["label_type"]) * main_weights[1]
        
    elif loss_function == 'weighted':
        # Use class-weighted CrossEntropyLoss
        loss_sev = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, 
            weight=severity_class_weights
        )(sev_logits, batch_data["label_severity"]) * main_weights[0]
        
        loss_act = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(act_logits, batch_data["label_type"]) * main_weights[1]
    
    elif loss_function == 'plain':
        # Use plain CrossEntropyLoss (no class weights)
        loss_sev = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(sev_logits, batch_data["label_severity"]) * main_weights[0]
        loss_act = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(act_logits, batch_data["label_type"]) * main_weights[1]
    
    else:
        raise ValueError(f"Unknown loss_function: {loss_function}. Must be 'focal', 'weighted', or 'plain'")
    
    total_loss = loss_sev + loss_act
    
    return total_loss, loss_sev, loss_act

class EarlyStopping:
    """Early stopping utility."""
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        """Save model weights."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

def train_one_epoch(model, dataloader, criterion_severity, criterion_action, optimizer, device, 
                   scaler=None, max_batches=None, loss_weights=[1.0, 1.0], gradient_clip_norm=1.0, 
                   label_smoothing=0.0, severity_class_weights=None, loss_function='weighted', focal_gamma=2.0):
    model.train()
    running_loss = 0.0
    running_sev_acc = 0.0
    running_act_acc = 0.0
    running_sev_f1 = 0.0
    running_act_f1 = 0.0
    processed_batches = 0

    start_time = time.time()
    
    # Clean training without progress bar spam
    total_batches = max_batches if max_batches else len(dataloader)
    
    for i, batch_data in enumerate(dataloader):
        if max_batches is not None and (i + 1) > max_batches:
            break 
        
        # Move all tensors in the batch to the device
        for key in batch_data:
            if isinstance(batch_data[key], torch.Tensor):
                batch_data[key] = batch_data[key].to(device, non_blocking=True)
            
        severity_labels = batch_data["label_severity"]
        action_labels = batch_data["label_type"]

        optimizer.zero_grad()

        # Mixed precision forward pass (fixed deprecation warning)
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                sev_logits, act_logits = model(batch_data)
                total_loss, loss_sev_weighted, loss_act_weighted = calculate_multitask_loss(
                    sev_logits, act_logits, batch_data, loss_weights, 
                    label_smoothing=label_smoothing, severity_class_weights=severity_class_weights,
                    loss_function=loss_function, focal_gamma=focal_gamma
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            sev_logits, act_logits = model(batch_data)
            total_loss, loss_sev_weighted, loss_act_weighted = calculate_multitask_loss(
                sev_logits, act_logits, batch_data, loss_weights, 
                label_smoothing=label_smoothing, severity_class_weights=severity_class_weights,
                loss_function=loss_function, focal_gamma=focal_gamma
            )

            total_loss.backward()
            
            # Gradient clipping
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            
            optimizer.step()

        # Calculate metrics
        running_loss += total_loss.item() * batch_data["clips"].size(0)
        sev_acc = calculate_accuracy(sev_logits, severity_labels)
        act_acc = calculate_accuracy(act_logits, action_labels)
        sev_f1 = calculate_f1_score(sev_logits, severity_labels, 5)  # 5 severity classes
        act_f1 = calculate_f1_score(act_logits, action_labels, 9)  # 9 action classes
        
        running_sev_acc += sev_acc
        running_act_acc += act_acc
        running_sev_f1 += sev_f1
        running_act_f1 += act_f1
        processed_batches += 1
        
        # Periodic memory cleanup during training
        if args.memory_cleanup_interval > 0 and (i + 1) % args.memory_cleanup_interval == 0:
            cleanup_memory()
        
        # Only print progress every 25% of batches
        if (i + 1) % max(1, total_batches // 4) == 0:
            current_avg_loss = running_loss / (processed_batches * batch_data["clips"].size(0))
            current_avg_sev_acc = running_sev_acc / processed_batches
            current_avg_act_acc = running_act_acc / processed_batches
            progress = (i + 1) / total_batches * 100
            logger.info(f"  Training Progress: {progress:.0f}% | Loss: {current_avg_loss:.3f} | SevAcc: {current_avg_sev_acc:.3f} | ActAcc: {current_avg_act_acc:.3f}")
    
    num_samples_processed = len(dataloader.dataset) if max_batches is None else processed_batches * dataloader.batch_size 
    epoch_loss = running_loss / num_samples_processed if num_samples_processed > 0 else 0
    epoch_sev_acc = running_sev_acc / processed_batches if processed_batches > 0 else 0
    epoch_act_acc = running_act_acc / processed_batches if processed_batches > 0 else 0
    epoch_sev_f1 = running_sev_f1 / processed_batches if processed_batches > 0 else 0
    epoch_act_f1 = running_act_f1 / processed_batches if processed_batches > 0 else 0
    
    epoch_time = time.time() - start_time
    return epoch_loss, epoch_sev_acc, epoch_act_acc, epoch_sev_f1, epoch_act_f1

def cleanup_memory():
    """Clean up GPU memory to prevent accumulation."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def validate_one_epoch(model, dataloader, criterion_severity, criterion_action, device, 
                      max_batches=None, loss_weights=[1.0, 1.0], label_smoothing=0.0, 
                      severity_class_weights=None, loss_function='weighted', focal_gamma=2.0):
    model.eval()
    running_loss = 0.0
    running_sev_acc = 0.0
    running_act_acc = 0.0
    running_sev_f1 = 0.0
    running_act_f1 = 0.0
    processed_batches = 0

    start_time = time.time()
    
    # Clean validation without progress bar spam
    total_batches = max_batches if max_batches else len(dataloader)
    
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            if max_batches is not None and (i + 1) > max_batches:
                break
            
            # Move all tensors in the batch to the device
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(device, non_blocking=True)
                
            severity_labels = batch_data["label_severity"]
            action_labels = batch_data["label_type"]

            # Apply autocast for consistency with training if using CUDA and mixed precision is enabled for training
            # We infer mixed precision enablement from whether scaler was used in train,
            # or more directly, if args.mixed_precision is True (assuming args is accessible or passed)
            # For simplicity here, we'll check device type.
            if device.type == 'cuda': # Assuming mixed precision is used if cuda is available and training uses it.
                with torch.amp.autocast('cuda'): # Enable AMP for the forward pass
                    sev_logits, act_logits = model(batch_data)
                    total_loss, loss_sev_weighted, loss_act_weighted = calculate_multitask_loss(
                        sev_logits, act_logits, batch_data, loss_weights, 
                        label_smoothing=label_smoothing, severity_class_weights=severity_class_weights,
                        loss_function=loss_function, focal_gamma=focal_gamma
                    )
            else: # CPU or other device, or if mixed precision is explicitly disabled for validation
                sev_logits, act_logits = model(batch_data)
                total_loss, loss_sev_weighted, loss_act_weighted = calculate_multitask_loss(
                    sev_logits, act_logits, batch_data, loss_weights, 
                    label_smoothing=label_smoothing, severity_class_weights=severity_class_weights,
                    loss_function=loss_function, focal_gamma=focal_gamma
                )

            running_loss += total_loss.item() * batch_data["clips"].size(0)
            sev_acc = calculate_accuracy(sev_logits, severity_labels)
            act_acc = calculate_accuracy(act_logits, action_labels)
            running_sev_acc += sev_acc
            running_act_acc += act_acc
            running_sev_f1 += calculate_f1_score(sev_logits, severity_labels, 5)
            running_act_f1 += calculate_f1_score(act_logits, action_labels, 9)  # 9 action classes
            processed_batches += 1
            
            # Clean up batch data explicitly
            del batch_data, sev_logits, act_logits, total_loss
            
            # Periodic memory cleanup during validation
            if args.memory_cleanup_interval > 0 and (i + 1) % args.memory_cleanup_interval == 0:
                cleanup_memory()

    num_samples_processed = len(dataloader.dataset) if max_batches is None else processed_batches * dataloader.batch_size
    epoch_loss = running_loss / num_samples_processed if num_samples_processed > 0 else 0
    epoch_sev_acc = running_sev_acc / processed_batches if processed_batches > 0 else 0
    epoch_act_acc = running_act_acc / processed_batches if processed_batches > 0 else 0
    epoch_sev_f1 = running_sev_f1 / processed_batches if processed_batches > 0 else 0
    epoch_act_f1 = running_act_f1 / processed_batches if processed_batches > 0 else 0
    
    # Critical: Clean up memory after validation
    cleanup_memory()
    
    epoch_time = time.time() - start_time
    return epoch_loss, epoch_sev_acc, epoch_act_acc, epoch_sev_f1, epoch_act_f1

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, filepath):
    """Save training checkpoint."""
    # Handle DataParallel models
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, scaler=None):
    """Load training checkpoint with DataParallel compatibility."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Handle DataParallel state dict key mismatch
    state_dict = checkpoint['model_state_dict']
    model_state_dict = model.state_dict()
    
    # Check if we need to add or remove 'module.' prefix
    model_keys = list(model_state_dict.keys())
    checkpoint_keys = list(state_dict.keys())
    
    if len(model_keys) > 0 and len(checkpoint_keys) > 0:
        model_has_module = model_keys[0].startswith('module.')
        checkpoint_has_module = checkpoint_keys[0].startswith('module.')
        
        if model_has_module and not checkpoint_has_module:
            # Model is DataParallel wrapped, checkpoint is not - add 'module.' prefix
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            logger.info("Added 'module.' prefix to checkpoint keys for DataParallel compatibility")
        elif not model_has_module and checkpoint_has_module:
            # Model is not DataParallel wrapped, checkpoint is - remove 'module.' prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            logger.info("Removed 'module.' prefix from checkpoint keys for DataParallel compatibility")
    
    model.load_state_dict(state_dict)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    logger.info(f"Checkpoint loaded from {filepath}")
    return checkpoint['epoch'], checkpoint.get('metrics', {})

def freeze_backbone(model):
    """Freeze all backbone parameters."""
    # Handle DataParallel wrapper
    actual_model = model.module if hasattr(model, 'module') else model
    
    for param in actual_model.backbone.parameters():
        param.requires_grad = False
    
    logger.info("[FREEZE] Backbone frozen - only training classification heads")

def unfreeze_backbone_gradually(model, num_blocks_to_unfreeze=2):
    """Gradually unfreeze the last N residual blocks of the backbone."""
    # Handle DataParallel wrapper
    actual_model = model.module if hasattr(model, 'module') else model
    
    # Get the ResNet3D backbone
    backbone = actual_model.backbone.backbone
    
    # For ResNet architectures, we want to unfreeze the last few layers
    # ResNet3D structure: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool
    layers_to_unfreeze = []
    
    if hasattr(backbone, 'layer4'):
        layers_to_unfreeze.append(backbone.layer4)
    if hasattr(backbone, 'layer3') and num_blocks_to_unfreeze > 1:
        layers_to_unfreeze.append(backbone.layer3)
    if hasattr(backbone, 'layer2') and num_blocks_to_unfreeze > 2:
        layers_to_unfreeze.append(backbone.layer2)
    
    unfrozen_params = 0
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()
    
    logger.info(f"[UNFREEZE] Unfroze last {len(layers_to_unfreeze)} backbone layers ({unfrozen_params:,} parameters)")

def setup_discriminative_optimizer(model, head_lr, backbone_lr):
    """Setup optimizer with discriminative learning rates for different model parts."""
    # Handle DataParallel wrapper
    actual_model = model.module if hasattr(model, 'module') else model
    
    param_groups = []
    
    # Classification heads with higher learning rate
    head_params = []
    head_params.extend(actual_model.severity_head.parameters())
    head_params.extend(actual_model.action_type_head.parameters())
    
    # Add embedding and aggregation parameters to head group (they're task-specific)
    head_params.extend(actual_model.embedding_manager.parameters())
    head_params.extend(actual_model.view_aggregator.parameters())
    
    param_groups.append({
        'params': head_params,
        'lr': head_lr,
        'name': 'heads'
    })
    
    # Backbone parameters with lower learning rate (only unfrozen ones)
    backbone_params = [p for p in actual_model.backbone.parameters() if p.requires_grad]
    
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': backbone_lr,
            'name': 'backbone'
        })
    
    logger.info(f"[OPTIM] Discriminative LR setup: Heads={head_lr:.1e}, Backbone={backbone_lr:.1e}")
    return param_groups

def get_phase_info(epoch, phase1_epochs, total_epochs):
    """Determine current training phase."""
    if epoch < phase1_epochs:
        return 1, f"Phase 1: Head-only training"
    else:
        return 2, f"Phase 2: Gradual unfreezing"

def log_trainable_parameters(model):
    """Log the number of trainable parameters."""
    # Handle DataParallel wrapper
    actual_model = model.module if hasattr(model, 'module') else model
    
    total_params = sum(p.numel() for p in actual_model.parameters())
    trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    logger.info(f"[PARAMS] Total={total_params:,}, Trainable={trainable_params:,}, Frozen={frozen_params:,}")
    return trainable_params, total_params

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance - often more stable than class weighting.
    
    Focal Loss = -α(1-pt)^γ * log(pt)
    where pt is the model's confidence for the correct class.
    
    Benefits over class weighting:
    - Automatically focuses on hard examples
    - No extreme weight ratios
    - More stable training dynamics
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha  # Class balancing weights (optional)
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # Standard cross entropy with label smoothing
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, 
                                reduction='none', label_smoothing=self.label_smoothing)
        
        # Calculate pt (model confidence for correct class)
        pt = torch.exp(-ce_loss)
        
        # Apply focal term: (1-pt)^gamma
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

if __name__ == "__main__":
    # Configure multiprocessing for DataLoader workers
    try:
        # Use 'spawn' method to avoid process spawning issues on vast.ai/cloud platforms
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn', force=True)
            logger.info("🔧 Set multiprocessing start method to 'spawn' for stability")
    except RuntimeError as e:
        logger.warning(f"Could not set multiprocessing method: {e}")
        # Continue anyway - this isn't critical
    
    args = parse_args()
    
    # Test run setup
    if args.test_run:
        logger.info("=" * 60)
        logger.info(f"PERFORMING TEST RUN (1 Epoch, {args.test_batches} Batches)")
        logger.info("Model checkpoints will NOT be saved.")
        logger.info("=" * 60)
        args.epochs = 1
        num_batches_to_run = args.test_batches
    else:
        num_batches_to_run = None
        
    # Set seed and create directories
    set_seed(args.seed)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Multi-GPU setup
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        logger.info(f"Found {num_gpus} GPUs! Using multi-GPU training.")
        # Adjust batch size for multi-GPU if not explicitly set
        if args.batch_size == 8:  # Default value
            recommended_batch_size = min(32, args.batch_size * num_gpus * 2)
            logger.info(f"Automatically scaling batch size from {args.batch_size} to {recommended_batch_size} for multi-GPU")
            args.batch_size = recommended_batch_size
        
        # Adjust learning rate for larger effective batch size (linear scaling rule)
        if args.lr == 2e-4:  # Default value
            lr_scale = args.batch_size / 8  # Scale from base batch size of 8
            args.lr = args.lr * lr_scale
            logger.info(f"Scaled learning rate to {args.lr:.6f} for larger batch size")
    else:
        logger.info("Using single GPU training.")

    # Initialize GradScaler for mixed-precision training
    # Old: scaler = GradScaler() if args.mixed_precision and device.type == 'cuda' else None
    # Updated to new syntax and explicitly using torch.amp
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision and device.type == 'cuda' else None

    logger.info(f"Using mixed precision training" if scaler else "Not using mixed precision training")

    # Enhanced transforms with AGGRESSIVE/EXTREME augmentation for small dataset
    # Multiple augmentation stages for maximum data variety
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
    train_transform = Compose(augmentation_stages)
    
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
    
    # Standard validation transforms (NO augmentation for consistent evaluation)
    val_transform = Compose([
        ConvertToFloatAndScale(),
        VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ShortSideScale(size=args.img_height),
        PerFrameCenterCrop((args.img_height, args.img_width))
    ])

    # Load datasets
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
        exit(1)
        
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.error("One or both datasets are empty after loading.")
        exit(1)

    # Create data loaders with class balancing for small datasets
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
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,  # Async prefetch when workers > 0
            drop_last=True,  # Better for training stability
            collate_fn=variable_views_collate_fn
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
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,  # Async prefetch when workers > 0
            drop_last=True,  # Better for training stability
            collate_fn=variable_views_collate_fn
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
        persistent_workers=val_num_workers > 0,
        prefetch_factor=2 if val_num_workers > 0 else None,
        collate_fn=variable_views_collate_fn
    )
    
    # Log data loading optimization details
    if args.num_workers > 0:
        logger.info(f"🚀 Async data loading enabled:")
        logger.info(f"   - Training workers: {args.num_workers} (prefetch_factor=4)")
        logger.info(f"   - Validation workers: {val_num_workers} (prefetch_factor=2)")
        logger.info(f"   - Pin memory: True (faster CPU->GPU transfers)")
        logger.info(f"   - Persistent workers: True (avoid respawning overhead)")
    else:
        logger.info(f"⚠️  Synchronous data loading (num_workers=0)")
        logger.info(f"   - This may cause GPU starvation and low utilization")
        logger.info(f"   - Consider setting num_workers >= 2 for better performance")
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Calculate class weights for severe imbalance handling
    severity_class_weights = None
    if args.loss_function in ['focal', 'weighted'] and not args.disable_class_balancing:
        if args.loss_function == 'focal':
            logger.info("🎯 Using Focal Loss for severity class imbalance handling!")
            logger.info(f"   - Focal gamma: {args.focal_gamma} (higher = more focus on hard examples)")
            severity_class_weights = calculate_class_weights(train_dataset, 6, device, args.class_weighting_strategy, args.max_weight_ratio)
            logger.info("   - Class weights will be used as alpha parameter in Focal Loss")
        elif args.loss_function == 'weighted':
            logger.info("🎯 Using class-weighted CrossEntropyLoss for severity imbalance!")
            logger.info(f"   - Weighting strategy: {args.class_weighting_strategy}")
            logger.info(f"   - Max weight ratio: {args.max_weight_ratio}")
            severity_class_weights = calculate_class_weights(train_dataset, 6, device, args.class_weighting_strategy, args.max_weight_ratio)
    else:
        if args.loss_function == 'plain':
            logger.info("📊 Using plain CrossEntropyLoss (no class balancing)")
        else:
            logger.info("📊 Class balancing disabled - using standard CrossEntropyLoss")
        logger.info("   - Consider enabling class balancing for imbalanced datasets")

    # === CLEAR LOSS FUNCTION LOGGING ===
    logger.info("=" * 60)
    logger.info("🔧 LOSS FUNCTION CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Loss function: {args.loss_function.upper()}")
    if args.loss_function == 'focal':
        logger.info(f"   - Focal gamma: {args.focal_gamma}")
        logger.info(f"   - Class weights as alpha: {'Yes' if severity_class_weights is not None else 'No'}")
    elif args.loss_function == 'weighted':
        logger.info(f"   - Class weighting strategy: {args.class_weighting_strategy}")
        logger.info(f"   - Max weight ratio: {args.max_weight_ratio}")
        logger.info(f"   - Class weights applied: {'Yes' if severity_class_weights is not None else 'No'}")
    elif args.loss_function == 'plain':
        logger.info("   - No class balancing applied")
    
    logger.info(f"Class balanced sampler: {'Enabled' if args.use_class_balanced_sampler else 'Disabled'}")
    logger.info(f"Augmentation: {'Disabled' if args.disable_augmentation else ('Aggressive' if args.aggressive_augmentation else 'Standard')}")
    logger.info(f"In-model augmentation: {'Disabled' if args.disable_in_model_augmentation else 'Enabled'}")
    logger.info(f"Label smoothing: {args.label_smoothing}")
    logger.info(f"Gradual fine-tuning: {'Enabled' if args.gradual_finetuning else 'Disabled'}")
    if args.simple_training:
        logger.info("⚡ SIMPLE TRAINING MODE: Minimal features for debugging")
    logger.info("=" * 60)

    # Create vocabulary sizes dictionary for the model
    vocab_sizes = {
        'contact': train_dataset.num_contact_classes,
        'bodypart': train_dataset.num_bodypart_classes,
        'upper_bodypart': train_dataset.num_upper_bodypart_classes,
        'lower_bodypart': train_dataset.num_lower_bodypart_classes,
        'multiple_fouls': train_dataset.num_multiple_fouls_classes,
        'try_to_play': train_dataset.num_try_to_play_classes,
        'touch_ball': train_dataset.num_touch_ball_classes,
        'handball': train_dataset.num_handball_classes,
        'handball_offence': train_dataset.num_handball_offence_classes,
    }

    # Model configuration
    model_config = ModelConfig(
        use_attention_aggregation=args.attention_aggregation,
        input_frames=args.frames_per_clip,
        input_height=args.img_height,
        input_width=args.img_width  # ResNet3D supports rectangular inputs
    )

    # Initialize model with proper configuration
    logger.info(f"Initializing ResNet3D model: {args.backbone_name}")
    model = MultiTaskMultiViewResNet3D.create_model(
        num_severity=6,  # 6 severity classes: "", 1.0, 2.0, 3.0, 4.0, 5.0
        num_action_type=10,  # 10 action types: "", Challenge, Dive, Dont know, Elbowing, High leg, Holding, Pushing, Standing tackling, Tackling
        vocab_sizes=vocab_sizes,
        backbone_name=args.backbone_name,
        config=model_config,
        use_augmentation=(not args.disable_in_model_augmentation),  # Control in-model augmentation
        disable_in_model_augmentation=args.disable_in_model_augmentation  # Pass the flag explicitly
    )
    model.to(device)
    
    # Wrap model with DataParallel for multi-GPU
    if num_gpus > 1:
        model = nn.DataParallel(model)
        logger.info(f"Model wrapped with DataParallel for {num_gpus} GPUs")

    # Gradual fine-tuning setup
    if args.gradual_finetuning:
        # Start with backbone frozen (Phase 1)
        freeze_backbone(model)
        log_trainable_parameters(model)
        
        # Setup optimizer for Phase 1 (heads only)
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], 
            lr=args.head_lr, 
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        logger.info(f"[PHASE1] Phase 1 optimizer initialized with LR={args.head_lr:.1e}")
    else:
        # Standard training - all parameters trainable
        log_trainable_parameters(model)
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )

    # Learning rate scheduler
    scheduler = None
    phase1_scheduler = None # Specific scheduler for Phase 1 of gradual fine-tuning
    scheduler_info = "None"

    if args.gradual_finetuning:
        # Phase 1 will use ReduceLROnPlateau
        phase1_scheduler = ReduceLROnPlateau(optimizer, mode='max', 
                                            factor=args.phase1_plateau_factor, 
                                            patience=args.phase1_plateau_patience, 
                                            min_lr=args.min_lr, verbose=True)
        scheduler_info = f"Phase 1: ReduceLROnPlateau (patience={args.phase1_plateau_patience}, factor={args.phase1_plateau_factor})"
        # Main scheduler (for Phase 2) will be initialized later
    else:
        # Standard (non-gradual) training: initialize the main scheduler now
        if args.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
            scheduler_info = f"CosineAnnealing (T_max={args.epochs}, eta_min={args.lr * 0.01:.1e})"
        elif args.scheduler == 'onecycle':
            steps_per_epoch = len(train_loader)
            scheduler = OneCycleLR(
                optimizer, 
                max_lr=args.lr,
                epochs=args.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=args.warmup_epochs / args.epochs if args.epochs > 0 else 0.1
            )
            scheduler_info = f"OneCycle (max_lr={args.lr:.1e}, warmup_epochs={args.warmup_epochs})"
        elif args.scheduler == 'step':
            scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
            scheduler_info = f"StepLR (step_size={args.step_size}, gamma={args.gamma})"
        elif args.scheduler == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
            scheduler_info = f"ExponentialLR (gamma={args.gamma})"
        elif args.scheduler == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.gamma, 
                                        patience=args.plateau_patience, min_lr=args.min_lr, verbose=True)
            scheduler_info = f"ReduceLROnPlateau (mode=max, factor={args.gamma}, patience={args.plateau_patience}, min_lr={args.min_lr:.1e})"

    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0  # Initialize here
    best_epoch = -1
    
    if args.resume:
        start_epoch, loaded_metrics = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        
        # Restore best validation accuracy from checkpoint with better fallback logic
        if 'best_val_acc' in loaded_metrics:
            best_val_acc = loaded_metrics['best_val_acc']
            best_epoch = loaded_metrics.get('best_epoch', loaded_metrics.get('epoch', start_epoch))
            logger.info(f"✅ Restored best validation accuracy: {best_val_acc:.4f} from epoch {best_epoch}")
        elif 'val_sev_acc' in loaded_metrics and 'val_act_acc' in loaded_metrics:
            # Fallback: Calculate combined accuracy from individual metrics if available
            restored_sev_acc = loaded_metrics['val_sev_acc']
            restored_act_acc = loaded_metrics['val_act_acc']
            best_val_acc = (restored_sev_acc + restored_act_acc) / 2
            best_epoch = loaded_metrics.get('epoch', start_epoch)
            logger.info(f"📊 Calculated best validation accuracy from checkpoint metrics: {best_val_acc:.4f} (sev: {restored_sev_acc:.3f}, act: {restored_act_acc:.3f})")
            logger.info(f"   - Using epoch {best_epoch} as best epoch")
        else:
            logger.warning("⚠️  No validation accuracy found in checkpoint - will start fresh tracking")
            logger.warning("   - This might happen with older checkpoint formats")
            best_val_acc = 0.0
            best_epoch = -1
        
        # Manual override if specified
        if args.resume_best_acc is not None:
            original_best = best_val_acc
            best_val_acc = args.resume_best_acc
            logger.info(f"🔧 MANUAL OVERRIDE: Best accuracy set to {best_val_acc:.4f} (was {original_best:.4f})")
        
        logger.info(f"🔄 Resuming training from epoch {start_epoch}")
        if best_val_acc > 0:
            logger.info(f"🎯 Current best to beat: {best_val_acc:.4f}")
        else:
            logger.info(f"🆕 Starting fresh best accuracy tracking")

    # Training history
    history = defaultdict(list)

    logger.info("Starting Training")
    logger.info(f"Configuration: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr}, Backbone={args.backbone_name}")
    if args.gradual_finetuning:
        logger.info(f"Gradual Fine-tuning: Phase1={args.phase1_epochs}e@{args.head_lr:.1e}, Phase2={args.phase2_epochs}e@{args.backbone_lr:.1e}")
    logger.info(f"Learning Rate Scheduler: {scheduler_info}")
    logger.info(f"Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    # Dataset size recommendations
    if len(train_dataset) < 100:
        logger.info("💡 RECOMMENDATION: Your dataset is very small (<100 samples)!")
        logger.info("   Consider using --extreme_augmentation for maximum data variety")
        logger.info("   Also try: --oversample_factor 6.0 --label_smoothing 0.15")
    elif len(train_dataset) < 500:
        logger.info("💡 RECOMMENDATION: Your dataset is small (<500 samples)!")
        logger.info("   Current aggressive augmentation should help")
        logger.info("   Consider: --oversample_factor 4.0 --label_smoothing 0.1")
    elif len(train_dataset) < 1000:
        logger.info("💡 RECOMMENDATION: Medium-sized dataset - current settings should work well")
    
    logger.info("=" * 80)

    # Log model info - handle DataParallel wrapper
    try:
        # Get the actual model (unwrap DataParallel if needed)
        actual_model = model.module if hasattr(model, 'module') else model
        model_info = actual_model.get_model_info()
        logger.info(f"Model initialized - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Combined feature dimension: {model_info['video_feature_dim'] + model_info['total_embedding_dim']}")
    except Exception as e:
        logger.warning(f"Could not get model info: {e}")
        logger.info(f"Model initialized - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss functions (kept for compatibility, but class weights are handled in calculate_multitask_loss)
    criterion_severity = nn.CrossEntropyLoss()
    criterion_action = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Gradual fine-tuning phase management
        if args.gradual_finetuning:
            current_phase, phase_description = get_phase_info(epoch, args.phase1_epochs, args.epochs)
            
            # Transition from Phase 1 to Phase 2
            if epoch == args.phase1_epochs and current_phase == 2:
                logger.info("[PHASE2] " + "="*60)
                logger.info("[PHASE2] TRANSITIONING TO PHASE 2: Gradual Unfreezing")
                logger.info("[PHASE2] " + "="*60)
                
                # Unfreeze backbone layers gradually
                unfreeze_backbone_gradually(model, args.unfreeze_blocks)
                
                # Setup discriminative learning rates for Phase 2
                actual_phase2_backbone_lr = args.backbone_lr * args.phase2_backbone_lr_scale_factor
                phase2_head_lr = actual_phase2_backbone_lr * args.phase2_head_lr_ratio
                
                param_groups = setup_discriminative_optimizer(model, phase2_head_lr, actual_phase2_backbone_lr)
                logger.info(f"[PHASE2_LR_SETUP] Phase 2 LRs: Backbone={actual_phase2_backbone_lr:.1e}, Head={phase2_head_lr:.1e}")
                
                # Recreate optimizer with new parameter groups
                optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.999))
                
                # Initialize main scheduler for Phase 2
                remaining_epochs = args.epochs - epoch
                if args.scheduler == 'cosine':
                    scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=actual_phase2_backbone_lr * 0.01)
                    scheduler_info = f"Phase 2: CosineAnnealing (T_max={remaining_epochs}, eta_min={actual_phase2_backbone_lr * 0.01:.1e})"
                elif args.scheduler == 'onecycle':
                    steps_per_epoch_phase2 = len(train_loader)
                    scheduler = OneCycleLR(
                        optimizer,
                        max_lr=[pg['lr'] for pg in param_groups],
                        epochs=remaining_epochs,
                        steps_per_epoch=steps_per_epoch_phase2,
                        pct_start=(args.warmup_epochs / remaining_epochs) if remaining_epochs > 0 and args.warmup_epochs < remaining_epochs else 0.1
                    )
                    scheduler_info = f"Phase 2: OneCycle (max_lr_config, warmup_epochs_scaled)"
                elif args.scheduler == 'step':
                    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
                    scheduler_info = f"Phase 2: StepLR (step_size={args.step_size}, gamma={args.gamma})"
                elif args.scheduler == 'exponential':
                    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
                    scheduler_info = f"Phase 2: ExponentialLR (gamma={args.gamma})"
                elif args.scheduler == 'reduce_on_plateau':
                    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.gamma,
                                                patience=args.plateau_patience, min_lr=args.min_lr, verbose=True)
                    scheduler_info = f"Phase 2: ReduceLROnPlateau (main_args)"
                else:
                    scheduler = None # No scheduler for phase 2 if 'none' or unknown
                    scheduler_info = "Phase 2: None"
                logger.info(f"[PHASE2_SCHEDULER] Initialized: {scheduler_info}")
                phase1_scheduler = None # Ensure Phase 1 scheduler is no longer used
                 
                log_trainable_parameters(model)
                logger.info("[PHASE2] Phase 2 setup complete!")
                logger.info("[PHASE2] " + "="*60)
        
        # Training
        train_metrics = train_one_epoch(
            model, train_loader, criterion_severity, criterion_action, optimizer, device,
            scaler=scaler, max_batches=num_batches_to_run, 
            loss_weights=args.main_task_weights, gradient_clip_norm=args.gradient_clip_norm, label_smoothing=args.label_smoothing,
            severity_class_weights=severity_class_weights, loss_function=args.loss_function, focal_gamma=args.focal_gamma
        )
        
        train_loss, train_sev_acc, train_act_acc, train_sev_f1, train_act_f1 = train_metrics
        
        # Validation
        val_metrics = validate_one_epoch(
            model, val_loader, criterion_severity, criterion_action, device,
            max_batches=num_batches_to_run, loss_weights=args.main_task_weights, label_smoothing=args.label_smoothing,
            severity_class_weights=severity_class_weights, loss_function=args.loss_function, focal_gamma=args.focal_gamma
        )
        
        val_loss, val_sev_acc, val_act_acc, val_sev_f1, val_act_f1 = val_metrics
        
        # Critical: Reset model to training mode and clean memory
        model.train()
        cleanup_memory()

        # Update learning rate
        if scheduler is not None:
            # Determine current phase for scheduler step
            current_phase_for_scheduler, _ = get_phase_info(epoch, args.phase1_epochs, args.epochs)

            if args.gradual_finetuning and current_phase_for_scheduler == 1 and phase1_scheduler is not None:
                phase1_scheduler.step(val_combined_acc)
                # logger.info(f"Stepped Phase 1 ReduceLROnPlateau scheduler with val_acc: {val_combined_acc:.4f}")
            elif scheduler is not None: # Handles Phase 2 or non-gradual training
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_combined_acc)
                elif not isinstance(scheduler, OneCycleLR): # OneCycleLR steps per batch
                    scheduler.step()
            
        # Calculate epoch time and combined metrics
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        prev_lr = history.get('learning_rate', [current_lr])[-1] if history.get('learning_rate') else current_lr
        train_combined_acc = (train_sev_acc + train_act_acc) / 2
        val_combined_acc = (val_sev_acc + val_act_acc) / 2

        # Check if this is a new best model
        is_new_best = val_combined_acc > best_val_acc
        best_indicator = " [NEW BEST!]" if is_new_best else ""

        # Check for learning rate changes
        lr_change_indicator = ""
        if abs(current_lr - prev_lr) > 1e-8:  # If LR changed significantly
            lr_change_indicator = f" [LR DOWN]"

        # Phase indicator for gradual fine-tuning
        phase_indicator = ""
        if args.gradual_finetuning:
            current_phase, _ = get_phase_info(epoch, args.phase1_epochs, args.epochs)
            phase_indicator = f" [P{current_phase}]"

        # Compact epoch summary
        logger.info(f"Epoch {epoch+1:2d}/{args.epochs} [{epoch_time:.1f}s]{phase_indicator} "
                   f"| Train: Loss={train_loss:.3f}, Acc={train_combined_acc:.3f} "
                   f"| Val: Loss={val_loss:.3f}, Acc={val_combined_acc:.3f} "
                   f"| LR={current_lr:.1e}{lr_change_indicator}{best_indicator}")

        # Store history (including learning rate)
        history['train_loss'].append(train_loss)
        history['train_sev_acc'].append(train_sev_acc)
        history['train_act_acc'].append(train_act_acc)
        history['val_loss'].append(val_loss)
        history['val_sev_acc'].append(val_sev_acc)
        history['val_act_acc'].append(val_act_acc)
        history['learning_rate'] = history.get('learning_rate', []) + [current_lr]

        # Model saving and early stopping
        if not args.test_run:
            current_val_acc = val_combined_acc
            
            # Save best model
            if current_val_acc > best_val_acc:
                improvement = current_val_acc - best_val_acc
                best_val_acc = current_val_acc
                best_epoch = epoch + 1
                
                metrics = {
                    'epoch': best_epoch,
                    'best_val_acc': best_val_acc,
                    'best_epoch': best_epoch,  # Explicitly save best_epoch for clarity
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_sev_acc': train_sev_acc,
                    'train_act_acc': train_act_acc,
                    'val_sev_acc': val_sev_acc,
                    'val_act_acc': val_act_acc
                }
                
                save_path = os.path.join(args.save_dir, f'best_model_epoch_{best_epoch}.pth')
                save_checkpoint(model, optimizer, scheduler, scaler, best_epoch, metrics, save_path)
                logger.info(f"[SAVE] Best model updated! Accuracy: {best_val_acc:.4f} (+{improvement:.4f}) - Saved to {save_path}")

            # Early stopping check
            if early_stopping(current_val_acc, model):
                logger.info(f"[EARLY_STOP] Early stopping triggered after {epoch + 1} epochs")
                break

            # Save regular checkpoint every 10 epochs - INCLUDE BEST ACCURACY INFO
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                metrics = {
                    'epoch': epoch + 1,
                    'best_val_acc': best_val_acc,  # Always include current best
                    'best_epoch': best_epoch,      # Always include best epoch
                    'current_train_loss': train_loss,
                    'current_val_loss': val_loss,
                    'current_train_sev_acc': train_sev_acc,
                    'current_train_act_acc': train_act_acc,
                    'current_val_sev_acc': val_sev_acc,
                    'current_val_act_acc': val_act_acc
                }
                save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, metrics, checkpoint_path)
                logger.info(f"[CHECKPOINT] Checkpoint saved at epoch {epoch + 1} (best so far: {best_val_acc:.4f})")

    # Save training history
    if not args.test_run:
        history_path = os.path.join(args.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            history_serializable = {k: [float(x) for x in v] for k, v in history.items()}
            json.dump(history_serializable, f, indent=2)
        logger.info(f"[SAVE] Training history saved to {history_path}")

    logger.info("=" * 80)
    logger.info("[COMPLETE] Training Finished!")
    if not args.test_run:
        logger.info(f"[BEST] Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    else:
        logger.info("[TEST] Test run completed successfully")
    logger.info("=" * 80) 
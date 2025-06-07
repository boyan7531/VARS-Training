"""
Configuration and argument parsing for Multi-Task Multi-View ResNet3D training.

This module handles all command-line arguments and training configuration setup.
"""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train Multi-Task Multi-View ResNet3D for Foul Recognition')
    
    # === BASIC TRAINING CONFIGURATION ===
    parser.add_argument('--dataset_root', type=str, default="", help='Root directory containing the mvfouls folder')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # === MODEL CONFIGURATION ===
    parser.add_argument('--backbone_name', type=str, default='r2plus1d_18', 
                        choices=['resnet3d_18', 'mc3_18', 'r2plus1d_18', 'resnet3d_50'], 
                        help="ResNet3D backbone variant (r2plus1d_18 recommended for best accuracy)")
    parser.add_argument('--frames_per_clip', type=int, default=16, help='Number of frames per clip')
    parser.add_argument('--target_fps', type=int, default=15, help='Target FPS for clips')
    parser.add_argument('--start_frame', type=int, default=67, help='Start frame index for foul-centered extraction (8 frames before foul at frame 75)')
    parser.add_argument('--end_frame', type=int, default=82, help='End frame index for foul-centered extraction (7 frames after foul at frame 75)')
    parser.add_argument('--img_height', type=int, default=224, help='Target image height')
    parser.add_argument('--img_width', type=int, default=398, help='Target image width (matches original VARS paper)')
    parser.add_argument('--max_views', type=int, default=None, help='Optional limit on max views per action (default: use all available)')
    parser.add_argument('--attention_aggregation', action='store_true', default=True, help='Use attention for view aggregation')
    
    # === TRAINING OPTIMIZATION ===
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd', 'adam'], help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                       help='Patience for early stopping (epochs with no improvement).')
    
    # === LEARNING RATE SCHEDULING ===
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['cosine', 'onecycle', 'step', 'exponential', 'reduce_on_plateau', 'none'], 
                       help='Learning rate scheduler type')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.5, help='Decay factor for StepLR/ExponentialLR')
    parser.add_argument('--plateau_patience', type=int, default=5, help='Patience for ReduceLROnPlateau')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    
    # === MULTI-TASK LOSS CONFIGURATION ===
    parser.add_argument('--main_task_weights', type=float, nargs=2, default=[3.0, 3.0], 
                       help='Loss weights for [severity, action_type] - main tasks')
    parser.add_argument('--auxiliary_task_weight', type=float, default=0.5,
                       help='Loss weight for all auxiliary tasks (currently not implemented)')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor (0.1 recommended for small datasets)')
    
    # === LOSS FUNCTION CONTROLS ===
    parser.add_argument('--loss_function', type=str, default='weighted', 
                       choices=['focal', 'weighted', 'plain'],
                       help='Loss function type: focal (FocalLoss), weighted (weighted CrossEntropyLoss), plain (standard CrossEntropyLoss)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Focal Loss gamma parameter (higher = more focus on hard examples)')
    
    # === CLASS IMBALANCE HANDLING ===
    parser.add_argument('--class_weighting_strategy', type=str, default='balanced_capped',
                       choices=['none', 'balanced_capped', 'sqrt', 'log', 'effective_number'],
                       help='Strategy for calculating class weights (balanced_capped recommended for stability)')
    parser.add_argument('--max_weight_ratio', type=float, default=10.0,
                       help='Maximum ratio between highest and lowest class weight (prevents training instability)')
    parser.add_argument('--use_class_balanced_sampler', action='store_true', default=True,
                       help='Use class-balanced sampler to oversample minority classes')
    parser.add_argument('--oversample_factor', type=float, default=4.0,
                       help='Factor by which to oversample minority classes (higher = more aggressive)')
    
    # === ENHANCED FREEZING STRATEGY OPTIONS ===
    parser.add_argument('--freezing_strategy', type=str, default='fixed', 
                       choices=['fixed', 'adaptive', 'progressive', 'none'],
                       help='Freezing strategy: fixed (original), adaptive (plateau-based), progressive (layer-wise), none (no freezing)')
    parser.add_argument('--adaptive_patience', type=int, default=3,
                       help='Patience for adaptive unfreezing (epochs without improvement before unfreezing next layer)')
    parser.add_argument('--adaptive_min_improvement', type=float, default=0.001,
                       help='Minimum validation accuracy improvement to reset adaptive patience')
    parser.add_argument('--exponential_lr_decay', action='store_true', default=False,
                       help='Use exponential LR decay for backbone layers (each layer gets 0.5x previous layer LR)')
    parser.add_argument('--compare_freezing', action='store_true', default=False,
                       help='Run comparison between freezing and no-freezing strategies (experimental)')
    
    # === GRADUAL FINE-TUNING (LEGACY/FIXED MODE) ===
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
    parser.add_argument('--unfreeze_blocks', type=int, default=3,
                       help='Number of final residual blocks to unfreeze in Phase 2')
    parser.add_argument('--phase2_backbone_lr_scale_factor', type=float, default=1.0,
                       help='Scale factor for args.backbone_lr in Phase 2 (e.g., 2.0 for 2x base backbone_lr)')
    parser.add_argument('--phase2_head_lr_ratio', type=float, default=5.0,
                       help='Ratio for Phase 2 head LR relative to Phase 2 backbone LR (e.g., head_lr = p2_backbone_lr * ratio)')
    parser.add_argument('--phase1_plateau_patience', type=int, default=3,
                       help='Patience for ReduceLROnPlateau scheduler in Phase 1 (gradual fine-tuning only).')
    parser.add_argument('--phase1_plateau_factor', type=float, default=0.2,
                       help='Factor for ReduceLROnPlateau scheduler in Phase 1 (gradual fine-tuning only).')
    
    # === AUGMENTATION CONFIGURATION ===
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
    parser.add_argument('--gpu_augmentation', action='store_true', default=False,
                       help='Use GPU-based augmentation instead of CPU augmentation (recommended for dual GPU setups)')
    
    # === DEBUGGING & FLEXIBILITY OPTIONS ===
    parser.add_argument('--simple_training', action='store_true', default=False,
                       help='Enable simple training mode: disables class balancing, augmentation, and uses plain CrossEntropyLoss')
    parser.add_argument('--disable_class_balancing', action='store_true', default=False,
                       help='Disable all class balancing (both class weights and balanced sampler)')
    parser.add_argument('--disable_augmentation', action='store_true', default=False,
                       help='Disable all augmentation (both in-dataset and in-model)')
    parser.add_argument('--disable_in_model_augmentation', action='store_true', default=False,
                       help='Disable VideoAugmentation applied inside the model forward pass (for bottleneck testing)')
    
    # === CHECKPOINTING AND LOGGING ===
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--resume_best_acc', type=float, default=None, 
                       help='Manually override best validation accuracy when resuming (useful for checkpoint issues)')
    parser.add_argument('--memory_cleanup_interval', type=int, default=20,
                       help='Interval (in batches) for calling memory cleanup. 0 or negative to disable in train/val loops.')
    
    # === TESTING AND DEVELOPMENT ===
    parser.add_argument('--test_run', action='store_true', help='Perform a quick test run (1 epoch, few batches, no saving)')
    parser.add_argument('--test_batches', type=int, default=2, help='Number of batches to run in test mode')
    parser.add_argument('--force_batch_size', action='store_true', 
                       help='Force specified batch size even with multi-GPU (disable auto-scaling)')
    
    # === ADVANCED OPTIONS ===
    parser.add_argument('--multi_scale', action='store_true', help='Enable multi-scale training for better accuracy')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for regularization')
    parser.add_argument('--lr_warmup', action='store_true', help='Enable learning rate warmup')
    
    # === LEGACY ARGUMENTS (for backward compatibility) ===
    parser.add_argument('--use_focal_loss', action='store_true', default=False,
                       help='DEPRECATED: Use --loss_function focal instead')
    parser.add_argument('--use_class_weighted_loss', action='store_true', default=True,
                       help='DEPRECATED: Use --loss_function weighted instead')
    
    args = parser.parse_args()
    
    # Apply configuration processing and validation
    args = process_config(args)
    
    return args

def process_config(args):
    """Process and validate configuration arguments."""
    
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
    
    # Validate and construct dataset path
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
            args.early_stopping_patience = args.phase1_epochs + args.phase2_epochs // 2
            logger.info(f"💡 Default early stopping patience set to {args.early_stopping_patience} for gradual fine-tuning.")
        else:
            args.early_stopping_patience = 10
            logger.info(f"💡 Default early stopping patience set to {args.early_stopping_patience} for standard training.")
    else:
        logger.info(f"💡 User-defined early stopping patience: {args.early_stopping_patience}")
    
    return args

def log_configuration_summary(args):
    """Log a comprehensive summary of the training configuration."""
    logger.info("=" * 60)
    logger.info("🔧 TRAINING CONFIGURATION SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"Dataset: {args.dataset_root}")
    logger.info(f"Model: {args.backbone_name} ({args.frames_per_clip} frames, {args.img_height}x{args.img_width})")
    logger.info(f"Training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    logger.info(f"Optimizer: {args.optimizer}, scheduler={args.scheduler}")
    
    # Loss function configuration
    logger.info(f"Loss function: {args.loss_function.upper()}")
    if args.loss_function == 'focal':
        logger.info(f"   - Focal gamma: {args.focal_gamma}")
    elif args.loss_function == 'weighted':
        logger.info(f"   - Class weighting: {args.class_weighting_strategy}")
        logger.info(f"   - Max weight ratio: {args.max_weight_ratio}")
    
    # Freezing strategy
    logger.info(f"Freezing strategy: {args.freezing_strategy}")
    if args.freezing_strategy == 'adaptive':
        logger.info(f"   - Patience: {args.adaptive_patience}")
        logger.info(f"   - Min improvement: {args.adaptive_min_improvement}")
    elif args.gradual_finetuning and args.freezing_strategy == 'fixed':
        logger.info(f"   - Phase 1: {args.phase1_epochs} epochs")
        logger.info(f"   - Phase 2: {args.phase2_epochs} epochs")
    
    # Augmentation
    aug_level = "None"
    if args.extreme_augmentation:
        aug_level = "Extreme"
    elif args.aggressive_augmentation:
        aug_level = "Aggressive"
    elif not args.disable_augmentation:
        aug_level = "Standard"
    logger.info(f"Augmentation: {aug_level}")
    
    # Class balancing
    logger.info(f"Class balancing: {'Enabled' if args.use_class_balanced_sampler else 'Disabled'}")
    if args.use_class_balanced_sampler:
        logger.info(f"   - Oversample factor: {args.oversample_factor}")
    
    logger.info("=" * 60) 
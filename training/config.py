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
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW (0.01 for all params except bias & norm)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--worker_timeout', type=int, default=0, help='Timeout for DataLoader workers (seconds)')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Number of batches loaded in advance by each worker')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for DataLoader')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # === MODEL CONFIGURATION ===
    parser.add_argument('--backbone_type', type=str, default='resnet3d', 
                        choices=['resnet3d', 'mvit'], 
                        help='Backbone architecture type: resnet3d (torchvision 3D ResNets) or mvit (MViTv2 from PyTorchVideo)')
    parser.add_argument('--backbone_name', type=str, default='r2plus1d_18', 
                        help="Specific backbone model name. For ResNet3D: 'resnet3d_18', 'mc3_18', 'r2plus1d_18', 'resnet3d_50'. "
                             "For MViT: 'mvit_base_16', 'mvit_base_16x4', 'mvit_base_32x3' (r2plus1d_18/mvit_base_16x4 recommended)")
    parser.add_argument('--frames_per_clip', type=int, default=16, help='Number of frames per clip')
    parser.add_argument('--target_fps', type=int, default=15, help='Target FPS for clips')
    parser.add_argument('--start_frame', type=int, default=67, help='Start frame index for foul-centered extraction (8 frames before foul at frame 75)')
    parser.add_argument('--end_frame', type=int, default=82, help='End frame index for foul-centered extraction (7 frames after foul at frame 75)')
    parser.add_argument('--img_height', type=int, default=224, help='Target image height')
    parser.add_argument('--img_width', type=int, default=224, help='Target image width (224x224 square for MViT compatibility)')
    parser.add_argument('--max_views', type=int, default=None, help='Optional limit on max views per action (default: use all available)')
    parser.add_argument('--attention_aggregation', action='store_true', default=True, help='Use attention for view aggregation')
    
    # === VIEW AGGREGATOR CONFIGURATION ===
    parser.add_argument('--aggregator_type', type=str, default='transformer', choices=['mlp', 'transformer', 'moe'],
                       help='Type of view aggregator: mlp (current MLP attention), transformer (Transformer-based), moe (Mixture of Experts)')
    parser.add_argument('--agg_heads', type=int, default=2, help='Number of attention heads for transformer aggregator')
    parser.add_argument('--agg_layers', type=int, default=1, help='Number of transformer encoder layers for aggregator')
    
    # === TRAINING OPTIMIZATION ===
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd', 'adam'], help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    
    # Advanced optimizer parameters
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam/AdamW beta1 parameter')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam/AdamW beta2 parameter (auto-adjusted for MViT)')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam/AdamW epsilon for numerical stability')
    parser.add_argument('--amsgrad', action='store_true', help='Use AMSGrad variant of Adam/AdamW')
    parser.add_argument('--nesterov', action='store_true', default=True, help='Use Nesterov momentum for SGD')
    parser.add_argument('--dampening', type=float, default=0, help='SGD dampening parameter')
    
    # Gradient clipping and EMA
    parser.add_argument('--grad_clip_norm', type=float, default=0.0, help='Gradient clipping max norm (0 to disable)')
    parser.add_argument('--use_ema', action='store_true', help='Use Exponential Moving Average of model weights')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay rate')
    parser.add_argument('--ema_eval', action='store_true', default=True,
                       help='Use EMA weights for validation/test inference (default: True, no downside)')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                       help='Patience for early stopping (epochs with no improvement).')
    
    # === LEARNING RATE SCHEDULING ===
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['cosine', 'onecycle', 'step', 'exponential', 'reduce_on_plateau', 'none'], 
                       help='Learning rate scheduler type')
    parser.add_argument('--scheduler_warmup_epochs', type=int, default=5, help='Number of warmup epochs for scheduler')
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
    
    # === VIEW CONSISTENCY LOSS ===
    parser.add_argument('--view_consistency', action='store_true', default=False,
                       help='Enable view consistency loss (KL agreement across camera views)')
    parser.add_argument('--vc_weight', type=float, default=0.3,
                       help='Weight for view consistency loss (default: 0.3)')
    
    # === CLASS IMBALANCE HANDLING ===
    parser.add_argument('--class_weighting_strategy', type=str, default='sqrt',
                       choices=['none', 'balanced_capped', 'sqrt', 'log', 'effective_number'],
                       help='Strategy for calculating class weights (sqrt recommended for safer training)')
    parser.add_argument('--max_weight_ratio', type=float, default=6.0,
                       help='Maximum ratio between highest and lowest class weight (reduced from 12 to prevent training instability)')
    parser.add_argument('--use_class_balanced_sampler', action='store_true', default=True,
                       help='Use class-balanced sampler to oversample minority classes')
    parser.add_argument('--oversample_factor', type=float, default=2.0,
                       help='Factor by which to oversample minority classes (reduced from 4.0 to prevent overfitting)')
    parser.add_argument('--use_alternating_sampler', action='store_true', default=False,
                       help='Use alternating sampler that switches between severity and action balancing per epoch')
    parser.add_argument('--action_oversample_factor', type=float, default=2.0,
                       help='Factor by which to oversample minority action classes (reduced from 4.0 to prevent overfitting)')
    parser.add_argument('--use_action_balanced_sampler_only', action='store_true', default=False,
                       help='Use only action-balanced sampler (ignore severity balancing)')
    parser.add_argument('--use_strong_action_weights', action='store_true', default=False,
                       help='Use strong action class weights (not normalized) to combat severe action imbalance')
    parser.add_argument('--action_weight_strategy', type=str, default='strong_inverse',
                       choices=['strong_inverse', 'focal_style', 'exponential'],
                       help='Strategy for calculating strong action weights')
    parser.add_argument('--action_weight_power', type=float, default=2.0,
                       help='Power factor for strong action weights (higher = more aggressive)')
    parser.add_argument('--disable_class_balancing', action='store_true', default=False,
                       help='Disable all class balancing techniques (use with caution)')
    parser.add_argument('--use_class_weights_only', action='store_true', default=False,
                       help='Use only class weights for balancing, disable progressive sampling (recommended fix for severe bias)')
    
    # === NEW STRATEGIES FOR CLASS IMBALANCE ===
    parser.add_argument('--progressive_class_balancing', action='store_true', default=True,
                       help='Enable progressive class-balanced sampling that increases minority representation over time (recommended over fixed oversampling)')
    parser.add_argument('--progressive_start_factor', type=float, default=1.5,
                       help='Starting balancing factor for progressive sampling (default: 1.5)')
    parser.add_argument('--progressive_end_factor', type=float, default=2.0,
                       help='Ending balancing factor for progressive sampling (default: 2.0, reduced from 3.0)')
    parser.add_argument('--progressive_epochs', type=int, default=15,
                       help='Number of epochs for progressive sampling transition')

    # Emergency unfreezing
    parser.add_argument('--emergency_unfreeze_epoch', type=int, default=4,
                       help='Epoch to force unfreeze if no layers unfrozen yet (default: 4)')
    parser.add_argument('--min_unfreeze_layers', type=int, default=1,
                       help='Minimum number of backbone layers to unfreeze (default: 1)')
    parser.add_argument('--emergency_unfreeze_gradual', action='store_true', default=True,
                       help='Use gradual emergency unfreezing (sub-blocks instead of full layers)')
    parser.add_argument('--validation_plateau_patience', type=int, default=2,
                       help='Epochs to wait before treating validation performance as plateaued')
    
    # === ENHANCED FREEZING STRATEGY OPTIONS ===
    parser.add_argument('--freezing_strategy', type=str, default='fixed', 
                       choices=['none', 'fixed', 'adaptive', 'progressive', 'gradient_guided', 'advanced', 'early_gradual'],
                       help='Strategy for parameter freezing (none=no freezing, fixed=timed phases, adaptive/progressive/gradient_guided/advanced, early_gradual=freeze 1 epoch then unfreeze 1 block per epoch)')
    parser.add_argument('--adaptive_patience', type=int, default=2,
                       help='Epochs to wait before unfreezing the next layer in adaptive mode (reduced for faster response)')
    parser.add_argument('--adaptive_min_improvement', type=float, default=0.005,
                       help='Minimum validation improvement to reset patience counter in adaptive mode (increased for more reliable signals)')
    
    # Early gradual freezing specific parameters
    parser.add_argument('--early_gradual_freeze_epochs', type=int, default=1,
                       help='Number of epochs to keep backbone frozen before starting gradual unfreezing (early_gradual strategy)')
    parser.add_argument('--early_gradual_blocks_per_epoch', type=int, default=1,
                       help='Number of blocks to unfreeze per epoch in early_gradual strategy')
    parser.add_argument('--early_gradual_target_ratio', type=float, default=0.5,
                       help='Target ratio of backbone parameters to unfreeze (0.5 = half the backbone)')
    
    # Gradient-guided freezing specific parameters
    parser.add_argument('--importance_threshold', type=float, default=0.01,
                        help='Minimum gradient importance threshold for unfreezing layers in gradient-guided mode')
    parser.add_argument('--layer_warmup_epochs', type=int, default=3,
                        help='Number of epochs to warmup newly unfrozen layers with reduced learning rate')
    parser.add_argument('--unfreeze_patience', type=int, default=2,
                        help='Minimum epochs between layer unfreezing operations')
    parser.add_argument('--max_layers_per_step', type=int, default=1,
                        help='Maximum number of layers to unfreeze in a single step')
    parser.add_argument('--sampling_epochs', type=int, default=2,
                        help='Number of epochs to sample gradients before first unfreezing decision')
    
    # Advanced freezing specific parameters
    parser.add_argument('--base_importance_threshold', type=float, default=0.002,
                        help='Base importance threshold for advanced freezing (will adapt dynamically)')
    parser.add_argument('--performance_threshold', type=float, default=0.001,
                        help='Minimum performance improvement threshold for advanced freezing decisions')
    parser.add_argument('--rollback_patience', type=int, default=2,
                        help='Epochs to wait before performing rollback in advanced freezing')
    parser.add_argument('--gradient_momentum', type=float, default=0.9,
                        help='Momentum factor for smoothing gradient importance in advanced freezing')
    parser.add_argument('--analysis_window', type=int, default=2,
                        help='Number of epochs to analyze before making unfreezing decisions')
    parser.add_argument('--enable_rollback', action='store_true', default=True,
                        help='Enable performance-based rollback in advanced freezing')
    parser.add_argument('--enable_dependency_analysis', action='store_true', default=True,
                        help='Enable layer dependency analysis in advanced freezing')
    
    # === GRADUAL FINE-TUNING (LEGACY/FIXED MODE) ===
    parser.add_argument('--gradual_finetuning', action='store_true', default=False, 
                       help='Enable gradual fine-tuning with discriminative learning rates')
    parser.add_argument('--phase1_epochs', type=int, default=8, 
                       help='Number of epochs for Phase 1 (head-only training)')
    parser.add_argument('--phase2_epochs', type=int, default=15, 
                       help='Number of epochs for Phase 2 (gradual unfreezing)')
    parser.add_argument('--head_lr', type=float, default=1e-3, 
                       help='Learning rate for classification heads (used when backbone is frozen)')
    parser.add_argument('--backbone_lr', type=float, default=1e-4,
                       help='Learning rate for unfrozen backbone layers (typically head_lr/10 = 1e-4)')
    parser.add_argument('--backbone_lr_ratio_after_half', type=float, default=0.6,
                       help='Backbone LR ratio relative to head LR after ≥50% of backbone is unfrozen (default: 0.6)')
    parser.add_argument('--enable_backbone_lr_boost', action='store_true', default=True,
                       help='Enable automatic backbone LR boost when ≥50% of backbone is unfrozen')
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
    parser.add_argument('--augmentation_strength', type=str, default='moderate',
                       choices=['none', 'mild', 'moderate', 'aggressive', 'extreme'],
                       help='Overall augmentation strength (none, mild, moderate, aggressive, extreme)')
    parser.add_argument('--aggressive_augmentation', action='store_true', default=False,
                       help='Enable aggressive augmentation pipeline for small datasets (deprecated)')
    parser.add_argument('--extreme_augmentation', action='store_true', default=False,
                       help='Enable EXTREME augmentation with all techniques (use for very small datasets) (deprecated)')
    parser.add_argument('--temporal_jitter_strength', type=int, default=3,
                       help='Max temporal jitter in frames (higher = more temporal variation)')
    parser.add_argument('--dropout_prob', type=float, default=0.05,
                       help='Frame dropout probability (reduced for medium dataset size)')
    parser.add_argument('--spatial_crop_strength', type=float, default=0.7,
                       help='Minimum crop scale (lower = more aggressive spatial crops)')
    parser.add_argument('--color_aug_strength', type=float, default=0.3,
                       help='Color augmentation strength (higher = more variation)')
    parser.add_argument('--noise_strength', type=float, default=0.02,
                       help='Maximum Gaussian noise standard deviation (reduced for medium dataset)')
    parser.add_argument('--gpu_augmentation', action='store_true', default=False,
                       help='Use GPU-based augmentation instead of CPU augmentation (recommended for dual GPU setups)')
    parser.add_argument('--severity_aware_augmentation', action='store_true', default=False,
                       help='Use severity-aware augmentation with class-specific strengths')
    parser.add_argument('--use_randaugment', action='store_true', default=False,
                       help='Enable RandAugment for stronger augmentation')
    parser.add_argument('--randaugment_n', type=int, default=2,
                       help='Number of augmentation transformations to apply in RandAugment')
    parser.add_argument('--randaugment_m', type=int, default=10,
                       help='Magnitude of augmentation transformations in RandAugment (1-30)')
    parser.add_argument('--strong_color_jitter', action='store_true', default=False,
                       help='Enable stronger color jittering (hue, saturation) in addition to brightness/contrast')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                       help='Alpha parameter for mixup augmentation (0 = disabled)')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0,
                       help='Alpha parameter for cutmix augmentation (0 = disabled)')
    parser.add_argument('--cutmix_prob', type=float, default=0.5,
                       help='Probability of applying cutmix when mixup/cutmix is enabled')
    
    # === DEBUGGING & FLEXIBILITY OPTIONS ===
    parser.add_argument('--simple_training', action='store_true', default=False,
                       help='Enable simple training mode: disables class balancing, augmentation, and uses plain CrossEntropyLoss')
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
    
    # === DATA LOADING OPTIMIZATION ===
    parser.add_argument('--enable_data_optimization', action='store_true', default=True,
                       help='Enable data loading optimization features')
    parser.add_argument('--disable_data_optimization', action='store_true', default=False,
                       help='Disable data loading optimization features')
    
    # === ERROR RECOVERY ===
    parser.add_argument('--enable_oom_recovery', action='store_true', default=True,
                       help='Enable automatic OOM recovery')
    parser.add_argument('--oom_reduction_factor', type=float, default=0.75,
                       help='Factor to reduce batch size on OOM (default: 0.75)')
    parser.add_argument('--min_batch_size', type=int, default=1,
                       help='Minimum batch size for OOM recovery')
    parser.add_argument('--enable_config_validation', action='store_true', default=True,
                       help='Enable configuration validation')
    parser.add_argument('--strict_config_validation', action='store_true', default=False,
                       help='Treat config warnings as errors')
    
    # === MVIT OPTIMIZATION ===
    parser.add_argument('--enable_gradient_checkpointing', action='store_true', default=False,
                       help='Enable gradient checkpointing for memory efficiency (recommended for MViT)')
    parser.add_argument('--disable_gradient_checkpointing', action='store_true', default=False,
                       help='Disable gradient checkpointing (for debugging or small models)')
    parser.add_argument('--enable_memory_optimization', action='store_true', default=True,
                       help='Enable memory optimization features (aggressive cleanup, efficient tensor operations)')
    parser.add_argument('--disable_memory_optimization', action='store_true', default=False,
                       help='Disable memory optimization features')
    parser.add_argument('--mvit_memory_cleanup_interval', type=int, default=5,
                       help='Memory cleanup interval for MViT training (batches, lower = more frequent cleanup)')
    parser.add_argument('--mvit_sequential_processing', action='store_true', default=True,
                       help='Use sequential view processing for MViT to reduce memory fragmentation')
    
    # === MULTI-CLIP TEMPORAL SAMPLING ===
    parser.add_argument('--clips_per_video', type=int, default=1,
                       help='Number of clips to sample per video (1=single clip, 3+ for temporal ensemble)')
    parser.add_argument('--clip_sampling', type=str, default='uniform', choices=['uniform', 'random'],
                       help='How to sample multiple clips: uniform (evenly spaced) or random positions')
    
    # === TESTING AND DEVELOPMENT ===
    parser.add_argument('--test_run', action='store_true', help='Perform a quick test run (1 epoch, few batches, no saving)')
    parser.add_argument('--test_batches', type=int, default=2, help='Number of batches to run in test mode')
    parser.add_argument('--force_batch_size', action='store_true', 
                       help='Force specified batch size even with multi-GPU (disable auto-scaling)')
    
    # === ADVANCED OPTIONS ===
    parser.add_argument('--multi_scale', action='store_true', help='Enable multi-scale training for better accuracy')
    parser.add_argument('--multi_scale_sizes', type=int, nargs='+', default=[224, 256, 288], 
                       help='List of image sizes for multi-scale training (default: [224, 256, 288])')
    parser.add_argument('--multi_scale_prob', type=float, default=0.5,
                       help='Probability of applying multi-scale cropping (default: 0.5)')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for regularization')
    parser.add_argument('--lr_warmup', action='store_true', help='Enable learning rate warmup')
    parser.add_argument('--lr_warmup_steps', type=int, default=0, help='Number of warm-up steps (computed automatically if 0)')
    parser.add_argument('--lr_warmup_pct', type=float, default=0.05, help='Percentage of total training steps for warm-up (default: 5%)')
    parser.add_argument('--lr_warmup_start_lr', type=float, default=None, help='Starting learning rate for warm-up (default: lr/100)')
    
    parser.add_argument('--discriminative_lr', action='store_true', default=False,
                       help='Enable discriminative learning rates for different layers (e.g., backbone vs. head)')
    
    # === PYTORCH LIGHTNING FEATURES ===
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                       help='Number of batches to accumulate gradients over before updating weights')
    parser.add_argument('--num_nodes', type=int, default=1,
                       help='Number of nodes for multi-node distributed training')
    parser.add_argument('--devices_per_node', type=int, default=-1,
                       help='Number of devices (GPUs) per node (-1 = all available)')
    parser.add_argument('--enable_profiler', action='store_true', default=False,
                       help='Enable PyTorch Lightning profiler for performance analysis')
    parser.add_argument('--log_every_n_steps', type=int, default=10,
                       help='How often to log training metrics (steps)')
    parser.add_argument('--val_check_interval', type=float, default=1.0,
                       help='How often to check validation set (fraction of epoch or number of batches)')
    parser.add_argument('--limit_train_batches', type=float, default=1.0,
                       help='Limit training batches (fraction or number of batches)')
    parser.add_argument('--limit_val_batches', type=float, default=1.0,
                       help='Limit validation batches (fraction or number of batches)')
    parser.add_argument('--max_time', type=str, default=None,
                       help='Maximum training time (e.g., "00:12:00:00" for 12 hours)')
    parser.add_argument('--auto_lr_find', action='store_true', default=False,
                       help='Automatically find optimal learning rate')
    parser.add_argument('--auto_scale_batch_size', action='store_true', default=False,
                       help='Automatically find optimal batch size')
    
    # === DISTRIBUTED TRAINING OPTIMIZATIONS ===
    parser.add_argument('--sync_batchnorm', action='store_true', default=False,
                       help='Convert BatchNorm to SyncBatchNorm for distributed training')
    parser.add_argument('--find_unused_parameters', action='store_true', default=False,
                       help='Find unused parameters in DDP (may slow down training)')
    parser.add_argument('--ddp_timeout', type=int, default=1800,
                       help='Timeout for DDP communication (seconds)')
    
    # === PERFORMANCE MONITORING ===
    parser.add_argument('--track_grad_norm', type=int, default=-1,
                       help='Track gradient norm (L2 norm order, -1 to disable)')
    parser.add_argument('--log_gpu_memory', action='store_true', default=False,
                       help='Log GPU memory usage during training')
    
    # === LEGACY ARGUMENTS (for backward compatibility) ===
    parser.add_argument('--use_focal_loss', action='store_true', default=False,
                       help='DEPRECATED: Use --loss_function focal instead')
    parser.add_argument('--use_class_weighted_loss', action='store_true', default=True,
                       help='DEPRECATED: Use --loss_function weighted instead')
    
    # === NEW CLI ARGUMENTS ===
    parser.add_argument('--fused_optim', action='store_true', help='Use PyTorch fused optimizer kernels (CUDA only)')
    parser.add_argument('--lookahead', action='store_true', help='Wrap optimizer with Lookahead for better generalization')
    parser.add_argument('--la_steps', type=int, default=5, help='Lookahead sync period k')
    parser.add_argument('--la_alpha', type=float, default=0.5, help='Lookahead slow weight alpha')
    
    parser.add_argument('--auto_lr_scale', action='store_true', help='Scale learning rate by batch_size/256 automatically')
    
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of warm-up epochs before main scheduler')
    parser.add_argument('--warmup_lr', type=float, default=1e-7, help='Initial LR during warm-up')
    
    parser.add_argument('--torch_compile', action='store_true', help='Compile model with torch.compile for speed (PyTorch 2.0+)')
    
    parser.add_argument('--no_class_balanced_sampler', dest='use_class_balanced_sampler', action='store_false',
                        help='Disable class-balanced sampler and use natural data distribution')
    
    args = parser.parse_args()
    
    # Apply configuration processing and validation
    args = process_config(args)
    
    return args

def process_config(args):
    """
    Process and validate configuration arguments.
    Apply cross-argument logic and compatibility fixes.
    """
    # DEBUG: Log that function is being called
    logger.info(f"DEBUG: process_config called with use_class_weights_only = {getattr(args, 'use_class_weights_only', 'NOT_SET')}")
    
    # === EARLY PROCESSING: Apply use_class_weights_only flag FIRST ===
    if args.use_class_weights_only:
        logger.info("🎯 Early config processing: Using class weights only mode")
        args.use_class_balanced_sampler = False
        args.progressive_class_balancing = False
        args.use_alternating_sampler = False
        args.use_action_balanced_sampler_only = False
        # Don't force loss_function change here - keep user's choice of focal vs weighted
        logger.info("   - All sampling techniques disabled")
        logger.info("   - Will use computed class weights for balanced training")
    
    # === CONFIGURATION PROCESSING ===
    
    # Set dataset path
    if args.dataset_root:
        from pathlib import Path
        args.mvfouls_path = str(Path(args.dataset_root) / "mvfouls")
    else:
        # Default to current directory
        from pathlib import Path
        args.mvfouls_path = str(Path(".") / "mvfouls")
    
    # Handle simple training mode
    if args.simple_training:
        logger.info("SIMPLE TRAINING MODE ENABLED - Disabling advanced features for debugging")
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
    
    # Process LR warmup configuration
    if args.lr_warmup and args.lr_warmup_steps == 0:
        # Calculate total training steps approximately
        # Note: This is an estimate - actual steps may vary with dataset size
        steps_per_epoch = max(1, 1000 // args.batch_size)  # Rough estimate, will be refined later
        total_steps = args.epochs * steps_per_epoch
        args.lr_warmup_steps = int(args.lr_warmup_pct * total_steps)
        logger.info(f"LR Warmup enabled: {args.lr_warmup_steps} steps ({args.lr_warmup_pct*100:.1f}% of estimated {total_steps} total steps)")
    
    # Set default warmup start LR if not specified
    if args.lr_warmup and args.lr_warmup_start_lr is None:
        args.lr_warmup_start_lr = args.lr / 100
        logger.info(f"LR Warmup start LR set to {args.lr_warmup_start_lr:.2e} (lr/100)")

    # Process scheduler configuration and provide recommendations
    args = process_scheduler_config(args)
    
    # Handle disable flags
    if args.disable_class_balancing:
        args.use_class_balanced_sampler = False
        args.class_weighting_strategy = 'none'
        logger.info("Class balancing disabled (both sampler and weights)")
    
    if args.disable_augmentation:
        args.aggressive_augmentation = False
        args.extreme_augmentation = False
        args.disable_in_model_augmentation = True
        logger.info("All augmentation disabled")
    
    # Handle MViT optimization disable flags
    if args.disable_gradient_checkpointing:
        args.enable_gradient_checkpointing = False
        logger.info("Gradient checkpointing disabled")
    
    if args.disable_memory_optimization:
        args.enable_memory_optimization = False
        logger.info("Memory optimization features disabled")
    
    # Validate MViT optimization settings
    if args.backbone_type == 'mvit':
        if not args.enable_gradient_checkpointing:
            logger.warning("⚠️  Gradient checkpointing disabled for MViT - may cause high memory usage")
        if not args.enable_memory_optimization:
            logger.warning("⚠️  Memory optimization disabled for MViT - may cause memory issues")
        if args.mvit_memory_cleanup_interval > 10:
            logger.warning(f"⚠️  MViT memory cleanup interval ({args.mvit_memory_cleanup_interval}) is high - consider reducing for better memory management")
        
        # Log MViT optimization status
        logger.info(f"🚀 MViT Optimizations - Gradient Checkpointing: {'✅' if args.enable_gradient_checkpointing else '❌'}, "
                   f"Memory Optimization: {'✅' if args.enable_memory_optimization else '❌'}, "
                   f"Sequential Processing: {'✅' if args.mvit_sequential_processing else '❌'}")
    
    # Handle data optimization disable flags
    if args.disable_data_optimization:
        args.enable_data_optimization = False
        logger.info("Data optimization disabled")
    
    # Set augmentation strength based on legacy flags (for backward compatibility)
    if not hasattr(args, 'augmentation_strength') or args.augmentation_strength is None:
        if args.extreme_augmentation:
            args.augmentation_strength = 'extreme'
        elif args.aggressive_augmentation:
            args.augmentation_strength = 'aggressive'
        else:
            args.augmentation_strength = 'moderate'
    
    # Handle new augmentation_strength argument
    if args.augmentation_strength == 'none':
        args.disable_augmentation = True
    elif args.augmentation_strength == 'mild':
        args.aggressive_augmentation = False
        args.extreme_augmentation = False
        args.temporal_jitter_strength = 1
        args.spatial_crop_strength = 0.9
        args.color_aug_strength = 0.1
        args.noise_strength = 0.01
        logger.info("Augmentation strength set to MILD.")
    elif args.augmentation_strength == 'moderate':
        args.aggressive_augmentation = False # Explicitly set to False
        args.extreme_augmentation = False # Explicitly set to False 
        logger.info("Augmentation strength set to MODERATE.")
    elif args.augmentation_strength == 'aggressive':
        args.aggressive_augmentation = True
        args.extreme_augmentation = False
        logger.info("Augmentation strength set to AGGRESSIVE.")
    elif args.augmentation_strength == 'extreme':
        args.aggressive_augmentation = True
        args.extreme_augmentation = True
        logger.info("Augmentation strength set to EXTREME.")
    
    # Validate learning rate ranges for error recovery
    if args.lr < 1e-6 or args.lr > 1e-2:
        logger.warning(f"⚠️  Learning rate {args.lr:.1e} is outside recommended range [1e-6, 1e-2]")
    
    # Force augmentation override if augmentation is manually disabled via command line
    if args.disable_augmentation:
        args.aggressive_augmentation = False
        args.extreme_augmentation = False
        args.disable_in_model_augmentation = True
        # args.augmentation_strength is purely informational at this point so we don't change it
    
    # Handle gradual fine-tuning configuration
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
            args.early_stopping_patience = 20
            logger.info(f"INFO: Default early stopping patience set to {args.early_stopping_patience} for standard training.")
    else:
        logger.info(f"💡 User-defined early stopping patience: {args.early_stopping_patience}")
    
    return args

def process_scheduler_config(args):
    """Process scheduler configuration and provide recommendations for layer unfreezing."""
    
    # Check for potentially problematic OneCycle + layer unfreezing combinations
    has_layer_unfreezing = (
        args.gradual_finetuning or 
        args.freezing_strategy in ['adaptive', 'progressive', 'gradient_guided', 'advanced']
    )
    
    if args.scheduler == 'onecycle' and has_layer_unfreezing:
        logger.warning("⚠️  SCHEDULER RECOMMENDATION:")
        logger.warning("   OneCycle scheduler with layer unfreezing can cause training instability")
        logger.warning("   because rebuilding the scheduler resets progress.")
        logger.warning("")
        logger.warning("   🎯 RECOMMENDED ALTERNATIVES:")
        logger.warning("   1. Use cosine decay: --scheduler cosine --epochs 30")
        logger.warning("   2. Use reduce_on_plateau for adaptive LR")
        logger.warning("")
        logger.warning("   The system will prevent OneCycle rebuilds, but cosine is more stable.")
        logger.warning("")
    
    # Provide learning rate recommendations based on configuration
    if has_layer_unfreezing:
        if args.head_lr == 1e-3 and args.backbone_lr < 1e-4:
            logger.info("💡 LEARNING RATE RECOMMENDATIONS:")
            logger.info(f"   Current: head_lr={args.head_lr:.1e}, backbone_lr={args.backbone_lr:.1e}")
            logger.info("   ✅ head_lr=1e-3 is good for frozen backbone training")
            logger.info("   💡 Consider backbone_lr=1e-4 (head_lr/10) when unfreezing layers")
    
    return args

def log_configuration_summary(args):
    """Log a comprehensive summary of the training configuration."""
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"Dataset: {args.dataset_root}")
    logger.info(f"Model: {args.backbone_name} ({args.frames_per_clip} frames, {args.img_height}x{args.img_width})")
    logger.info(f"Training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    logger.info(f"Optimizer: {args.optimizer}, scheduler={args.scheduler}")
    logger.info(f"Optimizer params: beta1={args.beta1}, beta2={args.beta2}, eps={args.eps}")
    if args.optimizer == 'sgd':
        logger.info(f"SGD params: momentum={args.momentum}, nesterov={args.nesterov}")
    if args.grad_clip_norm > 0:
        logger.info(f"Gradient clipping: max_norm={args.grad_clip_norm}")
    if args.use_ema:
        logger.info(f"EMA enabled: decay={args.ema_decay}")
    
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
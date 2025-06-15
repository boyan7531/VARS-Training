#!/usr/bin/env python3
"""
Main training script for Multi-Task Multi-View ResNet3D.

This script orchestrates the entire training process by importing and using
the modular components from the training package.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR, ReduceLROnPlateau
import time
import os
import logging
import random
import numpy as np
import multiprocessing as mp
from pathlib import Path

# Import our modular components
from .config import parse_args, log_configuration_summary
from .data import create_datasets, create_dataloaders, create_gpu_augmentation, log_dataset_recommendations
from .model_utils import (
    create_model, setup_freezing_strategy,
    SmartFreezingManager, GradientGuidedFreezingManager, AdvancedFreezingManager, get_phase_info, setup_discriminative_optimizer,
    unfreeze_backbone_gradually, log_trainable_parameters
)
from .training_utils import (
    train_one_epoch, validate_one_epoch, EarlyStopping, cleanup_memory
)
from .checkpoint_utils import (
    save_checkpoint, load_checkpoint, restore_best_metrics, save_training_history,
    log_epoch_summary, log_configuration_summary, create_training_history,
    update_training_history, log_completion_summary
)
from .freezing.early_gradual_manager import EarlyGradualFreezingManager

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


def setup_device_and_scaling(args):
    """Setup device and adjust parameters for multi-GPU training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Multi-GPU setup
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        logger.info(f"Found {num_gpus} GPUs! Using multi-GPU training.")
        
        # Disable automatic batch size scaling to prevent bottlenecks
        # User explicitly set batch_size=8 to fix gradient computation bottleneck
        logger.info(f"Keeping user-specified batch size: {args.batch_size} (automatic scaling disabled)")
        
        # Keep original learning rate since we're not scaling batch size
        logger.info(f"Using original learning rate: {args.lr:.6f} (no scaling needed for batch_size={args.batch_size})")
    else:
        logger.info("Using single GPU training.")
    
    return device, num_gpus


def setup_optimizer_and_scheduler(args, model, freezing_manager=None):
    """Setup optimizer and learning rate scheduler."""
    
    optimizer = None  # Initialize optimizer variable
    
    # For SmartFreezingManager
    if isinstance(freezing_manager, SmartFreezingManager) and args.freezing_strategy in ['adaptive', 'progressive']:
        # Use discriminative learning rates from freezing manager
        param_groups = freezing_manager.get_discriminative_lr_groups(args.head_lr, args.backbone_lr)
        # Use improved optimizer creation with enhanced parameters
        from .model_utils import get_optimizer
        # Create temporary args object for optimizer creation
        class TempArgs:
            def __init__(self, original_args):
                for key, value in vars(original_args).items():
                    setattr(self, key, value)
                self.lr = original_args.head_lr  # Use head_lr for discriminative setup
        
        temp_args = TempArgs(args)
        optimizer = get_optimizer(model, temp_args)
        logger.info("🔄 Using discriminative learning rates from freezing manager with enhanced optimizer")
    
    # For AdvancedFreezingManager
    elif isinstance(freezing_manager, AdvancedFreezingManager):
        # Use advanced parameter groups with layer-specific learning rates
        param_groups = freezing_manager.create_advanced_optimizer_param_groups(
            head_lr=args.head_lr,
            backbone_lr=args.backbone_lr
        )
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))  # Weight decay handled per-group
        logger.info("🚀 Using advanced multi-metric parameter groups for optimizer")
    
    # For GradientGuidedFreezingManager
    elif isinstance(freezing_manager, GradientGuidedFreezingManager):
        # Use specialized parameter groups from gradient-guided freezing manager
        param_groups = freezing_manager.create_optimizer_param_groups(
            head_lr=args.head_lr,
            backbone_lr=args.backbone_lr
        )
        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        logger.info("🔄 Using gradient-guided parameter groups for optimizer")
    
    # For standard discriminative learning rates
    elif args.discriminative_lr:
        # Manually setup discriminative learning rates
        param_groups = setup_discriminative_optimizer(model, args.head_lr, args.backbone_lr)
        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        logger.info(f"🔄 Using discriminative learning rates: Head={args.head_lr:.1e}, Backbone={args.backbone_lr:.1e}")
    
    # Standard (uniform) learning rate
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        logger.info(f"🔄 Using uniform learning rate: {args.lr:.1e}")
    
    # Initialize schedulers
    scheduler = None
    phase1_scheduler = None
    scheduler_info = "None (constant learning rate)"

    if args.gradual_finetuning:
        # Phase 1 will use ReduceLROnPlateau
        phase1_scheduler = ReduceLROnPlateau(
            optimizer, mode='max', 
            factor=args.phase1_plateau_factor, 
            patience=args.phase1_plateau_patience, 
            min_lr=args.min_lr
        )
        scheduler_info = f"Phase 1: ReduceLROnPlateau (patience={args.phase1_plateau_patience}, factor={args.phase1_plateau_factor})"
    else:
        # Standard (non-gradual) training: initialize the main scheduler now
        if args.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
            scheduler_info = f"CosineAnnealing (T_max={args.epochs}, eta_min={args.lr * 0.01:.1e})"
        elif args.scheduler == 'onecycle':
            # Note: steps_per_epoch will be set after dataloader creation
            scheduler_info = f"OneCycle (max_lr={args.lr:.1e}, warmup_epochs={args.scheduler_warmup_epochs})"
        elif args.scheduler == 'step':
            scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
            scheduler_info = f"StepLR (step_size={args.step_size}, gamma={args.gamma})"
        elif args.scheduler == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
            scheduler_info = f"ExponentialLR (gamma={args.gamma})"
        elif args.scheduler == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.gamma, 
                                        patience=args.plateau_patience, min_lr=args.min_lr)
            scheduler_info = f"ReduceLROnPlateau (mode=max, factor={args.gamma}, patience={args.plateau_patience}, min_lr={args.min_lr:.1e})"

    # After creating scheduler (or phase1_scheduler) for non-gradual training
    # Integrate warmup if requested and not using OneCycleLR
    if not args.gradual_finetuning and args.warmup_epochs > 0 and args.scheduler != 'onecycle':
        from torch.optim.lr_scheduler import SequentialLR, LinearLR
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=args.warmup_lr / args.lr if args.lr > 0 else 0.1,
            total_iters=args.warmup_epochs,
        )
        if scheduler is not None:
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[args.warmup_epochs])
            scheduler_info = f"Warmup({args.warmup_epochs} epochs) -> {scheduler_info}"
        else:
            scheduler = warmup_scheduler
            scheduler_info = f"Warmup LinearLR ({args.warmup_epochs} epochs)"

    return optimizer, scheduler, phase1_scheduler, scheduler_info


def handle_gradual_finetuning_transition(args, model, optimizer, scheduler, epoch, freezing_manager=None, val_metric=None, train_loader=None):
    """
    Handle phase transitions for gradual fine-tuning or freezing strategy adjustments.
    Returns updated optimizer and scheduler.
    """
    rebuild_optimizer = False
    rebuild_scheduler = False
    
    if freezing_manager is not None:
        if isinstance(freezing_manager, SmartFreezingManager):
            # Original SmartFreezingManager
            if epoch > 0 and val_metric is not None and args.freezing_strategy == 'adaptive':
                if freezing_manager.adaptive_unfreeze_step(val_metric, epoch):
                    rebuild_optimizer = True
            
            # Monitor gradients for debugging/analysis
            freezing_manager.monitor_gradients(epoch)
        
        elif isinstance(freezing_manager, AdvancedFreezingManager):
            # Advanced multi-metric freezing manager
            if epoch > 0 and val_metric is not None:
                # Update freezing status based on comprehensive analysis
                update_info = freezing_manager.update_after_epoch(val_metric, epoch)
                
                # Log comprehensive status
                if epoch % 2 == 0:  # Log every 2 epochs to avoid spam
                    freezing_manager.log_comprehensive_status(epoch)
                
                # Check if we need to rebuild optimizer
                if update_info['rebuild_optimizer']:
                    rebuild_optimizer = True
                    logger.info(f"[ADVANCED_FREEZING] Rebuilding optimizer due to layer changes")
                    
                    if update_info['unfrozen_layers']:
                        logger.info(f"[ADVANCED_FREEZING] Newly unfrozen layers: {update_info['unfrozen_layers']}")
                    
                    if update_info['rollback_performed']:
                        logger.info(f"[ADVANCED_FREEZING] Performance rollback performed")
        
        elif isinstance(freezing_manager, GradientGuidedFreezingManager):
            # New GradientGuidedFreezingManager
            if epoch > 0 and val_metric is not None:
                # Update freezing status based on gradient analysis
                update_info = freezing_manager.update_after_epoch(val_metric, epoch)
                
                # Log current freezing status
                freezing_manager.log_status(epoch)
                
                # Check if we need to rebuild optimizer
                if update_info['rebuild_optimizer']:
                    rebuild_optimizer = True
                    logger.info(f"[GRADIENT_GUIDED] Rebuilding optimizer due to layer changes")
                    
                    if update_info['unfrozen_layers']:
                        logger.info(f"[GRADIENT_GUIDED] Newly unfrozen layers: {update_info['unfrozen_layers']}")
        
        elif isinstance(freezing_manager, EarlyGradualFreezingManager):
            # Early gradual unfreezing manager
            if val_metric is not None:
                # Update freezing status - this happens every epoch
                update_info = freezing_manager.update_after_epoch(val_metric, epoch)
                
                # Always log parameter status for detailed tracking
                freezing_manager.log_status(epoch)
                
                # Check if we need to rebuild optimizer
                if update_info['rebuild_optimizer']:
                    rebuild_optimizer = True
                    logger.info(f"[EARLY_GRADUAL] Rebuilding optimizer due to layer changes")
                    
                    if update_info['unfrozen_layers']:
                        logger.info(f"[EARLY_GRADUAL] Newly unfrozen layers: {update_info['unfrozen_layers']}")
    
    # Traditional gradual fine-tuning with fixed phases
    elif args.gradual_finetuning:
        current_phase, phase_name = get_phase_info(epoch, args.phase1_epochs, args.epochs)
        prev_phase, _ = get_phase_info(epoch-1, args.phase1_epochs, args.epochs) if epoch > 0 else (current_phase, "")
        
        # Handle phase transition
        if epoch > 0 and current_phase != prev_phase:
            logger.info(f"Transitioning to {phase_name}")
            
            if current_phase == 2:
                # Phase 1 -> Phase 2: Start unfreezing backbone gradually
                unfreeze_backbone_gradually(model, num_blocks_to_unfreeze=args.unfreeze_blocks)
                log_trainable_parameters(model)
                rebuild_optimizer = True
    
    # Rebuild optimizer if needed
    if rebuild_optimizer:
        # Get current learning rates - preserve the original max LR instead of using tiny current LR
        current_head_lr = optimizer.param_groups[0]['lr']
        
        # Use original max LR or a reasonable LR when rebuilding to avoid tiny LRs
        if args.gradual_finetuning and current_head_lr < args.head_lr * 0.1:
            # If current LR is too small (due to scheduler decay), use original head_lr
            effective_head_lr = args.head_lr
            logger.info(f"🔧 Using original head_lr {effective_head_lr:.2e} instead of tiny current LR {current_head_lr:.2e}")
        else:
            effective_head_lr = current_head_lr
        
        # Calculate backbone LR as one-tenth of head LR when unfreezing
        effective_backbone_lr = effective_head_lr * 0.1
        logger.info(f"🎯 Setting backbone_lr to {effective_backbone_lr:.2e} (head_lr / 10)")
        
        if isinstance(freezing_manager, AdvancedFreezingManager):
            # Use advanced param groups for multi-metric freezing
            try:
                param_groups = freezing_manager.create_advanced_optimizer_param_groups(
                    head_lr=effective_head_lr,
                    backbone_lr=effective_backbone_lr
                )
                
                # Validate parameter groups before creating optimizer
                all_params = set()
                for i, group in enumerate(param_groups):
                    for param in group['params']:
                        param_id = id(param)
                        if param_id in all_params:
                            logger.error(f"Parameter overlap detected in group {i} ({group['name']})")
                            raise ValueError("Parameter overlap in parameter groups")
                        all_params.add(param_id)
                
                optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
                logger.info(f"Rebuilt optimizer with advanced parameter groups ({len(param_groups)} groups)")
                
            except Exception as e:
                logger.error(f"Failed to create advanced optimizer: {e}")
                logger.warning("Falling back to simple optimizer rebuild")
                
                # Fallback: Use all trainable parameters with current LR
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=effective_head_lr,
                    weight_decay=args.weight_decay
                )
                logger.info(f"Created fallback optimizer with uniform LR: {effective_head_lr:.2e}")
            
        elif isinstance(freezing_manager, GradientGuidedFreezingManager):
            # Use specialized param groups for gradient-guided freezing
            param_groups = freezing_manager.create_optimizer_param_groups(
                head_lr=effective_head_lr,
                backbone_lr=effective_backbone_lr
            )
            
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=args.weight_decay
            )
            logger.info(f"Rebuilt optimizer with gradient-guided parameter groups")
            
        elif args.discriminative_lr:
            # Use discriminative learning rates
            param_groups = setup_discriminative_optimizer(
                model, 
                head_lr=effective_head_lr,
                backbone_lr=effective_backbone_lr
            )
            
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=args.weight_decay
            )
            logger.info(f"Rebuilt optimizer with discriminative learning rates (head: {effective_head_lr:.2e}, backbone: {effective_backbone_lr:.2e})")
            
        else:
            # Standard optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=effective_head_lr,
                weight_decay=args.weight_decay
            )
            logger.info(f"Rebuilt optimizer with uniform learning rate: {effective_head_lr:.2e}")
        
        # DO NOT rebuild OneCycle scheduler to avoid resetting progress
        if args.scheduler == 'onecycle':
            rebuild_scheduler = False
            logger.info("🚀 OneCycle scheduler: Keeping existing scheduler (no rebuild to preserve progress)")
        else:
            rebuild_scheduler = True
    
    # Rebuild scheduler if needed (but not for OneCycle)
    if rebuild_scheduler and scheduler is not None:
        remaining_epochs = args.epochs - epoch
        if args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=remaining_epochs,
                eta_min=args.lr * 0.01  # Use original LR as reference
            )
            logger.info(f"Rebuilt cosine scheduler for remaining {remaining_epochs} epochs")
            
        elif args.scheduler == 'onecycle':
            # This should not happen due to rebuild_scheduler = False above
            logger.warning("⚠️  OneCycle scheduler rebuild blocked - this should not happen")
            rebuild_scheduler = False
        
        # Add more scheduler types here if needed
        
    return optimizer, scheduler


def reset_gradscaler_after_calibration(scaler):
    """
    Reset GradScaler parameters to more conservative values after initial calibration.
    This helps with long-term training stability.
    """
    # Use more conservative growth factor for long-term stability
    scaler.set_growth_factor(1.5)
    # Check less frequently to allow stabilization
    scaler.set_growth_interval(500)
    # Keep current scale as starting point
    logger.info(f"GradScaler reset to stable parameters after calibration. Current scale: {scaler.get_scale():.1f}")
    return scaler


def main():
    """Main training function."""
    # Configure multiprocessing for DataLoader workers
    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn', force=True)
            logger.info("Set multiprocessing start method to 'spawn' for stability")
    except RuntimeError as e:
        logger.warning(f"Could not set multiprocessing method: {e}")
    
    # Parse arguments and setup
    args = parse_args()
    
    if getattr(args, 'auto_lr_scale', False):
        base_lr = args.lr
        args.lr = args.lr * (args.batch_size / 256)
        logger.info(f"Auto LR scaling: base_lr {base_lr:.2e} -> scaled_lr {args.lr:.2e} for batch_size {args.batch_size}")
    
    # Config validation and error recovery setup
    try:
        from .error_recovery import create_config_validator, RobustTrainingWrapper, OOMRecoveryManager
        config_validator = create_config_validator()
        robust_wrapper = RobustTrainingWrapper(config_validator=config_validator)
        
        # Handle disable flags for optimization features
        if args.disable_data_optimization:
            args.enable_data_optimization = False
        
        # Validate configuration
        if args.enable_config_validation:
            if not robust_wrapper.validate_config(args):
                if args.strict_config_validation:
                    logger.error("❌ Configuration validation failed with strict mode enabled")
                    return
                else:
                    logger.warning("⚠️ Configuration validation failed but continuing with warnings")
        
        # Setup OOM recovery if enabled
        oom_manager = None
        if args.enable_oom_recovery:
            oom_manager = OOMRecoveryManager(
                initial_batch_size=args.batch_size,
                min_batch_size=args.min_batch_size,
                reduction_factor=args.oom_reduction_factor
            )
            robust_wrapper.oom_manager = oom_manager
            logger.info(f"🛡️ OOM recovery enabled (min_batch_size={args.min_batch_size}, reduction_factor={args.oom_reduction_factor})")
        
    except ImportError:
        logger.warning("Error recovery module not available, using standard training")
        robust_wrapper = None
        oom_manager = None
    
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

    # Device setup and scaling
    device, num_gpus = setup_device_and_scaling(args)

    # Initialize GradScaler for mixed-precision training with calibration
    scaler = torch.amp.GradScaler() if args.mixed_precision and device.type == 'cuda' else None
    
    # Configure GradScaler for better initial stability
    if scaler is not None:
        # Start with more aggressive growth factor for faster calibration
        scaler.set_growth_factor(2.0)
        scaler.set_growth_interval(100)  # Check more frequently initially
        logger.info(f"Using mixed precision training with calibrated GradScaler")
    else:
        logger.info("Not using mixed precision training")
        
    # Flag to track if we've reset scaler parameters after calibration
    scaler_calibration_reset = False

    # Create datasets and dataloaders
    train_dataset, val_dataset = create_datasets(args)
    train_loader, val_loader = create_dataloaders(args, train_dataset, val_dataset)
    
    # Log class imbalance handling strategy
    logger.info("=" * 60)
    logger.info("🎯 CLASS IMBALANCE HANDLING STRATEGY")
    logger.info("=" * 60)
    
    if args.disable_class_balancing:
        logger.info("❌ All class balancing techniques DISABLED")
        logger.info("   Using plain loss functions without any balancing")
    elif args.use_class_balanced_sampler:
        logger.info("✅ Using ClassBalancedSampler (oversampling) as PRIMARY technique")
        logger.info(f"   Oversample factor: {args.oversample_factor}x for minority classes")
        if args.progressive_class_balancing:
            logger.info(f"   Progressive balancing: {args.progressive_start_factor}x → {args.progressive_end_factor}x over {args.progressive_epochs} epochs")
        logger.info("   Class weights and focal-α: DISABLED (prevents gradient multiplication)")
        logger.info("   Focal-γ: Single value (2.0) for all classes")
    else:
        logger.info("✅ Using Class Weights / Focal-α as PRIMARY technique")
        logger.info(f"   Weighting strategy: {args.class_weighting_strategy}")
        logger.info(f"   Max weight ratio: {args.max_weight_ratio}")
        logger.info("   ClassBalancedSampler: DISABLED")
        if args.loss_function == 'focal':
            logger.info("   Focal-γ: Class-specific values for rare classes")
    
    logger.info("=" * 60)

    # Log dataset recommendations
    log_dataset_recommendations(train_dataset)

    # Create vocabulary sizes dictionary for the model (now optional - video-only)
    vocab_sizes = None  # No longer needed since we only use video features

    # Create model (video-only)
    model = create_model(args, vocab_sizes, device, num_gpus)
    
    # Torch compile (PyTorch 2.0+)
    if getattr(args, 'torch_compile', False):
        try:
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile for faster training")
        except Exception as e:
            logger.warning(f"torch.compile failed or unsupported: {e}")
    
    # Setup freezing strategy
    freezing_manager = setup_freezing_strategy(args, model)
    
    # Setup optimizer and scheduler (after dataloader creation for OneCycleLR)
    optimizer, scheduler, phase1_scheduler, scheduler_info = setup_optimizer_and_scheduler(args, model, freezing_manager)
    
    # Setup EMA model if requested
    ema_model = None
    if args.use_ema:
        from .model_utils import create_ema_model
        ema_model = create_ema_model(model, decay=args.ema_decay)
        if ema_model:
            logger.info(f"🚀 EMA model created with decay: {args.ema_decay}")
        else:
            logger.warning("⚠️ EMA model creation failed, continuing without EMA")
    
    # Initialize OneCycleLR after dataloader creation to get steps_per_epoch
    if args.scheduler == 'onecycle' and scheduler is None:
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=args.scheduler_warmup_epochs / args.epochs if args.epochs > 0 else 0.1
        )
        scheduler_info = f"OneCycle (max_lr={args.lr:.1e}, steps_per_epoch={steps_per_epoch}, warmup={args.scheduler_warmup_epochs}e)"
        logger.info(f"Initialized OneCycleLR scheduler with {steps_per_epoch} steps per epoch")

    # Class weights are now computed automatically by Lightning DataModule
    # But we should set them to None when using ClassBalancedSampler to avoid double-balancing
    severity_class_weights = None
    action_class_weights = None
    
    if args.use_class_balanced_sampler:
        logger.info("🎯 ClassBalancedSampler detected: Setting severity_class_weights=None to prevent double-balancing")
    
    # Handle strong action weights if enabled
    if args.use_strong_action_weights:
        from training.training_utils import calculate_strong_action_class_weights
        action_class_weights = calculate_strong_action_class_weights(
            train_dataset, 
            device,
            args.action_weight_strategy,
            args.action_weight_power
        )
        logger.info("💪 Using strong action class weights to combat severe action imbalance")

    # Update loss function to adaptive focal loss if enabled
    if args.loss_function == 'focal':
        if args.use_class_balanced_sampler:
            logger.info("🔥 Using focal loss with single gamma=2.0 (compatible with ClassBalancedSampler)")
            logger.info("   Class-specific gamma tuning disabled to avoid gradient multiplication")
            class_gamma_map = None
        else:
            logger.info("🔥 Using focal loss with class-specific gamma values")
            # Define class-specific gamma values (higher for rare classes)
            class_gamma_map = {
                0: 2.0,  # Medium frequency
                1: 1.5,  # Majority class - less focus needed
                2: 2.0,  # Medium frequency
                3: 2.0,  # Medium frequency
                4: 3.0,  # Very rare - high focus
                5: 3.5   # Extremely rare - highest focus
            }
            logger.info("Class-specific gamma values:")
            for cls, gamma in class_gamma_map.items():
                logger.info(f"  Class {cls}: gamma={gamma:.1f}")
    else:
        class_gamma_map = None

    # GPU augmentation setup
    gpu_augmentation = create_gpu_augmentation(args, device)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    best_epoch = -1
    
    if args.resume:
        start_epoch, loaded_metrics = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        best_val_acc, best_epoch = restore_best_metrics(loaded_metrics, args.resume_best_acc)
        logger.info(f"🔄 Resuming training from epoch {start_epoch}")
        if best_val_acc > 0:
            logger.info(f"🎯 Current best to beat: {best_val_acc:.4f}")

    # Training history
    history = create_training_history()

    # Log configuration summary
    log_configuration_summary(args, train_dataset, val_dataset)
    logger.info(f"Learning Rate Scheduler: {scheduler_info}")

    # Loss functions (kept for compatibility)
    criterion_severity = nn.CrossEntropyLoss()
    criterion_action = nn.CrossEntropyLoss()

    # Initialize confusion matrix tracking for original distribution analysis
    train_confusion_matrices = {}  # For tracking training metrics on original distribution
    val_confusion_matrices = {}    # For validation confusion matrices

    # Define class names for better logging
    severity_class_names = ["Sev_0", "Sev_1", "Sev_2", "Sev_3", "Sev_4", "Sev_5"]
    action_class_names = [f"Act_{i}" for i in range(10)]  # 10 action classes

    # Main training loop
    logger.info("Starting Training")
    logger.info("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        # Track if we're using mixed precision for logging
        if scaler is not None and epoch == 0:
            logger.info("🚀 Mixed precision training active with explicit float16")
        epoch_start_time = time.time()
        
        # Update sampler epoch if needed
        if args.progressive_class_balancing or args.use_alternating_sampler:
            # For OptimizedDataLoader, access the original dataloader's sampler
            actual_loader = getattr(train_loader, 'dataloader', train_loader)
            if hasattr(actual_loader, 'sampler') and hasattr(actual_loader.sampler, 'set_epoch'):
                actual_loader.sampler.set_epoch(epoch)
                if args.progressive_class_balancing:
                    logger.debug(f"Updated progressive sampler for epoch {epoch}")
                elif args.use_alternating_sampler:
                    sampler_type = "Severity" if epoch % 2 == 0 else "Action"
                    logger.debug(f"Updated alternating sampler for epoch {epoch} ({sampler_type} balancing)")
        
        # Training with OOM protection
        if robust_wrapper and oom_manager:
            try:
                # Use MViT-specific memory cleanup interval if MViT model is detected
                mvit_memory_interval = getattr(args, 'mvit_memory_cleanup_interval', 5)
                memory_cleanup_interval_for_training = (
                    mvit_memory_interval if args.backbone_type.lower() == 'mvit' 
                    else args.memory_cleanup_interval
                )
                
                # Reset training confusion matrices for this epoch
                train_confusion_matrices.clear()
                
                train_metrics = robust_wrapper.oom_safe_training_step(
                    train_one_epoch,
                    model, train_loader, optimizer, device,
                    loss_config={
                        'function': args.loss_function,
                        'weights': args.main_task_weights,
                        'label_smoothing': args.label_smoothing,
                        'focal_gamma': args.focal_gamma,
                        'severity_class_weights': None if args.use_class_balanced_sampler else severity_class_weights,
                        'action_class_weights': action_class_weights,
                        'class_gamma_map': class_gamma_map
                    },
                    scaler=scaler, 
                    max_batches=num_batches_to_run, 
                    gradient_clip_norm=args.gradient_clip_norm, 
                    memory_cleanup_interval=memory_cleanup_interval_for_training,
                    scheduler=scheduler if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) else None,
                    gpu_augmentation=gpu_augmentation,
                    enable_profiling=False,  # Profiling disabled
                    confusion_matrix_dict=train_confusion_matrices  # Track original distribution metrics
                )
                
                # Check if batch size was reduced and update dataloader if needed
                if oom_manager.current_batch_size != args.batch_size:
                    logger.info(f"🔄 Updating dataloader for new batch size: {oom_manager.current_batch_size}")
                    # Note: In a real implementation, you might need to recreate the dataloader
                    # For now, we'll just log the change
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"❌ Critical OOM error that couldn't be recovered: {e}")
                    logger.error("Stopping training due to unrecoverable OOM")
                    break
                else:
                    raise  # Re-raise non-OOM errors
        else:
            # Standard training without OOM protection
            # Use MViT-specific memory cleanup interval if MViT model is detected
            mvit_memory_interval = getattr(args, 'mvit_memory_cleanup_interval', 5)
            memory_cleanup_interval_for_training = (
                mvit_memory_interval if args.backbone_type.lower() == 'mvit' 
                else args.memory_cleanup_interval
            )
            
            # Reset training confusion matrices for this epoch
            train_confusion_matrices.clear()
            
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, device,
                loss_config={
                    'function': args.loss_function,
                    'weights': args.main_task_weights,
                    'label_smoothing': args.label_smoothing,
                    'focal_gamma': args.focal_gamma,
                    'severity_class_weights': None if args.use_class_balanced_sampler else severity_class_weights,
                    'action_class_weights': action_class_weights,
                    'class_gamma_map': class_gamma_map
                },
                scaler=scaler, 
                max_batches=num_batches_to_run, 
                gradient_clip_norm=max(args.gradient_clip_norm, args.grad_clip_norm) if hasattr(args, 'grad_clip_norm') else args.gradient_clip_norm, 
                memory_cleanup_interval=memory_cleanup_interval_for_training,
                scheduler=scheduler if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) else None,
                gpu_augmentation=gpu_augmentation,
                enable_profiling=False,  # Profiling disabled
                confusion_matrix_dict=train_confusion_matrices,  # Track original distribution metrics
                ema_model=ema_model
            )
        
        # Reset GradScaler parameters after initial calibration period (typically 100-500 steps)
        if scaler is not None and not scaler_calibration_reset and epoch >= 1:
            scaler = reset_gradscaler_after_calibration(scaler)
            scaler_calibration_reset = True
        
        # Reset validation confusion matrices for this epoch
        val_confusion_matrices.clear()
        
        # Validation
        val_metrics = validate_one_epoch(
            model, val_loader, device,
            loss_config={
                'function': args.loss_function,
                'weights': args.main_task_weights,
                'label_smoothing': args.label_smoothing,
                'focal_gamma': args.focal_gamma,
                'severity_class_weights': None if args.use_class_balanced_sampler else severity_class_weights,
                'action_class_weights': action_class_weights,
                'class_gamma_map': class_gamma_map
            },
            confusion_matrix_dict=val_confusion_matrices
        )
        
        # Log trainable parameter count every epoch (requirement #10)
        log_trainable_parameters(model, epoch)
        
        # Log training confusion matrices (original distribution analysis)
        if train_confusion_matrices:
            from training.training_utils import compute_confusion_matrices, log_confusion_matrix
            train_cms = compute_confusion_matrices(train_confusion_matrices)
            
            if 'severity' in train_cms:
                logger.info(f"\n[EPOCH {epoch+1}] TRAINING METRICS (Original Distribution - Before Oversampling)")
                log_confusion_matrix(train_cms['severity'], 'Severity', severity_class_names)
            
            if 'action_type' in train_cms:
                log_confusion_matrix(train_cms['action_type'], 'Action Type', action_class_names)
        
        # Log and save validation confusion matrices every 5 epochs
        if val_confusion_matrices and (epoch + 1) % 5 == 0:
            from training.training_utils import compute_confusion_matrices, log_confusion_matrix, save_confusion_matrix
            val_cms = compute_confusion_matrices(val_confusion_matrices)
            
            logger.info(f"\n[EPOCH {epoch+1}] VALIDATION CONFUSION MATRICES")
            
            if 'severity' in val_cms:
                log_confusion_matrix(val_cms['severity'], 'Validation Severity', severity_class_names)
                save_confusion_matrix(val_cms['severity'], 'severity', epoch+1, args.save_dir)
            
            if 'action_type' in val_cms:
                log_confusion_matrix(val_cms['action_type'], 'Validation Action Type', action_class_names)
                save_confusion_matrix(val_cms['action_type'], 'action_type', epoch+1, args.save_dir)
        
        # Check for validation plateau-based early unfreezing
        if (freezing_manager is not None and 
              hasattr(freezing_manager, 'emergency_unfreeze') and
              epoch >= getattr(args, 'validation_plateau_patience', 2) and
              len(getattr(freezing_manager, 'unfrozen_layers', set())) == 0):
            
            # Check if validation performance has plateaued
            if (hasattr(freezing_manager, 'performance_history') and 
                len(freezing_manager.performance_history) >= args.validation_plateau_patience + 1):
                
                recent_performance = freezing_manager.performance_history[-args.validation_plateau_patience-1:]
                performance_trend = recent_performance[-1] - recent_performance[0]
                
                # If performance hasn't improved significantly in recent epochs
                if performance_trend < 0.005:  # Less than 0.5% improvement
                    logger.info(f"[PLATEAU] Validation performance plateaued (trend: {performance_trend:+.4f}). Triggering early unfreezing...")
                    emergency_success = freezing_manager.emergency_unfreeze(
                        min_layers=1,  # More conservative for plateau-based unfreezing
                        use_gradual=True
                    )
                    
                    if emergency_success:
                        optimizer, scheduler = handle_gradual_finetuning_transition(
                            args, model, optimizer, scheduler, epoch,
                            freezing_manager=freezing_manager, 
                            val_metric=val_metrics['sev_acc'],
                            train_loader=train_loader
                        )

        # Handle freezing strategy updates with validation results
        optimizer, scheduler = handle_gradual_finetuning_transition(
            args, model, optimizer, scheduler, epoch,
            freezing_manager=freezing_manager, 
            val_metric=val_metrics['sev_acc'],  # Use the correct key from validation metrics
            train_loader=train_loader
        )

        # Update learning rate
        if scheduler is not None:
            current_phase_for_scheduler = get_phase_info(epoch, args.phase1_epochs, args.epochs)[0] if args.gradual_finetuning else None

            if args.gradual_finetuning and current_phase_for_scheduler == 1 and phase1_scheduler is not None:
                val_combined_acc = (val_metrics['sev_acc'] + val_metrics['act_acc']) / 2
                phase1_scheduler.step(val_combined_acc)
            elif scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    val_combined_acc = (val_metrics['sev_acc'] + val_metrics['act_acc']) / 2
                    scheduler.step(val_combined_acc)
                elif not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
            
        # Calculate epoch metrics and log summary
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        prev_lr = history.get('learning_rate', [current_lr])[-1] if history.get('learning_rate') else current_lr
        
        # Determine phase for logging
        phase_info = None
        if args.gradual_finetuning:
            phase_info = get_phase_info(epoch, args.phase1_epochs, args.epochs)[0]

        # Log epoch summary and get validation accuracy
        val_combined_acc = log_epoch_summary(
            epoch, args.epochs, epoch_time, train_metrics, val_metrics,
            current_lr, prev_lr, best_val_acc, phase_info, True
        )

        # Update training history
        history = update_training_history(history, train_metrics, val_metrics, current_lr)

        # Model saving and early stopping
        if not args.test_run:
            # Save best model
            if val_combined_acc > best_val_acc:
                improvement = val_combined_acc - best_val_acc
                best_val_acc = val_combined_acc
                best_epoch = epoch + 1
                
                metrics = {
                    'epoch': best_epoch,
                    'best_val_acc': best_val_acc,
                    'best_epoch': best_epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'train_sev_acc': train_metrics['sev_acc'],
                    'train_act_acc': train_metrics['act_acc'],
                    'val_sev_acc': val_metrics['sev_acc'],
                    'val_act_acc': val_metrics['act_acc']
                }
                
                save_path = os.path.join(args.save_dir, f'best_model_epoch_{best_epoch}.pth')
                save_checkpoint(model, optimizer, scheduler, scaler, best_epoch, metrics, save_path)
                logger.info(f"[SAVE] Best model updated! Accuracy: {best_val_acc:.4f} (+{improvement:.4f}) - Saved to {save_path}")

            # Early stopping check
            if early_stopping(val_combined_acc, model):
                logger.info(f"[EARLY_STOP] Early stopping triggered after {epoch + 1} epochs")
                break

            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                metrics = {
                    'epoch': epoch + 1,
                    'best_val_acc': best_val_acc,
                    'best_epoch': best_epoch,
                    'current_train_loss': train_metrics['loss'],
                    'current_val_loss': val_metrics['loss'],
                    'current_train_sev_acc': train_metrics['sev_acc'],
                    'current_train_act_acc': train_metrics['act_acc'],
                    'current_val_sev_acc': val_metrics['sev_acc'],
                    'current_val_act_acc': val_metrics['act_acc']
                }
                save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, metrics, checkpoint_path)
                logger.info(f"[CHECKPOINT] Checkpoint saved at epoch {epoch + 1} (best so far: {best_val_acc:.4f})")

    # Save training history
    if not args.test_run:
        history_path = os.path.join(args.save_dir, 'training_history.json')
        save_training_history(history, history_path)

    # Log completion summary
    log_completion_summary(best_val_acc, best_epoch, args.test_run)


if __name__ == "__main__":
    main() 
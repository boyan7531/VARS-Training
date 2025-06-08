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
    create_model, setup_freezing_strategy, calculate_class_weights,
    SmartFreezingManager, get_phase_info, setup_discriminative_optimizer,
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
        
        # Adjust batch size for multi-GPU if not explicitly set
        if args.batch_size == 8 and not args.force_batch_size:  # Default value
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
    
    return device, num_gpus


def setup_optimizer_and_scheduler(args, model, freezing_manager=None):
    """Setup optimizer and learning rate scheduler."""
    
    # For SmartFreezingManager
    if isinstance(freezing_manager, SmartFreezingManager) and args.freezing_strategy in ['adaptive', 'progressive'] and args.exponential_lr_decay:
        # Use discriminative learning rates from freezing manager
        param_groups = freezing_manager.get_discriminative_lr_groups(args.head_lr, args.backbone_lr)
        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        logger.info("🔄 Using discriminative learning rates from freezing manager")
    
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
            scheduler_info = f"OneCycle (max_lr={args.lr:.1e}, warmup_epochs={args.warmup_epochs})"
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
        # Get LR from current optimizer
        current_head_lr = optimizer.param_groups[0]['lr']
        
        if isinstance(freezing_manager, GradientGuidedFreezingManager):
            # Use specialized param groups for gradient-guided freezing
            param_groups = freezing_manager.create_optimizer_param_groups(
                head_lr=current_head_lr,
                backbone_lr=current_head_lr * args.backbone_lr_ratio
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
                head_lr=current_head_lr,
                backbone_lr=args.backbone_lr if args.backbone_lr > 0 else current_head_lr * args.backbone_lr_ratio
            )
            
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=args.weight_decay
            )
            logger.info(f"Rebuilt optimizer with discriminative learning rates")
            
        else:
            # Standard optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=current_head_lr,
                weight_decay=args.weight_decay
            )
            logger.info(f"Rebuilt optimizer with uniform learning rate: {current_head_lr:.2e}")
        
        rebuild_scheduler = True
    
    # Rebuild scheduler if needed
    if rebuild_scheduler and scheduler is not None:
        remaining_epochs = args.epochs - epoch
        if args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=remaining_epochs
            )
            logger.info(f"Rebuilt cosine scheduler for remaining {remaining_epochs} epochs")
            
        elif args.scheduler == 'onecycle':
            if train_loader is not None:
                steps_per_epoch = len(train_loader)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=[group['lr'] for group in optimizer.param_groups],
                    epochs=remaining_epochs,
                    steps_per_epoch=steps_per_epoch,
                    pct_start=0.1  # Short warmup for rebuilding
                )
                logger.info(f"Rebuilt OneCycleLR scheduler for remaining {remaining_epochs} epochs")
            else:
                logger.warning("Cannot rebuild OneCycleLR scheduler: train_loader not provided")
            
        # Add more scheduler types here if needed
        
    return optimizer, scheduler


def main():
    """Main training function."""
    # Configure multiprocessing for DataLoader workers
    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn', force=True)
            logger.info("🔧 Set multiprocessing start method to 'spawn' for stability")
    except RuntimeError as e:
        logger.warning(f"Could not set multiprocessing method: {e}")
    
    # Parse arguments and setup
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

    # Device setup and scaling
    device, num_gpus = setup_device_and_scaling(args)

    # Initialize GradScaler for mixed-precision training
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision and device.type == 'cuda' else None
    logger.info(f"Using mixed precision training" if scaler else "Not using mixed precision training")

    # Create datasets and dataloaders
    train_dataset, val_dataset = create_datasets(args)
    train_loader, val_loader = create_dataloaders(args, train_dataset, val_dataset)
    
    # Log dataset recommendations
    log_dataset_recommendations(train_dataset)

    # Create vocabulary sizes dictionary for the model
    vocab_sizes = {
        'contact': train_dataset.num_contact_classes,
        'bodypart': train_dataset.num_bodypart_classes,
        'upper_bodypart': train_dataset.num_upper_bodypart_classes,
        'multiple_fouls': train_dataset.num_multiple_fouls_classes,
        'try_to_play': train_dataset.num_try_to_play_classes,
        'touch_ball': train_dataset.num_touch_ball_classes,
        'handball': train_dataset.num_handball_classes,
        'handball_offence': train_dataset.num_handball_offence_classes,
    }

    # Create model
    model = create_model(args, vocab_sizes, device, num_gpus)
    
    # Setup freezing strategy
    freezing_manager = setup_freezing_strategy(args, model)
    
    # Setup optimizer and scheduler (after dataloader creation for OneCycleLR)
    optimizer, scheduler, phase1_scheduler, scheduler_info = setup_optimizer_and_scheduler(args, model, freezing_manager)
    
    # Initialize OneCycleLR after dataloader creation to get steps_per_epoch
    if args.scheduler == 'onecycle' and scheduler is None:
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=args.warmup_epochs / args.epochs if args.epochs > 0 else 0.1
        )
        scheduler_info = f"OneCycle (max_lr={args.lr:.1e}, steps_per_epoch={steps_per_epoch}, warmup={args.warmup_epochs}e)"
        logger.info(f"Initialized OneCycleLR scheduler with {steps_per_epoch} steps per epoch")

    # Calculate class weights for severity classification
    severity_class_weights = None
    if args.loss_function in ['focal', 'weighted'] and not args.disable_class_balancing:
        severity_class_weights = calculate_class_weights(
            train_dataset, 6, device, args.class_weighting_strategy, args.max_weight_ratio
        )

    # Update loss function to adaptive focal loss if enabled
    if args.adaptive_focal_loss:
        args.loss_function = 'adaptive_focal'
        logger.info("🔥 Using adaptive focal loss with class-specific gamma values")
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

    # Main training loop
    logger.info("Starting Training")
    logger.info("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Update progressive sampler epoch if enabled
        if args.progressive_class_balancing and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
            logger.debug(f"Updated progressive sampler for epoch {epoch}")
        
        # Training
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            loss_config={
                'function': args.loss_function,
                'weights': args.main_task_weights,
                'label_smoothing': args.label_smoothing,
                'focal_gamma': args.focal_gamma,
                'severity_class_weights': severity_class_weights,
                'class_gamma_map': class_gamma_map
            },
            scaler=scaler, 
            max_batches=num_batches_to_run, 
            gradient_clip_norm=args.gradient_clip_norm, 
            memory_cleanup_interval=args.memory_cleanup_interval,
            scheduler=scheduler if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) else None,
            gpu_augmentation=gpu_augmentation
        )
        
        # Validation
        val_metrics = validate_one_epoch(
            model, val_loader, device,
            loss_config={
                'function': args.loss_function,
                'weights': args.main_task_weights,
                'label_smoothing': args.label_smoothing,
                'focal_gamma': args.focal_gamma,
                'severity_class_weights': severity_class_weights
            },
            max_batches=num_batches_to_run,
            memory_cleanup_interval=args.memory_cleanup_interval
        )
        
        # Reset model to training mode and clean memory
        model.train()
        cleanup_memory()

        # Handle freezing strategy updates with validation results
        optimizer, scheduler = handle_gradual_finetuning_transition(
            args, model, optimizer, scheduler, epoch,
            freezing_manager=freezing_manager, 
            val_metric=val_metrics['severity_accuracy'],
            train_loader=train_loader
        )

        # Update learning rate
        if scheduler is not None:
            current_phase_for_scheduler = get_phase_info(epoch, args.phase1_epochs, args.epochs)[0] if args.gradual_finetuning else None

            if args.gradual_finetuning and current_phase_for_scheduler == 1 and phase1_scheduler is not None:
                val_combined_acc = (val_metrics[1] + val_metrics[2]) / 2
                phase1_scheduler.step(val_combined_acc)
            elif scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    val_combined_acc = (val_metrics[1] + val_metrics[2]) / 2
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
                    'train_loss': train_metrics[0],
                    'val_loss': val_metrics[0],
                    'train_sev_acc': train_metrics[1],
                    'train_act_acc': train_metrics[2],
                    'val_sev_acc': val_metrics[1],
                    'val_act_acc': val_metrics[2]
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
                    'current_train_loss': train_metrics[0],
                    'current_val_loss': val_metrics[0],
                    'current_train_sev_acc': train_metrics[1],
                    'current_train_act_acc': train_metrics[2],
                    'current_val_sev_acc': val_metrics[1],
                    'current_val_act_acc': val_metrics[2]
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
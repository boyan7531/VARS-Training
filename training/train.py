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
from torch.utils.data import DataLoader

# Import our modular components
from .config import parse_args, log_configuration_summary
from dataset import SoccerNetMVFoulDataset, variable_views_collate_fn, ClassBalancedSampler
from torchvision import transforms
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
    
    # Check num_workers for potential bottlenecks and provide feedback
    if args.num_workers == 0:
        logger.warning("⚠️  num_workers is 0. Data loading is synchronous and may cause a CPU bottleneck, leading to low GPU utilization.")
        logger.warning("   Consider setting --num-workers to a value like 4 for better performance.")
    elif args.num_workers > os.cpu_count():
        logger.warning(f"⚠️  num_workers ({args.num_workers}) is greater than the number of CPU cores ({os.cpu_count()}). This may cause resource contention.")
        logger.warning(f"   A value around {os.cpu_count()} is often optimal.")
    else:
        logger.info(f"🚀 Using {args.num_workers} worker processes for asynchronous data loading.")
    
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
    
    # Setup optimizer based on freezing strategy
    if args.freezing_strategy == 'none':
        # Standard training - all parameters trainable
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        
    elif args.freezing_strategy in ['adaptive', 'progressive'] and freezing_manager:
        # Smart freezing strategies
        if args.exponential_lr_decay:
            param_groups = freezing_manager.get_discriminative_lr_groups(args.head_lr, args.backbone_lr)
            optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.999))
            logger.info("🔄 Using exponential LR decay for backbone layers")
        else:
            optimizer = optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], 
                lr=args.head_lr, 
                weight_decay=args.weight_decay,
                betas=(0.9, 0.999)
            )
        
        logger.info(f"[SMART] Initial optimizer setup with head LR={args.head_lr:.1e}")
        
    elif args.gradual_finetuning:
        # Original gradual fine-tuning (fixed strategy)
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], 
            lr=args.head_lr, 
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        logger.info(f"[PHASE1] Phase 1 optimizer initialized with LR={args.head_lr:.1e}")
    else:
        # Fallback to standard training
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )

    # Setup learning rate scheduler
    scheduler = None
    phase1_scheduler = None
    scheduler_info = "None"

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


def handle_gradual_finetuning_transition(args, model, optimizer, scheduler, epoch):
    """Handle transition from Phase 1 to Phase 2 in gradual fine-tuning."""
    
    if not args.gradual_finetuning:
        return optimizer, scheduler
        
    current_phase, _ = get_phase_info(epoch, args.phase1_epochs, args.epochs)
    
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
        elif args.scheduler == 'onecycle':
            scheduler = OneCycleLR(
                optimizer,
                max_lr=[pg['lr'] for pg in param_groups],
                epochs=remaining_epochs,
                steps_per_epoch=1,  # Will be updated with actual steps
                pct_start=(args.warmup_epochs / remaining_epochs) if remaining_epochs > 0 and args.warmup_epochs < remaining_epochs else 0.1
            )
        elif args.scheduler == 'step':
            scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.scheduler == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
        elif args.scheduler == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.gamma,
                                        patience=args.plateau_patience, min_lr=args.min_lr)
        else:
            scheduler = None
            
        log_trainable_parameters(model)
        logger.info("[PHASE2] Phase 2 setup complete!")
        logger.info("[PHASE2] " + "="*60)
        
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
    # Minimal transform, as augmentations are handled inside the dataset class via use_severity_aware_aug=True
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    logger.info("Loading datasets...")
    train_dataset = SoccerNetMVFoulDataset(
        dataset_path=args.mvfouls_path,
        split='train',
        annotation_file_name="annotations.json",
        frames_per_clip=args.frames_per_clip,
        target_fps=args.target_fps,
        transform=transform,
        use_severity_aware_aug=True,  # Use the augmentation strategy from dataset.py
        max_views_to_load=args.max_views,
        target_height=args.img_height,
        target_width=args.img_width,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )

    val_dataset = SoccerNetMVFoulDataset(
        dataset_path=args.mvfouls_path,
        split='valid',
        annotation_file_name="annotations.json",
        frames_per_clip=args.frames_per_clip,
        target_fps=args.target_fps,
        transform=transform,
        use_severity_aware_aug=False, # No augmentation on validation
        max_views_to_load=args.max_views,
        target_height=args.img_height,
        target_width=args.img_width,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    
    # Create Dataloaders
    logger.info("Creating data loaders...")
    train_sampler = None
    if hasattr(args, 'use_class_balanced_sampler') and args.use_class_balanced_sampler:
        logger.info("🎯 Using ClassBalancedSampler to address class imbalance!")
        train_sampler = ClassBalancedSampler(train_dataset, oversample_factor=args.oversample_factor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=variable_views_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=variable_views_collate_fn
    )

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

    # Create model
    model = create_model(args, vocab_sizes, device, num_gpus, backbone_name=args.backbone_name)
    
    # Setup freezing strategy
    freezing_manager = setup_freezing_strategy(args, model)
    
    # Setup optimizer and scheduler
    optimizer, scheduler, phase1_scheduler, scheduler_info = setup_optimizer_and_scheduler(args, model, freezing_manager)
    
    # Update OneCycleLR with actual steps per epoch
    if args.scheduler == 'onecycle' and scheduler is None and not args.gradual_finetuning:
        steps_per_epoch = len(train_loader)
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=args.warmup_epochs / args.epochs if args.epochs > 0 else 0.1
        )

    # Calculate class weights for severity classification
    severity_class_weights = None
    if args.loss_function in ['focal', 'weighted'] and not args.disable_class_balancing:
        severity_class_weights = calculate_class_weights(
            train_dataset, 6, device, args.class_weighting_strategy, args.max_weight_ratio
        )

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
        
        # Handle gradual fine-tuning transitions
        optimizer, scheduler = handle_gradual_finetuning_transition(args, model, optimizer, scheduler, epoch)
        
        # Training
        train_metrics = train_one_epoch(
            model, train_loader, criterion_severity, criterion_action, optimizer, device,
            scaler=scaler, max_batches=num_batches_to_run, 
            loss_weights=args.main_task_weights, gradient_clip_norm=args.gradient_clip_norm, 
            label_smoothing=args.label_smoothing, severity_class_weights=severity_class_weights, 
            loss_function=args.loss_function, focal_gamma=args.focal_gamma,
            memory_cleanup_interval=args.memory_cleanup_interval
        )
        
        # Validation
        val_metrics = validate_one_epoch(
            model, val_loader, criterion_severity, criterion_action, device,
            max_batches=num_batches_to_run, loss_weights=args.main_task_weights, 
            label_smoothing=args.label_smoothing, severity_class_weights=severity_class_weights, 
            loss_function=args.loss_function, focal_gamma=args.focal_gamma,
            memory_cleanup_interval=args.memory_cleanup_interval
        )
        
        # Reset model to training mode and clean memory
        model.train()
        cleanup_memory()

        # Smart freezing logic
        optimizer_updated = False
        if freezing_manager is not None:
            val_combined_acc = (val_metrics[1] + val_metrics[2]) / 2  # sev_acc + act_acc
            
            # Monitor gradients
            freezing_manager.monitor_gradients(epoch)
            
            # Check for adaptive unfreezing
            if freezing_manager.adaptive_unfreeze_step(val_combined_acc, epoch):
                # Update optimizer with newly unfrozen parameters
                if args.exponential_lr_decay:
                    param_groups = freezing_manager.get_discriminative_lr_groups(args.head_lr, args.backbone_lr)
                    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay, betas=(0.9, 0.999))
                    logger.info("🔄 Updated optimizer with new unfrozen layers (exponential LR)")
                else:
                    # Add new parameters to existing optimizer
                    new_params = [p for p in model.parameters() if p.requires_grad and id(p) not in [id(p2) for group in optimizer.param_groups for p2 in group['params']]]
                    if new_params:
                        optimizer.add_param_group({'params': new_params, 'lr': args.backbone_lr})
                        logger.info(f"➕ Added {len(new_params)} newly unfrozen parameters to optimizer")
                
                optimizer_updated = True
                log_trainable_parameters(model)

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
                elif not isinstance(scheduler, OneCycleLR):
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
            current_lr, prev_lr, best_val_acc, phase_info, optimizer_updated
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
"""
Checkpoint and logging utilities for Multi-Task Multi-View ResNet3D training.

This module handles model checkpointing, saving/loading, and logging utilities.
"""

import torch
import os
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


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
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            logger.warning(f"Could not load optimizer state: {e}")
            logger.warning("Continuing with fresh optimizer state (this is normal when resuming across training phases)")
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    logger.info(f"Checkpoint loaded from {filepath}")
    return checkpoint['epoch'], checkpoint.get('metrics', {})


def save_training_history(history, filepath):
    """Save training history to JSON file."""
    # Convert numpy types to native Python types for JSON serialization
    history_serializable = {k: [float(x) for x in v] for k, v in history.items()}
    
    with open(filepath, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    
    logger.info(f"Training history saved to {filepath}")


def restore_best_metrics(checkpoint_metrics, resume_best_acc=None):
    """Restore best validation metrics from checkpoint."""
    best_val_acc = 0.0
    best_epoch = -1
    
    # Restore best validation accuracy from checkpoint with better fallback logic
    if 'best_val_acc' in checkpoint_metrics:
        best_val_acc = checkpoint_metrics['best_val_acc']
        best_epoch = checkpoint_metrics.get('best_epoch', checkpoint_metrics.get('epoch', 0))
        logger.info(f"✅ Restored best validation accuracy: {best_val_acc:.4f} from epoch {best_epoch}")
    elif 'val_sev_acc' in checkpoint_metrics and 'val_act_acc' in checkpoint_metrics:
        # Fallback: Calculate combined accuracy from individual metrics if available
        restored_sev_acc = checkpoint_metrics['val_sev_acc']
        restored_act_acc = checkpoint_metrics['val_act_acc']
        best_val_acc = (restored_sev_acc + restored_act_acc) / 2
        best_epoch = checkpoint_metrics.get('epoch', 0)
        logger.info(f"📊 Calculated best validation accuracy from checkpoint metrics: {best_val_acc:.4f} (sev: {restored_sev_acc:.3f}, act: {restored_act_acc:.3f})")
        logger.info(f"   - Using epoch {best_epoch} as best epoch")
    else:
        logger.warning("⚠️  No validation accuracy found in checkpoint - will start fresh tracking")
        logger.warning("   - This might happen with older checkpoint formats")
        best_val_acc = 0.0
        best_epoch = -1
    
    # Manual override if specified
    if resume_best_acc is not None:
        original_best = best_val_acc
        best_val_acc = resume_best_acc
        logger.info(f"🔧 MANUAL OVERRIDE: Best accuracy set to {best_val_acc:.4f} (was {original_best:.4f})")
    
    return best_val_acc, best_epoch


def log_epoch_summary(epoch, total_epochs, epoch_time, train_metrics, val_metrics, 
                     current_lr, prev_lr, best_val_acc, phase_info=None, optimizer_updated=False):
    """Log a comprehensive epoch summary."""
    
    train_loss, train_sev_acc, train_act_acc, train_sev_f1, train_act_f1 = train_metrics
    val_loss, val_sev_acc, val_act_acc, val_sev_f1, val_act_f1 = val_metrics
    
    # Calculate combined metrics
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
    if phase_info:
        phase_indicator = f" [P{phase_info}]"

    # Optimizer update indicator
    optim_indicator = " [OPTIM UPDATED]" if optimizer_updated else ""

    # Compact epoch summary
    logger.info(f"Epoch {epoch+1:2d}/{total_epochs} [{epoch_time:.1f}s]{phase_indicator} "
               f"| Train: Loss={train_loss:.3f}, Acc={train_combined_acc:.3f} "
               f"| Val: Loss={val_loss:.3f}, Acc={val_combined_acc:.3f} "
               f"| LR={current_lr:.1e}{lr_change_indicator}{best_indicator}{optim_indicator}")
    
    return val_combined_acc


def log_configuration_summary(args, train_dataset, val_dataset):
    """Log a comprehensive summary of the training configuration."""
    logger.info("=" * 80)
    logger.info("[CONFIGURATION] Training Configuration Summary")
    logger.info("=" * 80)
    
    logger.info(f"Configuration: Epochs={args.epochs}, Batch Size={args.batch_size}, LR={args.lr}, Backbone={args.backbone_name}")
    
    if args.gradual_finetuning:
        logger.info(f"Gradual Fine-tuning: Phase1={args.phase1_epochs}e@{args.head_lr:.1e}, Phase2={args.phase2_epochs}e@{args.backbone_lr:.1e}")
    
    logger.info(f"Freezing Strategy: {args.freezing_strategy}")
    
    # Show advanced class balancing settings
    logger.info(f"Class Balancing: {'Enabled' if args.use_class_balanced_sampler else 'Disabled'}")
    if args.use_class_balanced_sampler:
        if args.progressive_class_balancing:
            logger.info(f"  - Progressive Sampling: {args.oversample_factor_start}x → {args.oversample_factor}x over {args.progressive_duration_epochs} epochs")
        else:
            logger.info(f"  - Fixed Oversampling: {args.oversample_factor}x")
    
    # Show loss function settings
    if args.adaptive_focal_loss:
        logger.info(f"Loss Function: Adaptive Focal Loss (class-specific gamma values)")
    else:
        logger.info(f"Loss Function: {args.loss_function}")
    
    # Show augmentation settings
    aug_level = "None"
    if args.extreme_augmentation:
        aug_level = "Extreme"
    elif args.aggressive_augmentation:
        aug_level = "Aggressive"
    elif not args.disable_augmentation:
        aug_level = "Standard"
    
    logger.info(f"Augmentation: {aug_level}")
    if args.severity_aware_augmentation:
        logger.info(f"  - Using Severity-Aware Augmentation (class-specific strengths)")
    
    logger.info(f"Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}")
    logger.info("=" * 80)


def create_training_history():
    """Create an empty training history dictionary."""
    return defaultdict(list)


def update_training_history(history, train_metrics, val_metrics, current_lr):
    """Update training history with current epoch metrics."""
    train_loss, train_sev_acc, train_act_acc, train_sev_f1, train_act_f1 = train_metrics
    val_loss, val_sev_acc, val_act_acc, val_sev_f1, val_act_f1 = val_metrics
    
    # Store history (including learning rate)
    history['train_loss'].append(train_loss)
    history['train_sev_acc'].append(train_sev_acc)
    history['train_act_acc'].append(train_act_acc)
    history['val_loss'].append(val_loss)
    history['val_sev_acc'].append(val_sev_acc)
    history['val_act_acc'].append(val_act_acc)
    history['learning_rate'] = history.get('learning_rate', []) + [current_lr]
    
    return history


def log_completion_summary(best_val_acc, best_epoch, test_run=False):
    """Log training completion summary."""
    logger.info("=" * 80)
    logger.info("[COMPLETE] Training Finished!")
    if not test_run:
        logger.info(f"[BEST] Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    else:
        logger.info("[TEST] Test run completed successfully")
    logger.info("=" * 80) 
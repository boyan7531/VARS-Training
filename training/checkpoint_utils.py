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


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, filepath, ema_model=None):
    """Save training checkpoint with optional EMA weights."""
    # Handle DataParallel models
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save EMA weights if available
    if ema_model is not None:
        if hasattr(ema_model, 'shadow'):  # SimpleEMA
            checkpoint['ema_state_dict'] = ema_model.shadow
        elif hasattr(ema_model, 'state_dict'):  # timm ModelEmaV2
            checkpoint['ema_state_dict'] = ema_model.state_dict()
        checkpoint['has_ema'] = True
        logger.info("💾 Saving checkpoint with EMA weights")
    else:
        checkpoint['has_ema'] = False
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add scheduler metadata for debugging
        scheduler_info = {
            'scheduler_type': type(scheduler).__name__,
            'has_warmup': hasattr(scheduler, 'warmup_steps')
        }
        
        if hasattr(scheduler, 'warmup_steps'):
            scheduler_info.update({
                'warmup_steps': scheduler.warmup_steps,
                'start_lr': scheduler.start_lr,
                'after_scheduler_type': type(scheduler.after_scheduler).__name__
            })
        
        checkpoint['scheduler_info'] = scheduler_info
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")
    
    # Log scheduler info for debugging
    if scheduler is not None and hasattr(scheduler, 'warmup_steps'):
        logger.debug(f"Saved warmup scheduler: {scheduler.warmup_steps} warmup steps")


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, scaler=None, ema_model=None):
    """Load training checkpoint with DataParallel compatibility and EMA support."""
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
    
    # Load EMA weights if available and EMA model is provided
    if ema_model is not None and checkpoint.get('has_ema', False) and 'ema_state_dict' in checkpoint:
        ema_state_dict = checkpoint['ema_state_dict']
        if hasattr(ema_model, 'shadow'):  # SimpleEMA
            ema_model.shadow = ema_state_dict
            logger.info("📈 Restored EMA weights from checkpoint")
        elif hasattr(ema_model, 'load_state_dict'):  # timm ModelEmaV2
            ema_model.load_state_dict(ema_state_dict)
            logger.info("📈 Restored EMA weights from checkpoint")
    elif checkpoint.get('has_ema', False):
        logger.warning("⚠️ Checkpoint contains EMA weights but no EMA model provided")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            logger.warning(f"Could not load optimizer state: {e}")
            logger.warning("Continuing with fresh optimizer state (this is normal when resuming across training phases)")
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Log scheduler info if available
        if 'scheduler_info' in checkpoint:
            sched_info = checkpoint['scheduler_info']
            logger.debug(f"Loaded scheduler: {sched_info['scheduler_type']}")
            if sched_info.get('has_warmup', False):
                logger.debug(f"  Warmup steps: {sched_info.get('warmup_steps', 'unknown')}")
                logger.debug(f"  After scheduler: {sched_info.get('after_scheduler_type', 'unknown')}")
    
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
    from training.training_utils import check_overfitting_alert
    
    # Extract metrics from dictionaries
    train_loss = train_metrics['loss'] 
    train_sev_acc = train_metrics['sev_acc']
    train_act_acc = train_metrics['act_acc']
    train_sev_f1 = train_metrics['sev_f1']
    train_act_f1 = train_metrics['act_f1']
    
    val_loss = val_metrics['loss']
    val_sev_acc = val_metrics['sev_acc']
    val_act_acc = val_metrics['act_acc']
    val_sev_f1 = val_metrics['sev_f1'] 
    val_act_f1 = val_metrics['act_f1']
    
    # Calculate combined metrics
    train_combined_acc = (train_sev_acc + train_act_acc) / 2
    val_combined_acc = (val_sev_acc + val_act_acc) / 2
    
    # Check for overfitting alert
    overfitting_detected = check_overfitting_alert(train_loss, val_loss, epoch)
    
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
    
    # Overfitting indicator
    overfit_indicator = " [OVERFIT!]" if overfitting_detected else ""

    # Detailed epoch summary with explicit metrics
    logger.info("=" * 80)
    logger.info(f"EPOCH {epoch+1:2d}/{total_epochs} SUMMARY [{epoch_time:.1f}s]{phase_indicator}{best_indicator}{optim_indicator}{overfit_indicator}")
    logger.info("=" * 80)
    logger.info(f"📈 TRAINING METRICS:")
    logger.info(f"   Loss: {train_loss:.4f}")
    logger.info(f"   Severity Accuracy: {train_sev_acc:.4f} ({train_sev_acc*100:.2f}%)")
    logger.info(f"   Action/Offence Accuracy: {train_act_acc:.4f} ({train_act_acc*100:.2f}%)")
    logger.info(f"   Combined Accuracy: {train_combined_acc:.4f} ({train_combined_acc*100:.2f}%)")
    logger.info(f"   Severity F1: {train_sev_f1:.4f}")
    logger.info(f"   Action/Offence F1: {train_act_f1:.4f}")
    logger.info("")
    logger.info(f"📊 VALIDATION METRICS:")
    logger.info(f"   Loss: {val_loss:.4f}")
    logger.info(f"   Severity Accuracy: {val_sev_acc:.4f} ({val_sev_acc*100:.2f}%)")
    logger.info(f"   Action/Offence Accuracy: {val_act_acc:.4f} ({val_act_acc*100:.2f}%)")
    logger.info(f"   Combined Accuracy: {val_combined_acc:.4f} ({val_combined_acc*100:.2f}%)")
    logger.info(f"   Severity F1: {val_sev_f1:.4f}")
    logger.info(f"   Action/Offence F1: {val_act_f1:.4f}")
    logger.info("")
    logger.info(f"🎯 TRAINING STATUS:")
    logger.info(f"   Learning Rate: {current_lr:.2e}{lr_change_indicator}")
    logger.info(f"   Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    if is_new_best:
        improvement = val_combined_acc - best_val_acc
        logger.info(f"   🎉 NEW BEST MODEL! Improvement: +{improvement:.4f} (+{improvement*100:.2f}%)")
    logger.info("=" * 80)
    
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
            logger.info(f"  - Progressive Sampling: {args.progressive_start_factor}x → {args.progressive_end_factor}x over {args.progressive_epochs} epochs")
        else:
            logger.info(f"  - Fixed Oversampling: {args.oversample_factor}x")
    
    # Show loss function settings
    if args.loss_function == 'focal':
        logger.info(f"Loss Function: Focal Loss (class-specific gamma values)")
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
    # Extract metrics from dictionaries
    train_loss = train_metrics['loss']
    train_sev_acc = train_metrics['sev_acc']
    train_act_acc = train_metrics['act_acc']
    train_sev_f1 = train_metrics['sev_f1']
    train_act_f1 = train_metrics['act_f1']
    
    val_loss = val_metrics['loss']
    val_sev_acc = val_metrics['sev_acc']
    val_act_acc = val_metrics['act_acc']
    val_sev_f1 = val_metrics['sev_f1']
    val_act_f1 = val_metrics['act_f1']
    
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
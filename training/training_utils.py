"""
Training utilities for Multi-Task Multi-View ResNet3D training.

This module handles training loops, validation, loss functions, and metrics calculation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler
import time
import logging
import contextlib
import numpy as np
import os
from sklearn.metrics import f1_score, confusion_matrix
from torch.profiler import profile, record_function, ProfilerActivity
import gc

logger = logging.getLogger(__name__)


def is_main_process():
    """
    Check if current process is the main process (rank 0) in distributed training.
    Returns True for single-GPU training or rank 0 in multi-GPU training.
    """
    try:
        # Try PyTorch's distributed backend first (most reliable)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        
        # Try PyTorch Lightning's distributed utilities (version-agnostic)
        try:
            import pytorch_lightning as pl
            # Try different import paths for different PL versions
            try:
                from lightning.fabric.utilities.distributed import _get_rank
                return _get_rank() == 0
            except ImportError:
                try:
                    from pytorch_lightning.utilities.rank_zero import rank_zero_only
                    return getattr(rank_zero_only, 'rank', 0) == 0
                except ImportError:
                    pass
        except ImportError:
            pass
        
        # Single process case
        return True
        
    except Exception:
        # If anything fails, assume single process
        return True


def cleanup_memory():
    """Clean up GPU memory to prevent accumulation."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def calculate_accuracy(outputs, labels):
    """Calculates accuracy for a single task."""
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total if total > 0 else 0


def calculate_f1_score(outputs, labels, num_classes):
    """Calculate F1 score for multi-class classification."""
    _, predicted = torch.max(outputs.data, 1)
    return f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted', zero_division=0)


def update_confusion_matrix(confusion_matrix_dict, outputs, labels, task_name):
    """Update running confusion matrix for a task."""
    _, predicted = torch.max(outputs.data, 1)
    labels_np = labels.cpu().numpy()
    predicted_np = predicted.cpu().numpy()
    
    if task_name not in confusion_matrix_dict:
        confusion_matrix_dict[task_name] = {'true_labels': [], 'predicted_labels': []}
    
    confusion_matrix_dict[task_name]['true_labels'].extend(labels_np.tolist())
    confusion_matrix_dict[task_name]['predicted_labels'].extend(predicted_np.tolist())


def compute_confusion_matrices(confusion_matrix_dict):
    """Compute confusion matrices from accumulated predictions."""
    results = {}
    for task_name, data in confusion_matrix_dict.items():
        if len(data['true_labels']) > 0:
            num_classes = max(max(data['true_labels']), max(data['predicted_labels'])) + 1
            cm = confusion_matrix(
                data['true_labels'], 
                data['predicted_labels'], 
                labels=list(range(num_classes))
            )
            results[task_name] = cm
    return results


def log_confusion_matrix(cm, task_name, class_names=None):
    """Log confusion matrix with per-class recall (only from main process)."""
    # Only log from main process to avoid duplicates in distributed training
    if not is_main_process():
        return
        
    logger.info(f"\n[CONFUSION_MATRIX] {task_name.upper()}")
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(cm.shape[0])]
    
    # Log the confusion matrix
    logger.info(f"Confusion Matrix ({cm.shape[0]} classes):")
    for i, row in enumerate(cm):
        row_str = " ".join([f"{val:4d}" for val in row])
        logger.info(f"  {class_names[i]:12s}: {row_str}")
    
    # Calculate and log per-class recall
    logger.info("Per-class Recall:")
    for i in range(cm.shape[0]):
        total_true = cm[i].sum()
        if total_true > 0:
            recall = cm[i, i] / total_true
            logger.info(f"  {class_names[i]:12s}: {recall:.3f} ({cm[i, i]:3d}/{total_true:3d})")
        else:
            logger.info(f"  {class_names[i]:12s}: N/A   (0 samples)")


def save_confusion_matrix(cm, task_name, epoch, save_dir):
    """Save confusion matrix to file (only from main process)."""
    # Only save from main process to avoid file conflicts
    if not is_main_process():
        return
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filename = f"confusion_matrix_{task_name}_epoch_{epoch}.npy"
    filepath = os.path.join(save_dir, filename)
    np.save(filepath, cm)
    logger.info(f"[SAVE] Confusion matrix saved: {filepath}")


def check_overfitting_alert(train_loss, val_loss, epoch):
    """Check for potential overfitting and alert if detected (only from main process)."""
    # Only log from main process to avoid duplicate warnings
    if not is_main_process():
        return False
        
    if val_loss > 0 and train_loss < 0.2 * val_loss:
        logger.warning(f"🚨 [OVERFITTING ALERT] Epoch {epoch+1}: Train loss ({train_loss:.4f}) < 0.2 × Val loss ({val_loss:.4f})")
        logger.warning(f"    This suggests overfitting to the oversampled training distribution!")
        logger.warning(f"    Consider: reducing augmentation, increasing regularization, or reducing oversampling factor")
        return True
    return False


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


def view_consistency_loss(logits, mask, temperature=1.0):
    """
    Compute view consistency loss using symmetric KL divergence.
    
    Args:
        logits: [B, V, C] per-view logits
        mask: [B, V] view mask (True for valid views)
        temperature: Temperature for softmax
        
    Returns:
        Consistency loss value
    """
    batch_size, max_views, num_classes = logits.shape
    device = logits.device
    
    if mask.sum() < 2:  # Need at least 2 valid views for consistency
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    num_pairs = 0
    
    # For each batch
    for b in range(batch_size):
        valid_views = mask[b].nonzero(as_tuple=False).squeeze(-1)
        if len(valid_views) < 2:
            continue
            
        # Compare all pairs of valid views
        for i in range(len(valid_views)):
            for j in range(i + 1, len(valid_views)):
                view_i, view_j = valid_views[i], valid_views[j]
                
                # Get logits for this pair
                logits_i = logits[b, view_i] / temperature  # [C]
                logits_j = logits[b, view_j] / temperature  # [C]
                
                # Compute log-probabilities and probabilities
                log_p_i = torch.log_softmax(logits_i, dim=-1)
                log_p_j = torch.log_softmax(logits_j, dim=-1)
                p_i = torch.softmax(logits_i, dim=-1)
                p_j = torch.softmax(logits_j, dim=-1)
                
                # Symmetric KL divergence: KL(P_i||P_j) + KL(P_j||P_i)
                # Use manual calculation to avoid reduction parameter compatibility issues
                kl_ij = (p_j * (torch.log(p_j + 1e-8) - log_p_i)).sum()
                kl_ji = (p_i * (torch.log(p_i + 1e-8) - log_p_j)).sum()
                
                total_loss = total_loss + kl_ij + kl_ji
                num_pairs += 1
    
    # Average over all valid pairs
    if num_pairs > 0:
        return total_loss / num_pairs
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


class AdaptiveFocalLoss(nn.Module):
    """
    Enhanced Focal Loss with class-specific gamma values.
    
    Adapts the focusing parameter (gamma) per class to give more
    emphasis to rare classes and hard examples.
    
    Args:
        class_gamma_map: Dictionary mapping class indices to gamma values
        class_alpha_map: Optional dictionary mapping class indices to alpha weights
        reduction: Reduction mode ('mean', 'sum', or 'none')
        label_smoothing: Label smoothing parameter
    """
    def __init__(self, class_gamma_map=None, class_alpha_map=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        
        # Default gamma values (higher = more focus on hard examples)
        if class_gamma_map is None:
            self.class_gamma_map = {
                0: 2.0,  # Medium frequency
                1: 1.5,  # Majority class - less focus needed
                2: 2.0,  # Medium frequency
                3: 2.0,  # Medium frequency
                4: 3.0,  # Very rare - high focus
                5: 3.5   # Extremely rare - highest focus
            }
        else:
            self.class_gamma_map = class_gamma_map
            
        # Optional per-class alpha weighting (complementary to sampling)
        self.class_alpha_map = class_alpha_map
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        # Only log once when first created (not every batch)
        if not hasattr(AdaptiveFocalLoss, '_logged_gamma_values'):
            logger.info("Using AdaptiveFocalLoss with class-specific gamma values:")
            for cls, gamma in self.class_gamma_map.items():
                logger.info(f"  Class {cls}: gamma={gamma:.1f}")
            AdaptiveFocalLoss._logged_gamma_values = True
    
    def forward(self, inputs, targets):
        # Standard cross entropy component
        ce_loss = F.cross_entropy(
            inputs, targets, 
            reduction='none', 
            weight=self._get_alpha_weights(targets) if self.class_alpha_map else None,
            label_smoothing=self.label_smoothing
        )
        
        # Calculate pt (model confidence for correct class)
        pt = torch.exp(-ce_loss)
        
        # Apply class-specific gamma focusing
        batch_gammas = torch.tensor(
            [self.class_gamma_map.get(t.item(), 2.0) for t in targets],
            device=targets.device,
            dtype=inputs.dtype  # Ensure dtype consistency for mixed precision
        )
        
        # Apply focal term with class-specific gamma: (1-pt)^gamma_c
        focal_weights = torch.pow(1 - pt, batch_gammas)
        focal_loss = focal_weights * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def _get_alpha_weights(self, targets):
        # Convert class-specific alpha values to tensor weights
        # This creates a weight tensor matching the target shape
        return torch.tensor(
            [self.class_alpha_map.get(t.item(), 1.0) for t in targets],
            device=targets.device,
            dtype=torch.float32  # Keep alpha weights in float32 for stability
        )


def calculate_multitask_loss(sev_logits, act_logits, batch_data, loss_config: dict, view_logits=None):
    """
    Calculate weighted multi-task loss based on a configuration dictionary.
    
    Args:
        sev_logits: Severity classification logits
        act_logits: Action type classification logits  
        batch_data: Batch data containing labels
        loss_config (dict): Configuration for the loss function.
            - 'function': 'focal', 'weighted', 'adaptive_focal', or 'plain'
            - 'weights': [severity_weight, action_weight] for main tasks
            - 'label_smoothing': Label smoothing factor
            - 'focal_gamma': Focal Loss gamma parameter
            - 'severity_class_weights': Optional class weights for severity loss.
                                        If 'focal' is used, these become the 'alpha' parameter.
                                        If using a balanced sampler, this should typically be None.
            - 'view_consistency': Whether to compute view consistency loss
            - 'vc_weight': Weight for view consistency loss
        view_logits (dict, optional): Dict containing per-view logits if view consistency is enabled
    
    Returns:
        total_loss, loss_sev, loss_act, loss_components (dict with individual losses)
    """
    loss_function = loss_config.get('function', 'weighted')
    main_weights = loss_config.get('weights', [1.0, 1.0])
    label_smoothing = loss_config.get('label_smoothing', 0.0)
    
    # OPTIMIZATION: Disable label smoothing when using oversampling + class weights
    # This prevents gradient flattening for rare classes that are already amplified
    severity_class_weights = loss_config.get('severity_class_weights', None)
    action_class_weights = loss_config.get('action_class_weights', None)
    using_oversampling = loss_config.get('using_oversampling', False)
    
    # Auto-disable label smoothing if we're using both oversampling and class weights
    if label_smoothing > 0 and using_oversampling and (severity_class_weights is not None or action_class_weights is not None):
        label_smoothing = 0.0
        # Only log once per training run to avoid spam
        if not hasattr(calculate_multitask_loss, '_logged_smoothing_disabled'):
            if is_main_process():
                logger.info("🎯 Auto-disabled label smoothing: oversampling + class weights detected")
                logger.info("   This prevents gradient flattening for amplified rare classes")
            calculate_multitask_loss._logged_smoothing_disabled = True
    
    if loss_function == 'adaptive_focal':
        # Use adaptive focal loss with class-specific gamma values
        class_gamma_map = loss_config.get('class_gamma_map', None)
        
        adaptive_focal_criterion = AdaptiveFocalLoss(
            class_gamma_map=class_gamma_map,
            class_alpha_map=None,  # We already use severity_class_weights
            label_smoothing=label_smoothing
        )
        loss_sev = adaptive_focal_criterion(sev_logits, batch_data["label_severity"]) * main_weights[0]
        
        # Action type loss - use class weights if available
        if action_class_weights is not None:
            loss_act = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                weight=action_class_weights
            )(act_logits, batch_data["label_type"]) * main_weights[1]
        else:
            loss_act = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(act_logits, batch_data["label_type"]) * main_weights[1]
    
    elif loss_function == 'focal':
        focal_gamma = loss_config.get('focal_gamma', 2.0)
        
        # Guard: When using focal loss with ClassBalancedSampler, alpha should be None
        # to avoid double-balancing (oversampling + class weights = excessive minority class focus)
        # Note: severity_class_weights is already set to None when use_class_balanced_sampler=True
        if loss_function == 'focal' and severity_class_weights is None:
            alpha = None  # No class weights - using oversampling only
        else:
            alpha = severity_class_weights  # Use class weights - no oversampling
        
        focal_criterion = FocalLoss(
            alpha=alpha,
            gamma=focal_gamma,
            label_smoothing=label_smoothing
        )
        loss_sev = focal_criterion(sev_logits, batch_data["label_severity"]) * main_weights[0]
        
        # Action type loss - use class weights if available
        if action_class_weights is not None:
            loss_act = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                weight=action_class_weights
            )(act_logits, batch_data["label_type"]) * main_weights[1]
        else:
            loss_act = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(act_logits, batch_data["label_type"]) * main_weights[1]
    
    elif loss_function == 'weighted':
        
        loss_sev = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, 
            weight=severity_class_weights
        )(sev_logits, batch_data["label_severity"]) * main_weights[0]
        
        # Use action class weights if available, otherwise standard CE loss
        if action_class_weights is not None:
            loss_act = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing,
                weight=action_class_weights
            )(act_logits, batch_data["label_type"]) * main_weights[1]
        else:
            loss_act = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(act_logits, batch_data["label_type"]) * main_weights[1]
    
    elif loss_function == 'plain':
        loss_sev = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(sev_logits, batch_data["label_severity"]) * main_weights[0]
        loss_act = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(act_logits, batch_data["label_type"]) * main_weights[1]
    
    else:
        raise ValueError(f"Unknown loss_function: {loss_function}. Must be 'adaptive_focal', 'focal', 'weighted', or 'plain'")
    
    # Initialize loss components dictionary
    loss_components = {
        'loss_sev': loss_sev,
        'loss_act': loss_act
    }
    
    total_loss = loss_sev + loss_act
    
    # Add view consistency loss if enabled
    if loss_config.get('view_consistency', False) and view_logits is not None:
        vc_weight = loss_config.get('vc_weight', 0.3)
        
        # Compute view consistency loss for both tasks
        vc_sev = view_consistency_loss(view_logits['sev_logits_v'], view_logits['view_mask'])
        vc_act = view_consistency_loss(view_logits['act_logits_v'], view_logits['view_mask'])
        
        # Add to total loss
        vc_loss_total = vc_weight * (vc_sev + vc_act)
        total_loss = total_loss + vc_loss_total
        
        # Store individual components for logging
        loss_components.update({
            'vc_sev': vc_sev,
            'vc_act': vc_act,
            'vc_loss_total': vc_loss_total
        })
        
        # Log view consistency loss (only from main process)
        if is_main_process():
            logger.debug(f"View consistency loss - Severity: {vc_sev:.4f}, Action: {vc_act:.4f}, Total: {vc_loss_total:.4f}")
    
    return total_loss, loss_sev, loss_act, loss_components


class EarlyStopping:
    """Early stopping utility."""
    def __init__(self, patience=20, min_delta=0.01, restore_best_weights=True):
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


def train_one_epoch(model, dataloader, optimizer, device, loss_config: dict, scaler=None, 
                   max_batches=None, gradient_clip_norm=1.0, memory_cleanup_interval=20, scheduler=None,
                   gpu_augmentation=None, enable_profiling=False, confusion_matrix_dict=None, ema_model=None):
    """Train the model for one epoch with optional GPU-based augmentation and profiling."""
    model.train()
    running_loss = 0.0
    running_sev_acc = 0.0
    running_act_acc = 0.0
    running_sev_f1 = 0.0
    running_act_f1 = 0.0
    processed_batches = 0
    total_samples = 0  # Track actual number of samples processed

    # Detect if this is an MViT model for optimized memory management
    is_mvit_model = hasattr(model, 'mvit_processor') or (
        hasattr(model, 'module') and hasattr(model.module, 'mvit_processor')
    )
    
    # Optimize memory cleanup interval for MViT (more aggressive cleanup)
    if is_mvit_model:
        memory_cleanup_interval = min(memory_cleanup_interval, 5)  # More frequent cleanup for MViT
        logger.info(f"🚀 MViT detected - using optimized memory cleanup interval: {memory_cleanup_interval}")

    start_time = time.time()
    # Handle OptimizedDataLoader which may not have len()
    if max_batches:
        total_batches = max_batches
    else:
        try:
            total_batches = len(dataloader)
        except TypeError:
            # OptimizedDataLoader doesn't support len(), use underlying dataloader
            actual_loader = getattr(dataloader, 'dataloader', dataloader)
            total_batches = len(actual_loader) if hasattr(actual_loader, '__len__') else 1000  # fallback
    
    # Initialize profiler if enabled
    prof = None
    if enable_profiling:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
            record_shapes=True, 
            profile_memory=True
        )
        prof.start()
        logger.info("🔍 PyTorch Profiler started - will profile first 10 batches")
    
    for i, batch_data in enumerate(dataloader):
        if max_batches is not None and (i + 1) > max_batches:
            break
        
        with record_function("data_loading") if prof else contextlib.suppress():
            # Move all tensors in the batch to the device
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(device, non_blocking=True)

        severity_labels = batch_data["label_severity"]
        action_labels = batch_data["label_type"]
        
        current_batch_size = batch_data["clips"].size(0)
        total_samples += current_batch_size

        optimizer.zero_grad()
        
        # Use autocast for mixed precision if scaler is provided
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                with record_function("gpu_augmentation") if prof else contextlib.suppress():
                    # Apply GPU augmentation if provided
                    if gpu_augmentation is not None:
                        # Check if this is a severity-aware augmentation
                        if hasattr(gpu_augmentation, 'severity_multipliers'):
                            # This is a SeverityAwareGPUAugmentation
                            clips = gpu_augmentation(batch_data["clips"], batch_data["label_severity"])
                            batch_data["clips"] = clips
                        else:
                            # Standard augmentation
                            batch_data["clips"] = gpu_augmentation(batch_data["clips"])
                        
                with record_function("model_forward") if prof else contextlib.suppress():
                    # Check if view consistency is enabled
                    view_consistency_enabled = loss_config.get('view_consistency', False)
                    
                    if view_consistency_enabled:
                        model_output = model(batch_data, return_view_logits=True)
                        sev_logits = model_output['severity_logits']
                        act_logits = model_output['action_logits']
                        view_logits = {
                            'sev_logits_v': model_output['sev_logits_v'],
                            'act_logits_v': model_output['act_logits_v'],
                            'view_mask': model_output['view_mask']
                        }
                        total_loss, _, _, _ = calculate_multitask_loss(
                            sev_logits, act_logits, batch_data, loss_config, view_logits
                        )
                    else:
                        sev_logits, act_logits = model(batch_data)
                        total_loss, _, _, _ = calculate_multitask_loss(
                            sev_logits, act_logits, batch_data, loss_config
                        )

            with record_function("loss_backward") if prof else contextlib.suppress():
                scaler.scale(total_loss).backward()
                
                # Gradient clipping
                if gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                
                scaler.step(optimizer)
                scaler.update()
                
                # Update EMA model if provided
                if ema_model is not None:
                    from .model_utils import update_ema_model
                    update_ema_model(ema_model, model)
            
            # Step scheduler per batch for OneCycleLR and WarmupWrapper
            if scheduler is not None:
                from .lr_schedulers import WarmupWrapper
                if (isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) or 
                    isinstance(scheduler, WarmupWrapper)):
                    scheduler.step()
        else:
            # Apply GPU augmentation if provided
            if gpu_augmentation is not None:
                # Check if this is a severity-aware augmentation
                if hasattr(gpu_augmentation, 'severity_multipliers'):
                    # This is a SeverityAwareGPUAugmentation
                    clips = gpu_augmentation(batch_data["clips"], batch_data["label_severity"])
                    batch_data["clips"] = clips
                else:
                    # Standard augmentation
                    batch_data["clips"] = gpu_augmentation(batch_data["clips"])
                    
            # Check if view consistency is enabled
            view_consistency_enabled = loss_config.get('view_consistency', False)
            
            if view_consistency_enabled:
                model_output = model(batch_data, return_view_logits=True)
                sev_logits = model_output['severity_logits']
                act_logits = model_output['action_logits']
                view_logits = {
                    'sev_logits_v': model_output['sev_logits_v'],
                    'act_logits_v': model_output['act_logits_v'],
                    'view_mask': model_output['view_mask']
                }
                total_loss, _, _, _ = calculate_multitask_loss(
                    sev_logits, act_logits, batch_data, loss_config, view_logits
                )
            else:
                sev_logits, act_logits = model(batch_data)
                total_loss, _, _, _ = calculate_multitask_loss(
                    sev_logits, act_logits, batch_data, loss_config
                )

            total_loss.backward()
            
            # Gradient clipping
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            
            optimizer.step()
            
            # Update EMA model if provided
            if ema_model is not None:
                from .model_utils import update_ema_model
                update_ema_model(ema_model, model)
            
            # Step scheduler per batch for OneCycleLR and WarmupWrapper
            if scheduler is not None:
                from .lr_schedulers import WarmupWrapper
                if (isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) or 
                    isinstance(scheduler, WarmupWrapper)):
                    scheduler.step()

        # Stop profiling after 10 batches and print results
        if prof is not None and i >= 9:  # Profile first 10 batches (0-9)
            prof.stop()
            
            # Print detailed timing breakdown
            logger.info("🔍 PROFILING RESULTS - Data Loading vs Compute Time Analysis:")
            logger.info("=" * 80)
            
            # Get timing table sorted by CUDA time
            timing_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
            logger.info(timing_table)
            
            # Calculate timing ratios
            key_averages = prof.key_averages()
            data_loading_time = 0
            compute_time = 0
            augmentation_time = 0
            
            for event in key_averages:
                if "data_loading" in event.key:
                    data_loading_time = event.device_time
                elif "model_forward" in event.key:
                    compute_time = event.device_time
                elif "gpu_augmentation" in event.key:
                    augmentation_time = event.device_time
            
            total_time = data_loading_time + compute_time + augmentation_time
            if total_time > 0:
                data_pct = (data_loading_time / total_time) * 100
                compute_pct = (compute_time / total_time) * 100
                aug_pct = (augmentation_time / total_time) * 100
                
                logger.info("⚡ PERFORMANCE BREAKDOWN:")
                logger.info(f"   Data Loading:     {data_loading_time/1000:.1f}ms ({data_pct:.1f}%)")
                logger.info(f"   GPU Augmentation: {augmentation_time/1000:.1f}ms ({aug_pct:.1f}%)")
                logger.info(f"   Model Forward:    {compute_time/1000:.1f}ms ({compute_pct:.1f}%)")
                logger.info("=" * 80)
                
                # Enhanced diagnosis for MViT
                if is_mvit_model:
                    logger.info("🎯 MViT-SPECIFIC ANALYSIS:")
                    if data_pct > 25:
                        logger.warning("🚨 CRITICAL: Data loading bottleneck detected in MViT training!")
                        logger.warning("   MViT Recommendations:")
                        logger.warning("   - Increase --num_workers to 16-20")
                        logger.warning("   - Enable --gpu_augmentation")
                        logger.warning("   - Reduce temporal resolution (fewer frames)")
                        logger.warning("   - Use smaller spatial resolution during training")
                    elif compute_pct < 40:
                        logger.warning("⚠️  Low GPU compute utilization for MViT - memory-bound!")
                        logger.warning("   This is expected for MViT attention operations")
                        logger.warning("   Consider: gradient checkpointing, attention optimization")
                    else:
                        logger.info("✅ Good MViT performance balance")
                    logger.info("=" * 80)
                else:
                    # Original diagnosis for non-MViT models
                    if data_pct > 30:
                        logger.warning("🚨 DATA LOADING BOTTLENECK DETECTED!")
                        logger.warning("   Recommendations:")
                        logger.warning("   - Increase --num_workers (try 12-16 for dual RTX 4090)")
                        logger.warning("   - Increase --prefetch_factor (try 8-16)")
                        logger.warning("   - Enable --gpu_augmentation to move processing off CPU")
                        logger.warning("   - Reduce augmentation complexity")
                    elif data_pct > 15:
                        logger.warning("⚠️  Data loading taking significant time (>15%)")
                        logger.warning("   Consider increasing num_workers or prefetch_factor")
                    else:
                        logger.info("✅ Good data loading performance (<15% of total time)")
            
            prof = None  # Disable profiling for remaining batches

        # Debug removed for cleaner training logs

        # Calculate metrics
        running_loss += total_loss.item() * current_batch_size
        sev_acc = calculate_accuracy(sev_logits, severity_labels)
        act_acc = calculate_accuracy(act_logits, action_labels)
        sev_f1 = calculate_f1_score(sev_logits, severity_labels, 6)  # 6 severity classes (0-5)
        act_f1 = calculate_f1_score(act_logits, action_labels, 10)  # 10 action classes (0-9)
        
        # Update confusion matrices for original distribution analysis
        if confusion_matrix_dict is not None:
            update_confusion_matrix(confusion_matrix_dict, sev_logits, severity_labels, 'severity')
            update_confusion_matrix(confusion_matrix_dict, act_logits, action_labels, 'action_type')
        
        running_sev_acc += sev_acc
        running_act_acc += act_acc
        running_sev_f1 += sev_f1
        running_act_f1 += act_f1
        processed_batches += 1
        
        # Enhanced memory cleanup with MViT optimization
        if memory_cleanup_interval > 0 and (i + 1) % memory_cleanup_interval == 0:
            # Clean up intermediate tensors explicitly
            del sev_logits, act_logits, total_loss
            cleanup_memory()
            
            # Additional MViT-specific memory management
            if is_mvit_model and torch.cuda.is_available():
                # More aggressive memory management for MViT
                torch.cuda.synchronize()  # Ensure all operations complete
                torch.cuda.empty_cache()  # Clear cache more frequently
        
        # Only print progress every 25% of batches
        if (i + 1) % max(1, total_batches // 4) == 0:
            current_avg_loss = running_loss / total_samples
            current_avg_sev_acc = running_sev_acc / processed_batches
            current_avg_act_acc = running_act_acc / processed_batches
            progress = (i + 1) / total_batches * 100
            
            # Enhanced progress logging for MViT
            model_type = "MViT" if is_mvit_model else "Model"
            logger.info(f"  {model_type} Training Progress: {progress:.0f}% | Loss: {current_avg_loss:.3f} | SevAcc: {current_avg_sev_acc:.3f} | ActAcc: {current_avg_act_acc:.3f}")
    
    # Final cleanup
    if is_mvit_model:
        cleanup_memory()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
    # Use the actual number of samples we processed for loss calculation
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_sev_acc = running_sev_acc / processed_batches if processed_batches > 0 else 0
    epoch_act_acc = running_act_acc / processed_batches if processed_batches > 0 else 0
    epoch_sev_f1 = running_sev_f1 / processed_batches if processed_batches > 0 else 0
    epoch_act_f1 = running_act_f1 / processed_batches if processed_batches > 0 else 0
    
    return {
        'loss': epoch_loss,
        'sev_acc': epoch_sev_acc,
        'act_acc': epoch_act_acc,
        'sev_f1': epoch_sev_f1,
        'act_f1': epoch_act_f1
    }


def validate_one_epoch(model, dataloader, device, loss_config: dict, max_batches=None, memory_cleanup_interval=20, confusion_matrix_dict=None):
    """Validate the model for one epoch."""
    model.eval()
    running_loss = 0.0
    running_sev_acc = 0.0
    running_act_acc = 0.0
    running_sev_f1 = 0.0
    running_act_f1 = 0.0
    processed_batches = 0
    total_samples = 0  # Track actual number of samples processed
    
    # Detect if this is an MViT model for optimized memory management
    is_mvit_model = hasattr(model, 'mvit_processor') or (
        hasattr(model, 'module') and hasattr(model.module, 'mvit_processor')
    )
    
    # Optimize memory cleanup interval for MViT
    if is_mvit_model:
        memory_cleanup_interval = min(memory_cleanup_interval, 3)  # Even more aggressive for validation

    start_time = time.time()
    
    # Handle OptimizedDataLoader which may not have len()
    if max_batches:
        total_batches = max_batches
    else:
        try:
            total_batches = len(dataloader)
        except TypeError:
            # OptimizedDataLoader doesn't support len(), use underlying dataloader
            actual_loader = getattr(dataloader, 'dataloader', dataloader)
            total_batches = len(actual_loader) if hasattr(actual_loader, '__len__') else 1000  # fallback
    
    with torch.no_grad():  # Ensure no gradients for validation
        for i, batch_data in enumerate(dataloader):
            if max_batches is not None and (i + 1) > max_batches:
                break
            
            # Move all tensors in the batch to the device
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(device, non_blocking=True)
            
            severity_labels = batch_data["label_severity"]
            action_labels = batch_data["label_type"]
            
            current_batch_size = batch_data["clips"].size(0)
            total_samples += current_batch_size

            # Apply autocast for validation to match training config
            # Consistently use autocast with explicit dtype for validation when on CUDA
            if device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.float16): # Explicitly use float16 for consistency
                    # Note: For validation, we typically don't need view consistency loss
                    # so we skip return_view_logits to save computation
                    sev_logits, act_logits = model(batch_data)
                    total_loss, _, _, _ = calculate_multitask_loss(
                        sev_logits, act_logits, batch_data, loss_config
                    )
            else: # CPU or other device where mixed precision isn't available
                sev_logits, act_logits = model(batch_data)
                total_loss, _, _, _ = calculate_multitask_loss(
                    sev_logits, act_logits, batch_data, loss_config
                )

            running_loss += total_loss.item() * current_batch_size
            sev_acc = calculate_accuracy(sev_logits, severity_labels)
            act_acc = calculate_accuracy(act_logits, action_labels)
            
            # Update confusion matrices for validation analysis
            if confusion_matrix_dict is not None:
                update_confusion_matrix(confusion_matrix_dict, sev_logits, severity_labels, 'severity')
                update_confusion_matrix(confusion_matrix_dict, act_logits, action_labels, 'action_type')
            
            running_sev_acc += sev_acc
            running_act_acc += act_acc
            running_sev_f1 += calculate_f1_score(sev_logits, severity_labels, 6)  # 6 severity classes (0-5)
            running_act_f1 += calculate_f1_score(act_logits, action_labels, 10)  # 10 action classes (0-9)
            processed_batches += 1
            
            # Clean up batch data explicitly
            del batch_data, sev_logits, act_logits, total_loss
            
            # Enhanced memory cleanup for MViT during validation
            if memory_cleanup_interval > 0 and (i + 1) % memory_cleanup_interval == 0:
                cleanup_memory()
                
                # Additional cleanup for MViT
                if is_mvit_model and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

    # Final cleanup for MViT
    if is_mvit_model:
        cleanup_memory()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # Use the actual number of samples we processed for loss calculation
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_sev_acc = running_sev_acc / processed_batches if processed_batches > 0 else 0
    epoch_act_acc = running_act_acc / processed_batches if processed_batches > 0 else 0
    epoch_sev_f1 = running_sev_f1 / processed_batches if processed_batches > 0 else 0
    epoch_act_f1 = running_act_f1 / processed_batches if processed_batches > 0 else 0
    
    # Critical: Clean up memory after validation
    cleanup_memory()
    
    return {
        'loss': epoch_loss,
        'sev_acc': epoch_sev_acc,
        'act_acc': epoch_act_acc,
        'sev_f1': epoch_sev_f1,
        'act_f1': epoch_act_f1
    }


def debug_class_weights_impact(sev_logits, severity_labels, class_weights=None):
    """
    Debug function to check how class weights affect the loss for each class.
    This helps identify if class weights are being applied incorrectly.
    """
    import torch.nn.functional as F
    
    # Ensure type consistency for mixed precision training
    if class_weights is not None:
        class_weights = class_weights.to(sev_logits.dtype)
    
    # Calculate per-sample losses without weights
    ce_loss_no_weight = F.cross_entropy(sev_logits, severity_labels, reduction='none')
    
    # Calculate per-sample losses with weights
    if class_weights is not None:
        ce_loss_with_weight = F.cross_entropy(sev_logits, severity_labels, weight=class_weights, reduction='none')
    else:
        ce_loss_with_weight = ce_loss_no_weight
    
    # Get predictions and calculate class-wise statistics
    _, predictions = torch.max(sev_logits, 1)
    
    logger.info("🔍 CLASS WEIGHT DEBUG ANALYSIS:")
    logger.info(f"Batch size: {severity_labels.size(0)}")
    
    unique_labels = torch.unique(severity_labels)
    for label_class in unique_labels:
        mask = (severity_labels == label_class)
        if mask.sum() == 0:
            continue
            
        count = mask.sum().item()
        
        # Average losses for this class
        avg_loss_no_weight = ce_loss_no_weight[mask].mean().item()
        avg_loss_with_weight = ce_loss_with_weight[mask].mean().item()
        
        # Class weight
        weight = class_weights[label_class].item() if class_weights is not None else 1.0
        
        # Predictions for this class
        predicted_as_this = (predictions == label_class).sum().item()
        
        logger.info(f"  Class {label_class}: {count} samples, weight={weight:.2f}")
        logger.info(f"    Loss without weight: {avg_loss_no_weight:.4f}")
        logger.info(f"    Loss with weight: {avg_loss_with_weight:.4f}")
        logger.info(f"    Predicted as this class: {predicted_as_this}")
        
        # Check if weighting is working as expected
        expected_weighted_loss = avg_loss_no_weight * weight
        if abs(avg_loss_with_weight - expected_weighted_loss) > 0.001:
            logger.warning(f"    ⚠️  Weight application mismatch! Expected: {expected_weighted_loss:.4f}") 


def calculate_strong_action_class_weights(dataset, device, weighting_strategy='strong_inverse', power_factor=2.0):
    """
    Calculate stronger action class weights to combat severe imbalance.
    
    Unlike regular class weights that are normalized, these weights are much stronger
    and do not normalize to sum = #classes to provide more aggressive rebalancing.
    
    Args:
        dataset: Training dataset
        device: Device to put weights on
        weighting_strategy: Strategy for calculating weights
            - 'strong_inverse': Strong inverse frequency (power factor applied)
            - 'focal_style': Weights designed for focal loss (higher power)
            - 'exponential': Exponential scaling based on class rarity
        power_factor: Power to raise the inverse frequency to (higher = more aggressive)
    
    Returns:
        action_class_weights tensor
    """
    import torch
    
    # Count action type samples
    action_counts = torch.zeros(10)  # 10 action classes
    
    for action in dataset.actions:
        action_label = action['label_type']
        if 0 <= action_label < 10:
            action_counts[action_label] += 1
    
    # Avoid division by zero
    action_counts[action_counts == 0] = 1
    
    if weighting_strategy == 'strong_inverse':
        # Strong inverse frequency with power factor (not normalized)
        total_samples = action_counts.sum()
        raw_weights = total_samples / action_counts
        # Apply power factor to make minority class weights even stronger
        action_weights = torch.pow(raw_weights, power_factor)
        
        logger.info(f"Strong inverse action weighting (power factor: {power_factor})")
        
    elif weighting_strategy == 'focal_style':
        # Weights optimized for focal loss - emphasize hard examples more
        total_samples = action_counts.sum()
        frequencies = action_counts / total_samples
        # Use focal-loss style alpha: (1 - frequency)^power
        action_weights = torch.pow(1.0 - frequencies, power_factor) / frequencies
        
        logger.info(f"Focal-style action weighting (power factor: {power_factor})")
        
    elif weighting_strategy == 'exponential':
        # Exponential scaling - very aggressive for minority classes
        total_samples = action_counts.sum()
        max_count = action_counts.max()
        ratios = max_count / action_counts
        action_weights = torch.exp(ratios * power_factor / 10.0)  # Scale to prevent overflow
        
        logger.info(f"Exponential action weighting (power factor: {power_factor})")
        
    else:
        raise ValueError(f"Unknown weighting strategy: {weighting_strategy}")
    
    # Log class distribution and weights
    logger.info("Action class distribution and strong weights:")
    for i in range(10):
        if action_counts[i] > 0:
            count = int(action_counts[i])
            weight = action_weights[i].item()
            logger.info(f"  Action {i}: {count} samples → Weight: {weight:.2f}")
    
    weight_ratio = (action_weights.max() / action_weights.min()).item()
    logger.info(f"Action weight ratio (max/min): {weight_ratio:.1f}")
    
    if weight_ratio > 100:
        logger.warning(f"⚠️ Very large action weight ratio ({weight_ratio:.1f}) - this is intentional for severe imbalance")
        logger.warning("   Monitor training carefully for stability")
    
    return action_weights.to(device) 
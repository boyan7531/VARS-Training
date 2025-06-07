"""
Training utilities for Multi-Task Multi-View ResNet3D training.

This module handles training loops, validation, loss functions, and metrics calculation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


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


def calculate_multitask_loss(sev_logits, act_logits, batch_data, loss_config: dict):
    """
    Calculate weighted multi-task loss based on a configuration dictionary.
    
    Args:
        sev_logits: Severity classification logits
        act_logits: Action type classification logits  
        batch_data: Batch data containing labels
        loss_config (dict): Configuration for the loss function.
            - 'function': 'focal', 'weighted', or 'plain'
            - 'weights': [severity_weight, action_weight] for main tasks
            - 'label_smoothing': Label smoothing factor
            - 'focal_gamma': Focal Loss gamma parameter
            - 'severity_class_weights': Optional class weights for severity loss.
                                        If 'focal' is used, these become the 'alpha' parameter.
                                        If using a balanced sampler, this should typically be None.
    
    Returns:
        total_loss, loss_sev, loss_act
    """
    loss_function = loss_config.get('function', 'weighted')
    main_weights = loss_config.get('weights', [1.0, 1.0])
    label_smoothing = loss_config.get('label_smoothing', 0.0)
    
    if loss_function == 'focal':
        focal_gamma = loss_config.get('focal_gamma', 2.0)
        # When using Focal Loss with a balanced sampler, alpha (class weights) is often omitted.
        severity_class_weights = loss_config.get('severity_class_weights', None) 
        
        focal_criterion = FocalLoss(
            alpha=severity_class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing
        )
        loss_sev = focal_criterion(sev_logits, batch_data["label_severity"]) * main_weights[0]
        
        # Action type loss is typically less imbalanced, so standard CE is fine.
        loss_act = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(act_logits, batch_data["label_type"]) * main_weights[1]
        
    elif loss_function == 'weighted':
        severity_class_weights = loss_config.get('severity_class_weights') # Should be provided for this option
        loss_sev = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, 
            weight=severity_class_weights
        )(sev_logits, batch_data["label_severity"]) * main_weights[0]
        
        loss_act = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(act_logits, batch_data["label_type"]) * main_weights[1]
    
    elif loss_function == 'plain':
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


def train_one_epoch(model, dataloader, optimizer, device, loss_config: dict, scaler=None, 
                   max_batches=None, gradient_clip_norm=1.0, memory_cleanup_interval=20):
    """Train the model for one epoch."""
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
                total_loss, _, _ = calculate_multitask_loss(
                    sev_logits, act_logits, batch_data, loss_config
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
            total_loss, _, _ = calculate_multitask_loss(
                sev_logits, act_logits, batch_data, loss_config
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
        if memory_cleanup_interval > 0 and (i + 1) % memory_cleanup_interval == 0:
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
    
    return {
        'loss': epoch_loss,
        'sev_acc': epoch_sev_acc,
        'act_acc': epoch_act_acc,
        'sev_f1': epoch_sev_f1,
        'act_f1': epoch_act_f1
    }


def validate_one_epoch(model, dataloader, device, loss_config: dict, max_batches=None, memory_cleanup_interval=20):
    """Validate the model for one epoch."""
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
                    total_loss, _, _ = calculate_multitask_loss(
                        sev_logits, act_logits, batch_data, loss_config
                    )
            else: # CPU or other device, or if mixed precision is explicitly disabled for validation
                sev_logits, act_logits = model(batch_data)
                total_loss, _, _ = calculate_multitask_loss(
                    sev_logits, act_logits, batch_data, loss_config
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
            if memory_cleanup_interval > 0 and (i + 1) % memory_cleanup_interval == 0:
                cleanup_memory()

    num_samples_processed = len(dataloader.dataset) if max_batches is None else processed_batches * dataloader.batch_size
    epoch_loss = running_loss / num_samples_processed if num_samples_processed > 0 else 0
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
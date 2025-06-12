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
from sklearn.metrics import f1_score
from torch.profiler import profile, record_function, ProfilerActivity

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


def calculate_multitask_loss(sev_logits, act_logits, batch_data, loss_config: dict):
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
    
    Returns:
        total_loss, loss_sev, loss_act
    """
    loss_function = loss_config.get('function', 'weighted')
    main_weights = loss_config.get('weights', [1.0, 1.0])
    label_smoothing = loss_config.get('label_smoothing', 0.0)
    
    if loss_function == 'adaptive_focal':
        # Use adaptive focal loss with class-specific gamma values
        class_gamma_map = loss_config.get('class_gamma_map', None)
        severity_class_weights = loss_config.get('severity_class_weights', None)
        
        adaptive_focal_criterion = AdaptiveFocalLoss(
            class_gamma_map=class_gamma_map,
            class_alpha_map=None,  # We already use severity_class_weights
            label_smoothing=label_smoothing
        )
        loss_sev = adaptive_focal_criterion(sev_logits, batch_data["label_severity"]) * main_weights[0]
        
        # Action type loss is typically less imbalanced, so standard CE is fine
        loss_act = nn.CrossEntropyLoss(label_smoothing=label_smoothing)(act_logits, batch_data["label_type"]) * main_weights[1]
    
    elif loss_function == 'focal':
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
        raise ValueError(f"Unknown loss_function: {loss_function}. Must be 'adaptive_focal', 'focal', 'weighted', or 'plain'")
    
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
                   max_batches=None, gradient_clip_norm=1.0, memory_cleanup_interval=20, scheduler=None,
                   gpu_augmentation=None, enable_profiling=False):
    """Train the model for one epoch with optional GPU-based augmentation and profiling."""
    model.train()
    running_loss = 0.0
    running_sev_acc = 0.0
    running_act_acc = 0.0
    running_sev_f1 = 0.0
    running_act_f1 = 0.0
    processed_batches = 0
    total_samples = 0  # Track actual number of samples processed

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
        
        # Profile data loading
        if prof is not None:
            with record_function("data_loading"):
                # Move all tensors in the batch to the device
                for key in batch_data:
                    if isinstance(batch_data[key], torch.Tensor):
                        batch_data[key] = batch_data[key].to(device, non_blocking=True)
        else:
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
                # Profile GPU augmentation
                if prof is not None:
                    with record_function("gpu_augmentation"):
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
                
                # Profile model forward pass
                if prof is not None:
                    with record_function("model_forward"):
                        sev_logits, act_logits = model(batch_data)
                        total_loss, _, _ = calculate_multitask_loss(
                            sev_logits, act_logits, batch_data, loss_config
                        )
                else:
                    sev_logits, act_logits = model(batch_data)
                    total_loss, _, _ = calculate_multitask_loss(
                        sev_logits, act_logits, batch_data, loss_config
                    )

            # Profile backward pass
            if prof is not None:
                with record_function("loss_backward"):
                    scaler.scale(total_loss).backward()
                    
                    # Gradient clipping
                    if gradient_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
            else:
                scaler.scale(total_loss).backward()
                
                # Gradient clipping
                if gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                
                scaler.step(optimizer)
                scaler.update()
            
            # Stepping the OneCycleLR scheduler after optimizer.step()
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
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
                    
            sev_logits, act_logits = model(batch_data)
            total_loss, _, _ = calculate_multitask_loss(
                sev_logits, act_logits, batch_data, loss_config
            )

            total_loss.backward()
            
            # Gradient clipping
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            
            optimizer.step()
            
            # Stepping the OneCycleLR scheduler after optimizer.step()
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
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
                    data_loading_time = event.cuda_time_total
                elif "model_forward" in event.key:
                    compute_time = event.cuda_time_total
                elif "gpu_augmentation" in event.key:
                    augmentation_time = event.cuda_time_total
            
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
                
                # Diagnosis
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
            current_avg_loss = running_loss / total_samples
            current_avg_sev_acc = running_sev_acc / processed_batches
            current_avg_act_acc = running_act_acc / processed_batches
            progress = (i + 1) / total_batches * 100
            logger.info(f"  Training Progress: {progress:.0f}% | Loss: {current_avg_loss:.3f} | SevAcc: {current_avg_sev_acc:.3f} | ActAcc: {current_avg_act_acc:.3f}")
    
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


def validate_one_epoch(model, dataloader, device, loss_config: dict, max_batches=None, memory_cleanup_interval=20):
    """Validate the model for one epoch."""
    model.eval()
    running_loss = 0.0
    running_sev_acc = 0.0
    running_act_acc = 0.0
    running_sev_f1 = 0.0
    running_act_f1 = 0.0
    processed_batches = 0
    total_samples = 0  # Track actual number of samples processed

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
            
            current_batch_size = batch_data["clips"].size(0)
            total_samples += current_batch_size

            # Apply autocast for validation to match training config
            # Consistently use autocast with explicit dtype for validation when on CUDA
            if device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.float16): # Explicitly use float16 for consistency
                    sev_logits, act_logits = model(batch_data)
                    total_loss, _, _ = calculate_multitask_loss(
                        sev_logits, act_logits, batch_data, loss_config
                    )
            else: # CPU or other device where mixed precision isn't available
                sev_logits, act_logits = model(batch_data)
                total_loss, _, _ = calculate_multitask_loss(
                    sev_logits, act_logits, batch_data, loss_config
                )

            running_loss += total_loss.item() * current_batch_size
            sev_acc = calculate_accuracy(sev_logits, severity_labels)
            act_acc = calculate_accuracy(act_logits, action_labels)
            running_sev_acc += sev_acc
            running_act_acc += act_acc
            running_sev_f1 += calculate_f1_score(sev_logits, severity_labels, 6)  # 6 severity classes (0-5)
            running_act_f1 += calculate_f1_score(act_logits, action_labels, 10)  # 10 action classes (0-9)
            processed_batches += 1
            
            # Clean up batch data explicitly
            del batch_data, sev_logits, act_logits, total_loss
            
            # Periodic memory cleanup during validation
            if memory_cleanup_interval > 0 and (i + 1) % memory_cleanup_interval == 0:
                cleanup_memory()

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
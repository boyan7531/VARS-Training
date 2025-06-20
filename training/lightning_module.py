"""
PyTorch Lightning module for Multi-Task Multi-View ResNet3D training.

This module wraps the existing sophisticated training logic in PyTorch Lightning
while preserving all custom features like freezing strategies, multi-task loss,
and advanced callbacks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR, ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from collections import defaultdict
import time

# Import existing components
from .model_utils import (
    create_model, setup_freezing_strategy, calculate_class_weights,
    SmartFreezingManager, GradientGuidedFreezingManager, AdvancedFreezingManager, 
    get_phase_info, setup_discriminative_optimizer, log_trainable_parameters
)
from .training_utils import (
    calculate_multitask_loss, calculate_accuracy, calculate_f1_score,
    update_confusion_matrix, check_overfitting_alert, cleanup_memory
)
from .data import create_gpu_augmentation

logger = logging.getLogger(__name__)


class MultiTaskVideoLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for multi-task video classification.
    
    This module preserves all the sophisticated training logic from the original
    implementation while adding Lightning's built-in features like gradient
    accumulation, distributed training, and automatic mixed precision.
    """
    
    def __init__(
        self,
        args: Any,
        vocab_sizes: Optional[Dict[str, int]] = None,
        num_classes: Tuple[int, int] = (6, 10),  # (severity, action_type)
        **kwargs
    ):
        """
        Initialize the Lightning module.
        
        Args:
            args: Training arguments from config
            vocab_sizes: Optional vocabulary sizes (deprecated for video-only)
            num_classes: Tuple of (num_severity_classes, num_action_type_classes)
        """
        super().__init__()
        
        # Store configuration
        self.args = args
        self.vocab_sizes = vocab_sizes
        self.num_severity_classes, self.num_action_type_classes = num_classes
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['vocab_sizes'])
        
        # Initialize model components
        self._init_model()
        self._init_loss_config()
        self._init_freezing_strategy()
        self._init_tracking_variables()
        self._init_ema_model()
        
        # GPU augmentation
        self.gpu_augmentation = None
        if hasattr(args, 'gpu_augmentation') and args.gpu_augmentation:
            # Will be initialized in setup() when device is available
            pass
        
        logger.info(f"Initialized MultiTaskVideoLightningModule with {self.num_severity_classes} severity and {self.num_action_type_classes} action classes")
    
    def _init_model(self):
        """Initialize the underlying model."""
        # Create model using existing factory function
        # Note: We'll handle device placement through Lightning
        self.model = create_model(self.args, self.vocab_sizes, torch.device('cpu'), num_gpus=1)
        
        # Extract the actual model if wrapped in DataParallel
        if hasattr(self.model, 'module'):
            self.model = self.model.module
    
    def _init_loss_config(self):
        """Initialize loss function configuration."""
        # Calculate class weights if needed
        self.severity_class_weights = None
        self.class_gamma_map = None
        
        # Will be set in setup() when dataset is available
        self.loss_config = {
            'function': self.args.loss_function,
            'weights': self.args.main_task_weights,
            'label_smoothing': self.args.label_smoothing,
            'focal_gamma': self.args.focal_gamma,
            'severity_class_weights': None,  # Will be set in setup()
            'class_gamma_map': None,  # Will be set in setup()
            'view_consistency': self.args.view_consistency,
            'vc_weight': self.args.vc_weight
        }
    
    def _init_freezing_strategy(self):
        """Initialize freezing strategy manager."""
        self.freezing_manager = None
        # Will be initialized in setup() when model is on device
    
    def _init_tracking_variables(self):
        """Initialize tracking variables for training."""
        self.best_val_acc = 0.0
        self.best_epoch = -1
        self.train_confusion_matrices = {}
        self.val_confusion_matrices = {}
        
        # Phase tracking for gradual fine-tuning
        self.current_phase = 1 if self.args.gradual_finetuning else None
        self.phase1_scheduler = None
        
        # Performance tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def _init_ema_model(self):
        """Initialize EMA model."""
        self.ema_model = None
        # Will be initialized in setup() when model is on device
    
    def setup(self, stage: str):
        """Setup hook called before training/validation/testing."""
        # Import the rank checking function
        from .training_utils import is_main_process
        
        if is_main_process():
            logger.info(f"Setting up Lightning module for stage: {stage}")
        
        if stage in ['fit', 'validate']:
            # Initialize GPU augmentation if enabled
            if hasattr(self.args, 'gpu_augmentation') and self.args.gpu_augmentation:
                self.gpu_augmentation = create_gpu_augmentation(self.args, self.device)
                if is_main_process():
                    logger.info("GPU augmentation initialized")
            
            # Initialize EMA model if enabled
            if self.args.use_ema and self.ema_model is None:
                from .model_utils import create_ema_model
                self.ema_model = create_ema_model(self.model, decay=self.args.ema_decay)
                if self.ema_model and is_main_process():
                    logger.info(f"🚀 EMA model initialized with decay: {self.args.ema_decay}")
                    if self.args.ema_eval:
                        logger.info("✅ EMA weights will be used for validation in Lightning")
            
            # Initialize freezing strategy
            if self.freezing_manager is None:
                self.freezing_manager = setup_freezing_strategy(self.args, self.model)
                if is_main_process():
                    logger.info(f"Freezing strategy initialized: {self.args.freezing_strategy}")
            
            # Get automatic class weights from datamodule if available
            if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'get_class_weights'):
                class_weights = self.trainer.datamodule.get_class_weights()
                
                if class_weights and 'severity' in class_weights and 'action' in class_weights:
                    # Use automatic class weights from datamodule
                    self.severity_class_weights = class_weights['severity'].to(self.device)
                    self.action_class_weights = class_weights['action'].to(self.device)
                    if is_main_process():
                        logger.info("Using automatic class weights from datamodule")
                        logger.info(f"Severity weights: {self.severity_class_weights}")
                        logger.info(f"Action weights: {self.action_class_weights}")
                else:
                    if is_main_process():
                        logger.info("No automatic class weights available from datamodule")
                    
                    # Fallback to manual class weight calculation if needed
                    if (self.args.loss_function in ['focal', 'weighted'] and 
                        not self.args.disable_class_balancing and 
                        not self.args.use_class_balanced_sampler and
                        hasattr(self.trainer.datamodule, 'train_dataset')):
                        
                        train_dataset = self.trainer.datamodule.train_dataset
                        self.severity_class_weights = calculate_class_weights(
                            train_dataset, 6, self.device, 
                            self.args.class_weighting_strategy, 
                            self.args.max_weight_ratio
                        )
                        logger.info("Fallback: Manual class weights calculated for loss function")
                    else:
                        logger.info("Skipping class weight calculation (using ClassBalancedSampler or disabled)")
                
                # Setup class-specific gamma for focal loss
                if self.args.loss_function == 'focal':
                    if not self.args.use_class_balanced_sampler:
                        self.class_gamma_map = {
                            0: 2.0, 1: 1.5, 2: 2.0, 3: 2.0, 4: 3.0, 5: 3.5
                        }
                        logger.info("Class-specific gamma values set for focal loss")
                
                # Handle strong action weights if enabled
                action_weights_for_loss = None
                if self.args.use_strong_action_weights and hasattr(self.trainer.datamodule, 'train_dataset'):
                    from training.training_utils import calculate_strong_action_class_weights
                    train_dataset = self.trainer.datamodule.train_dataset
                    action_weights_for_loss = calculate_strong_action_class_weights(
                        train_dataset, 
                        self.device,
                        self.args.action_weight_strategy,
                        self.args.action_weight_power
                    )
                    if is_main_process():
                        logger.info("💪 Using strong action class weights to combat severe action imbalance")
                elif hasattr(self, 'action_class_weights'):
                    action_weights_for_loss = self.action_class_weights
                
                # Update loss config with both severity and action weights
                # Fix: When using ClassBalancedSampler, don't apply severity class weights to avoid double-balancing
                # But only when we're actually balancing severity (not when using action-only balancing)
                severity_weights_for_loss = (
                    None if (self.args.use_class_balanced_sampler and not self.args.use_action_balanced_sampler_only) 
                    else self.severity_class_weights
                )
                if self.args.use_class_balanced_sampler and not self.args.use_action_balanced_sampler_only and is_main_process():
                    logger.info("BALANCE: ClassBalancedSampler detected: Setting severity_class_weights=None to prevent double-balancing")
                
                self.loss_config.update({
                    'severity_class_weights': severity_weights_for_loss,
                    'action_class_weights': action_weights_for_loss,
                    'class_gamma_map': self.class_gamma_map,
                    'using_oversampling': self.args.use_class_balanced_sampler or self.args.use_action_balanced_sampler_only,
                    'view_consistency': self.args.view_consistency,
                    'vc_weight': self.args.vc_weight
                })
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Setup optimizer using existing logic
        optimizer, scheduler, phase1_scheduler, scheduler_info = self._setup_optimizer_and_scheduler()
        
        self.phase1_scheduler = phase1_scheduler
        
        # Return optimizer and scheduler configuration
        if scheduler is not None:
            # Handle different scheduler types
            if isinstance(scheduler, ReduceLROnPlateau):
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': 'val_combined_acc',
                        'mode': 'max',
                        'frequency': 1
                    }
                }
            elif isinstance(scheduler, OneCycleLR):
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',  # Update every step for OneCycleLR
                        'frequency': 1
                    }
                }
            else:
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'epoch',
                        'frequency': 1
                    }
                }
        
        return optimizer
    
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and scheduler using existing logic."""
        from .model_utils import setup_discriminative_optimizer
        
        # Create base optimizer
        if self.args.optimizer == 'adamw':
            if self.args.discriminative_lr:
                # Use backbone_lr if provided, otherwise default to lr * 0.1
                backbone_lr = getattr(self.args, 'backbone_lr', self.args.lr * 0.1)
                param_groups = setup_discriminative_optimizer(
                    self.model, 
                    head_lr=self.args.lr,
                    backbone_lr=backbone_lr,
                    weight_decay=self.args.weight_decay
                )
                optimizer = optim.AdamW(param_groups)
            else:
                optimizer = optim.AdamW(
                    self.model.parameters(), 
                    lr=self.args.lr, 
                    weight_decay=self.args.weight_decay
                )
        elif self.args.optimizer == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.args.lr, 
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.args.lr, 
                weight_decay=self.args.weight_decay
            )
        
        # Create scheduler
        scheduler = None
        phase1_scheduler = None
        scheduler_info = "None"
        
        if self.args.gradual_finetuning:
            # Phase 1 uses ReduceLROnPlateau
            phase1_scheduler = ReduceLROnPlateau(
                optimizer, mode='max',
                factor=self.args.phase1_plateau_factor,
                patience=self.args.phase1_plateau_patience,
                min_lr=self.args.min_lr
            )
            scheduler_info = f"Phase 1: ReduceLROnPlateau (patience={self.args.phase1_plateau_patience})"
        else:
            # Standard scheduling
            if self.args.scheduler == 'cosine':
                scheduler = CosineAnnealingLR(
                    optimizer, 
                    T_max=self.args.epochs, 
                    eta_min=self.args.lr * 0.01
                )
                scheduler_info = f"CosineAnnealing (T_max={self.args.epochs})"
            elif self.args.scheduler == 'onecycle':
                # OneCycleLR will be created after we know steps_per_epoch
                # For now, we'll handle this in on_train_start
                scheduler_info = f"OneCycle (max_lr={self.args.lr:.1e})"
            elif self.args.scheduler == 'step':
                scheduler = StepLR(
                    optimizer, 
                    step_size=self.args.step_size, 
                    gamma=self.args.gamma
                )
                scheduler_info = f"StepLR (step_size={self.args.step_size})"
            elif self.args.scheduler == 'reduce_on_plateau':
                scheduler = ReduceLROnPlateau(
                    optimizer, mode='max', 
                    factor=self.args.gamma,
                    patience=self.args.plateau_patience, 
                    min_lr=self.args.min_lr
                )
                scheduler_info = f"ReduceLROnPlateau (patience={self.args.plateau_patience})"
        
        logger.info(f"Optimizer: {self.args.optimizer}, Scheduler: {scheduler_info}")
        return optimizer, scheduler, phase1_scheduler, scheduler_info
    
    def on_train_start(self):
        """Called when training starts."""
        # Initialize OneCycleLR if needed (now we know steps_per_epoch)
        if self.args.scheduler == 'onecycle' and not self.args.gradual_finetuning:
            steps_per_epoch = len(self.trainer.train_dataloader)
            
            # Replace the scheduler
            optimizer = self.optimizers()
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.args.lr,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=self.args.scheduler_warmup_epochs / self.trainer.max_epochs if self.trainer.max_epochs > 0 else 0.1
            )
            
            # Update Lightning's scheduler
            self.trainer.lr_scheduler_configs[0].scheduler = scheduler
            logger.info(f"Initialized OneCycleLR with {steps_per_epoch} steps per epoch")
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """Training step for one batch."""
        # Enable anomaly detection for debugging NaN issues
        if batch_idx < 5:  # Only for first few batches to avoid performance impact
            torch.autograd.set_detect_anomaly(True)
        
        # [NaN-origin] Step 6: Safety net at Lightning level
        if torch.isnan(batch["clips"]).any():
            self.log("batch_has_nan", True)
            logger.error(f"[NaN-origin] NaN reached the model in batch {batch_idx}")
            raise RuntimeError("NaN reached the model")
        
        # Comprehensive input validation
        if "clips" in batch:
            clips = batch["clips"]
            if torch.isnan(clips).any():
                logger.error(f"NaN detected in input clips at batch {batch_idx}!")
                logger.error(f"Clips shape: {clips.shape}, NaN count: {torch.isnan(clips).sum().item()}")
                # Skip this batch entirely if input is corrupted
                return None
            
            # Check for extreme values that could cause instability
            clips_min, clips_max = clips.min(), clips.max()
            if clips_max > 100.0 or clips_min < -100.0:
                logger.warning(f"Extreme values in clips at batch {batch_idx}: min={clips_min:.3f}, max={clips_max:.3f}")
                # Clamp to reasonable range
                batch["clips"] = torch.clamp(clips, min=-10.0, max=10.0)
                logger.warning("Clamped input clips to [-10, 10] range")
        
        # Validate labels
        for label_key in ["label_severity", "label_type"]:
            if label_key in batch:
                labels = batch[label_key]
                if torch.isnan(labels).any():
                    logger.error(f"NaN detected in {label_key} at batch {batch_idx}!")
                    return None
        
        # Get model output with view consistency if enabled
        view_consistency_enabled = (hasattr(self.args, 'view_consistency') and 
                                  self.args.view_consistency)
        
        # Check if we have view mask for variable views
        view_mask = batch.get("view_mask", None)
        
        try:
            if view_consistency_enabled:
                model_output = self.model(batch, return_view_logits=True)
                sev_logits = model_output['severity_logits']
                act_logits = model_output['action_logits']
                
                # Get view logits if available
                view_logits = None
                if 'view_logits' in model_output:
                    view_logits = model_output['view_logits']
                    # Ensure view_mask is properly passed to view consistency loss
                    if view_mask is not None:
                        # Make sure view_mask matches the view_logits dimensions
                        if view_logits['sev_logits_v'].dim() == 3:  # [B, V, C]
                            batch_size, max_views = view_logits['sev_logits_v'].shape[:2]
                            if view_mask.shape != (batch_size, max_views):
                                if view_mask.dim() == 3:  # [B, clips_per_video, max_views]
                                    # Take the first clip's mask (assuming same for all clips)
                                    view_mask = view_mask[:, 0, :]
                        # Add view_mask to view_logits for consistency loss calculation
                        view_logits['view_mask'] = view_mask
                    else:
                        # Create a default mask if not provided
                        if 'sev_logits_v' in view_logits:
                            batch_size, max_views = view_logits['sev_logits_v'].shape[:2]
                            view_mask = torch.ones(batch_size, max_views, dtype=torch.bool, device=sev_logits.device)
                            view_logits['view_mask'] = view_mask
                
                total_loss, loss_sev, loss_act, loss_components = calculate_multitask_loss(
                    sev_logits, act_logits, batch, self.loss_config, view_logits
                )
            else:
                sev_logits, act_logits = self.model(batch)
                view_logits = None
                total_loss, loss_sev, loss_act, loss_components = calculate_multitask_loss(
                    sev_logits, act_logits, batch, self.loss_config
                )
            
            # Additional safety check for NaN in logits
            if torch.isnan(sev_logits).any() or torch.isnan(act_logits).any():
                logger.error(f"NaN detected in logits at batch {batch_idx}!")
                logger.error(f"Severity logits NaN count: {torch.isnan(sev_logits).sum()}")
                logger.error(f"Action logits NaN count: {torch.isnan(act_logits).sum()}")
                # Skip this batch
                return None
            
        except Exception as e:
            logger.error(f"Error in forward pass at batch {batch_idx}: Forward pass failed: {str(e)}")
            # Log more details about the error
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Skip this batch
            return None
        finally:
            # Disable anomaly detection after first few batches
            if batch_idx < 5:
                torch.autograd.set_detect_anomaly(False)
        
        # Calculate metrics
        sev_acc = calculate_accuracy(sev_logits, batch["label_severity"])
        act_acc = calculate_accuracy(act_logits, batch["label_type"])
        sev_f1 = calculate_f1_score(sev_logits, batch["label_severity"], self.num_severity_classes)
        act_f1 = calculate_f1_score(act_logits, batch["label_type"], self.num_action_type_classes)
        
        # Update confusion matrices for training analysis
        if batch_idx % 50 == 0:  # Update every 50 batches to avoid overhead
            update_confusion_matrix(self.train_confusion_matrices, sev_logits, batch["label_severity"], 'severity')
            update_confusion_matrix(self.train_confusion_matrices, act_logits, batch["label_type"], 'action_type')
        
        # Store outputs for epoch-level processing
        output = {
            'loss': total_loss,
            'loss_sev': loss_sev,
            'loss_act': loss_act,
            'sev_acc': sev_acc,
            'act_acc': act_acc,
            'sev_f1': sev_f1,
            'act_f1': act_f1,
        }
        
        # Update EMA if enabled
        if self.ema_model is not None:
            from .model_utils import update_ema_model
            update_ema_model(self.ema_model, self.model)
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_loss_sev', loss_sev, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_loss_act', loss_act, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_sev_acc', sev_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_act_acc', act_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_combined_acc', (sev_acc + act_acc) / 2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return output
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """Validation step for one batch."""
        # Forward pass (no augmentation in validation)
        # Note: For validation, we typically don't compute view consistency loss
        # to save computation, but we could enable it if needed for debugging
        sev_logits, act_logits = self.model(batch)
        
        # Calculate loss
        total_loss, loss_sev, loss_act, loss_components = calculate_multitask_loss(
            sev_logits, act_logits, batch, self.loss_config
        )
        
        # Calculate metrics
        sev_acc = calculate_accuracy(sev_logits, batch["label_severity"])
        act_acc = calculate_accuracy(act_logits, batch["label_type"])
        sev_f1 = calculate_f1_score(sev_logits, batch["label_severity"], self.num_severity_classes)
        act_f1 = calculate_f1_score(act_logits, batch["label_type"], self.num_action_type_classes)
        
        # Update confusion matrices
        update_confusion_matrix(self.val_confusion_matrices, sev_logits, batch["label_severity"], 'severity')
        update_confusion_matrix(self.val_confusion_matrices, act_logits, batch["label_type"], 'action_type')
        
        # Store validation predictions for debugging (accumulated across batches)
        if not hasattr(self, 'val_predictions'):
            self.val_predictions = {'sev_logits': [], 'act_logits': [], 'sev_labels': [], 'act_labels': []}
        
        self.val_predictions['sev_logits'].append(sev_logits.detach())
        self.val_predictions['act_logits'].append(act_logits.detach())
        self.val_predictions['sev_labels'].append(batch["label_severity"].detach())
        self.val_predictions['act_labels'].append(batch["label_type"].detach())
        
        # Store outputs
        output = {
            'val_loss': total_loss,
            'val_loss_sev': loss_sev,
            'val_loss_act': loss_act,
            'val_sev_acc': sev_acc,
            'val_act_acc': act_acc,
            'val_sev_f1': sev_f1,
            'val_act_f1': act_f1,
        }
        
        # Log metrics
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_loss_sev', loss_sev, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_loss_act', loss_act, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_sev_acc', sev_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_act_acc', act_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_combined_acc', (sev_acc + act_acc) / 2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return output
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Log trainable parameters
        log_trainable_parameters(self.model, self.current_epoch)
        
        # Handle freezing strategy updates
        if self.freezing_manager is not None:
            # Get current validation accuracy for freezing decisions
            val_combined_acc = self.trainer.callback_metrics.get('val_combined_acc', 0.0)
            
            # Update freezing manager
            update_info = self.freezing_manager.update_after_epoch(val_combined_acc, self.current_epoch)
            
            if update_info.get('rebuild_optimizer', False):
                logger.info("Rebuilding optimizer due to freezing strategy update")
                # Note: In Lightning, optimizer rebuilding needs to be handled differently
                # This is a limitation we'll need to work around
        
        # Handle gradual fine-tuning phase transitions
        if self.args.gradual_finetuning:
            self._handle_phase_transition()
        
        # OPTIMIZATION: Boost backbone LR when ≥50% of backbone is unfrozen
        from .freezing.base_utils import boost_backbone_lr_if_needed
        optimizer = self.optimizers()
        boost_backbone_lr_if_needed(self.args, optimizer, self.model, self.current_epoch)
        
        # Memory cleanup
        cleanup_memory()
    
    def on_validation_start(self):
        """Called at the start of validation."""
        # Apply EMA weights for validation if enabled
        from .model_utils import should_use_ema_for_validation, apply_ema_weights
        use_ema_for_val = should_use_ema_for_validation(self.args, self.ema_model, self.current_epoch)
        
        if use_ema_for_val:
            logger.debug(f"🔄 Using EMA weights for validation (epoch {self.current_epoch+1})")
            apply_ema_weights(self.model, self.ema_model)
        
        # Store flag for restoration in on_validation_end
        self._used_ema_for_validation = use_ema_for_val
    
    def on_validation_end(self):
        """Called at the end of validation."""
        # Restore online weights after validation
        if getattr(self, '_used_ema_for_validation', False):
            from .model_utils import restore_online_weights
            restore_online_weights(self.model, self.ema_model)
            logger.debug(f"🔄 Restored online weights after validation")
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Import the rank checking function
        from .training_utils import is_main_process, log_validation_prediction_stats
        
        # Log validation prediction statistics for debugging
        if hasattr(self, 'val_predictions') and is_main_process():
            # Concatenate all validation predictions from this epoch
            sev_logits = torch.cat(self.val_predictions['sev_logits'], dim=0)
            act_logits = torch.cat(self.val_predictions['act_logits'], dim=0)
            sev_labels = torch.cat(self.val_predictions['sev_labels'], dim=0)
            act_labels = torch.cat(self.val_predictions['act_labels'], dim=0)
            
            # Log prediction statistics every 5 epochs
            log_validation_prediction_stats(
                sev_logits, act_logits, sev_labels, act_labels, 
                self.current_epoch, log_frequency=5
            )
            
            # Clear predictions for next epoch
            self.val_predictions = {'sev_logits': [], 'act_logits': [], 'sev_labels': [], 'act_labels': []}
        
        # Log confusion matrices every 5 epochs
        if (self.current_epoch + 1) % 5 == 0:
            self._log_confusion_matrices()
        
        # Check for overfitting
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch', 0.0)
        val_loss = self.trainer.callback_metrics.get('val_loss', 0.0)
        check_overfitting_alert(train_loss, val_loss, self.current_epoch)
        
        # Update best metrics
        val_combined_acc = self.trainer.callback_metrics.get('val_combined_acc', 0.0)
        if val_combined_acc > self.best_val_acc:
            self.best_val_acc = val_combined_acc
            self.best_epoch = self.current_epoch + 1
            if is_main_process():
                logger.info(f"New best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
    
    def _handle_phase_transition(self):
        """Handle gradual fine-tuning phase transitions."""
        current_phase, phase_epoch, total_phase_epochs = get_phase_info(
            self.current_epoch, self.args.phase1_epochs, self.trainer.max_epochs
        )
        
        # Phase transition logic
        if current_phase != self.current_phase:
            logger.info(f"Transitioning from Phase {self.current_phase} to Phase {current_phase}")
            self.current_phase = current_phase
            
            if current_phase == 2:
                # Transition to Phase 2 - unfreeze backbone layers
                from .model_utils import unfreeze_backbone_gradually
                unfreeze_backbone_gradually(
                    self.model, 
                    num_blocks=self.args.unfreeze_blocks,
                    gradual=True
                )
                logger.info(f"Unfroze {self.args.unfreeze_blocks} backbone blocks for Phase 2")
    
    def _log_confusion_matrices(self):
        """Log confusion matrices for training and validation."""
        try:
            from .training_utils import compute_confusion_matrices, log_confusion_matrix
            
            # Define class names
            severity_class_names = ["Sev_0", "Sev_1", "Sev_2", "Sev_3", "Sev_4", "Sev_5"]
            action_class_names = [f"Act_{i}" for i in range(10)]
            
            # Log training confusion matrices
            if self.train_confusion_matrices:
                train_cms = compute_confusion_matrices(self.train_confusion_matrices)
                
                if 'severity' in train_cms:
                    logger.info(f"\n[EPOCH {self.current_epoch+1}] TRAINING METRICS (Original Distribution)")
                    log_confusion_matrix(train_cms['severity'], 'Severity', severity_class_names)
                
                if 'action_type' in train_cms:
                    log_confusion_matrix(train_cms['action_type'], 'Action Type', action_class_names)
                
                # Clear for next epoch
                self.train_confusion_matrices.clear()
            
            # Log validation confusion matrices
            if self.val_confusion_matrices:
                val_cms = compute_confusion_matrices(self.val_confusion_matrices)
                
                logger.info(f"\n[EPOCH {self.current_epoch+1}] VALIDATION CONFUSION MATRICES")
                
                if 'severity' in val_cms:
                    log_confusion_matrix(val_cms['severity'], 'Validation Severity', severity_class_names)
                
                if 'action_type' in val_cms:
                    log_confusion_matrix(val_cms['action_type'], 'Validation Action Type', action_class_names)
                
                # Clear for next epoch
                self.val_confusion_matrices.clear()
                
        except Exception as e:
            logger.warning(f"Failed to log confusion matrices: {e}")
    
    def forward(self, batch):
        """Forward pass through the model."""
        return self.model(batch)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step for inference."""
        sev_logits, act_logits = self.model(batch)
        
        # Convert to predictions
        sev_preds = torch.argmax(sev_logits, dim=1)
        act_preds = torch.argmax(act_logits, dim=1)
        
        return {
            'severity_predictions': sev_preds,
            'action_predictions': act_preds,
            'severity_logits': sev_logits,
            'action_logits': act_logits
        } 
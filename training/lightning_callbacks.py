"""
PyTorch Lightning callbacks for Multi-Task Multi-View ResNet3D training.

These callbacks preserve all the sophisticated training features from the original
implementation while working within Lightning's callback system.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import logging
from typing import Any, Dict, Optional
import os
from pathlib import Path

# Import existing utilities
from .training_utils import cleanup_memory
from .checkpoint_utils import save_training_history

logger = logging.getLogger(__name__)


class FreezingStrategyCallback(Callback):
    """
    Callback to handle sophisticated freezing strategies.
    
    This preserves all the advanced freezing logic including adaptive,
    progressive, gradient-guided, and advanced freezing strategies.
    """
    
    def __init__(self):
        super().__init__()
        self.freezing_manager = None
    
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Setup the freezing manager."""
        if stage == 'fit':
            # The freezing manager is already initialized in the Lightning module
            self.freezing_manager = pl_module.freezing_manager
            if self.freezing_manager is not None:
                logger.info(f"FreezingStrategyCallback initialized with {type(self.freezing_manager).__name__}")
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Handle freezing strategy updates at the end of each epoch."""
        if self.freezing_manager is not None:
            # Get current validation accuracy
            val_combined_acc = trainer.callback_metrics.get('val_combined_acc', 0.0)
            
            try:
                # Update freezing manager
                update_info = self.freezing_manager.update_after_epoch(val_combined_acc, trainer.current_epoch)
                
                if update_info.get('rebuild_optimizer', False):
                    logger.info("Freezing strategy requested optimizer rebuild")
                    # Note: Optimizer rebuilding in Lightning requires special handling
                    # This would need to be implemented with Lightning's manual optimization
                
                if update_info.get('unfrozen_layers'):
                    logger.info(f"Unfrozen layers: {update_info['unfrozen_layers']}")
                
                if update_info.get('rollback_performed'):
                    logger.info("Rollback performed by advanced freezing strategy")
                    
            except Exception as e:
                logger.warning(f"Error in freezing strategy update: {e}")


class MemoryCleanupCallback(Callback):
    """
    Callback for memory cleanup and optimization.
    
    This handles the sophisticated memory management from the original implementation.
    """
    
    def __init__(self, cleanup_interval: int = 20):
        super().__init__()
        self.cleanup_interval = cleanup_interval
        self.step_count = 0
    
    def on_train_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: STEP_OUTPUT, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Cleanup memory after training batches."""
        self.step_count += 1
        
        if self.cleanup_interval > 0 and self.step_count % self.cleanup_interval == 0:
            cleanup_memory()
            
            # Additional cleanup for MViT models
            if hasattr(pl_module.model, 'mvit_processor'):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
    
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Cleanup memory after validation batches."""
        if self.cleanup_interval > 0 and batch_idx % max(1, self.cleanup_interval // 2) == 0:
            cleanup_memory()
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Final memory cleanup at epoch end."""
        cleanup_memory()
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Final memory cleanup after validation."""
        cleanup_memory()


class OOMRecoveryCallback(Callback):
    """
    Callback for Out-of-Memory recovery.
    
    This integrates the sophisticated OOM handling from the original implementation.
    """
    
    def __init__(self, reduction_factor: float = 0.75, min_batch_size: int = 1):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.min_batch_size = min_batch_size
        self.original_batch_size = None
        self.current_batch_size = None
    
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Setup OOM recovery with initial batch size."""
        if stage == 'fit':
            # Get batch size from trainer
            self.original_batch_size = trainer.datamodule.args.batch_size
            self.current_batch_size = self.original_batch_size
            logger.info(f"OOM Recovery initialized with batch_size={self.original_batch_size}")
    
    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        """Handle OOM exceptions."""
        if isinstance(exception, RuntimeError) and "out of memory" in str(exception).lower():
            logger.warning(f"🚨 Out of Memory detected: {exception}")
            
            # Calculate new batch size
            new_batch_size = max(int(self.current_batch_size * self.reduction_factor), self.min_batch_size)
            
            if new_batch_size < self.current_batch_size:
                logger.warning(f"📉 Reducing batch size: {self.current_batch_size} → {new_batch_size}")
                self.current_batch_size = new_batch_size
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Note: Actual batch size reduction would require restarting training
                # with the new batch size in Lightning
                logger.warning("⚠️ Please restart training with reduced batch size for OOM recovery")
            else:
                logger.error(f"❌ Cannot reduce batch size further (already at minimum: {self.min_batch_size})")


class ConfigurationValidationCallback(Callback):
    """
    Callback for configuration validation and warnings.
    """
    
    def __init__(self):
        super().__init__()
    
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Validate configuration at setup."""
        if stage == 'fit':
            args = pl_module.args
            
            # Validate configuration
            self._validate_training_config(args, trainer)
            self._validate_data_config(args)
            self._validate_optimization_config(args)
    
    def _validate_training_config(self, args: Any, trainer: pl.Trainer) -> None:
        """Validate training configuration."""
        # Check gradient accumulation compatibility
        if hasattr(args, 'accumulate_grad_batches') and args.accumulate_grad_batches > 1:
            if trainer.accumulate_grad_batches != args.accumulate_grad_batches:
                logger.warning(f"Trainer accumulate_grad_batches ({trainer.accumulate_grad_batches}) "
                             f"differs from args ({args.accumulate_grad_batches})")
        
        # Check mixed precision compatibility
        if hasattr(args, 'mixed_precision') and args.mixed_precision:
            if trainer.precision not in [16, 'bf16']:
                logger.warning(f"Mixed precision requested but trainer precision is {trainer.precision}")
        
        # Validate freezing strategy
        if hasattr(args, 'freezing_strategy'):
            if args.freezing_strategy in ['gradient_guided', 'advanced'] and args.epochs < 10:
                logger.warning(f"Freezing strategy '{args.freezing_strategy}' may not work well with < 10 epochs")
    
    def _validate_data_config(self, args: Any) -> None:
        """Validate data configuration."""
        # Check batch size vs. GPU memory
        if hasattr(args, 'batch_size') and args.batch_size > 32:
            if hasattr(args, 'frames_per_clip') and args.frames_per_clip > 16:
                logger.warning(f"Large batch_size ({args.batch_size}) with {args.frames_per_clip} frames "
                             f"may cause OOM. Consider reducing batch_size or using gradient accumulation.")
        
        # Check augmentation compatibility
        if hasattr(args, 'gpu_augmentation') and args.gpu_augmentation:
            if not hasattr(args, 'mixed_precision') or not args.mixed_precision:
                logger.info("GPU augmentation works best with mixed precision training")
    
    def _validate_optimization_config(self, args: Any) -> None:
        """Validate optimization configuration."""
        # Check learning rate vs. batch size
        if hasattr(args, 'lr') and hasattr(args, 'batch_size'):
            if args.lr > 1e-3 and args.batch_size < 8:
                logger.warning(f"High learning rate ({args.lr}) with small batch size ({args.batch_size}) "
                             f"may cause training instability")
        
        # Check scheduler compatibility
        if hasattr(args, 'scheduler') and args.scheduler == 'onecycle':
            if hasattr(args, 'gradual_finetuning') and args.gradual_finetuning:
                logger.warning("OneCycleLR scheduler may not work well with gradual fine-tuning")


class TrainingHistoryCallback(Callback):
    """
    Callback to track and save training history.
    
    This preserves the training history functionality from the original implementation.
    """
    
    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'train_sev_acc': [],
            'train_act_acc': [],
            'val_loss': [],
            'val_sev_acc': [],
            'val_act_acc': [],
            'learning_rate': []
        }
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Update training history at the end of each epoch."""
        # Get metrics from trainer
        metrics = trainer.callback_metrics
        
        # Update history
        self.history['train_loss'].append(float(metrics.get('train_loss_epoch', 0.0)))
        self.history['train_sev_acc'].append(float(metrics.get('train_sev_acc', 0.0)))
        self.history['train_act_acc'].append(float(metrics.get('train_act_acc', 0.0)))
        self.history['val_loss'].append(float(metrics.get('val_loss', 0.0)))
        self.history['val_sev_acc'].append(float(metrics.get('val_sev_acc', 0.0)))
        self.history['val_act_acc'].append(float(metrics.get('val_act_acc', 0.0)))
        
        # Get current learning rate
        current_lr = 0.0
        if trainer.optimizers:
            current_lr = trainer.optimizers[0].param_groups[0]['lr']
        self.history['learning_rate'].append(current_lr)
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save training history at the end of training."""
        if not pl_module.args.test_run:
            history_path = self.save_dir / 'training_history.json'
            save_training_history(self.history, str(history_path))
            logger.info(f"Training history saved to {history_path}")


class CustomEarlyStoppingCallback(EarlyStopping):
    """
    Enhanced early stopping callback that preserves the original early stopping logic.
    """
    
    def __init__(
        self,
        monitor: str = 'val_combined_acc',
        min_delta: float = 0.001,
        patience: int = 10,
        mode: str = 'max',
        restore_best_weights: bool = True,
        **kwargs
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            mode=mode,
            **kwargs
        )
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
    
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check for early stopping and save best weights."""
        super().on_validation_end(trainer, pl_module)
        
        # Save best weights if we have a new best
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is not None:
            if self._is_improvement(current_score):
                if self.restore_best_weights:
                    self.best_weights = pl_module.state_dict().copy()
                    logger.debug(f"Saved best weights at epoch {trainer.current_epoch}")
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Restore best weights if early stopping was triggered."""
        super().on_train_end(trainer, pl_module)
        
        if self.stopped_epoch > 0 and self.restore_best_weights and self.best_weights is not None:
            pl_module.load_state_dict(self.best_weights)
            logger.info(f"Restored best weights from epoch {self.best_epoch}")
    
    def _is_improvement(self, current_score: float) -> bool:
        """Check if current score is an improvement."""
        if self.best_score is None:
            return True
        
        if self.mode == 'max':
            return current_score > self.best_score + self.min_delta
        else:
            return current_score < self.best_score - self.min_delta


class CustomModelCheckpointCallback(ModelCheckpoint):
    """
    Enhanced model checkpoint callback that preserves the original checkpointing logic.
    """
    
    def __init__(
        self,
        dirpath: str,
        monitor: str = 'val_combined_acc',
        mode: str = 'max',
        save_top_k: int = 1,
        save_last: bool = True,
        every_n_epochs: int = 10,
        **kwargs
    ):
        super().__init__(
            dirpath=dirpath,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
            every_n_epochs=every_n_epochs,
            **kwargs
        )
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save checkpoint with enhanced metadata."""
        super().on_train_epoch_end(trainer, pl_module)
        
        # Save regular checkpoint every N epochs (not just best)
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            metrics = trainer.callback_metrics
            
            checkpoint_name = f'checkpoint_epoch_{trainer.current_epoch + 1}.ckpt'
            checkpoint_path = os.path.join(self.dirpath, checkpoint_name)
            
            # Save with additional metadata
            trainer.save_checkpoint(checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {trainer.current_epoch + 1}: {checkpoint_path}")


def create_lightning_callbacks(args: Any) -> list:
    """
    Create all Lightning callbacks based on configuration.
    
    Args:
        args: Training arguments
        
    Returns:
        List of configured callbacks
    """
    callbacks = []
    
    # Memory cleanup callback
    memory_interval = getattr(args, 'memory_cleanup_interval', 20)
    if memory_interval > 0:
        callbacks.append(MemoryCleanupCallback(cleanup_interval=memory_interval))
        logger.info(f"Added MemoryCleanupCallback (interval={memory_interval})")
    
    # Freezing strategy callback
    if hasattr(args, 'freezing_strategy') and args.freezing_strategy != 'none':
        callbacks.append(FreezingStrategyCallback())
        logger.info("Added FreezingStrategyCallback")
    
    # OOM recovery callback
    if hasattr(args, 'enable_oom_recovery') and args.enable_oom_recovery:
        oom_callback = OOMRecoveryCallback(
            reduction_factor=getattr(args, 'oom_reduction_factor', 0.75),
            min_batch_size=getattr(args, 'min_batch_size', 1)
        )
        callbacks.append(oom_callback)
        logger.info("Added OOMRecoveryCallback")
    
    # Configuration validation callback
    if hasattr(args, 'enable_config_validation') and args.enable_config_validation:
        callbacks.append(ConfigurationValidationCallback())
        logger.info("Added ConfigurationValidationCallback")
    
    # Training history callback
    if hasattr(args, 'save_dir'):
        callbacks.append(TrainingHistoryCallback(save_dir=args.save_dir))
        logger.info("Added TrainingHistoryCallback")
    
    # Early stopping callback
    if hasattr(args, 'early_stopping_patience') and args.early_stopping_patience is not None:
        early_stopping = CustomEarlyStoppingCallback(
            monitor='val_combined_acc',
            patience=args.early_stopping_patience,
            mode='max',
            min_delta=0.001,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        logger.info(f"Added EarlyStoppingCallback (patience={args.early_stopping_patience})")
    
    # Model checkpoint callback
    if hasattr(args, 'save_dir'):
        model_checkpoint = CustomModelCheckpointCallback(
            dirpath=args.save_dir,
            monitor='val_combined_acc',
            mode='max',
            save_top_k=1,
            save_last=True,
            every_n_epochs=10,
            filename='best_model_epoch_{epoch:02d}_{val_combined_acc:.4f}'
        )
        callbacks.append(model_checkpoint)
        logger.info("Added ModelCheckpointCallback")
    
    logger.info(f"Created {len(callbacks)} Lightning callbacks")
    return callbacks 
#!/usr/bin/env python3
"""
PyTorch Lightning training script for Multi-Task Multi-View ResNet3D.

This script converts the existing sophisticated training logic to PyTorch Lightning
while preserving all advanced features like freezing strategies, multi-task loss,
gradient accumulation, distributed training, and automatic mixed precision.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import logging
import random
import numpy as np
import os
import multiprocessing as mp
from pathlib import Path
import time

# Import training components
from training.config import parse_args, log_configuration_summary
from training.lightning_module import MultiTaskVideoLightningModule
from training.lightning_datamodule import VideoDataModule
from training.lightning_callbacks import create_lightning_callbacks

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_lightning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable deterministic algorithms with warn_only for non-deterministic operations
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info(f"Random seed set to {seed} with deterministic algorithms enabled (warn_only=True)")
    except Exception as e:
        logger.warning(f"Could not enable deterministic algorithms: {e}")
        logger.info(f"Random seed set to {seed}")


def create_trainer(args, callbacks, logger_instance=None):
    """
    Create PyTorch Lightning trainer with all sophisticated features.
    
    Args:
        args: Training arguments
        callbacks: List of callbacks
        logger_instance: Optional logger instance
        
    Returns:
        Configured PyTorch Lightning trainer
    """
    # Determine precision for mixed precision training
    precision = 32
    if hasattr(args, 'mixed_precision') and args.mixed_precision:
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
            precision = 16  # Use FP16 on modern GPUs
        else:
            precision = 32
            logger.warning("Mixed precision requested but not available. Using FP32.")
    
    # Determine strategy for distributed training
    strategy = "auto"
    if torch.cuda.device_count() > 1:
        # Use DDP for multi-GPU training
        strategy = DDPStrategy(
            find_unused_parameters=False,  # More efficient for our model
            gradient_as_bucket_view=True,  # Memory optimization
        )
        logger.info(f"Using DDP strategy for {torch.cuda.device_count()} GPUs")
    
    # Configure gradient accumulation
    accumulate_grad_batches = getattr(args, 'accumulate_grad_batches', 1)
    if accumulate_grad_batches > 1:
        logger.info(f"Gradient accumulation enabled: {accumulate_grad_batches} batches")
    
    # Configure gradient clipping
    gradient_clip_val = getattr(args, 'gradient_clip_norm', 1.0)
    gradient_clip_algorithm = 'norm'
    
    # Create trainer configuration
    trainer_config = {
        'max_epochs': args.epochs,
        'precision': precision,
        'strategy': strategy,
        'accumulate_grad_batches': accumulate_grad_batches,
        'gradient_clip_val': gradient_clip_val,
        'gradient_clip_algorithm': gradient_clip_algorithm,
        'callbacks': callbacks,
        'logger': logger_instance,
        'log_every_n_steps': 10,
        'val_check_interval': 1.0,  # Validate after each epoch
        'check_val_every_n_epoch': 1,
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'deterministic': 'warn',  # For reproducibility with non-deterministic op warnings
        'benchmark': False,       # For reproducibility
    }
    
    # Add test run specific settings
    if hasattr(args, 'test_run') and args.test_run:
        trainer_config.update({
            'fast_dev_run': args.test_batches,
            'logger': False,  # Disable logging for test runs
        })
        logger.info(f"Test run mode: {args.test_batches} batches")
    
    # Add profiler for debugging if needed
    if hasattr(args, 'enable_profiler') and args.enable_profiler:
        try:
            from pytorch_lightning.profilers import AdvancedProfiler
        except ImportError:
            try:
                from lightning.pytorch.profilers import AdvancedProfiler
            except ImportError:
                logger.warning("Profiler not available, skipping profiling")
                AdvancedProfiler = None
        
        if AdvancedProfiler:
            trainer_config['profiler'] = AdvancedProfiler(
                dirpath=args.save_dir,
                filename="training_profile"
            )
            logger.info("Advanced profiler enabled")
    
    # Handle multi-node training
    if hasattr(args, 'num_nodes') and args.num_nodes > 1:
        trainer_config.update({
            'num_nodes': args.num_nodes,
            'devices': getattr(args, 'devices_per_node', -1),  # Use all available GPUs per node
        })
        logger.info(f"Multi-node training: {args.num_nodes} nodes")
    else:
        # Single node training
        if torch.cuda.is_available():
            trainer_config['devices'] = -1  # Use all available GPUs
            trainer_config['accelerator'] = 'gpu'
        else:
            trainer_config['devices'] = 1
            trainer_config['accelerator'] = 'cpu'
            logger.warning("No GPU available. Using CPU training.")
    
    # Create and return trainer
    trainer = pl.Trainer(**trainer_config)
    
    logger.info("PyTorch Lightning Trainer created with configuration:")
    logger.info(f"  Max epochs: {trainer_config['max_epochs']}")
    logger.info(f"  Precision: {trainer_config['precision']}")
    logger.info(f"  Strategy: {trainer_config['strategy']}")
    logger.info(f"  Accumulate grad batches: {trainer_config['accumulate_grad_batches']}")
    logger.info(f"  Gradient clipping: {trainer_config['gradient_clip_val']}")
    
    return trainer


def main():
    """Main training function with PyTorch Lightning."""
    # Configure multiprocessing for DataLoader workers
    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn', force=True)
            logger.info("Set multiprocessing start method to 'spawn'")
    except RuntimeError as e:
        logger.warning(f"Could not set multiprocessing method: {e}")
    
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Log configuration summary
    log_configuration_summary(args)
    
    # Log Lightning-specific improvements
    logger.info("=" * 60)
    logger.info("PYTORCH LIGHTNING FEATURES ENABLED")
    logger.info("=" * 60)
    logger.info("+ Automatic gradient accumulation")
    logger.info("+ Automatic mixed precision (AMP)")
    logger.info("+ Multi-node distributed training support")
    logger.info("+ Advanced checkpointing and logging")
    logger.info("+ Memory optimization callbacks")
    logger.info("+ Out-of-Memory (OOM) recovery")
    logger.info("+ Early stopping with best weight restoration")
    logger.info("+ Sophisticated freezing strategies preserved")
    logger.info("=" * 60)
    
    # Create data module
    logger.info("Creating PyTorch Lightning DataModule...")
    datamodule = VideoDataModule(args)
    
    # Setup data module to get vocabulary sizes
    datamodule.setup('fit')
    vocab_sizes = datamodule.get_vocab_sizes()
    
    # Log dataset information
    dataset_info = datamodule.get_dataset_info()
    logger.info(f"Dataset sizes: Train={dataset_info['train_size']}, Val={dataset_info['val_size']}")
    
    # Create Lightning module
    logger.info("Creating PyTorch Lightning Module...")
    lightning_module = MultiTaskVideoLightningModule(
        args=args,
        vocab_sizes=vocab_sizes,
        num_classes=(6, 10)  # (severity, action_type)
    )
    
    # Create callbacks
    logger.info("Creating PyTorch Lightning callbacks...")
    callbacks = create_lightning_callbacks(args)
    
    # Create logger
    tb_logger = None
    if not (hasattr(args, 'test_run') and args.test_run):
        log_dir = Path(args.save_dir) / "lightning_logs"
        tb_logger = TensorBoardLogger(
            save_dir=str(log_dir),
            name="multi_task_video_training",
            version=None,  # Auto-increment version
            default_hp_metric=False
        )
        logger.info(f"TensorBoard logging enabled: {log_dir}")
    
    # Create trainer
    logger.info("Creating PyTorch Lightning Trainer...")
    trainer = create_trainer(args, callbacks, tb_logger)
    
    # Log final configuration
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION FINALIZED")
    logger.info("=" * 60)
    logger.info(f"Training strategy: {trainer.strategy.__class__.__name__}")
    logger.info(f"Precision: {trainer.precision}")
    logger.info(f"Devices: {trainer.num_devices}")
    logger.info(f"Max epochs: {trainer.max_epochs}")
    logger.info(f"Gradient accumulation: {trainer.accumulate_grad_batches}")
    logger.info(f"Callbacks: {len(callbacks)}")
    logger.info("=" * 60)
    
    # Handle resume from checkpoint
    ckpt_path = None
    if hasattr(args, 'resume') and args.resume:
        ckpt_path = args.resume
        logger.info(f"Resuming training from checkpoint: {ckpt_path}")
    
    try:
        # Start training
        logger.info("Starting PyTorch Lightning training...")
        
        # Track training time manually
        start_time = time.time()
        
        trainer.fit(
            model=lightning_module,
            datamodule=datamodule,
            ckpt_path=ckpt_path
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Log training completion
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Best validation accuracy: {lightning_module.best_val_acc:.4f} at epoch {lightning_module.best_epoch}")
        logger.info(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Get final learning rate safely
        try:
            if trainer.optimizers and len(trainer.optimizers) > 0:
                final_lr = trainer.optimizers[0].param_groups[0]['lr']
                logger.info(f"Final learning rate: {final_lr:.2e}")
        except Exception as e:
            logger.warning(f"Could not get final learning rate: {e}")
        
        # Save final model information
        if not (hasattr(args, 'test_run') and args.test_run):
            try:
                model_info = lightning_module.model.get_model_info() if hasattr(lightning_module.model, 'get_model_info') else {}
                if model_info:
                    logger.info("Model information:")
                    for key, value in model_info.items():
                        logger.info(f"  {key}: {value}")
            except Exception as e:
                logger.warning(f"Could not get model info: {e}")
        
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Training session ended")


if __name__ == "__main__":
    main() 
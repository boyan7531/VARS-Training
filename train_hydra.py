#!/usr/bin/env python3
"""
Hydra-based Training Script for VARS

This script demonstrates the new clean configuration system using Hydra.
Compare this to the old train.py with hundreds of arguments!

Usage Examples:
    # Basic training with defaults
    python train_hydra.py
    
    # Quick smoke test
    python train_hydra.py --config-name quick_test
    
    # Production run with W&B logging
    python train_hydra.py --config-name production dataset.data_dir=/path/to/data
    
    # Override specific parameters
    python train_hydra.py training.max_epochs=100 dataset.batch_size=32
    
    # Use different model
    python train_hydra.py model=resnet3d
    
    # Hyperparameter sweep
    python train_hydra.py --multirun training.learning_rate=1e-3,5e-4,1e-4
"""

import os
import logging
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# Import our existing training components
from training.lightning_module import MultiTaskVideoLightningModule
from training.lightning_datamodule import VideoDataModule
from training.config_structured import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_callbacks(cfg: DictConfig) -> list:
    """Setup PyTorch Lightning callbacks based on config"""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.experiment.monitor,
        mode=cfg.experiment.mode,
        save_top_k=cfg.experiment.save_top_k,
        save_last=cfg.experiment.save_last,
        filename='{epoch:02d}-{val_sev_acc:.3f}',
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Early stopping (optional)
    if hasattr(cfg.training, 'early_stopping') and cfg.training.early_stopping:
        early_stop = EarlyStopping(
            monitor=cfg.experiment.monitor,
            patience=cfg.training.get('early_stopping_patience', 10),
            mode=cfg.experiment.mode,
            verbose=True
        )
        callbacks.append(early_stop)
    
    return callbacks


def setup_logger(cfg: DictConfig) -> Optional[pl.loggers.Logger]:
    """Setup experiment logger based on config"""
    if cfg.experiment.use_wandb:
        return WandbLogger(
            project=cfg.experiment.wandb_project,
            entity=cfg.experiment.wandb_entity,
            name=cfg.experiment.name,
            tags=cfg.experiment.tags,
            notes=cfg.experiment.notes
        )
    else:
        return TensorBoardLogger(
            save_dir="logs",
            name=cfg.experiment.name,
            version=None
        )


def convert_config_to_args(cfg: DictConfig) -> object:
    """
    Convert Hydra config to args object for compatibility with existing code.
    This is a temporary bridge during migration.
    """
    class Args:
        def __init__(self, config_dict):
            self._config = config_dict
            # Flatten the nested config for backward compatibility
            self._flatten_config(config_dict)
            
            # BEGIN PATCH: ensure legacy path attributes exist
            # Map new Hydra dataset path to legacy attribute expected by existing code
            if hasattr(self, "dataset_data_dir") and not hasattr(self, "mvfouls_path"):
                # Ensure mvfouls_path is stored as string
                self.mvfouls_path = str(getattr(self, "dataset_data_dir"))
            # END PATCH
        
            # ------------------------------------------------------------------
            # Legacy field aliases (until full migration away from argparse-style
            # attribute names). These map the newer Hydra configuration keys to
            # the attribute names used by existing training / data-loading code.
            # ------------------------------------------------------------------

            dataset_cfg = config_dict.get('dataset', {}) if isinstance(config_dict, dict) else {}

            # Path to MVFouls root directory
            if 'data_dir' in dataset_cfg:
                setattr(self, 'mvfouls_path', dataset_cfg['data_dir'])

            # Clip / video parameters
            setattr(self, 'frames_per_clip', dataset_cfg.get('num_frames', 16))
            setattr(self, 'target_fps', dataset_cfg.get('fps', 25))

            frame_size = dataset_cfg.get('frame_size', 224)
            setattr(self, 'img_height', frame_size)
            setattr(self, 'img_width', frame_size)

            # Data loader parameters
            setattr(self, 'batch_size', dataset_cfg.get('batch_size', 8))
            setattr(self, 'num_workers', dataset_cfg.get('num_workers', 8))
            setattr(self, 'prefetch_factor', dataset_cfg.get('prefetch_factor', 2))

            # Default foul-centered frame indices if not provided elsewhere
            setattr(self, 'start_frame', dataset_cfg.get('start_frame', 67))
            setattr(self, 'end_frame', dataset_cfg.get('end_frame', 82))
        
        def _flatten_config(self, config_dict, prefix=''):
            """Flatten nested config into attributes"""
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    self._flatten_config(value, f"{prefix}{key}_" if prefix else f"{key}_")
                else:
                    attr_name = f"{prefix}{key}" if prefix else key
                    setattr(self, attr_name, value)
        
        def __getattr__(self, name):
            # Fallback for any missing attributes
            return getattr(self._config, name, None)
    
    return Args(OmegaConf.to_container(cfg, resolve=True))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function"""
    
    # Print the configuration (much cleaner than hundreds of args!)
    logger.info("🚀 Starting VARS training with Hydra configuration:")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seed for reproducibility
    pl.seed_everything(cfg.system.seed, workers=True)
    
    # Convert config to args for backward compatibility
    # TODO: Remove this once all components are updated to use cfg directly
    args = convert_config_to_args(cfg)
    
    # Create data module
    logger.info("📊 Setting up data module...")
    datamodule = VideoDataModule(args)
    
    # Create model
    logger.info("🧠 Creating model...")
    model = MultiTaskVideoLightningModule(args)
    
    # Setup callbacks and logger
    callbacks = setup_callbacks(cfg)
    experiment_logger = setup_logger(cfg)
    
    # Create trainer
    logger.info("⚡ Setting up PyTorch Lightning trainer...")
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        gpus=cfg.system.gpus if isinstance(cfg.system.gpus, int) else len(cfg.system.gpus),
        precision=cfg.system.precision,
        strategy=cfg.system.strategy,
        callbacks=callbacks,
        logger=experiment_logger,
        log_every_n_steps=cfg.experiment.log_every_n_steps,
        val_check_interval=cfg.experiment.val_check_interval,
        gradient_clip_val=cfg.training.gradient_clip_val,
        deterministic=cfg.system.deterministic,
        # Debug options
        fast_dev_run=cfg.experiment.fast_dev_run,
        limit_train_batches=cfg.experiment.limit_train_batches,
        limit_val_batches=cfg.experiment.limit_val_batches,
    )
    
    # Start training
    logger.info("🎯 Starting training...")
    trainer.fit(model, datamodule)
    
    # Optional test evaluation if a 'test' split exists inside the dataset directory
    from pathlib import Path as _P
    test_annotations = _P(args.mvfouls_path) / "test" / "annotations.json"
    if test_annotations.exists():
        logger.info("🧪 Found test split – running test evaluation...")
        trainer.test(model, datamodule)
    
    logger.info("✅ Training completed!")
    
    # Print best checkpoint info
    if hasattr(trainer.checkpoint_callback, 'best_model_path'):
        logger.info(f"💾 Best model saved at: {trainer.checkpoint_callback.best_model_path}")
        logger.info(f"📈 Best {cfg.experiment.monitor}: {trainer.checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main() 
"""
Structured Configuration System for VARS Training

This module defines type-safe, hierarchical configuration dataclasses
that replace the hundreds of argparse arguments with organized YAML configs.

Usage:
    python train_hydra.py --config-path conf --config-name baseline
    python train_hydra.py dataset=mvfouls model=mvit training=progressive
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from omegaconf import MISSING


@dataclass
class DatasetConfig:
    """Dataset and data loading configuration"""
    # Core paths
    data_dir: str = MISSING
    
    # Video processing
    clip_duration: float = 2.0
    fps: int = 25
    frame_size: int = 224
    num_frames: int = 16
    
    # Data loading
    batch_size: int = 8
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Augmentations
    enable_augmentations: bool = True
    horizontal_flip_prob: float = 0.5
    rotation_degrees: int = 10
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1
    
    # Advanced augmentations
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    auto_augment: Optional[str] = None  # 'randaugment', 'trivialaugment', etc.

    # Annotation format
    # Each split folder (train / valid / test) must contain an 'annotations.json'
    # file following the SoccerNet MVFoul specification. The data loader will
    # automatically locate this file; no explicit annotation paths are needed.


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Backbone
    backbone: str = "mvit_base_16x4"
    pretrained: bool = True
    
    # Architecture
    num_severity_classes: int = 6
    num_action_classes: int = 10
    dropout_rate: float = 0.1
    
    # Multi-task heads
    severity_head_hidden_dim: int = 512
    action_head_hidden_dim: int = 512
    shared_features: bool = False
    
    # Advanced features
    use_attention_pooling: bool = False
    use_temporal_attention: bool = False
    feature_fusion_method: str = "concat"  # 'concat', 'add', 'attention'


@dataclass
class LossConfig:
    """Loss function configuration"""
    # Loss function type
    function: str = "focal"  # 'focal', 'weighted', 'adaptive_focal', 'plain'
    
    # Task weights
    severity_weight: float = 1.0
    action_weight: float = 1.0
    
    # Label smoothing (auto-disabled with oversampling)
    label_smoothing: float = 0.1
    
    # Focal loss parameters
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None
    
    # Class weights (computed automatically)
    use_class_weights: bool = True
    class_weight_method: str = "effective_num"  # 'effective_num', 'inverse_freq', 'balanced'
    effective_num_beta: float = 0.99


@dataclass
class SamplingConfig:
    """Class balancing and sampling configuration"""
    # Sampling strategy
    use_class_balanced_sampler: bool = False
    use_action_balanced_sampler_only: bool = False
    
    # Oversampling parameters
    oversample_factor: float = 2.0  # Reduced from 4.0
    action_oversample_factor: float = 2.0  # Reduced from 4.0
    
    # Progressive sampling
    progressive_class_balancing: bool = True  # Now default
    progressive_start_epoch: int = 5
    progressive_end_epoch: int = 20
    
    # Advanced sampling
    alternating_sampler: bool = False
    alternating_switch_epoch: int = 10


@dataclass
class FreezingConfig:
    """Gradual unfreezing configuration"""
    # Strategy
    gradual_finetuning: bool = True
    freezing_strategy: str = "progressive"  # 'progressive', 'adaptive', 'scheduled'
    
    # Progressive unfreezing
    unfreeze_blocks: int = 3
    unfreeze_schedule: List[int] = field(default_factory=lambda: [5, 10, 15])
    
    # Adaptive unfreezing
    patience_epochs: int = 3
    min_improvement: float = 0.01
    
    # Backbone LR boost (NEW OPTIMIZATION)
    enable_backbone_lr_boost: bool = True
    backbone_lr_ratio_after_half: float = 0.6


@dataclass
class TrainingConfig:
    """Training hyperparameters and optimization"""
    # Basic training
    max_epochs: int = 50
    learning_rate: float = 1e-3
    head_lr: float = 1e-3
    backbone_lr: float = 1e-4
    
    # Optimizer
    optimizer: str = "adamw"  # 'adamw', 'adam', 'sgd'
    weight_decay: float = 1e-4
    momentum: float = 0.9  # for SGD
    
    # Scheduler
    scheduler: str = "cosine"  # 'cosine', 'step', 'plateau', 'warmup_cosine'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Regularization
    gradient_clip_val: float = 1.0
    dropout_rate: float = 0.1
    
    # Advanced optimizers
    lookahead: bool = False
    use_sam: bool = False  # Sharpness-Aware Minimization


@dataclass
class SystemConfig:
    """System and hardware configuration"""
    # Hardware
    gpus: Union[int, List[int]] = 1
    precision: str = "16-mixed"  # '32', '16-mixed', 'bf16-mixed'
    
    # Distributed training
    strategy: str = "auto"  # 'auto', 'ddp', 'fsdp'
    sync_batchnorm: bool = True
    
    # Memory optimization
    gradient_checkpointing: bool = False
    compile_model: bool = False  # PyTorch 2.0 compile
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False


@dataclass
class ExperimentConfig:
    """Experiment tracking and checkpointing"""
    # Experiment info
    name: str = "vars_training"
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    # Checkpointing
    save_top_k: int = 3
    monitor: str = "val/sev_acc"
    mode: str = "max"
    save_last: bool = True
    
    # Logging
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0
    
    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "vars-training"
    wandb_entity: Optional[str] = None
    
    # Debug mode
    debug: bool = False
    fast_dev_run: bool = False
    limit_train_batches: Optional[float] = None
    limit_val_batches: Optional[float] = None


@dataclass
class Config:
    """Main configuration combining all sub-configs"""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    freezing: FreezingConfig = field(default_factory=FreezingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # Global overrides
    defaults: List[Any] = field(default_factory=lambda: [
        "_self_",
        "dataset: mvfouls",
        "model: mvit",
        "training: baseline",
        "loss: focal",
        "sampling: progressive",
        "freezing: progressive",
        "system: single_gpu",
        "experiment: default"
    ])


# Register structured configs with Hydra
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="dataset", name="base", node=DatasetConfig)
cs.store(group="model", name="base", node=ModelConfig)
cs.store(group="loss", name="base", node=LossConfig)
cs.store(group="sampling", name="base", node=SamplingConfig)
cs.store(group="freezing", name="base", node=FreezingConfig)
cs.store(group="training", name="base", node=TrainingConfig)
cs.store(group="system", name="base", node=SystemConfig)
cs.store(group="experiment", name="base", node=ExperimentConfig) 
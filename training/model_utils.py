"""
Model utilities, freezing strategies, and parameter management for Multi-Task Multi-View ResNet3D training.

This module now serves as a compatibility layer that imports from the new modular freezing package.
The original functionality has been split into multiple focused modules within the freezing package.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR, ExponentialLR, ReduceLROnPlateau
import logging

from .freezing import (
    # Base utilities
    calculate_class_weights,
    freeze_backbone,
    unfreeze_backbone_gradually,
    setup_discriminative_optimizer,
    get_phase_info,
    log_trainable_parameters,
    
    # Freezing managers
    SmartFreezingManager,
    GradientGuidedFreezingManager,
    AdvancedFreezingManager,
    
    # Model setup
    create_model,
    setup_freezing_strategy
)

# Initialize logger
logger = logging.getLogger(__name__)

# Re-export everything for backward compatibility
__all__ = [
    'get_model_config',
    'get_optimizer', 
    'get_scheduler',
    'setup_freezing_strategy',
    'create_model',
    'separate_weight_decay_params'
]

def get_model_config():
    """Get model configuration - placeholder function."""
    return {}


def get_optimizer(model, args):
    """Create optimizer based on arguments with proper weight decay handling."""
    # Use weight decay parameter separation for better regularization
    param_groups = separate_weight_decay_params(model, args.weight_decay)
    
    if args.optimizer == 'adamw':
        return optim.AdamW(param_groups, lr=args.lr)
    elif args.optimizer == 'adam':
        return optim.Adam(param_groups, lr=args.lr)  
    elif args.optimizer == 'sgd':
        return optim.SGD(
            param_groups, 
            lr=args.lr,
            momentum=getattr(args, 'momentum', 0.9)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def get_scheduler(optimizer, args):
    """Create learning rate scheduler based on arguments."""
    if args.scheduler == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=getattr(args, 'min_lr', 1e-6))
    elif args.scheduler == 'onecycle':
        return OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs)
    elif args.scheduler == 'step':
        return StepLR(optimizer, step_size=getattr(args, 'step_size', 10), gamma=getattr(args, 'gamma', 0.1))
    elif args.scheduler == 'exponential':
        return ExponentialLR(optimizer, gamma=getattr(args, 'gamma', 0.95))
    elif args.scheduler == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=getattr(args, 'gamma', 0.5),
            patience=getattr(args, 'plateau_patience', 5),
            min_lr=getattr(args, 'min_lr', 1e-6)
        )
    elif args.scheduler == 'none':
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def create_model(args, vocab_sizes, device, num_gpus):
    """Create and initialize the model."""
    logger.info(f"Initializing {args.backbone_type.upper()} model: {args.backbone_name}")
    
    try:
        # Import the unified model interface
        from model import create_unified_model
        
        # Create model using the unified interface
        model = create_unified_model(
            backbone_type=args.backbone_type,
            num_severity=6,  # 6 severity classes (0-5)
            num_action_type=10,  # 10 action type classes
            vocab_sizes=vocab_sizes,
            backbone_name=args.backbone_name,
            use_attention_aggregation=args.attention_aggregation,
            use_augmentation=not args.disable_in_model_augmentation,
            disable_in_model_augmentation=args.disable_in_model_augmentation,
            enable_gradient_checkpointing=getattr(args, 'enable_gradient_checkpointing', True),
            enable_memory_optimization=getattr(args, 'enable_memory_optimization', True)
        )
        
        # Get model info for logging
        model_info = model.get_model_info()
        
        # Log model architecture details
        logger.info(f"{args.backbone_type.upper()} components initialized successfully. Combined feature dim: {model_info.get('video_feature_dim', 0) + model_info.get('total_embedding_dim', 0)}")
        
        # Log MViT-specific optimization status
        if args.backbone_type.lower() == 'mvit':
            logger.info(f"🚀 MViT Optimizations Active:")
            logger.info(f"   - Gradient Checkpointing: {'✅' if model_info.get('gradient_checkpointing_enabled', False) else '❌'}")
            logger.info(f"   - Memory Optimization: {'✅' if model_info.get('memory_optimization_enabled', False) else '❌'}")
            logger.info(f"   - Sequential Processing: {'✅' if getattr(args, 'mvit_sequential_processing', True) else '❌'}")
        
        # Move model to device
        model = model.to(device)
        
        # Multi-GPU setup
        if num_gpus > 1:
            model = torch.nn.DataParallel(model)
            logger.info(f"Model wrapped with DataParallel for {num_gpus} GPUs")
        
        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model initialized - Total parameters: {total_params:,}")
        logger.info(f"Combined feature dimension: {model_info.get('video_feature_dim', 0) + model_info.get('total_embedding_dim', 0)}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to create {args.backbone_type} model: {str(e)}")
        logger.error(f"Available backbone types: resnet3d, mvit")
        if args.backbone_type == 'resnet3d':
            logger.error(f"Available ResNet3D models: resnet3d_18, mc3_18, r2plus1d_18, resnet3d_50")
        elif args.backbone_type == 'mvit':
            logger.error(f"Available MViT models: mvit_base_16x4, mvit_base_32x3, mvit_small_16x4")
        raise


def separate_weight_decay_params(model, weight_decay=0.01):
    """
    Separate model parameters for weight decay application.
    Apply weight decay to all parameters except bias and normalization layers.
    
    Args:
        model: PyTorch model
        weight_decay: Weight decay value (default 0.01)
    
    Returns:
        List of parameter groups for optimizer
    """
    # Parameters that should have weight decay
    decay_params = []
    # Parameters that should NOT have weight decay (bias and norm layers)
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check if parameter is bias or from normalization layer
        if (name.endswith('.bias') or 
            'norm' in name.lower() or 
            'bn' in name.lower() or 
            'layernorm' in name.lower() or
            'batchnorm' in name.lower() or
            'groupnorm' in name.lower()):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    logger.info(f"Weight decay setup: {len(decay_params)} params with decay, {len(no_decay_params)} params without")
    return param_groups
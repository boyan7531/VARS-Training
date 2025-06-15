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
    'separate_weight_decay_params',
    'apply_gradient_clipping',
    'create_ema_model',
    'update_ema_model'
]

def get_model_config():
    """Get model configuration - placeholder function."""
    return {}


def get_optimizer(model, args):
    """Create optimizer based on arguments with proper weight decay handling and optimizations."""
    # Use weight decay parameter separation for better regularization
    param_groups = separate_weight_decay_params(model, args.weight_decay)
    
    fused_flag = getattr(args, 'fused_optim', False) and torch.cuda.is_available()
    if fused_flag:
        logger.info("Using fused optimizer kernels (CUDA)")
    
    optimizer = None  # Will hold the created optimizer instance

    if args.optimizer == 'adamw':
        # Adaptive betas for convergence
        beta1 = getattr(args, 'beta1', 0.9)
        beta2 = getattr(args, 'beta2', 0.999)
        if hasattr(args, 'backbone_type') and args.backbone_type.lower() == 'mvit':
            beta2 = getattr(args, 'beta2', 0.98)
            logger.info(f"Using MViT-optimized beta2: {beta2}")
        eps = getattr(args, 'eps', 1e-8)
        adamw_kwargs = dict(
            params=param_groups,
            lr=args.lr,
            betas=(beta1, beta2),
            eps=eps,
            amsgrad=getattr(args, 'amsgrad', False),
        )
        if 'fused' in optim.AdamW.__init__.__code__.co_varnames:
            adamw_kwargs['fused'] = fused_flag
        optimizer = optim.AdamW(**adamw_kwargs)
        logger.info(f"AdamW optimizer: lr={args.lr}, betas=({beta1}, {beta2}), eps={eps}")

    elif args.optimizer == 'adam':
        beta1 = getattr(args, 'beta1', 0.9)
        beta2 = getattr(args, 'beta2', 0.999)
        eps = getattr(args, 'eps', 1e-8)
        optimizer = optim.Adam(
            param_groups,
            lr=args.lr,
            betas=(beta1, beta2),
            eps=eps,
            amsgrad=getattr(args, 'amsgrad', False),
            fused=fused_flag if 'fused' in optim.Adam.__init__.__code__.co_varnames else False,
        )
        logger.info(f"Adam optimizer: lr={args.lr}, betas=({beta1}, {beta2}), eps={eps}")

    elif args.optimizer == 'sgd':
        momentum = getattr(args, 'momentum', 0.9)
        nesterov = getattr(args, 'nesterov', True)
        sgd_kwargs = dict(
            params=param_groups,
            lr=args.lr,
            momentum=momentum,
            nesterov=nesterov,
            dampening=getattr(args, 'dampening', 0),
            # Note: weight_decay is handled by parameter groups, not passed here
        )
        if 'fused' in optim.SGD.__init__.__code__.co_varnames:
            sgd_kwargs['fused'] = fused_flag
        optimizer = optim.SGD(**sgd_kwargs)
        logger.info(f"SGD optimizer: lr={args.lr}, momentum={momentum}, nesterov={nesterov}")

    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # === Lookahead ===
    if getattr(args, 'lookahead', False):
        try:
            from timm.optim import Lookahead  # type: ignore
            optimizer = Lookahead(
                optimizer,
                k=getattr(args, 'la_steps', 5),
                alpha=getattr(args, 'la_alpha', 0.5),
            )
            logger.info(
                f"Lookahead enabled (k={getattr(args, 'la_steps', 5)}, alpha={getattr(args, 'la_alpha', 0.5)})"
            )
        except ImportError:
            logger.warning(
                "timm not installed, Lookahead disabled. Install with: pip install timm>=0.6.0"
            )

    return optimizer


def apply_gradient_clipping(model, optimizer, max_norm=1.0, norm_type=2.0):
    """Apply gradient clipping for training stability."""
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=norm_type)
        return True
    return False


def create_ema_model(model, decay=0.9999):
    """Create Exponential Moving Average model for better performance."""
    try:
        import timm  # type: ignore
        from timm.utils import ModelEmaV2  # type: ignore
        return ModelEmaV2(model, decay=decay)
    except:
        # Fallback: simple EMA implementation
        logger.info("Using built-in EMA implementation")
        return SimpleEMA(model, decay=decay)


class SimpleEMA:
    """Simple EMA implementation without timm dependency."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data


def update_ema_model(ema_model, model, update_fn=None):
    """Update EMA model weights."""
    if ema_model is not None:
        if update_fn is not None:
            update_fn(ema_model, model)
        else:
            ema_model.update(model)


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
            enable_gradient_checkpointing=getattr(args, 'enable_gradient_checkpointing', False),
            enable_memory_optimization=getattr(args, 'enable_memory_optimization', True),
            dropout_rate=getattr(args, 'dropout_rate', 0.1)
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
            logger.error(f"Available MViT models: mvit_base_16, mvit_base_16x4, mvit_base_32x3")
        raise


def separate_weight_decay_params(model, weight_decay=0.01):
    """
    Separate model parameters for weight decay application.
    Apply weight decay to all parameters except bias and normalization layers.
    
    This follows best practices from modern vision transformers and CNNs where
    bias terms and normalization layer parameters should not be regularized.
    
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
    
    # Normalization layer types to exclude from weight decay
    norm_layer_keywords = {
        'bias',           # All bias terms
        'norm',           # Generic norm layers
        'bn',             # BatchNorm
        'batchnorm',      # BatchNorm variants
        'layernorm',      # LayerNorm
        'groupnorm',      # GroupNorm
        'instancenorm',   # InstanceNorm
        'ln',             # LayerNorm abbreviation
        'gn',             # GroupNorm abbreviation
        'in',             # InstanceNorm abbreviation (be careful with this one)
        'rmsnorm',        # RMSNorm (used in some transformers)
        'scale',          # Scale parameters in some norm layers
        'gamma',          # Gamma parameters in norm layers
        'beta',           # Beta parameters in norm layers
    }
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Convert name to lowercase for case-insensitive matching
        name_lower = name.lower()
        
        # Check if parameter should be excluded from weight decay
        should_exclude = False
        
        # Check for bias parameters
        if name.endswith('.bias'):
            should_exclude = True
        
        # Check for normalization layer parameters
        for keyword in norm_layer_keywords:
            if keyword in name_lower:
                # Special handling for 'in' to avoid false positives
                if keyword == 'in':
                    # Only match if it's a clear instance norm pattern
                    if ('instancenorm' in name_lower or 
                        name_lower.endswith('.in.weight') or 
                        name_lower.endswith('.in.bias')):
                        should_exclude = True
                        break
                else:
                    should_exclude = True
                    break
        
        if should_exclude:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    logger.info(f"Weight decay setup: {len(decay_params)} params with decay, {len(no_decay_params)} params without decay")
    return param_groups
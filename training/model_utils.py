"""
Model utilities, freezing strategies, and parameter management for Multi-Task Multi-View ResNet3D training.

This module now serves as a compatibility layer that imports from the new modular freezing package.
The original functionality has been split into multiple focused modules within the freezing package.
"""

# Import all freezing functionality from the new modular structure
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

# Re-export everything for backward compatibility
__all__ = [
    'get_model_config',
    'get_optimizer', 
    'get_scheduler',
    'setup_freezing_strategy',
    'create_model'
]

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
            disable_in_model_augmentation=args.disable_in_model_augmentation
        )
        
        # Get model info for logging
        model_info = model.get_model_info()
        
        # Log model architecture details
        logger.info(f"{args.backbone_type.upper()} components initialized successfully. Combined feature dim: {model_info.get('video_feature_dim', 0) + model_info.get('total_embedding_dim', 0)}")
        
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
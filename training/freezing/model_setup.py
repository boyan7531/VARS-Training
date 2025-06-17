"""
Model creation and freezing strategy setup utilities.

This module provides functions for creating models and setting up
different freezing strategies based on configuration arguments.
"""

import torch
import torch.nn as nn
import logging

from model import MultiTaskMultiViewResNet3D, ModelConfig
from .base_utils import freeze_backbone, log_trainable_parameters
from .smart_manager import SmartFreezingManager
from .gradient_guided_manager import GradientGuidedFreezingManager
from .advanced_manager import AdvancedFreezingManager
from .early_gradual_manager import EarlyGradualFreezingManager

# Import AdvancedFreezingManager dynamically to avoid circular imports
try:
    from .advanced_manager import AdvancedFreezingManager
except ImportError:
    AdvancedFreezingManager = None

logger = logging.getLogger(__name__)


def create_model(args, vocab_sizes, device, num_gpus=1):
    """Create and initialize the model with proper configuration."""
    
    # Model configuration
    model_config = ModelConfig(
        use_attention_aggregation=args.attention_aggregation,
        input_frames=args.frames_per_clip,
        input_height=args.img_height,
        input_width=args.img_width  # ResNet3D supports rectangular inputs
    )

    # Initialize model with proper configuration
    logger.info(f"Initializing ResNet3D model: {args.backbone_name}")
    model = MultiTaskMultiViewResNet3D.create_model(
        num_severity=6,  # 6 severity classes: "", 1.0, 2.0, 3.0, 4.0, 5.0
        num_action_type=10,  # 10 action types: "", Challenge, Dive, Dont know, Elbowing, High leg, Holding, Pushing, Standing tackling, Tackling
        vocab_sizes=vocab_sizes,
        backbone_name=args.backbone_name,
        config=model_config,
        use_augmentation=(not args.disable_in_model_augmentation),  # Control in-model augmentation
        disable_in_model_augmentation=args.disable_in_model_augmentation  # Pass the flag explicitly
    )
    # Keep master weights in fp32 for better precision during updates
    model.to(device, dtype=torch.float32)
    
    # Wrap model with DataParallel for multi-GPU
    if num_gpus > 1:
        model = nn.DataParallel(model)
        logger.info(f"Model wrapped with DataParallel for {num_gpus} GPUs")

    # Log model info - handle DataParallel wrapper
    try:
        # Get the actual model (unwrap DataParallel if needed)
        actual_model = model.module if hasattr(model, 'module') else model
        model_info = actual_model.get_model_info()
        logger.info(f"Model initialized - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Combined feature dimension: {model_info['video_feature_dim'] + model_info['total_embedding_dim']}")
    except Exception as e:
        logger.warning(f"Could not get model info: {e}")
        logger.info(f"Model initialized - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def setup_freezing_strategy(args, model):
    """Setup the appropriate freezing strategy based on configuration."""
    
    freezing_manager = None
    
    logger.info("=" * 60)
    logger.info("FREEZING STRATEGY CONFIGURATION")
    logger.info("=" * 60)
    
    if args.freezing_strategy == 'none':
        # No freezing - train all parameters from start
        logger.info("❄️  No parameter freezing - training all parameters from start")
        log_trainable_parameters(model)
        
    elif args.freezing_strategy == 'advanced':
        # Advanced multi-metric freezing strategy
        if AdvancedFreezingManager is None:
            logger.error("AdvancedFreezingManager not available. Please check the import.")
            raise ImportError("AdvancedFreezingManager could not be imported")
            
        logger.info("🚀 Using ADVANCED multi-metric freezing strategy")
        logger.info(f"   - Base importance threshold: {args.base_importance_threshold}")
        logger.info(f"   - Performance threshold: {args.performance_threshold}")
        logger.info(f"   - Analysis window: {args.analysis_window} epochs")
        logger.info(f"   - Rollback enabled: {args.enable_rollback}")
        logger.info(f"   - Dependency analysis: {args.enable_dependency_analysis}")
        logger.info(f"   - Gradient momentum: {args.gradient_momentum}")
        
        freezing_manager = AdvancedFreezingManager(
            model,
            base_importance_threshold=args.base_importance_threshold,
            performance_threshold=args.performance_threshold,
            max_layers_per_step=args.max_layers_per_step,
            warmup_epochs=args.layer_warmup_epochs,
            patience_epochs=args.unfreeze_patience,
            rollback_patience=args.rollback_patience,
            gradient_momentum=args.gradient_momentum,
            analysis_window=args.analysis_window,
            enable_rollback=args.enable_rollback,
            enable_dependency_analysis=args.enable_dependency_analysis
        )
        
        # Start with backbone frozen
        freezing_manager.freeze_all_backbone()
        log_trainable_parameters(model)
        
    elif args.freezing_strategy == 'gradient_guided':
        # Gradient-guided freezing strategy
        logger.info("🧠 Using gradient-guided freezing strategy")
        logger.info(f"   - Importance threshold: {args.importance_threshold}")
        logger.info(f"   - Warmup epochs: {args.layer_warmup_epochs}")
        logger.info(f"   - Unfreeze patience: {args.unfreeze_patience}")
        logger.info(f"   - Sampling epochs: {args.sampling_epochs}")
        
        freezing_manager = GradientGuidedFreezingManager(
            model,
            importance_threshold=args.importance_threshold,
            warmup_epochs=args.layer_warmup_epochs,
            unfreeze_patience=args.unfreeze_patience,
            max_layers_per_step=args.max_layers_per_step,
            sampling_epochs=args.sampling_epochs
        )
        
        # Start with backbone frozen
        freezing_manager.freeze_all_backbone()
        log_trainable_parameters(model)
        
    elif args.freezing_strategy in ['adaptive', 'progressive']:
        # Smart freezing with new strategies
        logger.info(f"🧠 Smart freezing strategy: {args.freezing_strategy.upper()}")
        
        if args.freezing_strategy == 'adaptive':
            logger.info(f"   - Patience: {args.adaptive_patience} epochs")
            logger.info(f"   - Min improvement: {args.adaptive_min_improvement}")
            
        freezing_manager = SmartFreezingManager(
            model, 
            strategy=args.freezing_strategy,
            patience=args.adaptive_patience,
            min_improvement=args.adaptive_min_improvement
        )
        
        # Start with backbone frozen
        freezing_manager.freeze_all_backbone()
        log_trainable_parameters(model)
        
        logger.info(f"[SMART] Smart freezing manager initialized")
        
    elif args.freezing_strategy == 'early_gradual':
        # Early gradual unfreezing strategy
        logger.info("⚡ Early gradual unfreezing strategy")
        logger.info(f"   - Freeze epochs: {args.early_gradual_freeze_epochs}")
        logger.info(f"   - Blocks per epoch: {args.early_gradual_blocks_per_epoch}")
        logger.info(f"   - Target ratio: {args.early_gradual_target_ratio*100:.0f}% of backbone")
        
        freezing_manager = EarlyGradualFreezingManager(
            model,
            freeze_epochs=args.early_gradual_freeze_epochs,
            blocks_per_epoch=args.early_gradual_blocks_per_epoch,
            target_ratio=args.early_gradual_target_ratio
        )
        
        # Start with backbone frozen
        freezing_manager.freeze_all_backbone()
        log_trainable_parameters(model)
        
        logger.info(f"[EARLY_GRADUAL] Early gradual freezing manager initialized")
        
    elif args.gradual_finetuning:
        # Original gradual fine-tuning (fixed strategy)
        logger.info("🔧 Original gradual fine-tuning strategy (fixed phases)")
        logger.info(f"   - Phase 1: {args.phase1_epochs} epochs (heads only)")
        logger.info(f"   - Phase 2: {args.phase2_epochs} epochs (gradual unfreezing)")
        
        # Start with backbone frozen (Phase 1)
        freeze_backbone(model)
        log_trainable_parameters(model)
        
    else:
        # Standard training - all parameters trainable
        logger.info("SETUP: Standard training - all parameters trainable from start")
        log_trainable_parameters(model)
    
    logger.info("=" * 60)
    
    return freezing_manager 
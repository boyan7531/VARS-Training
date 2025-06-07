"""
Model utilities, freezing strategies, and parameter management for Multi-Task Multi-View ResNet3D training.

This module handles model creation, parameter freezing/unfreezing, and advanced freezing strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import numpy as np
import logging

from model import MultiTaskMultiViewResNet3D, ModelConfig

logger = logging.getLogger(__name__)


def calculate_class_weights(dataset, num_classes, device, weighting_strategy='balanced_capped', max_weight_ratio=10.0):
    """
    Calculate class weights for loss balancing with multiple safe strategies.
    
    Args:
        dataset: Training dataset
        num_classes: Number of classes
        device: Device to put weights on
        weighting_strategy: Strategy for calculating weights
            - 'none': No weighting (all weights = 1.0)
            - 'balanced_capped': Inverse frequency with capped ratios (RECOMMENDED)
            - 'sqrt': Square root of inverse frequency (gentler)
            - 'log': Logarithmic weighting (very gentle)
            - 'effective_number': Uses effective number of samples
        max_weight_ratio: Maximum ratio between highest and lowest weight
    
    Returns:
        class_weights tensor
    """
    class_counts = torch.zeros(num_classes)
    
    for action in dataset.actions:
        severity_label = action['label_severity']
        if 0 <= severity_label < num_classes:
            class_counts[severity_label] += 1
    
    # Avoid division by zero
    class_counts[class_counts == 0] = 1
    
    if weighting_strategy == 'none':
        class_weights = torch.ones(num_classes)
        logger.info("Using no class weighting (all weights = 1.0)")
        
    elif weighting_strategy == 'balanced_capped':
        # Inverse frequency with capping - MUCH SAFER!
        total_samples = class_counts.sum()
        class_weights = total_samples / (num_classes * class_counts)
        
        # Cap the weights to prevent extreme ratios
        max_weight = class_weights.min() * max_weight_ratio
        class_weights = torch.clamp(class_weights, max=max_weight)
        
        # Normalize so minimum weight is 1.0
        class_weights = class_weights / class_weights.min()
        
        logger.info(f"Balanced capped weighting (max ratio: {max_weight_ratio})")
        
    elif weighting_strategy == 'sqrt':
        # Square root weighting - gentler than inverse frequency
        total_samples = class_counts.sum()
        class_weights = torch.sqrt(total_samples / (num_classes * class_counts))
        class_weights = class_weights / class_weights.min()
        
        logger.info("Square root weighting (gentler)")
        
    elif weighting_strategy == 'log':
        # Logarithmic weighting - very gentle
        total_samples = class_counts.sum()
        raw_weights = total_samples / (num_classes * class_counts)
        class_weights = torch.log(raw_weights + 1)  # +1 to avoid log(0)
        class_weights = class_weights / class_weights.min()
        
        logger.info("Logarithmic weighting (very gentle)")
        
    elif weighting_strategy == 'effective_number':
        # Effective number of samples - sophisticated approach
        beta = 0.999  # Hyperparameter
        effective_num = 1.0 - torch.pow(beta, class_counts.float())
        class_weights = (1.0 - beta) / effective_num
        class_weights = class_weights / class_weights.min()
        
        logger.info(f"Effective number weighting (beta={beta})")
        
    else:
        raise ValueError(f"Unknown weighting strategy: {weighting_strategy}")
    
    # Log class distribution and weights
    logger.info("Class distribution and weights:")
    for i in range(num_classes):
        if class_counts[i] > 0:
            logger.info(f"  Class {i}: {int(class_counts[i])} samples → Weight: {class_weights[i]:.2f}")
    
    max_ratio = (class_weights.max() / class_weights.min()).item()
    logger.info(f"Weight ratio (max/min): {max_ratio:.1f}")
    
    if max_ratio > 50:
        logger.warning(f"⚠️  Large weight ratio ({max_ratio:.1f}) may cause training instability!")
        logger.warning("   Consider using 'sqrt' or 'log' weighting strategy")
    
    return class_weights.to(device)


def freeze_backbone(model):
    """Freeze all backbone parameters."""
    # Handle DataParallel wrapper
    actual_model = model.module if hasattr(model, 'module') else model
    
    for param in actual_model.backbone.parameters():
        param.requires_grad = False
    
    logger.info("[FREEZE] Backbone frozen - only training classification heads")


def unfreeze_backbone_gradually(model, num_blocks_to_unfreeze=2):
    """Gradually unfreeze the last N residual blocks of the backbone."""
    # Handle DataParallel wrapper
    actual_model = model.module if hasattr(model, 'module') else model
    
    # Get the ResNet3D backbone
    backbone = actual_model.backbone.backbone
    
    # For ResNet architectures, we want to unfreeze the last few layers
    # ResNet3D structure: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool
    layers_to_unfreeze = []
    
    if hasattr(backbone, 'layer4'):
        layers_to_unfreeze.append(backbone.layer4)
    if hasattr(backbone, 'layer3') and num_blocks_to_unfreeze > 1:
        layers_to_unfreeze.append(backbone.layer3)
    if hasattr(backbone, 'layer2') and num_blocks_to_unfreeze > 2:
        layers_to_unfreeze.append(backbone.layer2)
    
    unfrozen_params = 0
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()
    
    logger.info(f"[UNFREEZE] Unfroze last {len(layers_to_unfreeze)} backbone layers ({unfrozen_params:,} parameters)")


def setup_discriminative_optimizer(model, head_lr, backbone_lr):
    """Setup optimizer with discriminative learning rates for different model parts."""
    # Handle DataParallel wrapper
    actual_model = model.module if hasattr(model, 'module') else model
    
    param_groups = []
    
    # Classification heads with higher learning rate
    head_params = []
    head_params.extend(actual_model.severity_head.parameters())
    head_params.extend(actual_model.action_type_head.parameters())
    
    # Add embedding and aggregation parameters to head group (they're task-specific)
    head_params.extend(actual_model.embedding_manager.parameters())
    head_params.extend(actual_model.view_aggregator.parameters())
    
    param_groups.append({
        'params': head_params,
        'lr': head_lr,
        'name': 'heads'
    })
    
    # Backbone parameters with lower learning rate (only unfrozen ones)
    backbone_params = [p for p in actual_model.backbone.parameters() if p.requires_grad]
    
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': backbone_lr,
            'name': 'backbone'
        })
    
    logger.info(f"[OPTIM] Discriminative LR setup: Heads={head_lr:.1e}, Backbone={backbone_lr:.1e}")
    return param_groups


def get_phase_info(epoch, phase1_epochs, total_epochs):
    """Determine current training phase."""
    if epoch < phase1_epochs:
        return 1, f"Phase 1: Head-only training"
    else:
        return 2, f"Phase 2: Gradual unfreezing"


def log_trainable_parameters(model):
    """Log the number of trainable parameters."""
    # Handle DataParallel wrapper
    actual_model = model.module if hasattr(model, 'module') else model
    
    total_params = sum(p.numel() for p in actual_model.parameters())
    trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    logger.info(f"[PARAMS] Total={total_params:,}, Trainable={trainable_params:,}, Frozen={frozen_params:,}")
    return trainable_params, total_params


# Enhanced freezing strategies with multiple approaches
class SmartFreezingManager:
    """
    Advanced parameter freezing manager with multiple strategies:
    1. Adaptive unfreezing based on validation performance
    2. Layer-wise progressive unfreezing 
    3. Gradient-based unfreezing decisions
    4. Comparison mode to validate freezing benefits
    """
    def __init__(self, model, strategy='adaptive', patience=3, min_improvement=0.001):
        self.model = model
        self.strategy = strategy
        self.patience = patience
        self.min_improvement = min_improvement
        self.no_improvement_count = 0
        self.best_val_acc = 0.0
        self.unfrozen_layers = []
        self.gradient_history = defaultdict(list)
        
    def get_backbone_layers(self):
        """Get ordered list of backbone layers for progressive unfreezing."""
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        backbone = actual_model.backbone.backbone
        
        layers = []
        # ResNet3D layer order: conv1, bn1, layer1, layer2, layer3, layer4
        if hasattr(backbone, 'layer4'): layers.append(('layer4', backbone.layer4))
        if hasattr(backbone, 'layer3'): layers.append(('layer3', backbone.layer3))
        if hasattr(backbone, 'layer2'): layers.append(('layer2', backbone.layer2))
        if hasattr(backbone, 'layer1'): layers.append(('layer1', backbone.layer1))
        if hasattr(backbone, 'conv1'): layers.append(('conv1', backbone.conv1))
        
        return layers
    
    def freeze_all_backbone(self):
        """Freeze all backbone parameters."""
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        frozen_params = 0
        
        for param in actual_model.backbone.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
            
        logger.info(f"[SMART_FREEZE] Froze entire backbone ({frozen_params:,} parameters)")
        return frozen_params
    
    def unfreeze_layer(self, layer_name, layer):
        """Unfreeze a specific layer with warmup."""
        if layer_name in self.unfrozen_layers:
            return 0
            
        unfrozen_params = 0
        for param in layer.parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()
            
        self.unfrozen_layers.append(layer_name)
        logger.info(f"[SMART_UNFREEZE] Unfroze {layer_name} ({unfrozen_params:,} parameters)")
        return unfrozen_params
    
    def should_unfreeze_next_layer(self, current_val_acc):
        """Decide whether to unfreeze the next layer based on validation performance."""
        if self.strategy == 'fixed':
            return False  # Let the original logic handle this
            
        elif self.strategy == 'adaptive':
            # Unfreeze when validation improvement plateaus
            improvement = current_val_acc - self.best_val_acc
            
            if improvement > self.min_improvement:
                self.best_val_acc = current_val_acc
                self.no_improvement_count = 0
                return False
            else:
                self.no_improvement_count += 1
                if self.no_improvement_count >= self.patience:
                    self.no_improvement_count = 0  # Reset counter
                    return True
                return False
                
        elif self.strategy == 'progressive':
            # Unfreeze one layer every N epochs
            return len(self.unfrozen_layers) < 4  # Max 4 layers
            
        return False
    
    def monitor_gradients(self, epoch):
        """Monitor gradient norms to identify layers that need unfreezing."""
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Monitor head gradients
        head_grad_norm = 0
        for name, param in actual_model.named_parameters():
            if 'head' in name and param.grad is not None:
                head_grad_norm += param.grad.norm().item()
        
        self.gradient_history['heads'].append(head_grad_norm)
        
        # Log gradient trends
        if epoch % 5 == 0 and len(self.gradient_history['heads']) > 1:
            recent_avg = np.mean(self.gradient_history['heads'][-3:])
            logger.info(f"[GRAD_MONITOR] Head gradient norm (recent avg): {recent_avg:.6f}")
    
    def get_discriminative_lr_groups(self, head_lr, backbone_lr_base):
        """Create parameter groups with exponentially decaying learning rates."""
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        param_groups = []
        
        # Classification heads (highest LR)
        head_params = []
        head_params.extend(actual_model.severity_head.parameters())
        head_params.extend(actual_model.action_type_head.parameters())
        head_params.extend(actual_model.embedding_manager.parameters())
        head_params.extend(actual_model.view_aggregator.parameters())
        
        param_groups.append({
            'params': head_params,
            'lr': head_lr,
            'name': 'heads'
        })
        
        # Backbone layers (exponentially decreasing LR)
        layers = self.get_backbone_layers()
        for i, (layer_name, layer) in enumerate(layers):
            layer_params = [p for p in layer.parameters() if p.requires_grad]
            if layer_params:
                # Exponential decay: each layer gets 0.5x the previous layer's LR
                layer_lr = backbone_lr_base * (0.5 ** i)
                param_groups.append({
                    'params': layer_params,
                    'lr': layer_lr,
                    'name': f'backbone_{layer_name}'
                })
                
        return param_groups
    
    def adaptive_unfreeze_step(self, current_val_acc, epoch):
        """Perform one step of adaptive unfreezing."""
        if not self.should_unfreeze_next_layer(current_val_acc):
            return False
            
        layers = self.get_backbone_layers()
        for layer_name, layer in layers:
            if layer_name not in self.unfrozen_layers:
                self.unfreeze_layer(layer_name, layer)
                logger.info(f"[ADAPTIVE] Unfroze {layer_name} at epoch {epoch+1} due to validation plateau")
                return True
        
        logger.info(f"[ADAPTIVE] All layers already unfrozen")
        return False


def create_model(args, vocab_sizes, device, num_gpus=1):
    """
    Create a multi-task, multi-view ResNet3D model.
    
    This function creates the model with Sports-1M pretrained weights for better transfer learning.
    
    Args:
        args: Command-line arguments
        vocab_sizes: Dictionary of vocabulary sizes for categorical features
        device: Device to place the model on
        num_gpus: Number of GPUs to use
        
    Returns:
        model: The initialized model
    """
    try:
        from model import create_model as model_create_model
        
        # Use the enhanced create_model function from model.loader that loads Sports-1M weights
        model = model_create_model(args, vocab_sizes, device, num_gpus)
        
        # Freeze backbone initially if using gradual fine-tuning
        if args.gradual_finetuning and args.freezing_strategy == 'fixed':
            logger.info("[PHASE1] Freezing backbone for Phase 1 (training heads only)")
            freeze_backbone(model)
            log_trainable_parameters(model)
        
        return model
        
    except ImportError:
        # Fallback to old implementation if import fails
        logger.warning("Could not import enhanced create_model. Using legacy implementation.")
        return _legacy_create_model(args, vocab_sizes, device, num_gpus)

def _legacy_create_model(args, vocab_sizes, device, num_gpus=1):
    """Legacy model creation for backward compatibility."""
    # Create model with the specified configuration
    logger.info(f"Creating model with backbone {args.backbone_name}")
    
    model = MultiTaskMultiViewResNet3D.create_model(
        num_severity=6,  # Number of severity classes including None (0-5)
        num_action_type=10,  # Number of action type classes including None (0-9)
        vocab_sizes=vocab_sizes,
        backbone_name=args.backbone_name,
        use_attention_aggregation=args.attention_aggregation,
        use_augmentation=not args.disable_in_model_augmentation,
        max_views=args.max_views,
        dropout_rate=args.dropout_rate,
    )
    
    # Apply DataParallel if using multiple GPUs
    if num_gpus > 1 and not args.force_batch_size:
        model = torch.nn.DataParallel(model)
        logger.info(f"Using DataParallel across {num_gpus} GPUs")
    
    # Move model to the specified device
    model = model.to(device)
    logger.info(f"Model moved to {device}")
    
    # Freeze backbone initially if using gradual fine-tuning
    if args.gradual_finetuning and args.freezing_strategy == 'fixed':
        logger.info("[PHASE1] Freezing backbone for Phase 1 (training heads only)")
        freeze_backbone(model)
        log_trainable_parameters(model)
    
    return model


def setup_freezing_strategy(args, model):
    """Setup the appropriate freezing strategy based on configuration."""
    
    freezing_manager = None
    
    logger.info("=" * 60)
    logger.info("🧊 FREEZING STRATEGY CONFIGURATION")
    logger.info("=" * 60)
    
    if args.freezing_strategy == 'none':
        # No freezing - train all parameters from start
        logger.info("❄️  No parameter freezing - training all parameters from start")
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
        logger.info("🔧 Standard training - all parameters trainable from start")
        log_trainable_parameters(model)
    
    logger.info("=" * 60)
    
    return freezing_manager 
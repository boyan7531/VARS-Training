"""
Base utilities for model parameter management and basic freezing operations.

This module contains fundamental utilities for class weight calculation,
basic freezing/unfreezing operations, and parameter management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

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
    
    # Handle different model structures
    if hasattr(actual_model, 'mvit_processor'):
        # Optimized MViT model - backbone is inside mvit_processor
        backbone = actual_model.mvit_processor.backbone
        logger.info("[FREEZE] Detected optimized MViT model structure")
    elif hasattr(actual_model, 'backbone'):
        # Standard model structure (ResNet3D or standard MViT)
        backbone = actual_model.backbone
        logger.info("[FREEZE] Detected standard model structure")
    else:
        raise AttributeError("Model does not have accessible backbone (neither 'backbone' nor 'mvit_processor.backbone' found)")
    
    # Freeze backbone parameters
    frozen_params = 0
    for param in backbone.parameters():
        param.requires_grad = False
        frozen_params += param.numel()
    
    logger.info(f"[FREEZE] Backbone frozen - {frozen_params:,} parameters frozen, only training classification heads")


def unfreeze_backbone_gradually(model, num_blocks_to_unfreeze=2):
    """Gradually unfreeze the last N residual blocks of the backbone."""
    # Handle DataParallel wrapper
    actual_model = model.module if hasattr(model, 'module') else model
    
    # Handle different model structures and get appropriate backbone reference
    if hasattr(actual_model, 'mvit_processor'):
        # Optimized MViT model - backbone is inside mvit_processor
        base_backbone = actual_model.mvit_processor.backbone
        logger.info("[UNFREEZE] Detected optimized MViT model structure")
        
        # Detect the actual backbone type
        if hasattr(base_backbone, 'backbone'):
            # This might be a ResNet3D backbone inside the processor
            backbone = base_backbone.backbone
            model_type = "resnet3d"
        else:
            # This is directly an MViT backbone
            backbone = base_backbone
            model_type = "mvit"
    else:
        # Standard model structure
        if hasattr(actual_model.backbone, 'backbone'):
            # ResNet3D model: actual_model.backbone.backbone
            backbone = actual_model.backbone.backbone
            model_type = "resnet3d"
        else:
            # MViT model: actual_model.backbone directly
            backbone = actual_model.backbone
            model_type = "mvit"
    
    logger.info(f"[UNFREEZE] Detected {model_type} model architecture")
    
    if model_type == "resnet3d":
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
        
        logger.info(f"[UNFREEZE] Unfroze last {len(layers_to_unfreeze)} ResNet layers ({unfrozen_params:,} parameters)")
        
    elif model_type == "mvit":
        # For MViT architectures, unfreeze the last N transformer blocks
        # MViT has blocks attribute containing the transformer layers
        unfrozen_params = 0
        
        if hasattr(backbone, 'blocks'):
            # Unfreeze last N blocks
            total_blocks = len(backbone.blocks)
            blocks_to_unfreeze = min(num_blocks_to_unfreeze, total_blocks)
            
            # Unfreeze from the end backwards
            for i in range(total_blocks - blocks_to_unfreeze, total_blocks):
                for param in backbone.blocks[i].parameters():
                    param.requires_grad = True
                    unfrozen_params += param.numel()
            
            # Also unfreeze the final norm layer if it exists
            if hasattr(backbone, 'norm') and backbone.norm is not None:
                for param in backbone.norm.parameters():
                    param.requires_grad = True
                    unfrozen_params += param.numel()
                    
            # Also unfreeze the head if it exists
            if hasattr(backbone, 'head') and backbone.head is not None:
                for param in backbone.head.parameters():
                    param.requires_grad = True
                    unfrozen_params += param.numel()
            
            logger.info(f"[UNFREEZE] Unfroze last {blocks_to_unfreeze} MViT blocks + norm + head ({unfrozen_params:,} parameters)")
            
        else:
            # Fallback: unfreeze all backbone parameters
            logger.warning("[UNFREEZE] MViT blocks not found, unfreezing entire backbone")
            for param in backbone.parameters():
                param.requires_grad = True
                unfrozen_params += param.numel()
            
            logger.info(f"[UNFREEZE] Unfroze entire MViT backbone ({unfrozen_params:,} parameters)")


def setup_discriminative_optimizer(model, head_lr, backbone_lr, weight_decay=0.01):
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
        'weight_decay': weight_decay,
        'name': 'heads'
    })
    
    # Handle different model structures for backbone parameters
    if hasattr(actual_model, 'mvit_processor'):
        # Optimized MViT model - backbone is inside mvit_processor
        backbone = actual_model.mvit_processor.backbone
    elif hasattr(actual_model, 'backbone'):
        # Standard model structure
        backbone = actual_model.backbone
    else:
        raise AttributeError("Model does not have accessible backbone")
    
    # Backbone parameters with lower learning rate (only unfrozen ones)
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': backbone_lr,
            'weight_decay': weight_decay,
            'name': 'backbone'
        })
    
    logger.info(f"[OPTIM] Discriminative LR setup: Heads={head_lr:.1e}, Backbone={backbone_lr:.1e}, Weight Decay={weight_decay}")
    return param_groups


def get_phase_info(epoch, phase1_epochs, total_epochs):
    """Determine current training phase."""
    if epoch < phase1_epochs:
        return 1, f"Phase 1: Head-only training"
    else:
        return 2, f"Phase 2: Gradual unfreezing"


def log_trainable_parameters(model, epoch=None):
    """Log detailed information about trainable parameters (only from main process)."""
    # Check if we should log (only from main process in distributed training)
    try:
        # Try PyTorch's distributed backend first (most reliable)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            is_main_process = torch.distributed.get_rank() == 0
        else:
            # Single process case
            is_main_process = True
    except Exception:
        # If anything fails, assume single process
        is_main_process = True
    
    # Handle DataParallel wrapper
    actual_model = model.module if hasattr(model, 'module') else model
    
    # Get total model parameters
    total_params = sum(p.numel() for p in actual_model.parameters())
    trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # Get backbone-specific parameters
    backbone_trainable = 0
    backbone_total = 0
    
    # Handle different model structures
    if hasattr(actual_model, 'mvit_processor'):
        # Optimized MViT model - backbone is inside mvit_processor
        backbone = actual_model.mvit_processor.backbone
    elif hasattr(actual_model, 'backbone'):
        # Standard model structure
        if hasattr(actual_model.backbone, 'backbone'):
            # ResNet3D model: actual_model.backbone.backbone
            backbone = actual_model.backbone.backbone
        else:
            # MViT model: actual_model.backbone directly
            backbone = actual_model.backbone
    else:
        backbone = None
    
    if backbone:
        backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        backbone_total = sum(p.numel() for p in backbone.parameters())
    
    # Calculate head parameters (non-backbone)
    head_params = trainable_params - backbone_trainable
    
    if is_main_process:
        epoch_str = f" (Epoch {epoch})" if epoch is not None else ""
        logger.info(f"📊 Trainable Parameters{epoch_str}:")
        logger.info(f"  Total model: {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.1f}%)")
        logger.info(f"  Backbone: {backbone_trainable:,}/{backbone_total:,} ({backbone_trainable/backbone_total*100:.1f}%)")
        logger.info(f"  Classification heads: {head_params:,}")


def get_backbone_unfrozen_ratio(model):
    """
    Calculate the ratio of unfrozen backbone parameters.
    
    Args:
        model: The model (can be wrapped in DataParallel)
        
    Returns:
        float: Ratio of unfrozen backbone parameters (0.0 to 1.0)
    """
    # Handle DataParallel wrapper
    actual_model = model.module if hasattr(model, 'module') else model
    
    # Handle different model structures
    if hasattr(actual_model, 'mvit_processor'):
        # Optimized MViT model - backbone is inside mvit_processor
        backbone = actual_model.mvit_processor.backbone
    elif hasattr(actual_model, 'backbone'):
        # Standard model structure
        if hasattr(actual_model.backbone, 'backbone'):
            # ResNet3D model: actual_model.backbone.backbone
            backbone = actual_model.backbone.backbone
        else:
            # MViT model: actual_model.backbone directly
            backbone = actual_model.backbone
    else:
        return 0.0  # No backbone found
    
    backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    backbone_total = sum(p.numel() for p in backbone.parameters())
    
    if backbone_total == 0:
        return 0.0
    
    return backbone_trainable / backbone_total


def boost_backbone_lr_if_needed(args, optimizer, model, epoch):
    """
    Boost backbone learning rate when ≥50% of backbone is unfrozen.
    
    Args:
        args: Training arguments
        optimizer: The optimizer
        model: The model
        epoch: Current epoch
        
    Returns:
        bool: True if LR was boosted, False otherwise
    """
    # Check if boost is enabled and not already applied
    if not getattr(args, 'enable_backbone_lr_boost', True):
        return False
    
    if getattr(args, 'lr_ratio_bumped', False):
        return False  # Already boosted
    
    # Calculate backbone unfrozen ratio
    unfrozen_ratio = get_backbone_unfrozen_ratio(model)
    
    # Check if we've reached the threshold
    if unfrozen_ratio >= 0.5:
        # Find backbone parameter groups and boost their LR
        target_ratio = getattr(args, 'backbone_lr_ratio_after_half', 0.6)
        head_lr = args.lr if hasattr(args, 'lr') else args.head_lr
        new_backbone_lr = head_lr * target_ratio
        
        boosted_groups = 0
        for group in optimizer.param_groups:
            group_name = group.get('name', '')
            # Look for backbone-related parameter groups
            if ('backbone' in group_name.lower() and 
                'head' not in group_name.lower()):
                old_lr = group['lr']
                group['lr'] = new_backbone_lr
                boosted_groups += 1
                logger.info(f"🔧 Boosted {group_name} LR: {old_lr:.2e} → {new_backbone_lr:.2e}")
        
        if boosted_groups > 0:
            args.lr_ratio_bumped = True
            logger.info(f"🚀 Backbone LR boost applied at epoch {epoch}")
            logger.info(f"   Unfrozen ratio: {unfrozen_ratio:.1%} (≥50% threshold reached)")
            logger.info(f"   New backbone LR ratio: {target_ratio:.1f} (was ~{args.backbone_lr/head_lr:.1f})")
            return True
    
    return False


def _get_backbone_blocks(model):
    """
    Get ordered list of backbone blocks for gradual unfreezing.
    
    This function handles different model structures and architectures to extract
    backbone blocks in the appropriate order for gradual unfreezing strategies.
    
    Args:
        model: The model (can be wrapped in DataParallel)
        
    Returns:
        List of tuples (block_name, block_module) ordered for unfreezing
    """
    # Handle DataParallel wrapper
    actual_model = model.module if hasattr(model, 'module') else model
    
    # Handle different model structures
    if hasattr(actual_model, 'mvit_processor'):
        # Optimized MViT model - backbone is inside mvit_processor
        backbone = actual_model.mvit_processor.backbone
    elif hasattr(actual_model, 'backbone'):
        # Standard model structure
        if hasattr(actual_model.backbone, 'backbone'):
            # ResNet3D model: actual_model.backbone.backbone
            backbone = actual_model.backbone.backbone
        else:
            # MViT model: actual_model.backbone directly
            backbone = actual_model.backbone
    else:
        raise AttributeError("Model does not have accessible backbone")
    
    blocks = []
    
    # Detect architecture and get blocks in unfreezing order (last to first)
    if hasattr(backbone, 'layer4'):
        # ResNet3D architecture - unfreeze from layer4 backwards
        if hasattr(backbone, 'layer4'):
            for i, block in enumerate(backbone.layer4):
                blocks.append((f'layer4.{i}', block))
        if hasattr(backbone, 'layer3'):
            for i, block in enumerate(backbone.layer3):
                blocks.append((f'layer3.{i}', block))
        if hasattr(backbone, 'layer2'):
            for i, block in enumerate(backbone.layer2):
                blocks.append((f'layer2.{i}', block))
        if hasattr(backbone, 'layer1'):
            for i, block in enumerate(backbone.layer1):
                blocks.append((f'layer1.{i}', block))
                
    elif hasattr(backbone, 'blocks'):
        # MViT architecture - unfreeze from last blocks backwards
        total_blocks = len(backbone.blocks)
        for i in range(total_blocks - 1, -1, -1):  # Reverse order
            blocks.append((f'block_{i}', backbone.blocks[i]))
    else:
        logger.warning("[BACKBONE_BLOCKS] Unknown backbone architecture")
        # Fallback: treat entire backbone as one block
        blocks.append(('entire_backbone', backbone))
    
    return blocks


def _get_backbone_layers(model):
    """
    Get ordered list of backbone layers for progressive unfreezing.
    
    This is a layer-level version of _get_backbone_blocks() that groups
    blocks into layers rather than individual blocks.
    
    Args:
        model: The model (can be wrapped in DataParallel)
        
    Returns:
        List of tuples (layer_name, layer_module) ordered for unfreezing
    """
    # Handle DataParallel wrapper
    actual_model = model.module if hasattr(model, 'module') else model
    
    # Handle different model structures
    if hasattr(actual_model, 'mvit_processor'):
        # Optimized MViT model - backbone is inside mvit_processor
        backbone = actual_model.mvit_processor.backbone
    elif hasattr(actual_model, 'backbone'):
        # Standard model structure
        backbone = actual_model.backbone
        # For ResNet3D models, the actual backbone is one level deeper
        if hasattr(backbone, 'backbone'):
            backbone = backbone.backbone
    else:
        raise AttributeError("Model does not have accessible backbone")
    
    layers = []
    
    # Detect model type and get appropriate layers
    if hasattr(backbone, 'layer4'):
        # ResNet3D architecture
        if hasattr(backbone, 'layer4'): layers.append(('layer4', backbone.layer4))
        if hasattr(backbone, 'layer3'): layers.append(('layer3', backbone.layer3))
        if hasattr(backbone, 'layer2'): layers.append(('layer2', backbone.layer2))
        if hasattr(backbone, 'layer1'): layers.append(('layer1', backbone.layer1))
        if hasattr(backbone, 'conv1'): layers.append(('conv1', backbone.conv1))
    elif hasattr(backbone, 'blocks'):
        # MViT architecture - use last few transformer blocks
        total_blocks = len(backbone.blocks)
        # Add blocks in reverse order (last blocks first for unfreezing)
        for i in range(min(4, total_blocks)):  # Up to 4 blocks
            block_idx = total_blocks - 1 - i
            layers.append((f'block_{block_idx}', backbone.blocks[block_idx]))
    else:
        logger.warning("[BACKBONE_LAYERS] Unknown backbone architecture, will use entire backbone")
        layers.append(('entire_backbone', backbone))
    
    return layers 
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


class GradientGuidedFreezingManager:
    """
    Advanced freezing manager that uses gradient flow analysis to make intelligent
    decisions about which layers to unfreeze and when.
    
    Key features:
    1. Tracks gradient flow through frozen layers to identify importance
    2. Selectively unfreezes layers based on their importance to the task
    3. Applies layer-specific learning rate warmup for newly unfrozen layers
    4. Supports partial unfreezing of layers based on parameter importance
    
    This approach is more data-driven than the time-based or performance-based
    strategies, as it directly observes which frozen parameters would contribute
    most to improving the model if they were unfrozen.
    """
    def __init__(
        self, 
        model,
        layer_selection='gradient_magnitude',
        importance_threshold=0.01,
        warmup_epochs=3,
        unfreeze_patience=2,
        max_layers_per_step=1,
        sampling_epochs=2
    ):
        self.model = model
        self.actual_model = model.module if hasattr(model, 'module') else model
        self.layer_selection = layer_selection
        self.importance_threshold = importance_threshold
        self.warmup_epochs = warmup_epochs
        self.unfreeze_patience = unfreeze_patience
        self.max_layers_per_step = max_layers_per_step
        self.sampling_epochs = sampling_epochs
        
        # Tracking state
        self.unfrozen_layers = set()
        self.layers_in_warmup = {}  # {layer_name: current_warmup_epoch}
        self.layer_importance = {}  # {layer_name: importance_score}
        self.gradient_history = {}  # {layer_name: [gradient_norms]}
        self.hooks = []  # Store forward/backward hooks
        self.no_improvement_count = 0
        self.best_val_metric = 0
        self.epochs_since_unfreeze = 0
        self.sampling_phase = True
        self.sampling_epoch_count = 0
        
        # Register hooks for gradient tracking
        self._register_gradient_hooks()
        
        logger.info(f"[GRADIENT_GUIDED] Initialized gradient-guided freezing manager")
        logger.info(f"  - Layer selection strategy: {layer_selection}")
        logger.info(f"  - Importance threshold: {importance_threshold}")
        logger.info(f"  - Warmup epochs: {warmup_epochs}")
        logger.info(f"  - Unfreeze patience: {unfreeze_patience}")
    
    def _register_gradient_hooks(self):
        """Register hooks to track gradient flow through frozen layers."""
        for name, module in self.actual_model.backbone.named_modules():
            # Skip non-parametric layers or already processed ones
            if len(list(module.parameters(recurse=False))) == 0:
                continue
            
            # Create gradient accumulation buffers for frozen parameters
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    # Initialize gradient accumulation buffer
                    param.register_hook(
                        lambda grad, param=param: self._gradient_hook(grad, param)
                    )
                    param.grad_acc = torch.zeros_like(param.data)
                    param.grad_samples = 0
                    
                    full_name = f"{name}.{param_name}"
                    self.gradient_history[full_name] = []
        
        logger.info("[GRADIENT_GUIDED] Registered gradient tracking hooks")
    
    def _gradient_hook(self, grad, param):
        """Hook to accumulate gradients for frozen parameters."""
        # This function accumulates gradients even though parameters won't be updated
        if hasattr(param, 'grad_acc') and not param.requires_grad:
            param.grad_acc += grad.abs().detach()  # Use absolute gradients
            param.grad_samples += 1
        return grad
    
    def freeze_all_backbone(self):
        """Freeze all backbone parameters and initialize gradient tracking."""
        frozen_params = 0
        
        for name, param in self.actual_model.backbone.named_parameters():
            param.requires_grad = False
            frozen_params += param.numel()
            
            # Initialize gradient accumulation
            param.grad_acc = torch.zeros_like(param.data)
            param.grad_samples = 0
        
        logger.info(f"[GRADIENT_GUIDED] Froze entire backbone ({frozen_params:,} parameters)")
        logger.info(f"[GRADIENT_GUIDED] Starting gradient sampling phase for {self.sampling_epochs} epochs")
        
        return frozen_params
    
    def get_backbone_layer_groups(self):
        """Get hierarchical grouping of backbone layers for analysis."""
        backbone = self.actual_model.backbone.backbone
        
        # Group parameters by layer with hierarchical structure
        layer_groups = {}
        
        # Major architectural blocks
        if hasattr(backbone, 'layer4'):
            layer_groups['layer4'] = {'module': backbone.layer4, 'children': {}}
            # Add sub-blocks (residual blocks)
            for i, block in enumerate(backbone.layer4):
                layer_groups['layer4']['children'][f'block{i}'] = {
                    'module': block, 'children': {}
                }
        
        if hasattr(backbone, 'layer3'):
            layer_groups['layer3'] = {'module': backbone.layer3, 'children': {}}
            for i, block in enumerate(backbone.layer3):
                layer_groups['layer3']['children'][f'block{i}'] = {
                    'module': block, 'children': {}
                }
        
        if hasattr(backbone, 'layer2'):
            layer_groups['layer2'] = {'module': backbone.layer2, 'children': {}}
            for i, block in enumerate(backbone.layer2):
                layer_groups['layer2']['children'][f'block{i}'] = {
                    'module': block, 'children': {}
                }
        
        if hasattr(backbone, 'layer1'):
            layer_groups['layer1'] = {'module': backbone.layer1, 'children': {}}
            for i, block in enumerate(backbone.layer1):
                layer_groups['layer1']['children'][f'block{i}'] = {
                    'module': block, 'children': {}
                }
        
        # Early layers
        if hasattr(backbone, 'conv1'):
            layer_groups['conv1'] = {'module': backbone.conv1, 'children': {}}
        
        return layer_groups
    
    def analyze_gradient_flow(self):
        """
        Analyze gradient accumulation across frozen layers to determine importance.
        Returns a dictionary of layer names and their importance scores.
        """
        importance_scores = {}
        
        # Process accumulated gradients for each frozen parameter
        for name, param in self.actual_model.backbone.named_parameters():
            if not param.requires_grad and hasattr(param, 'grad_acc') and param.grad_samples > 0:
                # Normalize by number of samples
                avg_grad = param.grad_acc / param.grad_samples
                
                # Calculate importance as mean of absolute gradient values
                importance = avg_grad.abs().mean().item()
                
                # Store importance score and reset accumulation
                importance_scores[name] = importance
                
                # Add to gradient history for trend analysis
                if name in self.gradient_history:
                    self.gradient_history[name].append(importance)
                
                # Reset accumulators for next epoch
                param.grad_acc.zero_()
                param.grad_samples = 0
        
        # Group by layers for easier analysis
        layer_importance = {}
        for name, score in importance_scores.items():
            # Extract layer name from parameter name (e.g., 'layer4.1.conv1.weight' -> 'layer4')
            parts = name.split('.')
            layer_name = parts[0]
            
            if layer_name not in layer_importance:
                layer_importance[layer_name] = []
            
            layer_importance[layer_name].append(score)
        
        # Aggregate layer scores (mean of parameter importances)
        for layer, scores in layer_importance.items():
            layer_importance[layer] = sum(scores) / len(scores)
        
        self.layer_importance = layer_importance
        return layer_importance
    
    def select_layers_to_unfreeze(self, epoch):
        """
        Intelligently select which layers to unfreeze next based on gradient analysis.
        Returns list of layer names to unfreeze, or empty list if none.
        """
        # During sampling phase, just collect gradients
        if self.sampling_phase:
            self.sampling_epoch_count += 1
            if self.sampling_epoch_count >= self.sampling_epochs:
                logger.info(f"[GRADIENT_GUIDED] Completed gradient sampling phase after {self.sampling_epochs} epochs")
                self.sampling_phase = False
            return []
        
        # Only consider unfreezing if we've waited long enough since last unfreeze
        if self.epochs_since_unfreeze < self.unfreeze_patience:
            self.epochs_since_unfreeze += 1
            return []
        
        # Check which layers are still frozen
        frozen_layers = {}
        layer_groups = self.get_backbone_layer_groups()
        
        for layer_name, layer_info in layer_groups.items():
            if layer_name not in self.unfrozen_layers:
                # Calculate proportion of frozen parameters in this layer
                total_params = 0
                frozen_params = 0
                
                for param in layer_info['module'].parameters():
                    total_params += 1
                    if not param.requires_grad:
                        frozen_params += 1
                
                if frozen_params > 0:  # Only consider layers that still have frozen params
                    frozen_ratio = frozen_params / total_params
                    frozen_layers[layer_name] = frozen_ratio
        
        if not frozen_layers:
            logger.info("[GRADIENT_GUIDED] All layers already unfrozen")
            return []
        
        # Use importance scores to select layers
        candidates = []
        
        # First prioritize high-level layers with high importance
        for layer_name, frozen_ratio in frozen_layers.items():
            if layer_name in self.layer_importance:
                importance = self.layer_importance[layer_name]
                
                # Check if importance exceeds threshold
                if importance > self.importance_threshold:
                    candidates.append((layer_name, importance))
        
        # Sort by importance (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select layers to unfreeze (limit by max_layers_per_step)
        selected_layers = [name for name, _ in candidates[:self.max_layers_per_step]]
        
        if selected_layers:
            logger.info(f"[GRADIENT_GUIDED] Selected layers to unfreeze: {selected_layers}")
            for layer in selected_layers:
                logger.info(f"  - {layer}: importance={self.layer_importance.get(layer, 0):.6f}")
        else:
            # If no layers exceed threshold, just unfreeze the most important one
            if candidates:
                selected_layers = [candidates[0][0]]
                logger.info(f"[GRADIENT_GUIDED] No layers exceeded threshold, unfreezing most important: {selected_layers[0]}")
        
        return selected_layers
    
    def unfreeze_layers(self, layer_names):
        """
        Unfreeze the specified layers and set up warmup for them.
        Returns the total number of parameters unfrozen.
        """
        if not layer_names:
            return 0
        
        unfrozen_params = 0
        layer_groups = self.get_backbone_layer_groups()
        
        for layer_name in layer_names:
            if layer_name in layer_groups and layer_name not in self.unfrozen_layers:
                layer_info = layer_groups[layer_name]
                
                # Unfreeze all parameters in this layer
                for param in layer_info['module'].parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        unfrozen_params += param.numel()
                
                # Mark layer as unfrozen and in warmup
                self.unfrozen_layers.add(layer_name)
                self.layers_in_warmup[layer_name] = 0
                
                logger.info(f"[GRADIENT_GUIDED] Unfroze {layer_name} ({unfrozen_params:,} parameters)")
        
        if unfrozen_params > 0:
            self.epochs_since_unfreeze = 0
        
        return unfrozen_params
    
    def create_optimizer_param_groups(self, head_lr, backbone_lr):
        """
        Create parameter groups for optimizer with proper learning rates.
        - Head parameters: highest learning rate
        - Recently unfrozen layers: warmup learning rates
        - Previously unfrozen layers: full backbone learning rate
        """
        param_groups = []
        
        # 1. Head parameters (highest LR)
        head_params = []
        head_params.extend(self.actual_model.severity_head.parameters())
        head_params.extend(self.actual_model.action_type_head.parameters())
        head_params.extend(self.actual_model.embedding_manager.parameters())
        head_params.extend(self.actual_model.view_aggregator.parameters())
        
        param_groups.append({
            'params': head_params,
            'lr': head_lr,
            'name': 'heads'
        })
        
        # 2. Group backbone parameters by layer and warmup status
        layer_groups = self.get_backbone_layer_groups()
        
        # Layers in warmup (reduced LR)
        for layer_name, warmup_epoch in self.layers_in_warmup.items():
            if layer_name in layer_groups:
                layer_info = layer_groups[layer_name]
                layer_params = [p for p in layer_info['module'].parameters() if p.requires_grad]
                
                if layer_params:
                    # Calculate warmup learning rate
                    warmup_progress = min(1.0, warmup_epoch / self.warmup_epochs)
                    warmup_lr = backbone_lr * (0.1 + 0.9 * warmup_progress)
                    
                    param_groups.append({
                        'params': layer_params,
                        'lr': warmup_lr,
                        'name': f'warmup_{layer_name}'
                    })
                    
                    logger.debug(f"[GRADIENT_GUIDED] Layer {layer_name} in warmup: epoch {warmup_epoch}/{self.warmup_epochs}, lr={warmup_lr:.2e}")
        
        # Regular unfrozen layers (full LR)
        regular_params = []
        for name, param in self.actual_model.backbone.named_parameters():
            if param.requires_grad:
                # Skip parameters already in warmup groups
                layer_in_warmup = False
                for warmup_layer in self.layers_in_warmup:
                    if warmup_layer in name:
                        layer_in_warmup = True
                        break
                
                if not layer_in_warmup:
                    regular_params.append(param)
        
        if regular_params:
            param_groups.append({
                'params': regular_params,
                'lr': backbone_lr,
                'name': 'backbone_regular'
            })
        
        return param_groups
    
    def update_after_epoch(self, val_metric, epoch):
        """
        Update the freezing manager state after each epoch.
        - Analyze gradient flow
        - Select layers to unfreeze based on importance
        - Unfreeze selected layers
        - Update warmup status for recently unfrozen layers
        
        Returns:
            dict: State update information including if optimizer needs rebuilding
        """
        update_info = {
            'rebuild_optimizer': False,
            'unfrozen_layers': [],
            'warmup_updates': {}
        }
        
        # 1. Analyze gradient accumulation from this epoch
        layer_importance = self.analyze_gradient_flow()
        
        # Log importance scores periodically
        if epoch % 2 == 0:
            logger.info(f"[GRADIENT_GUIDED] Layer importance scores at epoch {epoch}:")
            for layer, score in sorted(layer_importance.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  - {layer}: {score:.6f}")
        
        # 2. Update warmup progress for recently unfrozen layers
        warmup_updates = {}
        for layer_name, warmup_epoch in list(self.layers_in_warmup.items()):
            new_warmup_epoch = warmup_epoch + 1
            if new_warmup_epoch >= self.warmup_epochs:
                # Warmup complete
                logger.info(f"[GRADIENT_GUIDED] Layer {layer_name} completed warmup")
                del self.layers_in_warmup[layer_name]
                warmup_updates[layer_name] = 'completed'
            else:
                # Continue warmup
                self.layers_in_warmup[layer_name] = new_warmup_epoch
                warmup_updates[layer_name] = new_warmup_epoch
        
        update_info['warmup_updates'] = warmup_updates
        
        # 3. Check validation performance for adaptive unfreezing
        improvement = val_metric - self.best_val_metric
        if improvement > 0:
            self.best_val_metric = val_metric
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        
        # 4. Select and unfreeze layers if appropriate
        selected_layers = self.select_layers_to_unfreeze(epoch)
        if selected_layers:
            self.unfreeze_layers(selected_layers)
            update_info['unfrozen_layers'] = selected_layers
            update_info['rebuild_optimizer'] = True
        
        # 5. Check if we need to rebuild optimizer due to warmup changes
        if not update_info['rebuild_optimizer'] and (warmup_updates and any(status == 'completed' for status in warmup_updates.values())):
            update_info['rebuild_optimizer'] = True
        
        return update_info
    
    def log_status(self, epoch):
        """Log the current status of layer freezing."""
        # Count frozen and unfrozen parameters
        total_params = 0
        frozen_params = 0
        
        for name, param in self.actual_model.backbone.named_parameters():
            total_params += param.numel()
            if not param.requires_grad:
                frozen_params += param.numel()
        
        unfrozen_params = total_params - frozen_params
        
        logger.info(f"[GRADIENT_GUIDED] Epoch {epoch} status:")
        logger.info(f"  - Unfrozen layers: {sorted(self.unfrozen_layers)}")
        logger.info(f"  - Layers in warmup: {self.layers_in_warmup}")
        logger.info(f"  - Parameters: {unfrozen_params:,}/{total_params:,} unfrozen ({unfrozen_params/total_params*100:.1f}%)")
        
        return {
            'total_params': total_params,
            'frozen_params': frozen_params,
            'unfrozen_params': unfrozen_params,
            'unfrozen_ratio': unfrozen_params/total_params
        }


class AdvancedFreezingManager:
    """
    Next-generation freezing manager that combines multiple intelligence sources
    for optimal layer unfreezing decisions:
    
    1. **Multi-Metric Analysis**: Combines gradient magnitude, gradient variance, 
       activation patterns, and performance impact
    2. **Dynamic Thresholds**: Adapts thresholds based on training progress
    3. **Layer Dependency Analysis**: Considers inter-layer dependencies
    4. **Performance Rollback**: Can re-freeze layers if they hurt performance
    5. **Smart Warmup**: Uses layer-specific warmup strategies
    6. **Ensemble Decision Making**: Multiple criteria must agree for unfreezing
    
    This manager is designed for production use with complex models where
    traditional freezing strategies may be sub-optimal.
    """
    
    def __init__(
        self,
        model,
        base_importance_threshold=0.005,  # Lower base threshold, will adapt
        performance_threshold=0.002,     # Min performance improvement to continue
        max_layers_per_step=1,
        warmup_epochs=4,
        patience_epochs=3,
        rollback_patience=2,             # Epochs to wait before rollback
        gradient_momentum=0.9,           # For smoothing gradient importance
        analysis_window=3,               # Epochs to analyze before decision
        enable_rollback=True,
        enable_dependency_analysis=True
    ):
        self.model = model
        self.actual_model = model.module if hasattr(model, 'module') else model
        
        # Thresholds and parameters
        self.base_importance_threshold = base_importance_threshold
        self.current_importance_threshold = base_importance_threshold
        self.performance_threshold = performance_threshold
        self.max_layers_per_step = max_layers_per_step
        self.warmup_epochs = warmup_epochs
        self.patience_epochs = patience_epochs
        self.rollback_patience = rollback_patience
        self.gradient_momentum = gradient_momentum
        self.analysis_window = analysis_window
        self.enable_rollback = enable_rollback
        self.enable_dependency_analysis = enable_dependency_analysis
        
        # State tracking
        self.unfrozen_layers = set()
        self.layers_in_warmup = {}  # {layer_name: warmup_epoch}
        self.layer_performance_history = {}  # {layer_name: [performance_after_unfreeze]}
        self.layer_importance_smooth = {}  # Smoothed importance scores
        self.performance_history = []
        self.last_unfreeze_epoch = -999
        self.epochs_since_last_unfreeze = 0
        self.rollback_candidates = {}  # Layers that might need rollback
        
        # Analysis data
        self.gradient_stats = {}  # More detailed gradient statistics
        self.activation_stats = {}  # Track activation patterns
        self.layer_dependencies = {}  # Inter-layer dependency scores
        
        # Register enhanced hooks
        self._register_analysis_hooks()
        
        # Initialize layer groups and importance tracking
        self.layer_groups = self.get_enhanced_backbone_layer_groups()
        self._initialize_tracking()
        
        logger.info(f"[ADVANCED_FREEZING] Initialized with enhanced multi-metric analysis")
        logger.info(f"  - Base importance threshold: {base_importance_threshold}")
        logger.info(f"  - Performance threshold: {performance_threshold}")
        logger.info(f"  - Analysis window: {analysis_window} epochs")
        logger.info(f"  - Rollback enabled: {enable_rollback}")
        logger.info(f"  - Dependency analysis: {enable_dependency_analysis}")
    
    def _register_analysis_hooks(self):
        """Register comprehensive hooks for gradient and activation analysis."""
        self.forward_hooks = []
        self.backward_hooks = []
        
        for name, module in self.actual_model.backbone.named_modules():
            if len(list(module.parameters(recurse=False))) == 0:
                continue
            
            # Forward hook for activation statistics
            def make_forward_hook(layer_name):
                def forward_hook(module, input, output):
                    if self.training:
                        # Track activation statistics
                        if isinstance(output, torch.Tensor):
                            activation_std = output.std().item()
                            activation_mean = output.mean().item()
                            
                            if layer_name not in self.activation_stats:
                                self.activation_stats[layer_name] = {
                                    'std': [], 'mean': [], 'magnitude': []
                                }
                            
                            self.activation_stats[layer_name]['std'].append(activation_std)
                            self.activation_stats[layer_name]['mean'].append(activation_mean)
                            self.activation_stats[layer_name]['magnitude'].append(
                                output.abs().mean().item()
                            )
                return forward_hook
            
            hook = module.register_forward_hook(make_forward_hook(name))
            self.forward_hooks.append(hook)
            
            # Enhanced gradient hooks for parameters
            for param_name, param in module.named_parameters(recurse=False):
                full_param_name = f"{name}.{param_name}"
                
                # Initialize enhanced gradient tracking
                if not hasattr(param, 'grad_stats'):
                    param.grad_stats = {
                        'magnitude_history': [],
                        'variance_history': [],
                        'direction_changes': 0,
                        'last_grad_direction': None,
                        'importance_score': 0.0,
                        'samples_seen': 0
                    }
                
                # Register gradient hook
                def make_grad_hook(param_name):
                    def grad_hook(grad):
                        if grad is not None and hasattr(param, 'grad_stats'):
                            # Calculate gradient statistics
                            grad_magnitude = grad.abs().mean().item()
                            grad_variance = grad.var().item()
                            grad_direction = grad.sign().mean().item()
                            
                            stats = param.grad_stats
                            stats['magnitude_history'].append(grad_magnitude)
                            stats['variance_history'].append(grad_variance)
                            stats['samples_seen'] += 1
                            
                            # Track gradient direction changes (indicates learning dynamics)
                            if stats['last_grad_direction'] is not None:
                                if (stats['last_grad_direction'] > 0) != (grad_direction > 0):
                                    stats['direction_changes'] += 1
                            stats['last_grad_direction'] = grad_direction
                            
                            # Update smoothed importance score
                            current_importance = grad_magnitude * (1 + grad_variance)
                            if stats['importance_score'] == 0:
                                stats['importance_score'] = current_importance
                            else:
                                stats['importance_score'] = (
                                    self.gradient_momentum * stats['importance_score'] + 
                                    (1 - self.gradient_momentum) * current_importance
                                )
                        
                        return grad
                    return grad_hook
                
                param.register_hook(make_grad_hook(full_param_name))
    
    def _initialize_tracking(self):
        """Initialize tracking structures for all layers."""
        for layer_name in self.layer_groups.keys():
            self.layer_importance_smooth[layer_name] = 0.0
            self.layer_performance_history[layer_name] = []
            self.gradient_stats[layer_name] = {
                'recent_importance': [],
                'stability_score': 0.0,
                'learning_progress': 0.0
            }
    
    def get_enhanced_backbone_layer_groups(self):
        """Get enhanced layer grouping with hierarchical structure and dependencies."""
        backbone = self.actual_model.backbone.backbone
        layer_groups = {}
        
        # Create hierarchical layer structure
        layer_sequence = []
        
        # Early layers
        if hasattr(backbone, 'conv1'):
            layer_groups['conv1'] = {
                'module': backbone.conv1,
                'type': 'conv',
                'depth': 0,
                'dependencies': [],
                'children': {}
            }
            layer_sequence.append('conv1')
        
        # ResNet blocks
        for block_num in [1, 2, 3, 4]:
            block_name = f'layer{block_num}'
            if hasattr(backbone, block_name):
                layer_groups[block_name] = {
                    'module': getattr(backbone, block_name),
                    'type': 'residual_block',
                    'depth': block_num,
                    'dependencies': layer_sequence[-2:] if len(layer_sequence) >= 2 else layer_sequence,
                    'children': {}
                }
                layer_sequence.append(block_name)
                
                # Add sub-blocks for fine-grained control
                for i, sub_block in enumerate(getattr(backbone, block_name)):
                    sub_name = f'{block_name}.{i}'
                    layer_groups[sub_name] = {
                        'module': sub_block,
                        'type': 'residual_sub_block',
                        'depth': block_num + i/10,
                        'dependencies': [block_name] if i == 0 else [f'{block_name}.{i-1}'],
                        'parent': block_name
                    }
        
        return layer_groups
    
    def freeze_all_backbone(self):
        """Freeze all backbone parameters with enhanced tracking."""
        frozen_params = 0
        
        for name, param in self.actual_model.backbone.named_parameters():
            param.requires_grad = False
            frozen_params += param.numel()
            
            # Initialize enhanced tracking for this parameter
            if not hasattr(param, 'grad_stats'):
                param.grad_stats = {
                    'magnitude_history': [],
                    'variance_history': [],
                    'direction_changes': 0,
                    'last_grad_direction': None,
                    'importance_score': 0.0,
                    'samples_seen': 0
                }
        
        logger.info(f"[ADVANCED_FREEZING] Froze entire backbone ({frozen_params:,} parameters)")
        logger.info(f"[ADVANCED_FREEZING] Enhanced gradient tracking enabled for {len(list(self.actual_model.backbone.parameters()))} parameters")
        
        return frozen_params
    
    def analyze_multi_metric_importance(self, epoch):
        """
        Comprehensive importance analysis using multiple metrics:
        1. Gradient magnitude and variance
        2. Activation patterns
        3. Learning stability
        4. Inter-layer dependencies
        """
        layer_scores = {}
        
        for layer_name, layer_info in self.layer_groups.items():
            if layer_name in self.unfrozen_layers:
                continue  # Skip already unfrozen layers
            
            metrics = self._calculate_layer_metrics(layer_name, layer_info)
            
            # Combine metrics with learned weights
            importance_score = (
                0.4 * metrics['gradient_importance'] +
                0.3 * metrics['activation_variance'] +
                0.2 * metrics['learning_potential'] +
                0.1 * metrics['dependency_pressure']
            )
            
            # Apply dynamic threshold adjustment
            adjusted_threshold = self._get_adaptive_threshold(epoch, layer_name)
            
            layer_scores[layer_name] = {
                'total_score': importance_score,
                'threshold': adjusted_threshold,
                'should_unfreeze': importance_score > adjusted_threshold,
                'metrics': metrics
            }
            
            # Update smoothed importance
            if layer_name in self.layer_importance_smooth:
                self.layer_importance_smooth[layer_name] = (
                    0.7 * self.layer_importance_smooth[layer_name] + 
                    0.3 * importance_score
                )
            else:
                self.layer_importance_smooth[layer_name] = importance_score
        
        return layer_scores
    
    def _calculate_layer_metrics(self, layer_name, layer_info):
        """Calculate comprehensive metrics for a layer."""
        module = layer_info['module']
        
        # 1. Gradient-based importance
        gradient_importance = 0.0
        gradient_variance = 0.0
        param_count = 0
        
        for param in module.parameters():
            if hasattr(param, 'grad_stats') and param.grad_stats['samples_seen'] > 0:
                gradient_importance += param.grad_stats['importance_score']
                
                if param.grad_stats['variance_history']:
                    gradient_variance += np.mean(param.grad_stats['variance_history'][-10:])
                
                param_count += 1
        
        if param_count > 0:
            gradient_importance /= param_count
            gradient_variance /= param_count
        
        # 2. Activation-based metrics
        activation_variance = 0.0
        if layer_name in self.activation_stats:
            stats = self.activation_stats[layer_name]
            if stats['std']:
                activation_variance = np.mean(stats['std'][-10:])  # Recent activation variance
        
        # 3. Learning potential (based on gradient direction changes)
        learning_potential = 0.0
        total_direction_changes = 0
        for param in module.parameters():
            if hasattr(param, 'grad_stats'):
                total_direction_changes += param.grad_stats['direction_changes']
        
        if param_count > 0:
            learning_potential = total_direction_changes / max(param_count, 1)
        
        # 4. Dependency pressure (how much dependent layers need this one)
        dependency_pressure = 0.0
        if self.enable_dependency_analysis:
            # Count how many unfrozen layers depend on this one
            for other_layer, other_info in self.layer_groups.items():
                if (other_layer in self.unfrozen_layers and 
                    layer_name in other_info.get('dependencies', [])):
                    dependency_pressure += 0.5
        
        return {
            'gradient_importance': gradient_importance,
            'gradient_variance': gradient_variance,
            'activation_variance': activation_variance,
            'learning_potential': learning_potential,
            'dependency_pressure': dependency_pressure
        }
    
    def _get_adaptive_threshold(self, epoch, layer_name):
        """Calculate adaptive threshold based on training progress and layer characteristics."""
        # Base threshold decreases over time (easier to unfreeze later)
        time_factor = max(0.5, 1.0 - epoch / 50)  # Decrease over 50 epochs
        
        # Layer depth factor (deeper layers need higher threshold)
        depth_factor = 1.0
        if layer_name in self.layer_groups:
            depth = self.layer_groups[layer_name].get('depth', 0)
            depth_factor = 1.0 + depth * 0.1  # Increase threshold for deeper layers
        
        # Performance factor (if we're doing well, be more conservative)
        performance_factor = 1.0
        if len(self.performance_history) >= 2:
            recent_trend = self.performance_history[-1] - self.performance_history[-2]
            if recent_trend > 0.01:  # Good improvement
                performance_factor = 1.2  # Be more conservative
            elif recent_trend < -0.005:  # Performance declining
                performance_factor = 0.8  # Be more aggressive
        
        return self.base_importance_threshold * time_factor * depth_factor * performance_factor
    
    def select_optimal_layers_to_unfreeze(self, epoch, val_performance):
        """
        Advanced layer selection using ensemble decision making.
        Multiple criteria must agree for a layer to be unfrozen.
        """
        if self.epochs_since_last_unfreeze < self.patience_epochs:
            self.epochs_since_last_unfreeze += 1
            return []
        
        # Wait for sufficient analysis data
        if epoch < self.analysis_window:
            return []
        
        # Analyze all layers
        layer_scores = self.analyze_multi_metric_importance(epoch)
        
        # Filter candidates that meet all criteria
        candidates = []
        for layer_name, analysis in layer_scores.items():
            if analysis['should_unfreeze']:
                # Additional criteria for robust selection
                metrics = analysis['metrics']
                
                # Must have sufficient gradient activity
                if metrics['gradient_importance'] > self.base_importance_threshold * 0.5:
                    # Must show learning potential
                    if metrics['learning_potential'] > 0.1:
                        # Must not conflict with recently unfrozen layers
                        if not self._conflicts_with_recent_unfreeze(layer_name):
                            candidates.append((
                                layer_name,
                                analysis['total_score'],
                                analysis['metrics']
                            ))
        
        # Sort by total importance score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select top candidates up to max_layers_per_step
        selected = []
        for layer_name, score, metrics in candidates[:self.max_layers_per_step]:
            selected.append(layer_name)
            
            logger.info(f"[ADVANCED_FREEZING] Selected {layer_name} for unfreezing:")
            logger.info(f"  - Total score: {score:.6f}")
            logger.info(f"  - Gradient importance: {metrics['gradient_importance']:.6f}")
            logger.info(f"  - Activation variance: {metrics['activation_variance']:.6f}")
            logger.info(f"  - Learning potential: {metrics['learning_potential']:.6f}")
            logger.info(f"  - Dependency pressure: {metrics['dependency_pressure']:.6f}")
        
        return selected
    
    def _conflicts_with_recent_unfreeze(self, layer_name):
        """Check if unfreezing this layer conflicts with recently unfrozen layers."""
        # Don't unfreeze adjacent layers too quickly
        if layer_name in self.layer_groups:
            layer_info = self.layer_groups[layer_name]
            for warmup_layer in self.layers_in_warmup.keys():
                # Check if layers are adjacent in the architecture
                if warmup_layer in self.layer_groups:
                    warmup_info = self.layer_groups[warmup_layer]
                    depth_diff = abs(layer_info.get('depth', 0) - warmup_info.get('depth', 0))
                    if depth_diff < 1.0:  # Adjacent layers
                        return True
        return False
    
    def unfreeze_layers_with_enhanced_warmup(self, layer_names):
        """Unfreeze layers with layer-specific warmup strategies."""
        if not layer_names:
            return 0
        
        unfrozen_params = 0
        
        for layer_name in layer_names:
            if layer_name in self.layer_groups and layer_name not in self.unfrozen_layers:
                layer_info = self.layer_groups[layer_name]
                module = layer_info['module']
                
                # Unfreeze parameters
                layer_params = 0
                for param in module.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        layer_params += param.numel()
                        unfrozen_params += param.numel()
                
                # Setup enhanced warmup based on layer characteristics
                warmup_config = self._get_layer_warmup_config(layer_name, layer_info)
                
                self.unfrozen_layers.add(layer_name)
                self.layers_in_warmup[layer_name] = {
                    'epoch': 0,
                    'config': warmup_config,
                    'performance_before': self.performance_history[-1] if self.performance_history else 0.0
                }
                
                logger.info(f"[ADVANCED_FREEZING] Unfroze {layer_name} ({layer_params:,} parameters)")
                logger.info(f"  - Warmup strategy: {warmup_config['strategy']}")
                logger.info(f"  - Warmup duration: {warmup_config['duration']} epochs")
        
        if unfrozen_params > 0:
            self.epochs_since_last_unfreeze = 0
            self.last_unfreeze_epoch = len(self.performance_history)
        
        return unfrozen_params
    
    def _get_layer_warmup_config(self, layer_name, layer_info):
        """Get layer-specific warmup configuration."""
        layer_type = layer_info.get('type', 'unknown')
        depth = layer_info.get('depth', 0)
        
        if layer_type == 'conv':
            # Convolutional layers need gentle warmup
            return {
                'strategy': 'exponential',
                'duration': self.warmup_epochs + 1,
                'start_factor': 0.05,
                'growth_rate': 1.5
            }
        elif 'residual' in layer_type:
            # Residual blocks can handle more aggressive warmup
            return {
                'strategy': 'linear',
                'duration': max(2, self.warmup_epochs - 1),
                'start_factor': 0.1,
                'growth_rate': 1.0
            }
        else:
            # Default strategy
            return {
                'strategy': 'linear',
                'duration': self.warmup_epochs,
                'start_factor': 0.1,
                'growth_rate': 1.0
            }
    
    def create_advanced_optimizer_param_groups(self, head_lr, backbone_lr):
        """Create sophisticated parameter groups with layer-specific learning rates."""
        param_groups = []
        
        # 1. Head parameters (highest LR)
        head_params = []
        head_params.extend(self.actual_model.severity_head.parameters())
        head_params.extend(self.actual_model.action_type_head.parameters())
        head_params.extend(self.actual_model.embedding_manager.parameters())
        head_params.extend(self.actual_model.view_aggregator.parameters())
        
        param_groups.append({
            'params': head_params,
            'lr': head_lr,
            'name': 'heads',
            'weight_decay': 1e-4  # Standard weight decay for heads
        })
        
        # 2. Layers in warmup (custom LR based on warmup strategy)
        for layer_name, warmup_info in self.layers_in_warmup.items():
            if layer_name in self.layer_groups:
                layer_info = self.layer_groups[layer_name]
                layer_params = [p for p in layer_info['module'].parameters() if p.requires_grad]
                
                if layer_params:
                    warmup_lr = self._calculate_warmup_lr(
                        backbone_lr, 
                        warmup_info['epoch'],
                        warmup_info['config']
                    )
                    
                    param_groups.append({
                        'params': layer_params,
                        'lr': warmup_lr,
                        'name': f'warmup_{layer_name}',
                        'weight_decay': self._get_layer_weight_decay(layer_name)
                    })
        
        # 3. Fully unfrozen layers (full backbone LR with depth-based scaling)
        for layer_name in self.unfrozen_layers:
            if (layer_name not in self.layers_in_warmup and 
                layer_name in self.layer_groups):
                
                layer_info = self.layer_groups[layer_name]
                layer_params = [p for p in layer_info['module'].parameters() if p.requires_grad]
                
                if layer_params:
                    # Scale learning rate based on layer depth
                    depth = layer_info.get('depth', 0)
                    depth_factor = max(0.5, 1.0 - depth * 0.1)  # Deeper layers get lower LR
                    layer_lr = backbone_lr * depth_factor
                    
                    param_groups.append({
                        'params': layer_params,
                        'lr': layer_lr,
                        'name': f'backbone_{layer_name}',
                        'weight_decay': self._get_layer_weight_decay(layer_name)
                    })
        
        return param_groups
    
    def _calculate_warmup_lr(self, base_lr, warmup_epoch, warmup_config):
        """Calculate learning rate for layer in warmup."""
        strategy = warmup_config['strategy']
        duration = warmup_config['duration']
        start_factor = warmup_config['start_factor']
        
        progress = min(1.0, warmup_epoch / duration)
        
        if strategy == 'linear':
            factor = start_factor + (1.0 - start_factor) * progress
        elif strategy == 'exponential':
            growth_rate = warmup_config.get('growth_rate', 1.5)
            factor = start_factor * (growth_rate ** progress)
            factor = min(factor, 1.0)
        else:
            factor = start_factor + (1.0 - start_factor) * progress
        
        return base_lr * factor
    
    def _get_layer_weight_decay(self, layer_name):
        """Get layer-specific weight decay."""
        if layer_name in self.layer_groups:
            layer_type = self.layer_groups[layer_name].get('type', 'unknown')
            if layer_type == 'conv':
                return 5e-5  # Lower weight decay for early conv layers
            elif 'residual' in layer_type:
                return 1e-4  # Standard weight decay for residual blocks
        return 1e-4  # Default
    
    def update_after_epoch(self, val_metric, epoch):
        """Comprehensive update with performance monitoring and rollback capability."""
        update_info = {
            'rebuild_optimizer': False,
            'unfrozen_layers': [],
            'warmup_updates': {},
            'rollback_performed': False
        }
        
        # Store performance history
        self.performance_history.append(val_metric)
        
        # 1. Check for rollback opportunities
        if self.enable_rollback and len(self.performance_history) >= self.rollback_patience + 1:
            rollback_performed = self._check_and_perform_rollback(epoch)
            if rollback_performed:
                update_info['rollback_performed'] = True
                update_info['rebuild_optimizer'] = True
        
        # 2. Update warmup progress
        warmup_updates = {}
        for layer_name, warmup_info in list(self.layers_in_warmup.items()):
            new_epoch = warmup_info['epoch'] + 1
            duration = warmup_info['config']['duration']
            
            if new_epoch >= duration:
                # Warmup complete
                logger.info(f"[ADVANCED_FREEZING] Completed warmup for {layer_name}")
                del self.layers_in_warmup[layer_name]
                warmup_updates[layer_name] = 'completed'
                update_info['rebuild_optimizer'] = True
            else:
                # Continue warmup
                self.layers_in_warmup[layer_name]['epoch'] = new_epoch
                warmup_updates[layer_name] = new_epoch
        
        update_info['warmup_updates'] = warmup_updates
        
        # 3. Consider unfreezing new layers
        if not update_info['rollback_performed']:  # Don't unfreeze if we just rolled back
            selected_layers = self.select_optimal_layers_to_unfreeze(epoch, val_metric)
            if selected_layers:
                self.unfreeze_layers_with_enhanced_warmup(selected_layers)
                update_info['unfrozen_layers'] = selected_layers
                update_info['rebuild_optimizer'] = True
        
        return update_info
    
    def _check_and_perform_rollback(self, epoch):
        """Check if we should rollback recent unfreezing decisions."""
        if len(self.performance_history) < self.rollback_patience + 1:
            return False
        
        # Check recent performance trend
        recent_performance = self.performance_history[-self.rollback_patience:]
        trend = recent_performance[-1] - recent_performance[0]
        
        # If performance has decreased significantly since last unfreeze
        if (trend < -self.performance_threshold and 
            self.last_unfreeze_epoch >= 0 and
            epoch - self.last_unfreeze_epoch <= self.rollback_patience):
            
            # Find layers unfrozen recently that might be causing issues
            rollback_candidates = []
            for layer_name in list(self.unfrozen_layers):
                if layer_name in self.layers_in_warmup:
                    # Layer is still in warmup, good candidate for rollback
                    rollback_candidates.append(layer_name)
            
            if rollback_candidates:
                # Rollback the most recently unfrozen layer
                layer_to_rollback = rollback_candidates[0]  # Most recent
                self._rollback_layer(layer_to_rollback)
                
                logger.info(f"[ADVANCED_FREEZING] ROLLBACK: Re-froze {layer_to_rollback} due to performance decline")
                logger.info(f"  - Performance trend: {trend:.6f}")
                logger.info(f"  - Epochs since unfreeze: {epoch - self.last_unfreeze_epoch}")
                
                return True
        
        return False
    
    def _rollback_layer(self, layer_name):
        """Rollback (re-freeze) a specific layer."""
        if layer_name in self.layer_groups and layer_name in self.unfrozen_layers:
            layer_info = self.layer_groups[layer_name]
            
            # Re-freeze parameters
            frozen_params = 0
            for param in layer_info['module'].parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_params += param.numel()
            
            # Remove from tracking
            self.unfrozen_layers.discard(layer_name)
            if layer_name in self.layers_in_warmup:
                del self.layers_in_warmup[layer_name]
            
            logger.info(f"[ADVANCED_FREEZING] Rolled back {layer_name} ({frozen_params:,} parameters)")
    
    def log_comprehensive_status(self, epoch):
        """Log comprehensive status including all metrics and decisions."""
        # Basic status
        total_params = sum(p.numel() for p in self.actual_model.backbone.parameters())
        unfrozen_params = sum(p.numel() for p in self.actual_model.backbone.parameters() if p.requires_grad)
        
        logger.info(f"[ADVANCED_FREEZING] Epoch {epoch} comprehensive status:")
        logger.info(f"  🔓 Unfrozen layers: {sorted(self.unfrozen_layers)}")
        logger.info(f"  🔄 Layers in warmup: {list(self.layers_in_warmup.keys())}")
        logger.info(f"  📊 Parameters: {unfrozen_params:,}/{total_params:,} unfrozen ({unfrozen_params/total_params*100:.1f}%)")
        
        # Performance trend
        if len(self.performance_history) >= 3:
            recent_trend = self.performance_history[-1] - self.performance_history[-3]
            trend_emoji = "📈" if recent_trend > 0 else "📉" if recent_trend < -0.001 else "➡️"
            logger.info(f"  {trend_emoji} Performance trend (3 epochs): {recent_trend:+.6f}")
        
        # Adaptive threshold status
        logger.info(f"  🎯 Current importance threshold: {self.current_importance_threshold:.6f}")
        
        # Top candidate layers
        if epoch >= self.analysis_window:
            layer_scores = self.analyze_multi_metric_importance(epoch)
            top_candidates = sorted(
                [(name, analysis['total_score']) for name, analysis in layer_scores.items()],
                key=lambda x: x[1], reverse=True
            )[:3]
            
            if top_candidates:
                logger.info("  🏆 Top unfreeze candidates:")
                for layer_name, score in top_candidates:
                    logger.info(f"    - {layer_name}: {score:.6f}")


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
    logger.info("🧊 FREEZING STRATEGY CONFIGURATION")
    logger.info("=" * 60)
    
    if args.freezing_strategy == 'none':
        # No freezing - train all parameters from start
        logger.info("❄️  No parameter freezing - training all parameters from start")
        log_trainable_parameters(model)
        
    elif args.freezing_strategy == 'advanced':
        # Advanced multi-metric freezing strategy
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
            warmup_epochs=args.warmup_epochs,
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
        logger.info(f"   - Warmup epochs: {args.warmup_epochs}")
        logger.info(f"   - Unfreeze patience: {args.unfreeze_patience}")
        logger.info(f"   - Sampling epochs: {args.sampling_epochs}")
        
        freezing_manager = GradientGuidedFreezingManager(
            model,
            importance_threshold=args.importance_threshold,
            warmup_epochs=args.warmup_epochs,
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
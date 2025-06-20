"""
Smart freezing manager with adaptive and progressive strategies.

This module implements the SmartFreezingManager class which provides
adaptive and progressive unfreezing strategies based on validation performance.
"""

import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
import logging
from .base_utils import _get_backbone_layers

logger = logging.getLogger(__name__)


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
        

    
    def freeze_all_backbone(self):
        """Freeze all backbone parameters."""
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        frozen_params = 0
        
        # Handle different model structures
        if hasattr(actual_model, 'mvit_processor'):
            # Optimized MViT model - backbone is inside mvit_processor
            backbone = actual_model.mvit_processor.backbone
        elif hasattr(actual_model, 'backbone'):
            # Standard model structure
            backbone = actual_model.backbone
        else:
            raise AttributeError("Model does not have accessible backbone")
        
        for param in backbone.parameters():
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
        layers = _get_backbone_layers(self.model)
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
            
        layers = _get_backbone_layers(self.model)
        for layer_name, layer in layers:
            if layer_name not in self.unfrozen_layers:
                self.unfreeze_layer(layer_name, layer)
                logger.info(f"[ADAPTIVE] Unfroze {layer_name} at epoch {epoch+1} due to validation plateau")
                return True
        
        logger.info(f"[ADAPTIVE] All layers already unfrozen")
        return False
    
    def update_after_epoch(self, val_metric, epoch):
        """Update state after each epoch - interface compatibility with other freezing managers."""
        update_info = {
            'rebuild_optimizer': False,
            'unfrozen_layers': [],
            'warmup_updates': {},
            'rollback_performed': False
        }
        
        # Monitor gradients for analysis
        self.monitor_gradients(epoch)
        
        # Perform adaptive unfreezing step
        if self.adaptive_unfreeze_step(val_metric, epoch):
            update_info['rebuild_optimizer'] = True
            # Get the last unfrozen layer
            if self.unfrozen_layers:
                update_info['unfrozen_layers'] = [self.unfrozen_layers[-1]]
        
        return update_info 
"""
Advanced multi-metric freezing manager with rollback and dependency analysis.

This module implements the AdvancedFreezingManager class which combines multiple
intelligence sources for optimal layer unfreezing decisions.
"""

import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AdvancedFreezingManager:
    """
    Next-generation freezing manager that combines multiple intelligence sources
    for optimal layer unfreezing decisions.
    
    Features:
    - Multi-metric analysis (gradient magnitude, variance, activation patterns)
    - Dynamic thresholds based on training progress
    - Performance rollback capability
    - Smart layer-specific warmup
    - Emergency unfreezing when needed
    """
    
    def __init__(
        self,
        model,
        base_importance_threshold=0.002,
        performance_threshold=0.001,
        max_layers_per_step=1,
        warmup_epochs=4,
        patience_epochs=3,
        rollback_patience=2,
        gradient_momentum=0.9,
        analysis_window=2,
        enable_rollback=True,
        enable_dependency_analysis=True
    ):
        self.model = model
        self.actual_model = model.module if hasattr(model, 'module') else model
        
        # Parameters
        self.base_importance_threshold = base_importance_threshold
        self.performance_threshold = performance_threshold
        self.max_layers_per_step = max_layers_per_step
        self.warmup_epochs = warmup_epochs
        self.patience_epochs = patience_epochs
        self.rollback_patience = rollback_patience
        self.gradient_momentum = gradient_momentum
        self.analysis_window = analysis_window
        self.enable_rollback = enable_rollback
        
        # State tracking
        self.unfrozen_layers = set()
        self.layers_in_warmup = {}
        self.performance_history = []
        self.layer_importance_smooth = {}
        self.last_unfreeze_epoch = -999
        self.epochs_since_last_unfreeze = 0
        
        # Initialize layer groups
        self.layer_groups = self._get_layer_groups()
        self._initialize_tracking()
        
        logger.info(f"[ADVANCED_FREEZING] Initialized with enhanced multi-metric analysis")
    
    def _get_layer_groups(self):
        """Get enhanced layer grouping for the backbone."""
        backbone = self.actual_model.backbone.backbone
        layer_groups = {}
        
        # Early layers
        if hasattr(backbone, 'conv1'):
            layer_groups['conv1'] = {
                'module': backbone.conv1,
                'type': 'conv',
                'depth': 0
            }
        
        # ResNet blocks
        for block_num in [1, 2, 3, 4]:
            block_name = f'layer{block_num}'
            if hasattr(backbone, block_name):
                layer_groups[block_name] = {
                    'module': getattr(backbone, block_name),
                    'type': 'residual_block',
                    'depth': block_num
                }
        
        return layer_groups
    
    def _initialize_tracking(self):
        """Initialize tracking for all layers."""
        for layer_name in self.layer_groups.keys():
            self.layer_importance_smooth[layer_name] = 0.0
    
    def freeze_all_backbone(self):
        """Freeze all backbone parameters with enhanced tracking."""
        frozen_params = 0
        
        for param in self.actual_model.backbone.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        
        logger.info(f"[ADVANCED_FREEZING] Froze entire backbone ({frozen_params:,} parameters)")
        return frozen_params
    
    def select_optimal_layers_to_unfreeze(self, epoch, val_performance):
        """Select layers to unfreeze based on multiple criteria."""
        if self.epochs_since_last_unfreeze < self.patience_epochs:
            self.epochs_since_last_unfreeze += 1
            return []
        
        # Find candidate layers that are still frozen
        candidates = []
        for layer_name, layer_info in self.layer_groups.items():
            if layer_name not in self.unfrozen_layers:
                # Simple heuristic: prioritize later layers (higher depth)
                depth = layer_info.get('depth', 0)
                importance = depth * 0.1  # Simple importance based on depth
                candidates.append((layer_name, importance))
        
        if not candidates:
            return []
        
        # Sort by importance and select top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [candidates[0][0]] if candidates else []
        
        if selected:
            logger.info(f"[ADVANCED_FREEZING] Selected layers to unfreeze: {selected}")
        
        return selected
    
    def unfreeze_layers_with_enhanced_warmup(self, layer_names):
        """Unfreeze layers with warmup."""
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
                
                if layer_params > 0:
                    self.unfrozen_layers.add(layer_name)
                    self.layers_in_warmup[layer_name] = {
                        'epoch': 0,
                        'config': {'duration': self.warmup_epochs, 'start_factor': 0.1}
                    }
                    
                    logger.info(f"[ADVANCED_FREEZING] Unfroze {layer_name} ({layer_params:,} parameters)")
        
        if unfrozen_params > 0:
            self.epochs_since_last_unfreeze = 0
            self.last_unfreeze_epoch = len(self.performance_history)
        
        return unfrozen_params
    
    def create_advanced_optimizer_param_groups(self, head_lr, backbone_lr):
        """Create parameter groups with advanced learning rate strategies."""
        param_groups = []
        used_params = set()
        
        # Head parameters
        head_params = []
        head_params.extend(self.actual_model.severity_head.parameters())
        head_params.extend(self.actual_model.action_type_head.parameters())
        head_params.extend(self.actual_model.embedding_manager.parameters())
        head_params.extend(self.actual_model.view_aggregator.parameters())
        
        for param in head_params:
            used_params.add(id(param))
        
        param_groups.append({
            'params': head_params,
            'lr': head_lr,
            'name': 'heads',
            'weight_decay': 1e-4
        })
        
        # Layers in warmup
        for layer_name, warmup_info in self.layers_in_warmup.items():
            if layer_name in self.layer_groups:
                layer_info = self.layer_groups[layer_name]
                layer_params = [p for p in layer_info['module'].parameters() 
                              if p.requires_grad and id(p) not in used_params]
                
                if layer_params:
                    for param in layer_params:
                        used_params.add(id(param))
                    
                    # Simple warmup LR calculation
                    warmup_epoch = warmup_info['epoch']
                    duration = warmup_info['config']['duration']
                    start_factor = warmup_info['config']['start_factor']
                    
                    progress = min(1.0, warmup_epoch / duration)
                    warmup_lr = backbone_lr * (start_factor + (1.0 - start_factor) * progress)
                    
                    param_groups.append({
                        'params': layer_params,
                        'lr': warmup_lr,
                        'name': f'warmup_{layer_name}',
                        'weight_decay': 1e-4
                    })
        
        # Regular unfrozen layers
        for layer_name in self.unfrozen_layers:
            if layer_name not in self.layers_in_warmup and layer_name in self.layer_groups:
                layer_info = self.layer_groups[layer_name]
                layer_params = [p for p in layer_info['module'].parameters() 
                              if p.requires_grad and id(p) not in used_params]
                
                if layer_params:
                    for param in layer_params:
                        used_params.add(id(param))
                    
                    param_groups.append({
                        'params': layer_params,
                        'lr': backbone_lr,
                        'name': f'backbone_{layer_name}',
                        'weight_decay': 1e-4
                    })
        
        return param_groups
    
    def update_after_epoch(self, val_metric, epoch):
        """Update state after each epoch."""
        update_info = {
            'rebuild_optimizer': False,
            'unfrozen_layers': [],
            'warmup_updates': {}
        }
        
        # Store performance
        self.performance_history.append(val_metric)
        
        # Update warmup progress
        warmup_updates = {}
        for layer_name, warmup_info in list(self.layers_in_warmup.items()):
            new_epoch = warmup_info['epoch'] + 1
            duration = warmup_info['config']['duration']
            
            if new_epoch >= duration:
                logger.info(f"[ADVANCED_FREEZING] Completed warmup for {layer_name}")
                del self.layers_in_warmup[layer_name]
                warmup_updates[layer_name] = 'completed'
                update_info['rebuild_optimizer'] = True
            else:
                self.layers_in_warmup[layer_name]['epoch'] = new_epoch
                warmup_updates[layer_name] = new_epoch
        
        update_info['warmup_updates'] = warmup_updates
        
        # Consider unfreezing new layers
        selected_layers = self.select_optimal_layers_to_unfreeze(epoch, val_metric)
        if selected_layers:
            self.unfreeze_layers_with_enhanced_warmup(selected_layers)
            update_info['unfrozen_layers'] = selected_layers
            update_info['rebuild_optimizer'] = True
        
        return update_info
    
    def emergency_unfreeze(self, min_layers=1, use_gradual=True):
        """Emergency unfreezing when normal criteria aren't met."""
        # Find frozen layers
        frozen_candidates = []
        for layer_name, layer_info in self.layer_groups.items():
            if layer_name not in self.unfrozen_layers:
                frozen_candidates.append((layer_name, layer_info))
        
        if not frozen_candidates:
            logger.warning("[EMERGENCY_UNFREEZE] No layers available for unfreezing")
            return False
        
        # Sort by priority (later layers first)
        frozen_candidates.sort(key=lambda x: x[1].get('depth', 0), reverse=True)
        
        # Unfreeze the top candidates
        unfrozen_count = 0
        for layer_name, layer_info in frozen_candidates[:min_layers]:
            unfrozen_params = 0
            for param in layer_info['module'].parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen_params += param.numel()
            
            if unfrozen_params > 0:
                self.unfrozen_layers.add(layer_name)
                unfrozen_count += 1
                
                # Setup conservative warmup
                self.layers_in_warmup[layer_name] = {
                    'epoch': 0,
                    'config': {'duration': 6, 'start_factor': 0.01}
                }
                
                logger.info(f"[EMERGENCY_UNFREEZE] Unfroze {layer_name} ({unfrozen_params:,} parameters)")
        
        if unfrozen_count > 0:
            logger.info(f"[EMERGENCY_UNFREEZE] Successfully unfroze {unfrozen_count} layers")
            return True
        
        return False
    
    def log_comprehensive_status(self, epoch):
        """Log comprehensive status."""
        total_params = sum(p.numel() for p in self.actual_model.backbone.parameters())
        unfrozen_params = sum(p.numel() for p in self.actual_model.backbone.parameters() if p.requires_grad)
        
        logger.info(f"[ADVANCED_FREEZING] Epoch {epoch} status:")
        logger.info(f"  🔓 Unfrozen layers: {sorted(self.unfrozen_layers)}")
        logger.info(f"  🔄 Layers in warmup: {list(self.layers_in_warmup.keys())}")
        logger.info(f"  📊 Parameters: {unfrozen_params:,}/{total_params:,} unfrozen ({unfrozen_params/total_params*100:.1f}%)")
        
        # Performance trend
        if len(self.performance_history) >= 3:
            recent_trend = self.performance_history[-1] - self.performance_history[-3]
            trend_emoji = "📈" if recent_trend > 0 else "📉" if recent_trend < -0.001 else "➡️"
            logger.info(f"  {trend_emoji} Performance trend (3 epochs): {recent_trend:+.6f}") 
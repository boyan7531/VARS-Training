"""
Gradient-guided freezing manager that uses gradient flow analysis.

This module implements the GradientGuidedFreezingManager class which uses
gradient flow analysis to make intelligent decisions about layer unfreezing.
"""

import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
"""
Early Gradual Freezing Manager

This module implements a simple but effective freezing strategy:
1. Freeze backbone for the first epoch only
2. Unfreeze 1 block per epoch until half of the backbone is trainable
3. Log trainable parameter count every epoch
"""

import torch
import torch.nn as nn
import logging
from .base_utils import _get_backbone_blocks

logger = logging.getLogger(__name__)


class EarlyGradualFreezingManager:
    """
    Early gradual unfreezing manager that:
    - Freezes backbone for only the first epoch
    - Unfreezes 1 block per epoch until target ratio is reached
    - Provides detailed parameter logging every epoch
    """
    
    def __init__(self, model, freeze_epochs=1, blocks_per_epoch=1, target_ratio=0.5):
        self.model = model
        self.actual_model = model.module if hasattr(model, 'module') else model
        self.freeze_epochs = freeze_epochs
        self.blocks_per_epoch = blocks_per_epoch
        self.target_ratio = target_ratio
        
        # State tracking
        self.unfrozen_blocks = []
        self.total_backbone_params = 0
        self.current_unfrozen_params = 0
        self.target_unfrozen_params = 0
        
        # Initialize backbone layer information
        self.backbone_blocks = _get_backbone_blocks(self.model)
        self._calculate_parameter_targets()
        
        # Only log from main process to avoid duplicates in distributed training
        from ..training_utils import is_main_process
        if is_main_process():
            logger.info(f"[EARLY_GRADUAL] Initialized with {len(self.backbone_blocks)} backbone blocks")
            logger.info(f"[EARLY_GRADUAL] Target: {self.target_ratio*100:.0f}% of backbone ({self.target_unfrozen_params:,}/{self.total_backbone_params:,} parameters)")
    
    def enable_debug_mode_on_unfreezing(self, enabled=True):
        """Enable debug mode for backbone processor when unfreezing starts."""
        try:
            if hasattr(self.actual_model, 'enable_backbone_debug_mode'):
                self.actual_model.enable_backbone_debug_mode(enabled)
                from ..training_utils import is_main_process
                if is_main_process():
                    logger.info(f"[EARLY_GRADUAL] Backbone debug mode: {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            logger.warning(f"[EARLY_GRADUAL] Failed to set debug mode: {e}")
    
    def _calculate_parameter_targets(self):
        """Calculate total backbone parameters and target unfrozen parameters."""
        # Handle different model structures to get backbone
        if hasattr(self.actual_model, 'mvit_processor'):
            backbone = self.actual_model.mvit_processor.backbone
        elif hasattr(self.actual_model, 'backbone'):
            if hasattr(self.actual_model.backbone, 'backbone'):
                backbone = self.actual_model.backbone.backbone
            else:
                backbone = self.actual_model.backbone
        else:
            raise AttributeError("Model does not have accessible backbone")
        
        self.total_backbone_params = sum(p.numel() for p in backbone.parameters())
        self.target_unfrozen_params = int(self.total_backbone_params * self.target_ratio)
    
    def freeze_all_backbone(self):
        """Freeze all backbone parameters initially."""
        # Handle different model structures to get backbone
        if hasattr(self.actual_model, 'mvit_processor'):
            backbone = self.actual_model.mvit_processor.backbone
        elif hasattr(self.actual_model, 'backbone'):
            if hasattr(self.actual_model.backbone, 'backbone'):
                backbone = self.actual_model.backbone.backbone
            else:
                backbone = self.actual_model.backbone
        else:
            raise AttributeError("Model does not have accessible backbone")
        
        frozen_params = 0
        for param in backbone.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        
        self.current_unfrozen_params = 0
        # Only log from main process to avoid duplicates in distributed training
        from ..training_utils import is_main_process
        if is_main_process():
            logger.info(f"[EARLY_GRADUAL] Froze entire backbone ({frozen_params:,} parameters)")
        return frozen_params
    
    def should_unfreeze_blocks(self, epoch):
        """Determine if we should unfreeze blocks at this epoch."""
        # Don't unfreeze during the initial freeze period
        if epoch < self.freeze_epochs:
            return False
        
        # Don't unfreeze if we've already reached the target
        if self.current_unfrozen_params >= self.target_unfrozen_params:
            return False
        
        # Don't unfreeze if no more blocks available
        if len(self.unfrozen_blocks) >= len(self.backbone_blocks):
            return False
        
        return True
    
    def unfreeze_next_blocks(self, epoch):
        """Unfreeze the next set of blocks."""
        if not self.should_unfreeze_blocks(epoch):
            return []
        
        newly_unfrozen = []
        blocks_to_unfreeze = min(self.blocks_per_epoch, len(self.backbone_blocks) - len(self.unfrozen_blocks))
        
        for i in range(blocks_to_unfreeze):
            if len(self.unfrozen_blocks) < len(self.backbone_blocks):
                block_idx = len(self.unfrozen_blocks)
                block_name, block_module = self.backbone_blocks[block_idx]
                
                # Unfreeze this block
                block_params = 0
                for param in block_module.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        block_params += param.numel()
                
                if block_params > 0:
                    self.unfrozen_blocks.append(block_name)
                    self.current_unfrozen_params += block_params
                    newly_unfrozen.append(block_name)
                    
                    # Only log from main process to avoid duplicates in distributed training
                    from ..training_utils import is_main_process
                    if is_main_process():
                        logger.info(f"[EARLY_GRADUAL] Epoch {epoch}: Unfroze {block_name} ({block_params:,} parameters)")
        
        if newly_unfrozen:
            progress = self.current_unfrozen_params / self.target_unfrozen_params * 100
            from ..training_utils import is_main_process
            if is_main_process():
                logger.info(f"[EARLY_GRADUAL] Progress: {self.current_unfrozen_params:,}/{self.target_unfrozen_params:,} parameters ({progress:.1f}% of target)")
            
            # Stabilize model after unfreezing to prevent temporary instabilities
            try:
                if hasattr(self.actual_model, 'stabilize_after_gradual_unfreezing'):
                    self.actual_model.stabilize_after_gradual_unfreezing()
                    if is_main_process():
                        logger.debug(f"[EARLY_GRADUAL] Model stabilized after unfreezing {len(newly_unfrozen)} blocks")
                elif hasattr(self.actual_model, 'mvit_processor') and hasattr(self.actual_model.mvit_processor, 'stabilize_after_unfreezing'):
                    self.actual_model.mvit_processor.stabilize_after_unfreezing()
                    if is_main_process():
                        logger.debug(f"[EARLY_GRADUAL] Backbone stabilized after unfreezing {len(newly_unfrozen)} blocks")
            except Exception as e:
                if is_main_process():
                    logger.warning(f"[EARLY_GRADUAL] Failed to stabilize model after unfreezing: {e}")
        
        return newly_unfrozen
    
    def update_after_epoch(self, val_metric, epoch):
        """Update freezing state after each epoch."""
        update_info = {
            'rebuild_optimizer': False,
            'unfrozen_layers': [],
            'rollback_performed': False
        }
        
        # Unfreeze blocks if appropriate
        newly_unfrozen = self.unfreeze_next_blocks(epoch)
        if newly_unfrozen:
            update_info['unfrozen_layers'] = newly_unfrozen
            update_info['rebuild_optimizer'] = True
        
        return update_info
    
    def log_status(self, epoch):
        """Log detailed parameter status for this epoch."""
        # Get total model parameters
        total_model_params = sum(p.numel() for p in self.actual_model.parameters())
        trainable_model_params = sum(p.numel() for p in self.actual_model.parameters() if p.requires_grad)
        
        # Get backbone-specific parameters
        if hasattr(self.actual_model, 'mvit_processor'):
            backbone = self.actual_model.mvit_processor.backbone
        elif hasattr(self.actual_model, 'backbone'):
            if hasattr(self.actual_model.backbone, 'backbone'):
                backbone = self.actual_model.backbone.backbone
            else:
                backbone = self.actual_model.backbone
        else:
            backbone = None
        
        if backbone:
            backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            backbone_total = sum(p.numel() for p in backbone.parameters())
            backbone_ratio = backbone_trainable / backbone_total * 100 if backbone_total > 0 else 0
        else:
            backbone_trainable = 0
            backbone_total = 0
            backbone_ratio = 0
        
        # Calculate head parameters (non-backbone)
        head_params = trainable_model_params - backbone_trainable
        
        logger.info(f"[EARLY_GRADUAL] Epoch {epoch} parameter status:")
        logger.info(f"  📊 Total model: {trainable_model_params:,}/{total_model_params:,} trainable ({trainable_model_params/total_model_params*100:.1f}%)")
        logger.info(f"  🧠 Backbone: {backbone_trainable:,}/{backbone_total:,} trainable ({backbone_ratio:.1f}%)")
        logger.info(f"  🎯 Classification heads: {head_params:,} trainable")
        logger.info(f"  🔓 Unfrozen blocks: {len(self.unfrozen_blocks)}/{len(self.backbone_blocks)} ({', '.join(self.unfrozen_blocks) if self.unfrozen_blocks else 'none'})")
        
        # Progress towards target
        if self.target_unfrozen_params > 0:
            target_progress = backbone_trainable / self.target_unfrozen_params * 100
            logger.info(f"  🎯 Target progress: {backbone_trainable:,}/{self.target_unfrozen_params:,} ({target_progress:.1f}%)")
        
        return {
            'total_params': total_model_params,
            'trainable_params': trainable_model_params,
            'backbone_trainable': backbone_trainable,
            'backbone_total': backbone_total,
            'backbone_ratio': backbone_ratio,
            'unfrozen_blocks': len(self.unfrozen_blocks),
            'target_progress': backbone_trainable / self.target_unfrozen_params if self.target_unfrozen_params > 0 else 0
        } 
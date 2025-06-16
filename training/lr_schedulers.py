"""
Learning Rate Schedulers with Warmup Support

This module provides the WarmupWrapper class that implements linear learning rate
warmup before delegating to any PyTorch scheduler.
"""

import torch
import torch.optim.lr_scheduler
import logging
from typing import Optional, Union, List

logger = logging.getLogger(__name__)


class WarmupWrapper(torch.optim.lr_scheduler._LRScheduler):
    """
    Wraps any PyTorch scheduler with linear learning rate warmup.
    
    During the warmup period, linearly interpolates from start_lr to the base learning rates.
    After warmup, delegates to the wrapped scheduler.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of warmup steps
        after_scheduler: The scheduler to use after warmup
        start_lr: Starting learning rate for warmup
        last_epoch: The index of last epoch
    
    Note:
        This wrapper calls step() on every batch during warmup and then delegates to the
        wrapped scheduler. For ReduceLROnPlateau, this means it will continue to be
        stepped per epoch after warmup (as intended), but other schedulers like
        CosineAnnealingLR will be stepped per batch after warmup.
        
        Important: ReduceLROnPlateau should ideally continue to be stepped per epoch
        after warmup, not per batch. The current implementation works correctly for
        ReduceLROnPlateau because the training loop handles it separately, but future
        improvements could add scheduler-specific stepping logic here.
    """
    
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer,
        warmup_steps: int, 
        after_scheduler: torch.optim.lr_scheduler._LRScheduler,
        start_lr: float,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        self.start_lr = start_lr
        
        # Store original base_lrs for warmup calculation
        self._original_base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Set initial learning rate to start_lr
        for group in optimizer.param_groups:
            group['lr'] = start_lr
        
        # Initialize parent with the start learning rate as base_lrs
        super().__init__(optimizer, last_epoch)
        
        logger.info(f"WarmupWrapper initialized: {warmup_steps} warmup steps, "
                   f"start_lr={start_lr:.2e}, target_lrs={self._original_base_lrs}")
    
    def get_lr(self) -> List[float]:
        """Get learning rates for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear interpolation from start_lr to base_lr during warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [
                self.start_lr + warmup_factor * (base_lr - self.start_lr)
                for base_lr in self._original_base_lrs
            ]
        else:
            # Delegate to after_scheduler
            # Ensure after_scheduler has correct base_lrs and step count
            if self.last_epoch == self.warmup_steps:
                # First step after warmup - initialize after_scheduler
                self.after_scheduler.base_lrs = self._original_base_lrs
                for group, base_lr in zip(self.optimizer.param_groups, self._original_base_lrs):
                    group['lr'] = base_lr
                # Reset after_scheduler to start from step 0
                self.after_scheduler.last_epoch = -1
                
            # Get LR from after_scheduler at adjusted step
            adjusted_step = self.last_epoch - self.warmup_steps
            
            # Temporarily set after_scheduler's last_epoch to get correct LR
            original_last_epoch = self.after_scheduler.last_epoch
            self.after_scheduler.last_epoch = adjusted_step
            lrs = self.after_scheduler.get_lr()
            self.after_scheduler.last_epoch = original_last_epoch
            
            return lrs
    
    def step(self, epoch: Optional[int] = None) -> None:
        """
        Step the scheduler.
        
        Args:
            epoch: Optional epoch number. If None, increments internal counter.
        """
        if self.last_epoch < self.warmup_steps:
            # During warmup, use parent step method
            super().step(epoch)
        else:
            # After warmup, step the wrapped scheduler
            if epoch is None:
                self.after_scheduler.step()
                self.last_epoch += 1
            else:
                # Adjust epoch for after_scheduler
                adjusted_epoch = epoch - self.warmup_steps
                self.after_scheduler.step(adjusted_epoch)
                self.last_epoch = epoch
            
            # Update our last_lr to match after_scheduler
            self._last_lr = self.after_scheduler.get_last_lr()
    
    def get_last_lr(self) -> List[float]:
        """Get the last computed learning rates."""
        if hasattr(self, '_last_lr'):
            return self._last_lr
        else:
            return self.get_lr()
    
    def state_dict(self) -> dict:
        """Return state dict for checkpointing."""
        state_dict = super().state_dict()
        state_dict['warmup_steps'] = self.warmup_steps
        state_dict['start_lr'] = self.start_lr
        state_dict['_original_base_lrs'] = self._original_base_lrs
        state_dict['after_scheduler'] = self.after_scheduler.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict from checkpoint."""
        after_scheduler_state = state_dict.pop('after_scheduler')
        self.warmup_steps = state_dict.pop('warmup_steps')
        self.start_lr = state_dict.pop('start_lr')
        self._original_base_lrs = state_dict.pop('_original_base_lrs')
        
        super().load_state_dict(state_dict)
        self.after_scheduler.load_state_dict(after_scheduler_state) 
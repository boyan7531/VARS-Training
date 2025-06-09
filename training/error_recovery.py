"""
Error recovery and robustness utilities for training.

This module provides:
- Graceful GPU OOM error handling
- Configuration validation for incompatible arguments
- Fallback strategies when components fail
"""

import torch
import logging
import traceback
from functools import wraps
from typing import Dict, Any, Optional, Callable
import gc

logger = logging.getLogger(__name__)


class OOMRecoveryManager:
    """Manages GPU Out-of-Memory error recovery."""
    
    def __init__(self, initial_batch_size, min_batch_size=1, reduction_factor=0.75):
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.reduction_factor = reduction_factor
        self.oom_count = 0
        self.max_oom_retries = 3
        self.recovery_strategies = []
        
    def add_recovery_strategy(self, strategy_func, description):
        """Add a recovery strategy function."""
        self.recovery_strategies.append((strategy_func, description))
    
    def handle_oom(self, model=None, optimizer=None, scheduler=None):
        """Handle OOM error with progressive recovery strategies."""
        self.oom_count += 1
        logger.error(f"🚨 GPU OOM Error #{self.oom_count} detected!")
        
        if self.oom_count > self.max_oom_retries:
            logger.error(f"❌ Maximum OOM retries ({self.max_oom_retries}) exceeded. Training cannot continue.")
            raise RuntimeError("Unable to recover from repeated OOM errors")
        
        # Clear GPU memory
        self._emergency_memory_cleanup()
        
        # Try recovery strategies in order
        for i, (strategy_func, description) in enumerate(self.recovery_strategies):
            logger.info(f"🔧 Attempting recovery strategy {i+1}: {description}")
            try:
                success = strategy_func(model, optimizer, scheduler)
                if success:
                    logger.info(f"✅ Recovery successful using strategy: {description}")
                    return True
            except Exception as e:
                logger.warning(f"⚠️ Recovery strategy failed: {e}")
                continue
        
        # Final fallback: reduce batch size
        return self._reduce_batch_size()
    
    def _emergency_memory_cleanup(self):
        """Emergency GPU memory cleanup."""
        logger.info("🧹 Emergency memory cleanup...")
        
        # Clear Python garbage
        gc.collect()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Log memory status
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            
            logger.info(f"   Memory after cleanup: {allocated/1e9:.1f}GB allocated, {cached/1e9:.1f}GB cached, {total/1e9:.1f}GB total")
    
    def _reduce_batch_size(self):
        """Reduce batch size as last resort."""
        new_batch_size = max(int(self.current_batch_size * self.reduction_factor), self.min_batch_size)
        
        if new_batch_size < self.current_batch_size:
            logger.warning(f"📉 Reducing batch size: {self.current_batch_size} → {new_batch_size}")
            self.current_batch_size = new_batch_size
            return True
        else:
            logger.error(f"❌ Cannot reduce batch size further (already at minimum: {self.min_batch_size})")
            return False
    
    def get_accumulation_steps(self):
        """Get gradient accumulation steps to maintain effective batch size."""
        return max(1, self.initial_batch_size // self.current_batch_size)


def oom_safe_forward(oom_manager: OOMRecoveryManager):
    """Decorator for OOM-safe forward passes."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"OOM during {func.__name__}: {e}")
                    # The OOM manager will handle recovery at a higher level
                    raise  # Re-raise for higher-level handling
                else:
                    raise  # Re-raise non-OOM errors
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                logger.error(traceback.format_exc())
                raise
        return wrapper
    return decorator


class ConfigValidator:
    """Validates training configuration for incompatible argument combinations."""
    
    def __init__(self):
        self.validation_rules = []
        self.warnings = []
        self.errors = []
    
    def add_rule(self, rule_func, error_level='error'):
        """Add a validation rule."""
        self.validation_rules.append((rule_func, error_level))
    
    def validate(self, args) -> bool:
        """Validate configuration and return True if valid."""
        self.warnings.clear()
        self.errors.clear()
        
        # Run all validation rules
        for rule_func, error_level in self.validation_rules:
            try:
                result = rule_func(args)
                if result is not True:  # Rule failed
                    message = result if isinstance(result, str) else "Validation rule failed"
                    if error_level == 'error':
                        self.errors.append(message)
                    else:
                        self.warnings.append(message)
            except Exception as e:
                self.errors.append(f"Validation rule error: {e}")
        
        # Log results
        for warning in self.warnings:
            logger.warning(f"⚠️ CONFIG WARNING: {warning}")
        
        for error in self.errors:
            logger.error(f"❌ CONFIG ERROR: {error}")
        
        return len(self.errors) == 0
    
    def get_validation_summary(self):
        """Get summary of validation results."""
        return {
            'valid': len(self.errors) == 0,
            'warnings': self.warnings.copy(),
            'errors': self.errors.copy()
        }


def create_config_validator():
    """Create a config validator with standard rules."""
    validator = ConfigValidator()
    
    # Rule 1: Batch size vs GPU memory
    def validate_batch_size(args):
        if hasattr(args, 'batch_size') and args.batch_size > 32:
            if not torch.cuda.is_available():
                return "Large batch size specified but no GPU available"
            
            # Check available GPU memory
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                if total_memory < 8e9 and args.batch_size > 16:  # Less than 8GB
                    return "warning: Large batch size may cause OOM on GPU with <8GB memory"
        return True
    
    # Rule 2: Freezing strategy compatibility
    def validate_freezing_strategy(args):
        if hasattr(args, 'freezing_strategy') and hasattr(args, 'gradual_finetuning'):
            if args.freezing_strategy == 'none' and args.gradual_finetuning:
                return "Gradual fine-tuning requires a freezing strategy other than 'none'"
        return True
    
    # Rule 3: Augmentation compatibility
    def validate_augmentation(args):
        if hasattr(args, 'disable_augmentation') and hasattr(args, 'extreme_augmentation'):
            if args.disable_augmentation and (args.extreme_augmentation or args.aggressive_augmentation):
                return "Cannot disable augmentation while enabling aggressive/extreme augmentation"
        return True
    
    # Rule 4: Learning rate sanity
    def validate_learning_rates(args):
        if hasattr(args, 'lr') and hasattr(args, 'backbone_lr'):
            if args.lr < args.backbone_lr:
                return "warning: Head learning rate is lower than backbone learning rate (unusual)"
            if args.lr > 1e-1:
                return "warning: Very high learning rate may cause training instability"
            if args.backbone_lr > 1e-2:
                return "warning: Very high backbone learning rate may cause instability"
        return True
    
    # Rule 5: Class balancing compatibility
    def validate_class_balancing(args):
        if hasattr(args, 'use_class_balanced_sampler') and hasattr(args, 'loss_function'):
            if args.use_class_balanced_sampler and args.loss_function == 'weighted':
                return "warning: Using both class-balanced sampler AND weighted loss may over-compensate for imbalance"
        return True
    
    # Rule 6: Multi-GPU compatibility
    def validate_multi_gpu(args):
        if hasattr(args, 'num_workers'):
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if gpu_count > 1 and args.num_workers == 0:
                return "warning: Multi-GPU setup with num_workers=0 may cause data loading bottlenecks"
        return True
    
    # Rule 7: Test run compatibility
    def validate_test_run(args):
        if hasattr(args, 'test_run') and args.test_run:
            if hasattr(args, 'epochs') and args.epochs > 5:
                return "warning: Test run with many epochs - consider reducing for faster testing"
            if hasattr(args, 'save_checkpoints') and args.save_checkpoints:
                return "warning: Test run with checkpoint saving enabled - may clutter filesystem"
        return True
    
    # Add all rules
    validator.add_rule(validate_batch_size, 'error')
    validator.add_rule(validate_freezing_strategy, 'error')
    validator.add_rule(validate_augmentation, 'error')
    validator.add_rule(validate_learning_rates, 'warning')
    validator.add_rule(validate_class_balancing, 'warning')
    validator.add_rule(validate_multi_gpu, 'warning')
    validator.add_rule(validate_test_run, 'warning')
    
    return validator


class FallbackManager:
    """Manages fallback strategies when components fail."""
    
    def __init__(self):
        self.fallback_registry = {}
        self.failure_count = {}
    
    def register_fallback(self, component_name: str, fallback_func: Callable, description: str):
        """Register a fallback strategy for a component."""
        if component_name not in self.fallback_registry:
            self.fallback_registry[component_name] = []
        
        self.fallback_registry[component_name].append({
            'func': fallback_func,
            'description': description
        })
    
    def try_fallback(self, component_name: str, *args, **kwargs):
        """Try fallback strategies for a failed component."""
        if component_name not in self.fallback_registry:
            logger.error(f"❌ No fallback strategies registered for component: {component_name}")
            return None
        
        self.failure_count[component_name] = self.failure_count.get(component_name, 0) + 1
        logger.warning(f"🔄 Component '{component_name}' failed. Trying fallback strategies... (failure #{self.failure_count[component_name]})")
        
        for i, fallback in enumerate(self.fallback_registry[component_name]):
            try:
                logger.info(f"   Trying fallback {i+1}: {fallback['description']}")
                result = fallback['func'](*args, **kwargs)
                if result is not None:
                    logger.info(f"✅ Fallback successful: {fallback['description']}")
                    return result
            except Exception as e:
                logger.warning(f"⚠️ Fallback {i+1} failed: {e}")
                continue
        
        logger.error(f"❌ All fallback strategies failed for component: {component_name}")
        return None


def create_fallback_manager():
    """Create a fallback manager with standard strategies."""
    manager = FallbackManager()
    
    # Freezing strategy fallbacks
    def fallback_to_fixed_freezing(*args, **kwargs):
        """Fallback to basic fixed freezing when advanced strategies fail."""
        try:
            from .model_utils import SmartFreezingManager
            logger.info("Falling back to basic SmartFreezingManager")
            return SmartFreezingManager(*args, **kwargs)
        except Exception as e:
            logger.error(f"Fixed freezing fallback failed: {e}")
            return None
    
    def fallback_to_no_freezing(*args, **kwargs):
        """Ultimate fallback: no freezing at all."""
        logger.warning("Using ultimate fallback: NO FREEZING")
        return None
    
    manager.register_fallback('freezing_manager', fallback_to_fixed_freezing, 'Basic fixed freezing')
    manager.register_fallback('freezing_manager', fallback_to_no_freezing, 'No freezing (ultimate fallback)')
    
    # Optimizer fallbacks
    def fallback_to_adam(*args, **kwargs):
        """Fallback to basic Adam optimizer."""
        try:
            model = args[0] if args else kwargs.get('model')
            lr = kwargs.get('lr', 1e-3)
            return torch.optim.Adam(model.parameters(), lr=lr)
        except Exception as e:
            logger.error(f"Adam fallback failed: {e}")
            return None
    
    def fallback_to_sgd(*args, **kwargs):
        """Ultimate fallback to SGD."""
        try:
            model = args[0] if args else kwargs.get('model')
            lr = kwargs.get('lr', 1e-3)
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        except Exception as e:
            logger.error(f"SGD fallback failed: {e}")
            return None
    
    manager.register_fallback('optimizer', fallback_to_adam, 'Basic Adam optimizer')
    manager.register_fallback('optimizer', fallback_to_sgd, 'SGD optimizer (ultimate fallback)')
    
    # Scheduler fallbacks
    def fallback_to_step_lr(*args, **kwargs):
        """Fallback to simple step learning rate scheduler."""
        try:
            optimizer = args[0] if args else kwargs.get('optimizer')
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        except Exception as e:
            logger.error(f"StepLR fallback failed: {e}")
            return None
    
    def fallback_to_no_scheduler(*args, **kwargs):
        """Ultimate fallback: no scheduler."""
        logger.warning("Using ultimate fallback: NO SCHEDULER")
        return None
    
    manager.register_fallback('scheduler', fallback_to_step_lr, 'Simple step learning rate scheduler')
    manager.register_fallback('scheduler', fallback_to_no_scheduler, 'No scheduler (ultimate fallback)')
    
    return manager


class RobustTrainingWrapper:
    """Wrapper that adds robustness to training components."""
    
    def __init__(self, config_validator=None, oom_manager=None, fallback_manager=None):
        self.config_validator = config_validator or create_config_validator()
        self.oom_manager = oom_manager
        self.fallback_manager = fallback_manager or create_fallback_manager()
        self.error_history = []
    
    def validate_config(self, args):
        """Validate configuration before training."""
        return self.config_validator.validate(args)
    
    def robust_component_creation(self, component_name: str, create_func: Callable, *args, **kwargs):
        """Create a component with fallback on failure."""
        try:
            return create_func(*args, **kwargs)
        except Exception as e:
            self.error_history.append(f"{component_name}: {e}")
            logger.error(f"❌ Failed to create {component_name}: {e}")
            
            # Try fallback
            result = self.fallback_manager.try_fallback(component_name, *args, **kwargs)
            if result is None:
                raise RuntimeError(f"Failed to create {component_name} and all fallbacks failed")
            return result
    
    def oom_safe_training_step(self, training_func, *args, **kwargs):
        """Execute training step with OOM protection."""
        if not self.oom_manager:
            return training_func(*args, **kwargs)
        
        try:
            return training_func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"OOM during training step: {e}")
                
                # Try to recover
                recovery_success = self.oom_manager.handle_oom(
                    model=kwargs.get('model'),
                    optimizer=kwargs.get('optimizer'),
                    scheduler=kwargs.get('scheduler')
                )
                
                if recovery_success:
                    logger.info("🔄 Retrying training step after OOM recovery...")
                    # Update batch size in kwargs if available
                    if 'batch_size' in kwargs:
                        kwargs['batch_size'] = self.oom_manager.current_batch_size
                    return training_func(*args, **kwargs)
                else:
                    raise RuntimeError("Unable to recover from OOM error")
            else:
                raise  # Re-raise non-OOM errors
    
    def get_error_summary(self):
        """Get summary of all errors encountered."""
        return {
            'config_validation': self.config_validator.get_validation_summary(),
            'component_errors': self.error_history.copy(),
            'oom_events': self.oom_manager.oom_count if self.oom_manager else 0
        } 
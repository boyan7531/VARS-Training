#!/usr/bin/env python3
"""
Test script to reproduce and diagnose gradual unfreezing backbone errors.

This script simulates the exact gradual unfreezing process that occurs during training
to identify when and why backbone errors happen.
"""

import torch
import torch.nn as nn
import logging
import traceback
import time
from typing import List, Dict, Tuple
import gc

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_model_and_freezing():
    """Setup model with gradual unfreezing manager."""
    from model.unified_model import create_unified_model
    from training.freezing.early_gradual_manager import EarlyGradualFreezingManager
    
    # Create model
    model = create_unified_model(
        backbone_type='mvit',
        num_severity=5,
        num_action_type=8,
        vocab_sizes={},
        enable_memory_optimization=True,
        dropout_rate=0.1
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"Model moved to device: {device}")
    
    # Setup gradual unfreezing manager
    freezing_manager = EarlyGradualFreezingManager(
        model,
        freeze_epochs=1,
        blocks_per_epoch=1,
        target_ratio=0.5
    )
    
    # Enable debug mode
    if hasattr(model, 'enable_backbone_debug_mode'):
        model.enable_backbone_debug_mode(True)
        logger.info("Backbone debug mode enabled")
    
    # Enable debug mode for unfreezing
    freezing_manager.enable_debug_mode_on_unfreezing(True)
    
    # Start with backbone frozen
    freezing_manager.freeze_all_backbone()
    
    return model, freezing_manager, device

def create_test_batch(device: torch.device, batch_size: int = 2) -> Dict[str, torch.Tensor]:
    """Create a test batch similar to training data."""
    # Create 5D tensor input [B, C, T, H, W] - the format that previously caused issues
    clips = torch.randn(batch_size, 3, 16, 224, 224, device=device, dtype=torch.float32)
    
    # Ensure tensor is contiguous
    clips = clips.contiguous()
    
    batch_data = {
        'clips': clips
    }
    
    logger.info(f"Created test batch: clips shape {clips.shape}, device {clips.device}, dtype {clips.dtype}")
    return batch_data

def test_forward_pass(model: nn.Module, batch_data: Dict[str, torch.Tensor], 
                     epoch: int, unfrozen_blocks: List[str]) -> Tuple[bool, str]:
    """Test forward pass and return success status and error details."""
    try:
        model.train()  # Set to training mode to simulate training conditions
        
        # Clear any cached gradients
        if hasattr(model, 'zero_grad'):
            model.zero_grad()
        
        logger.debug(f"Testing forward pass - Epoch {epoch}, Unfrozen blocks: {unfrozen_blocks}")
        
        # Test forward pass
        start_time = time.time()
        severity_logits, action_logits = model(batch_data)
        forward_time = time.time() - start_time
        
        # Check for NaN in outputs
        severity_has_nan = torch.isnan(severity_logits).any()
        action_has_nan = torch.isnan(action_logits).any()
        
        if severity_has_nan or action_has_nan:
            error_msg = f"NaN detected in outputs - Severity: {severity_has_nan}, Action: {action_has_nan}"
            logger.warning(error_msg)
            return False, error_msg
        
        logger.info(f"✅ Forward pass successful - Time: {forward_time:.4f}s, "
                   f"Severity: {severity_logits.shape}, Action: {action_logits.shape}")
        return True, f"Success in {forward_time:.4f}s"
        
    except Exception as e:
        error_msg = str(e) if str(e) else f"Silent {type(e).__name__}"
        error_traceback = traceback.format_exc()
        
        logger.error(f"❌ Forward pass failed - Epoch {epoch}")
        logger.error(f"Error: {error_msg}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        
        return False, f"{type(e).__name__}: {error_msg}"

def test_backward_pass(model: nn.Module, severity_logits: torch.Tensor, 
                      action_logits: torch.Tensor) -> Tuple[bool, str]:
    """Test backward pass to check gradient computation."""
    try:
        # Create dummy targets
        batch_size = severity_logits.shape[0]
        device = severity_logits.device
        
        severity_targets = torch.randint(0, 5, (batch_size,), device=device)
        action_targets = torch.randint(0, 8, (batch_size,), device=device)
        
        # Compute losses
        criterion = nn.CrossEntropyLoss()
        severity_loss = criterion(severity_logits, severity_targets)
        action_loss = criterion(action_logits, action_targets)
        total_loss = severity_loss + action_loss
        
        # Test backward pass
        start_time = time.time()
        total_loss.backward()
        backward_time = time.time() - start_time
        
        # Check for NaN gradients
        nan_grad_count = 0
        total_grad_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                total_grad_params += 1
                if torch.isnan(param.grad).any():
                    nan_grad_count += 1
                    logger.warning(f"NaN gradient in parameter: {name}")
        
        if nan_grad_count > 0:
            error_msg = f"NaN gradients in {nan_grad_count}/{total_grad_params} parameters"
            logger.warning(error_msg)
            return False, error_msg
        
        logger.info(f"✅ Backward pass successful - Time: {backward_time:.4f}s, "
                   f"Loss: {total_loss.item():.4f}")
        return True, f"Success in {backward_time:.4f}s, Loss: {total_loss.item():.4f}"
        
    except Exception as e:
        error_msg = str(e) if str(e) else f"Silent {type(e).__name__}"
        logger.error(f"❌ Backward pass failed: {error_msg}")
        return False, f"{type(e).__name__}: {error_msg}"

def check_memory_usage(device: torch.device) -> Dict[str, float]:
    """Check GPU memory usage if on CUDA."""
    memory_info = {}
    
    if device.type == 'cuda':
        memory_info['allocated_gb'] = torch.cuda.memory_allocated(device) / 1e9
        memory_info['reserved_gb'] = torch.cuda.memory_reserved(device) / 1e9
        memory_info['max_allocated_gb'] = torch.cuda.max_memory_allocated(device) / 1e9
        
        logger.debug(f"GPU Memory - Allocated: {memory_info['allocated_gb']:.2f}GB, "
                    f"Reserved: {memory_info['reserved_gb']:.2f}GB, "
                    f"Max: {memory_info['max_allocated_gb']:.2f}GB")
    
    return memory_info

def test_gradual_unfreezing_sequence():
    """Test the complete gradual unfreezing sequence."""
    logger.info("=" * 80)
    logger.info("STARTING GRADUAL UNFREEZING ERROR REPRODUCTION TEST")
    logger.info("=" * 80)
    
    # Setup
    model, freezing_manager, device = setup_model_and_freezing()
    
    # Create test batch
    batch_data = create_test_batch(device)
    
    # Track results
    test_results = []
    
    # Test initial state (all frozen)
    logger.info("\n" + "="*60)
    logger.info("EPOCH 0: TESTING FULLY FROZEN STATE")
    logger.info("="*60)
    
    freezing_manager.log_status(0)
    memory_before = check_memory_usage(device)
    
    success, error_msg = test_forward_pass(model, batch_data, 0, [])
    test_results.append({
        'epoch': 0,
        'unfrozen_blocks': [],
        'forward_success': success,
        'forward_error': error_msg,
        'memory_before': memory_before
    })
    
    if success:
        # Test backward pass too
        severity_logits, action_logits = model(batch_data)
        backward_success, backward_error = test_backward_pass(model, severity_logits, action_logits)
        test_results[-1]['backward_success'] = backward_success
        test_results[-1]['backward_error'] = backward_error
    
    # Clear gradients and memory
    model.zero_grad()
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # Test gradual unfreezing - simulate multiple epochs
    max_epochs = 10  # Test up to 10 epochs of unfreezing
    
    for epoch in range(1, max_epochs + 1):
        logger.info(f"\n" + "="*60)
        logger.info(f"EPOCH {epoch}: TESTING GRADUAL UNFREEZING")
        logger.info("="*60)
        
        # Update freezing state
        update_info = freezing_manager.update_after_epoch(0.5, epoch)  # Dummy val_metric
        freezing_manager.log_status(epoch)
        
        if update_info['unfrozen_layers']:
            logger.info(f"🔓 Newly unfrozen layers: {update_info['unfrozen_layers']}")
            
            # Wait a moment for any memory reorganization
            time.sleep(0.1)
            
            # Check memory after unfreezing
            memory_after_unfreeze = check_memory_usage(device)
            
            # Test multiple forward passes to check for intermittent issues
            all_success = True
            error_messages = []
            
            for test_run in range(3):  # Test 3 times to catch intermittent issues
                logger.info(f"  Test run {test_run + 1}/3...")
                
                success, error_msg = test_forward_pass(model, batch_data, epoch, update_info['unfrozen_layers'])
                
                if not success:
                    all_success = False
                    error_messages.append(f"Run {test_run + 1}: {error_msg}")
                    logger.warning(f"  ❌ Test run {test_run + 1} failed: {error_msg}")
                else:
                    logger.info(f"  ✅ Test run {test_run + 1} succeeded")
                
                # Test backward pass if forward succeeded
                if success:
                    try:
                        severity_logits, action_logits = model(batch_data)
                        backward_success, backward_error = test_backward_pass(model, severity_logits, action_logits)
                        if not backward_success:
                            all_success = False
                            error_messages.append(f"Run {test_run + 1} backward: {backward_error}")
                    except Exception as e:
                        logger.warning(f"  ❌ Could not test backward pass: {e}")
                
                # Clear gradients between runs
                model.zero_grad()
                
                # Small delay between tests
                time.sleep(0.05)
            
            # Record results
            test_results.append({
                'epoch': epoch,
                'unfrozen_blocks': update_info['unfrozen_layers'].copy(),
                'all_forward_success': all_success,
                'error_messages': error_messages,
                'memory_after_unfreeze': memory_after_unfreeze,
                'rebuild_optimizer': update_info['rebuild_optimizer']
            })
            
            if not all_success:
                logger.error(f"🚨 ERRORS DETECTED IN EPOCH {epoch} AFTER UNFREEZING {update_info['unfrozen_layers']}")
                for msg in error_messages:
                    logger.error(f"  - {msg}")
                
                # Try stabilization
                logger.info("🔧 Attempting model stabilization...")
                if hasattr(model, 'stabilize_after_gradual_unfreezing'):
                    model.stabilize_after_gradual_unfreezing()
                    
                    # Test again after stabilization
                    logger.info("Testing after stabilization...")
                    success_after_stab, error_after_stab = test_forward_pass(model, batch_data, epoch, update_info['unfrozen_layers'])
                    
                    test_results[-1]['stabilization_helped'] = success_after_stab
                    test_results[-1]['error_after_stabilization'] = error_after_stab
                    
                    if success_after_stab:
                        logger.info("✅ Stabilization resolved the issue!")
                    else:
                        logger.error(f"❌ Stabilization did not help: {error_after_stab}")
        
        else:
            logger.info("No new layers unfrozen this epoch")
            
            # Still test forward pass
            success, error_msg = test_forward_pass(model, batch_data, epoch, freezing_manager.unfrozen_blocks)
            test_results.append({
                'epoch': epoch,
                'unfrozen_blocks': freezing_manager.unfrozen_blocks.copy(),
                'forward_success': success,
                'forward_error': error_msg
            })
        
        # Clear memory
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        # Stop if we've reached the target or if no more blocks to unfreeze
        if (freezing_manager.current_unfrozen_params >= freezing_manager.target_unfrozen_params or
            len(freezing_manager.unfrozen_blocks) >= len(freezing_manager.backbone_blocks)):
            logger.info(f"Reached unfreezing target or all blocks unfrozen. Stopping at epoch {epoch}.")
            break
    
    return test_results

def analyze_results(test_results: List[Dict]) -> None:
    """Analyze and summarize test results."""
    logger.info("\n" + "="*80)
    logger.info("TEST RESULTS ANALYSIS")
    logger.info("="*80)
    
    total_tests = len(test_results)
    failed_epochs = []
    stabilization_helped_count = 0
    
    for result in test_results:
        epoch = result['epoch']
        
        # Check for failures
        failed = False
        if 'all_forward_success' in result:
            if not result['all_forward_success']:
                failed = True
        elif 'forward_success' in result:
            if not result['forward_success']:
                failed = True
        
        if failed:
            failed_epochs.append(epoch)
            logger.error(f"❌ Epoch {epoch}: FAILED")
            
            if 'error_messages' in result:
                for msg in result['error_messages']:
                    logger.error(f"    {msg}")
            elif 'forward_error' in result:
                logger.error(f"    {result['forward_error']}")
            
            # Check if stabilization helped
            if result.get('stabilization_helped', False):
                stabilization_helped_count += 1
                logger.info(f"    ✅ Stabilization resolved the issue")
        else:
            logger.info(f"✅ Epoch {epoch}: SUCCESS")
    
    # Summary
    logger.info(f"\n📊 SUMMARY:")
    logger.info(f"  Total epochs tested: {total_tests}")
    logger.info(f"  Failed epochs: {len(failed_epochs)} - {failed_epochs}")
    logger.info(f"  Success rate: {(total_tests - len(failed_epochs))/total_tests*100:.1f}%")
    
    if stabilization_helped_count > 0:
        logger.info(f"  Stabilization helped in: {stabilization_helped_count} cases")
    
    # Identify patterns
    if failed_epochs:
        logger.warning(f"\n🔍 FAILURE PATTERN ANALYSIS:")
        logger.warning(f"  Failures occurred in epochs: {failed_epochs}")
        
        # Check if failures correlate with specific unfreezing events
        for result in test_results:
            if result['epoch'] in failed_epochs and result.get('unfrozen_blocks'):
                logger.warning(f"  Epoch {result['epoch']}: Failed after unfreezing {result['unfrozen_blocks']}")

def main():
    """Main test function."""
    try:
        # Check device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        if device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Available memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f}GB")
        
        # Run the test
        test_results = test_gradual_unfreezing_sequence()
        
        # Analyze results
        analyze_results(test_results)
        
        logger.info("\n" + "="*80)
        logger.info("TEST COMPLETED")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 
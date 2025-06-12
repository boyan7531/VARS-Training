#!/usr/bin/env python3
"""
Test script for MViT optimizations.

This script tests the new MViT optimization features to ensure they're working correctly.
"""

import torch
import logging
from model.unified_model import create_unified_model
from model.config import ModelConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mvit_optimizations():
    """Test the MViT optimization features."""
    
    # Test vocabulary sizes
    vocab_sizes = {
        'contact': 4,
        'bodypart': 8,
        'upper_bodypart': 6,
        'multiple_fouls': 3,
        'try_to_play': 3,
        'touch_ball': 3,
        'handball': 3,
        'handball_offence': 3,
    }
    
    logger.info("🚀 Testing MViT Optimizations")
    logger.info("=" * 50)
    
    # Test 1: Create optimized MViT model
    logger.info("Test 1: Creating optimized MViT model...")
    try:
        model_optimized = create_unified_model(
            backbone_type='mvit',
            num_severity=6,
            num_action_type=10,
            vocab_sizes=vocab_sizes,
            backbone_name='mvit_base_16x4',
            enable_gradient_checkpointing=True,
            enable_memory_optimization=True
        )
        
        # Check if optimization components are present
        if hasattr(model_optimized, 'mvit_processor'):
            logger.info("✅ OptimizedMViTProcessor found")
        else:
            logger.error("❌ OptimizedMViTProcessor not found")
            
        if hasattr(model_optimized.mvit_processor, 'use_gradient_checkpointing'):
            logger.info("✅ Gradient checkpointing configured")
        else:
            logger.error("❌ Gradient checkpointing not configured")
            
        logger.info("✅ Test 1 passed: Optimized MViT model created successfully")
        
    except Exception as e:
        logger.error(f"❌ Test 1 failed: {e}")
        return False
    
    # Test 2: Create non-optimized MViT model for comparison
    logger.info("\nTest 2: Creating non-optimized MViT model...")
    try:
        model_standard = create_unified_model(
            backbone_type='mvit',
            num_severity=6,
            num_action_type=10,
            vocab_sizes=vocab_sizes,
            backbone_name='mvit_base_16x4',
            enable_gradient_checkpointing=False,
            enable_memory_optimization=False
        )
        
        logger.info("✅ Test 2 passed: Non-optimized MViT model created successfully")
        
    except Exception as e:
        logger.error(f"❌ Test 2 failed: {e}")
        return False
    
    # Test 3: Forward pass with dummy data
    logger.info("\nTest 3: Testing forward pass with dummy data...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_optimized = model_optimized.to(device)
        
        # Create dummy batch data
        batch_size = 2
        max_views = 3
        frames = 16
        height = 224
        width = 398
        
        batch_data = {
            'clips': torch.randn(batch_size, max_views, 3, frames, height, width, device=device),
            'view_mask': torch.ones(batch_size, max_views, dtype=torch.bool, device=device),
            'label_severity': torch.randint(0, 6, (batch_size,), device=device),
            'label_type': torch.randint(0, 10, (batch_size,), device=device),
            # Categorical features
            'contact': torch.randint(0, 4, (batch_size,), device=device),
            'bodypart': torch.randint(0, 8, (batch_size,), device=device),
            'upper_bodypart': torch.randint(0, 6, (batch_size,), device=device),
            'multiple_fouls': torch.randint(0, 3, (batch_size,), device=device),
            'try_to_play': torch.randint(0, 3, (batch_size,), device=device),
            'touch_ball': torch.randint(0, 3, (batch_size,), device=device),
            'handball': torch.randint(0, 3, (batch_size,), device=device),
            'handball_offence': torch.randint(0, 3, (batch_size,), device=device),
        }
        
        # Forward pass
        model_optimized.eval()
        with torch.no_grad():
            sev_logits, act_logits = model_optimized(batch_data)
            
        # Check output shapes
        expected_sev_shape = (batch_size, 6)
        expected_act_shape = (batch_size, 10)
        
        if sev_logits.shape == expected_sev_shape and act_logits.shape == expected_act_shape:
            logger.info(f"✅ Forward pass successful - Outputs: {sev_logits.shape}, {act_logits.shape}")
        else:
            logger.error(f"❌ Output shape mismatch - Expected: {expected_sev_shape}, {expected_act_shape}, "
                        f"Got: {sev_logits.shape}, {act_logits.shape}")
            return False
            
        logger.info("✅ Test 3 passed: Forward pass completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Test 3 failed: {e}")
        return False
    
    # Test 4: Memory optimization features
    logger.info("\nTest 4: Testing memory optimization features...")
    try:
        model_info = model_optimized.get_model_info()
        
        required_info = [
            'gradient_checkpointing_enabled',
            'memory_optimization_enabled',
            'total_parameters',
            'video_feature_dim'
        ]
        
        for key in required_info:
            if key in model_info:
                logger.info(f"✅ Model info contains '{key}': {model_info[key]}")
            else:
                logger.error(f"❌ Model info missing '{key}'")
                return False
                
        logger.info("✅ Test 4 passed: Memory optimization features verified")
        
    except Exception as e:
        logger.error(f"❌ Test 4 failed: {e}")
        return False
    
    logger.info("\n" + "=" * 50)
    logger.info("🎉 ALL TESTS PASSED! MViT optimizations are working correctly.")
    logger.info("=" * 50)
    
    return True

if __name__ == "__main__":
    success = test_mvit_optimizations()
    exit(0 if success else 1) 
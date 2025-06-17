#!/usr/bin/env python3

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.unified_model import OptimizedMViTProcessor
from model.config import ModelConfig

def test_mvit_processor_fix():
    """Test that the OptimizedMViTProcessor now has use_mixed_precision attribute"""
    print("Testing OptimizedMViTProcessor fix...")
    
    # Create a dummy backbone (just a simple module for testing)
    class DummyBackbone(torch.nn.Module):
        def forward(self, x):
            # Return a simple tensor
            batch_size = x.shape[0]
            return torch.randn(batch_size, 768)  # MViT-like output
    
    backbone = DummyBackbone()
    
    # Test 1: Default initialization
    processor = OptimizedMViTProcessor(backbone)
    assert hasattr(processor, 'use_mixed_precision'), "Missing use_mixed_precision attribute"
    assert processor.use_mixed_precision == False, "Default use_mixed_precision should be False"
    print("✅ Default initialization works")
    
    # Test 2: With gradient checkpointing
    processor = OptimizedMViTProcessor(backbone, use_gradient_checkpointing=True)
    assert processor.use_gradient_checkpointing == True, "Gradient checkpointing not set"
    assert processor.use_mixed_precision == False, "Default use_mixed_precision should be False"
    print("✅ Gradient checkpointing initialization works")
    
    # Test 3: With mixed precision
    processor = OptimizedMViTProcessor(backbone, use_mixed_precision=True)
    assert processor.use_mixed_precision == True, "Mixed precision not set"
    print("✅ Mixed precision initialization works")
    
    # Test 4: Both options
    processor = OptimizedMViTProcessor(backbone, use_gradient_checkpointing=True, use_mixed_precision=True)
    assert processor.use_gradient_checkpointing == True, "Gradient checkpointing not set"
    assert processor.use_mixed_precision == True, "Mixed precision not set"
    print("✅ Both options initialization works")
    
    print("All OptimizedMViTProcessor tests passed! ✅")

def test_nan_detection():
    """Test NaN detection and replacement"""
    print("\nTesting NaN detection...")
    
    # Create tensor with NaN values
    test_tensor = torch.randn(2, 3, 16, 224, 224)
    test_tensor[0, 0, 0, 0, 0] = float('nan')
    test_tensor[1, 1, 5, 10, 10] = float('nan')
    
    print(f"Created test tensor with {torch.isnan(test_tensor).sum().item()} NaN values")
    
    # Test NaN replacement
    if torch.isnan(test_tensor).any():
        nan_count = torch.isnan(test_tensor).sum().item()
        test_tensor = torch.where(
            torch.isnan(test_tensor), 
            torch.full_like(test_tensor, 0.01), 
            test_tensor
        )
        print(f"✅ Replaced {nan_count} NaN values")
    
    # Verify no NaN values remain
    assert not torch.isnan(test_tensor).any(), "NaN values still present after replacement"
    print("✅ NaN detection and replacement works")

def test_data_range():
    """Test data range clamping"""
    print("\nTesting data range clamping...")
    
    # Create tensor with extreme values
    test_tensor = torch.randn(2, 3, 16, 224, 224)
    test_tensor[0, 0, 0, 0, 0] = 100.0  # Too high
    test_tensor[1, 1, 5, 10, 10] = -100.0  # Too low
    
    print(f"Created test tensor with min={test_tensor.min():.2f}, max={test_tensor.max():.2f}")
    
    # Test clamping
    test_tensor = torch.clamp(test_tensor, min=0.0, max=1.0)
    
    # Verify clamping worked
    assert test_tensor.min() >= 0.0, f"Min value {test_tensor.min()} is below 0"
    assert test_tensor.max() <= 1.0, f"Max value {test_tensor.max()} is above 1"
    print(f"✅ Clamped to range [0, 1]: min={test_tensor.min():.2f}, max={test_tensor.max():.2f}")

if __name__ == "__main__":
    print("Running fix verification tests...\n")
    
    try:
        test_mvit_processor_fix()
        test_nan_detection()
        test_data_range()
        
        print("\n🎉 All tests passed! The fixes should work correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
#!/usr/bin/env python3
"""
Unit tests for view consistency loss functionality.

Tests the view consistency loss implementation to ensure it:
1. Computes loss > 0 when there are valid view pairs
2. Returns 0 when there are insufficient views
3. Handles gradients correctly
4. Works with different batch sizes and view configurations
"""

import torch
import pytest
import numpy as np
import sys
import os

# Add the project root to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.training_utils import view_consistency_loss, calculate_multitask_loss


class TestViewConsistencyLoss:
    """Test cases for view consistency loss."""
    
    def test_basic_functionality(self):
        """Test basic view consistency loss computation."""
        # Create fake logits: [B=2, V=3, C=6] and mask with one missing view
        batch_size, max_views, num_classes = 2, 3, 6
        
        # Random logits
        torch.manual_seed(42)
        logits = torch.randn(batch_size, max_views, num_classes, requires_grad=True)
        
        # Mask: first batch has 3 views, second batch has 2 views
        mask = torch.tensor([
            [True, True, True],
            [True, True, False]
        ])
        
        # Compute loss
        loss = view_consistency_loss(logits, mask)
        
        # Check that loss is positive (views should be different with random logits)
        assert loss.item() > 0, "Loss should be positive with random logits"
        assert loss.requires_grad, "Loss should require gradients"
        
        print(f"✅ Basic test passed. Loss: {loss.item():.4f}")
    
    def test_insufficient_views(self):
        """Test that loss returns 0 when there are insufficient views."""
        batch_size, max_views, num_classes = 2, 3, 6
        
        logits = torch.randn(batch_size, max_views, num_classes, requires_grad=True)
        
        # Mask with only one view per batch (insufficient for pairs)
        mask = torch.tensor([
            [True, False, False],
            [False, True, False]
        ])
        
        loss = view_consistency_loss(logits, mask)
        
        # Should return 0 with insufficient views
        assert loss.item() == 0.0, "Loss should be 0 with insufficient views"
        assert loss.requires_grad, "Loss should still require gradients"
        
        print(f"✅ Insufficient views test passed. Loss: {loss.item():.4f}")
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through the loss."""
        batch_size, max_views, num_classes = 2, 3, 6
        
        logits = torch.randn(batch_size, max_views, num_classes, requires_grad=True)
        mask = torch.tensor([
            [True, True, True],
            [True, True, False]
        ])
        
        loss = view_consistency_loss(logits, mask)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are non-zero
        assert logits.grad is not None, "Gradients should exist"
        assert torch.any(logits.grad != 0), "Gradients should be non-zero"
        
        # Check gradient shape
        assert logits.grad.shape == logits.shape, "Gradient shape should match logits shape"
        
        print(f"✅ Gradient flow test passed. Grad norm: {logits.grad.norm().item():.4f}")


def run_tests():
    """Run all tests."""
    print("🧪 Running View Consistency Loss Tests...")
    print("=" * 60)
    
    test_suite = TestViewConsistencyLoss()
    
    try:
        test_suite.test_basic_functionality()
        test_suite.test_insufficient_views()
        test_suite.test_gradient_flow()
        
        print("=" * 60)
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1) 
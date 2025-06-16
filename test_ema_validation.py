#!/usr/bin/env python3
"""
Test script to verify EMA validation implementation.
This script trains a simple model for a few epochs to verify that EMA improves validation accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import logging

# Import our EMA utilities
from training.model_utils import create_ema_model, update_ema_model, apply_ema_weights, restore_online_weights, should_use_ema_for_validation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleTestModel(nn.Module):
    """Simple test model for EMA validation testing."""
    def __init__(self, input_size=10, hidden_size=32, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


def create_test_dataset(num_samples=1000, input_size=10, num_classes=3, noise_level=0.1):
    """Create a simple synthetic dataset."""
    # Generate random data with some structure
    X = torch.randn(num_samples, input_size)
    
    # Create labels with some correlation to input
    weights = torch.randn(input_size, num_classes)
    logits = X @ weights + noise_level * torch.randn(num_samples, num_classes)
    y = torch.argmax(logits, dim=1)
    
    return TensorDataset(X, y)


def test_ema_validation():
    """Test EMA validation with a simple model."""
    # Create test args object
    class TestArgs:
        def __init__(self):
            self.use_ema = True
            self.ema_eval = True
            self.ema_decay = 0.999
    
    args = TestArgs()
    
    # Create model and data
    model = SimpleTestModel()
    
    # Create datasets
    train_dataset = create_test_dataset(800, noise_level=0.2)
    val_dataset = create_test_dataset(200, noise_level=0.1)  # Less noisy validation
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Create EMA model
    ema_model = create_ema_model(model, decay=args.ema_decay)
    logger.info(f"✅ Created EMA model with decay {args.ema_decay}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    def validate(epoch):
        """Validation function."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # Test with online weights
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        online_acc = 100. * correct / total
        online_loss = total_loss / len(val_loader)
        
        # Test with EMA weights if available
        ema_acc = None
        ema_loss = None
        if should_use_ema_for_validation(args, ema_model, epoch):
            apply_ema_weights(model, ema_model)
            
            total_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_loader):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    total_loss += loss.item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            
            ema_acc = 100. * correct / total
            ema_loss = total_loss / len(val_loader)
            
            # Restore online weights
            restore_online_weights(model, ema_model)
        
        return online_acc, online_loss, ema_acc, ema_loss
    
    # Training loop
    logger.info("Starting EMA validation test...")
    
    for epoch in range(10):
        model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Update EMA
            if ema_model is not None:
                update_ema_model(ema_model, model)
        
        # Validation
        online_acc, online_loss, ema_acc, ema_loss = validate(epoch)
        
        if ema_acc is not None:
            improvement = ema_acc - online_acc
            logger.info(f"Epoch {epoch+1:2d}: Online: {online_acc:.2f}% | EMA: {ema_acc:.2f}% | Improvement: {improvement:+.2f}%")
        else:
            logger.info(f"Epoch {epoch+1:2d}: Online: {online_acc:.2f}% | EMA: Not used (epoch < 2)")
    
    logger.info("✅ EMA validation test completed!")


if __name__ == "__main__":
    test_ema_validation() 
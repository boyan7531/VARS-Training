#!/usr/bin/env python3
"""
Comprehensive test script that covers all critical training components
to ensure no crashes during full training.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import SoccerNetMVFoulDataset, variable_views_collate_fn
from model import MultiTaskMultiViewMViT, ModelConfig
from pytorchvideo.transforms import ShortSideScale, Normalize as VideoNormalize
from torchvision.transforms import Compose, CenterCrop

class ConvertToFloatAndScale(torch.nn.Module):
    def __call__(self, clip_cthw_uint8):
        if clip_cthw_uint8.dtype != torch.uint8:
            return clip_cthw_uint8
        return clip_cthw_uint8.float() / 255.0

class PerFrameCenterCrop(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.cropper = CenterCrop(size)

    def forward(self, clip_cthw):
        clip_tchw = clip_cthw.permute(1, 0, 2, 3)
        cropped_frames = [self.cropper(frame) for frame in clip_tchw]
        cropped_clip_tchw = torch.stack(cropped_frames)
        return cropped_clip_tchw.permute(1, 0, 2, 3)

def test_dataset_and_dataloader():
    """Test dataset + DataLoader with collate function."""
    print("🔍 Testing Dataset + DataLoader...")
    
    transform = Compose([
        ConvertToFloatAndScale(),
        VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ShortSideScale(size=224),
        PerFrameCenterCrop((224, 224))
    ])
    
    dataset = SoccerNetMVFoulDataset(
        dataset_path="mvfouls",
        split='train',
        frames_per_clip=16,
        target_fps=15,
        max_views_to_load=4,
        target_height=224,
        target_width=224,
        transform=transform
    )
    
    # Test DataLoader with collate function
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=2,
        collate_fn=variable_views_collate_fn
    )
    
    # Test batch loading
    batch = next(iter(dataloader))
    print(f"✅ DataLoader works: batch clips shape = {batch['clips'].shape}")
    return dataloader, dataset

def test_model_components(dataloader, dataset):
    """Test model, multi-GPU, mixed precision, optimizer."""
    print("\n🧠 Testing Model Components...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Vocab sizes
    vocab_sizes = {
        'contact': dataset.num_contact_classes,
        'bodypart': dataset.num_bodypart_classes,
        'upper_bodypart': dataset.num_upper_bodypart_classes,
        'lower_bodypart': dataset.num_lower_bodypart_classes,
        'multiple_fouls': dataset.num_multiple_fouls_classes,
        'try_to_play': dataset.num_try_to_play_classes,
        'touch_ball': dataset.num_touch_ball_classes,
        'handball': dataset.num_handball_classes,
        'handball_offence': dataset.num_handball_offence_classes,
    }
    
    # Model setup
    config = ModelConfig(pretrained_model_name='mvit_base_16x4')
    model = MultiTaskMultiViewMViT(
        num_severity=4,
        num_action_type=8,
        vocab_sizes=vocab_sizes,
        config=config
    )
    model.to(device)
    
    # Multi-GPU test
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Testing multi-GPU with {num_gpus} GPUs...")
        model = nn.DataParallel(model)
        print("✅ DataParallel wrapper successful")
    
    # Mixed precision setup
    scaler = GradScaler() if device.type == 'cuda' else None
    if scaler:
        print("✅ Mixed precision scaler created")
    
    return model, device, scaler

def test_training_components(model, dataloader, device, scaler):
    """Test complete training loop components."""
    print("\n🏃 Testing Training Components...")
    
    # Loss functions and optimizer
    criterion_severity = nn.CrossEntropyLoss()
    criterion_action = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    
    # Test training step
    model.train()
    batch = next(iter(dataloader))
    
    # Move batch to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device, non_blocking=True)
    
    print(f"Batch moved to device: {device}")
    
    # Test forward + backward pass
    optimizer.zero_grad()
    
    if scaler is not None:
        print("Testing mixed precision training...")
        with autocast():
            sev_logits, act_logits = model(batch)
            loss_sev = criterion_severity(sev_logits, batch["label_severity"])
            loss_act = criterion_action(act_logits, batch["label_type"])
            total_loss = loss_sev + loss_act
        
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print("✅ Mixed precision training step successful")
    else:
        print("Testing standard precision training...")
        sev_logits, act_logits = model(batch)
        loss_sev = criterion_severity(sev_logits, batch["label_severity"])
        loss_act = criterion_action(act_logits, batch["label_type"])
        total_loss = loss_sev + loss_act
        
        total_loss.backward()
        optimizer.step()
        print("✅ Standard training step successful")
    
    print(f"Loss: {total_loss.item():.4f}")
    return model, optimizer

def test_validation_mode(model, dataloader, device):
    """Test validation mode."""
    print("\n✅ Testing Validation Mode...")
    
    model.eval()
    criterion_severity = nn.CrossEntropyLoss()
    criterion_action = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        
        sev_logits, act_logits = model(batch)
        loss_sev = criterion_severity(sev_logits, batch["label_severity"])
        loss_act = criterion_action(act_logits, batch["label_type"])
        total_loss = loss_sev + loss_act
    
    print(f"✅ Validation step successful, Loss: {total_loss.item():.4f}")

def test_memory_stress():
    """Test memory handling over multiple batches."""
    print("\n🧠 Testing Memory Stress (5 batches)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Initial GPU memory: {initial_memory:.1f} MB")
    
    # Reload fresh components for stress test
    transform = Compose([
        ConvertToFloatAndScale(),
        VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ShortSideScale(size=224),
        PerFrameCenterCrop((224, 224))
    ])
    
    dataset = SoccerNetMVFoulDataset(
        dataset_path="mvfouls", split='train', frames_per_clip=16,
        target_fps=15, max_views_to_load=4, transform=transform,
        target_height=224, target_width=224
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, 
                          collate_fn=variable_views_collate_fn)
    
    for i, batch in enumerate(dataloader):
        if i >= 5:  # Test 5 batches
            break
        
        # Simulate memory cleanup
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        if device.type == 'cuda':
            current_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"Batch {i+1}: GPU memory = {current_memory:.1f} MB")
        
        # Force cleanup
        del batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print("✅ Memory stress test completed")

def main():
    print("🚀 COMPREHENSIVE VARS TRAINING TEST")
    print("=" * 60)
    
    try:
        # Test 1: Dataset + DataLoader
        dataloader, dataset = test_dataset_and_dataloader()
        
        # Test 2: Model + Multi-GPU + Mixed Precision
        model, device, scaler = test_model_components(dataloader, dataset)
        
        # Test 3: Training Loop Components
        model, optimizer = test_training_components(model, dataloader, device, scaler)
        
        # Test 4: Validation Mode
        test_validation_mode(model, dataloader, device)
        
        # Test 5: Memory Stress Test
        test_memory_stress()
        
        print("\n" + "=" * 60)
        print("🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print("✅ Your setup should handle full training without crashes")
        print("💡 Ready for: python train.py --dataset_root . --epochs 50")
        
    except Exception as e:
        print(f"\n❌ COMPREHENSIVE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️  Fix issues before running full training")

if __name__ == "__main__":
    main() 
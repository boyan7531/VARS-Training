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
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import SoccerNetMVFoulDataset, variable_views_collate_fn
from model import MultiTaskMultiViewResNet3D, ModelConfig
from pytorchvideo.transforms import ShortSideScale, Normalize as VideoNormalize
from torchvision.transforms import Compose, CenterCrop

def find_mvfouls_directory():
    """Find mvfouls directory regardless of current working directory."""
    # Try current directory first
    if Path("mvfouls").exists():
        return "mvfouls"
    
    # Try parent directory (if running from scripts/)
    if Path("../mvfouls").exists():
        return "../mvfouls"
    
    # Try relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    mvfouls_path = project_root / "mvfouls"
    if mvfouls_path.exists():
        return str(mvfouls_path)
    
    # Last resort - search common locations
    for path in [".", "..", "../..", "../../.."]:
        test_path = Path(path) / "mvfouls"
        if test_path.exists():
            return str(test_path)
    
    raise FileNotFoundError("Cannot find mvfouls directory. Please ensure the dataset is downloaded.")

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
    
    # Find mvfouls directory dynamically
    mvfouls_path = find_mvfouls_directory()
    print(f"Found mvfouls directory at: {mvfouls_path}")
    
    transform = Compose([
        ConvertToFloatAndScale(),
        VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ShortSideScale(size=224),
        PerFrameCenterCrop((224, 398))  # ResNet3D supports rectangular inputs
    ])
    
    dataset = SoccerNetMVFoulDataset(
        dataset_path=mvfouls_path,
        split='train',
        frames_per_clip=16,
        target_fps=15,
        max_views_to_load=4,
        target_height=224,
        target_width=398,  # ResNet3D supports rectangular inputs
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
    
    # Vocab sizes (for vocabulary-based features)
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
    
    print(f"Vocab sizes: {vocab_sizes}")
    
    # Model setup - updated for ResNet3D
    config = ModelConfig(
        use_attention_aggregation=True,
        input_frames=16,
        input_height=224,
        input_width=398  # ResNet3D supports rectangular inputs
    )
    model = MultiTaskMultiViewResNet3D(
        num_severity=5,  # 5 severity classes: 1.0, 2.0, 3.0, 4.0, 5.0
        num_action_type=9,  # 9 action types: Challenge, Dive, Dont know, Elbowing, High leg, Holding, Pushing, Standing tackling, Tackling
        vocab_sizes=vocab_sizes,
        backbone_name='resnet3d_18',
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
    
    # UPDATED: Add annotation validation before training
    try:
        severity_labels = batch["label_severity"]
        action_labels = batch["label_type"]
        
        # Check for out-of-bounds labels that would crash CrossEntropyLoss
        if severity_labels.max().item() >= 5 or severity_labels.min().item() < 0:
            raise ValueError(f"Wrong annotations: severity labels {severity_labels.min().item()}-{severity_labels.max().item()} outside range 0-4")
        
        if action_labels.max().item() >= 9 or action_labels.min().item() < 0:
            raise ValueError(f"Wrong annotations: action labels {action_labels.min().item()}-{action_labels.max().item()} outside range 0-8")
    
    except Exception as e:
        print(f"❌ ANNOTATION ISSUE DETECTED during training: {e}")
        raise
    
    # Test forward + backward pass
    optimizer.zero_grad()
    
    try:
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
    except RuntimeError as e:
        if "out of range" in str(e).lower() or "index" in str(e).lower():
            print(f"❌ ANNOTATION ISSUE: CrossEntropyLoss failed - {e}")
            print("❌ This indicates wrong annotations with labels outside expected class ranges")
            raise ValueError("Wrong annotations detected: Labels outside model class ranges")
        else:
            raise
    
    return model, optimizer

def test_validation_mode(model, dataloader, device):
    """Test validation mode."""
    print("\n✅ Testing Validation Mode...")
    
    model.eval()
    criterion_severity = nn.CrossEntropyLoss()
    criterion_action = nn.CrossEntropyLoss()
    
    try:
        with torch.no_grad():
            batch = next(iter(dataloader))
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            
            # UPDATED: Validate annotations before forward pass
            severity_labels = batch["label_severity"]
            action_labels = batch["label_type"]
            
            if severity_labels.max().item() >= 5 or severity_labels.min().item() < 0:
                raise ValueError(f"Wrong annotations in validation: severity labels {severity_labels.min().item()}-{severity_labels.max().item()} outside range 0-4")
            
            if action_labels.max().item() >= 9 or action_labels.min().item() < 0:
                raise ValueError(f"Wrong annotations in validation: action labels {action_labels.min().item()}-{action_labels.max().item()} outside range 0-8")
            
            sev_logits, act_logits = model(batch)
            loss_sev = criterion_severity(sev_logits, batch["label_severity"])
            loss_act = criterion_action(act_logits, batch["label_type"])
            total_loss = loss_sev + loss_act
        
        print(f"✅ Validation step successful, Loss: {total_loss.item():.4f}")
    except RuntimeError as e:
        if "out of range" in str(e).lower() or "index" in str(e).lower():
            print(f"❌ ANNOTATION ISSUE in validation: CrossEntropyLoss failed - {e}")
            print("❌ This indicates wrong annotations with labels outside expected class ranges")
            raise ValueError("Wrong annotations detected in validation: Labels outside model class ranges")
        else:
            raise
    except ValueError as e:
        print(f"❌ ANNOTATION VALIDATION FAILED: {e}")
        raise

def test_memory_stress():
    """Test memory handling over multiple batches."""
    print("\n🧠 Testing Memory Stress (5 batches)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Initial GPU memory: {initial_memory:.1f} MB")
    
    # Find mvfouls directory dynamically
    mvfouls_path = find_mvfouls_directory()
    
    # Reload fresh components for stress test
    transform = Compose([
        ConvertToFloatAndScale(),
        VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ShortSideScale(size=224),
        PerFrameCenterCrop((224, 398))  # ResNet3D supports rectangular inputs
    ])
    
    dataset = SoccerNetMVFoulDataset(
        dataset_path=mvfouls_path, split='train', frames_per_clip=16,
        target_fps=15, max_views_to_load=4, transform=transform,
        target_height=224, target_width=398  # ResNet3D supports rectangular inputs
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

def test_categorical_features(dataset):
    """Test updated categorical features and both index types."""
    print("\n📊 Testing Updated Categorical Features...")
    
    # Test a sample to ensure both vocab and standard indices work
    sample = dataset[0]
    
    print("Testing categorical feature indices:")
    print(f"  Severity: {sample['label_severity'].item()}")
    print(f"  Action Type: {sample['label_type'].item()}")
    
    # Test vocabulary-based indices
    vocab_features = [
        'contact_idx', 'bodypart_idx', 'upper_bodypart_idx', 'lower_bodypart_idx',
        'multiple_fouls_idx', 'try_to_play_idx', 'touch_ball_idx', 
        'handball_idx', 'handball_offence_idx'
    ]
    
    print("\nVocabulary-based indices:")
    for feature in vocab_features:
        if feature in sample:
            print(f"  {feature}: {sample[feature].item()}")
    
    # Test standardized indices
    standard_features = [
        'offence_standard_idx', 'contact_standard_idx', 'bodypart_standard_idx',
        'upper_bodypart_standard_idx', 'lower_bodypart_standard_idx',
        'multiple_fouls_standard_idx', 'try_to_play_standard_idx', 
        'touch_ball_standard_idx', 'handball_standard_idx', 'handball_offence_standard_idx'
    ]
    
    print("\nStandardized indices:")
    for feature in standard_features:
        if feature in sample:
            print(f"  {feature}: {sample[feature].item()}")
    
    print("✅ All categorical features loaded successfully")
    
    # UPDATED: More robust validation with better error messages
    severity_val = sample['label_severity'].item()
    action_val = sample['label_type'].item()
    
    # Check for annotation issues
    if not (0 <= severity_val < 5):
        print(f"⚠️  WARNING: Invalid severity annotation: {severity_val} (expected 0-4)")
        print("❌ ANNOTATION ISSUE DETECTED: Severity labels are outside expected range")
        raise ValueError(f"Wrong annotations detected: severity {severity_val} not in range 0-4")
    
    if not (0 <= action_val < 9):
        print(f"⚠️  WARNING: Invalid action type annotation: {action_val} (expected 0-8)")
        print("❌ ANNOTATION ISSUE DETECTED: Action type labels are outside expected range")
        raise ValueError(f"Wrong annotations detected: action_type {action_val} not in range 0-8")
    
    print("✅ Label ranges validated (severity: 0-4, action_type: 0-8)")
    
    # Test vocab sizes are reasonable
    print(f"\nVocabulary sizes:")
    print(f"  Contact: {dataset.num_contact_classes}")
    print(f"  Bodypart: {dataset.num_bodypart_classes}")
    print(f"  Upper bodypart: {dataset.num_upper_bodypart_classes}")
    print(f"  Lower bodypart: {dataset.num_lower_bodypart_classes}")
    print(f"  Multiple fouls: {dataset.num_multiple_fouls_classes}")
    print(f"  Try to play: {dataset.num_try_to_play_classes}")
    print(f"  Touch ball: {dataset.num_touch_ball_classes}")
    print(f"  Handball: {dataset.num_handball_classes}")
    print(f"  Handball offence: {dataset.num_handball_offence_classes}")
    
    # UPDATED: Better vocab size validation with specific error messages
    vocab_checks = [
        ('contact', dataset.num_contact_classes),
        ('bodypart', dataset.num_bodypart_classes),
        ('upper_bodypart', dataset.num_upper_bodypart_classes),
        ('lower_bodypart', dataset.num_lower_bodypart_classes),
        ('multiple_fouls', dataset.num_multiple_fouls_classes),
        ('try_to_play', dataset.num_try_to_play_classes),
        ('touch_ball', dataset.num_touch_ball_classes),
        ('handball', dataset.num_handball_classes),
        ('handball_offence', dataset.num_handball_offence_classes)
    ]
    
    for vocab_name, size in vocab_checks:
        if not (1 <= size <= 20):
            print(f"⚠️  WARNING: Unreasonable vocab size for {vocab_name}: {size}")
            print("❌ ANNOTATION ISSUE DETECTED: Vocabulary sizes suggest wrong annotation mapping")
            raise ValueError(f"Wrong annotations detected: {vocab_name} vocab size {size} not in range 1-20")
    
    print("✅ Vocabulary sizes validated")
    
    return sample

def main():
    print("🚀 COMPREHENSIVE VARS TRAINING TEST")
    print("=" * 60)
    
    try:
        # Test 1: Dataset + DataLoader
        dataloader, dataset = test_dataset_and_dataloader()
        
        # Test 2: Categorical Features
        test_categorical_features(dataset)
        
        # Test 3: Model + Multi-GPU + Mixed Precision
        model, device, scaler = test_model_components(dataloader, dataset)
        
        # Test 4: Training Loop Components
        model, optimizer = test_training_components(model, dataloader, device, scaler)
        
        # Test 5: Validation Mode
        test_validation_mode(model, dataloader, device)
        
        # Test 6: Memory Stress Test
        test_memory_stress()
        
        print("\n" + "=" * 60)
        print("🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print("✅ Your setup should handle full training without crashes")
        print("✅ Updated categorical mappings working correctly")
        print("💡 Ready for: python train.py --dataset_root . --epochs 50")
        
    except Exception as e:
        print(f"\n❌ COMPREHENSIVE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️  Fix issues before running full training")

if __name__ == "__main__":
    main() 
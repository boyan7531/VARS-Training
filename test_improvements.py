#!/usr/bin/env python3
"""
Test script to validate the image resolution and loss balancing improvements
"""

import torch
import argparse
from dataset import SoccerNetMVFoulDataset
from model import MultiTaskMultiViewMViT, ModelConfig
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from pytorchvideo.transforms import ShortSideScale, Normalize as VideoNormalize
import sys
import os

# Add transforms
class ConvertToFloatAndScale(torch.nn.Module):
    def __call__(self, clip_cthw_uint8):
        if clip_cthw_uint8.dtype != torch.uint8:
            return clip_cthw_uint8
        return clip_cthw_uint8.float() / 255.0

class PerFrameCenterCrop(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        from torchvision.transforms import CenterCrop
        self.cropper = CenterCrop(size)

    def forward(self, clip_cthw):
        clip_tchw = clip_cthw.permute(1, 0, 2, 3)
        cropped_frames = [self.cropper(frame) for frame in clip_tchw]
        cropped_clip_tchw = torch.stack(cropped_frames)
        return cropped_clip_tchw.permute(1, 0, 2, 3)

def test_resolution_change():
    """Test that the new resolution (224x398) works correctly"""
    print("🔧 Testing Image Resolution Change (224x224 → 224x398)")
    
    # Test transforms
    transform = Compose([
        ConvertToFloatAndScale(),
        VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ShortSideScale(size=224),
        PerFrameCenterCrop((224, 398))  # New resolution
    ])
    
    # Test dataset with new resolution
    try:
        dataset = SoccerNetMVFoulDataset(
            dataset_path="mvfouls",
            split='train',
            frames_per_clip=16,
            target_fps=17,
            start_frame=67,
            end_frame=82,
            max_views_to_load=2,  # Limit for testing
            transform=transform,
            target_height=224,
            target_width=398  # New width
        )
        
        print(f"✅ Dataset loaded successfully with {len(dataset)} samples")
        
        # Test a single sample
        sample = dataset[0]
        clips_shape = sample["clips"].shape
        print(f"✅ Sample clips shape: {clips_shape}")
        print(f"   Expected: (num_views, 3, 16, 224, 398)")
        
        if clips_shape[-2:] == (224, 398):
            print("✅ Resolution change successful!")
        else:
            print(f"❌ Wrong resolution: got {clips_shape[-2:]}, expected (224, 398)")
            
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")

def test_model_with_new_resolution():
    """Test that the model works with new resolution"""
    print("\n🤖 Testing Model with New Resolution")
    
    # Create model config with new resolution
    config = ModelConfig(
        pretrained_model_name='mvit_base_16x4',
        use_attention_aggregation=True,
        input_frames=16,
        input_height=224,
        input_width=398  # New width
    )
    
    print(f"✅ Model config created with resolution {config.input_height}x{config.input_width}")
    
    # Test vocab sizes (dummy)
    vocab_sizes = {
        'contact': 3, 'bodypart': 4, 'upper_bodypart': 5, 'lower_bodypart': 1,
        'multiple_fouls': 4, 'try_to_play': 4, 'touch_ball': 5,
        'handball': 3, 'handball_offence': 3
    }
    
    try:
        model = MultiTaskMultiViewMViT(
            num_severity=5,
            num_action_type=9,
            vocab_sizes=vocab_sizes,
            config=config
        )
        print("✅ Model created successfully with new resolution")
        
        # Test forward pass with dummy data
        batch_size = 2
        num_views = 3
        dummy_clips = torch.randn(batch_size, num_views, 3, 16, 224, 398)
        dummy_batch = {
            "clips": dummy_clips,
            "contact_idx": torch.randint(0, 3, (batch_size,)),
            "bodypart_idx": torch.randint(0, 4, (batch_size,)),
            "upper_bodypart_idx": torch.randint(0, 5, (batch_size,)),
            "lower_bodypart_idx": torch.zeros(batch_size, dtype=torch.long),
            "multiple_fouls_idx": torch.randint(0, 4, (batch_size,)),
            "try_to_play_idx": torch.randint(0, 4, (batch_size,)),
            "touch_ball_idx": torch.randint(0, 5, (batch_size,)),
            "handball_idx": torch.randint(0, 3, (batch_size,)),
            "handball_offence_idx": torch.randint(0, 3, (batch_size,)),
            # Standard indices (reuse same values)
            "offence_standard_idx": torch.randint(0, 3, (batch_size,)),
            "contact_standard_idx": torch.randint(0, 2, (batch_size,)),
            "bodypart_standard_idx": torch.randint(0, 3, (batch_size,)),
            "upper_bodypart_standard_idx": torch.randint(0, 3, (batch_size,)),
            "lower_bodypart_standard_idx": torch.randint(0, 4, (batch_size,)),
            "multiple_fouls_standard_idx": torch.randint(0, 2, (batch_size,)),
            "try_to_play_standard_idx": torch.randint(0, 2, (batch_size,)),
            "touch_ball_standard_idx": torch.randint(0, 3, (batch_size,)),
            "handball_standard_idx": torch.randint(0, 2, (batch_size,)),
            "handball_offence_standard_idx": torch.randint(0, 2, (batch_size,))
        }
        
        sev_logits, act_logits = model(dummy_batch)
        print(f"✅ Forward pass successful!")
        print(f"   Severity logits shape: {sev_logits.shape}")
        print(f"   Action logits shape: {act_logits.shape}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")

def test_loss_balancing():
    """Test the new multi-task loss balancing"""
    print("\n⚖️ Testing Multi-Task Loss Balancing")
    
    from train import calculate_multitask_loss
    
    # Create dummy logits and batch data
    batch_size = 4
    sev_logits = torch.randn(batch_size, 5)  # 5 severity classes
    act_logits = torch.randn(batch_size, 9)  # 9 action classes
    
    batch_data = {
        "label_severity": torch.randint(0, 5, (batch_size,)),
        "label_type": torch.randint(0, 9, (batch_size,))
    }
    
    # Test different weight configurations
    weight_configs = [
        ([1.0, 1.0], "Equal weights (original)"),
        ([3.0, 3.0], "High main task weights"),
        ([5.0, 2.0], "Severity emphasis"),
        ([2.0, 5.0], "Action emphasis")
    ]
    
    for main_weights, description in weight_configs:
        total_loss, sev_loss, act_loss = calculate_multitask_loss(
            sev_logits, act_logits, batch_data, main_weights
        )
        
        print(f"✅ {description}:")
        print(f"   Main weights: {main_weights}")
        print(f"   Severity loss: {sev_loss.item():.4f}")
        print(f"   Action loss: {act_loss.item():.4f}")
        print(f"   Total loss: {total_loss.item():.4f}")
        print()

if __name__ == "__main__":
    print("🚀 Testing VARS Training Improvements")
    print("=" * 50)
    
    test_resolution_change()
    test_model_with_new_resolution()
    test_loss_balancing()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("\n📝 Next steps:")
    print("1. Run training with attention ON:")
    print("   python train.py --dataset_root . --attention_aggregation --img_height 224 --img_width 398 --main_task_weights 3.0 3.0")
    print("\n2. Run training with attention OFF:")
    print("   python train.py --dataset_root . --no-attention_aggregation --img_height 224 --img_width 398 --main_task_weights 3.0 3.0") 
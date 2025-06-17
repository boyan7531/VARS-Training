import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_video, read_video_timestamps
import json
from pathlib import Path
import random
from collections import defaultdict
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import logging

# Initialize logger for NaN detection
logger = logging.getLogger(__name__)

# Optional import for advanced augmentations
try:
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some advanced augmentations will use fallback implementations.")

# Define label mappings based on actual dataset annotations
SEVERITY_LABELS = {
    "": 0,           # Empty/unknown severity
    "1.0": 1,        # Lowest severity 
    "2.0": 2,        # Low severity
    "3.0": 3,        # Medium severity  
    "4.0": 4,        # High severity
    "5.0": 5         # Highest severity (Red card level)
}

ACTION_TYPE_LABELS = {
    "": 0,              # Empty/unknown action
    "Challenge": 1,
    "Dive": 2, 
    "Dont know": 3,
    "Elbowing": 4,
    "High leg": 5,
    "Holding": 6,
    "Pushing": 7,
    "Standing tackling": 8,
    "Tackling": 9
}
# Inverse maps for potential debugging or inspection
INV_SEVERITY_LABELS = {v: k for k, v in SEVERITY_LABELS.items()}
INV_ACTION_TYPE_LABELS = {v: k for k, v in ACTION_TYPE_LABELS.items()}

# Define keys for categorical features
CONTACT_FIELD = "Contact"
BODYPART_FIELD = "Bodypart"
UPPER_BODYPART_FIELD = "Upper body part"
# LOWER_BODYPART_FIELD = "Lower body part"  # Removed as it's 100% N/A in the dataset
MULTIPLE_FOULS_FIELD = "Multiple fouls"
TRY_TO_PLAY_FIELD = "Try to play"
TOUCH_BALL_FIELD = "Touch ball"
HANDBALL_FIELD = "Handball"
HANDBALL_OFFENCE_FIELD = "Handball offence"
REPLAY_SPEED_FIELD = "Replay speed"
UNKNOWN_TOKEN = "<UNK>"

# Standard mappings for common fields (updated to handle all possible values)
OFFENCE_VALUES = {"Offence": 1, "No offence": 0, "Between": 2}
CONTACT_VALUES = {"With contact": 1, "Without contact": 0}
BODYPART_VALUES = {"Upper body": 1, "Under body": 2, "": 0}  # Empty string maps to 0
UPPER_BODYPART_VALUES = {"Use of shoulder": 1, "Use of arms": 2, "": 0, "Use of shoulders": 1} # Map plural to singular
# LOWER_BODYPART_VALUES = {"Use of leg": 1, "Use of knee": 2, "Use of foot": 3} # Removed
MULTIPLE_FOULS_VALUES = {"Yes": 1, "yes": 1, "No": 0, "": 0}  # Handle inconsistencies and map "" to No
TRY_TO_PLAY_VALUES = {"Yes": 1, "No": 0, "": 0}  # Empty string maps to 0 (No)
TOUCH_BALL_VALUES = {"Yes": 1, "No": 0, "Maybe": 2, "": 0}  # Empty string maps to 0 (No)
HANDBALL_VALUES = {"Handball": 1, "No handball": 0}
HANDBALL_OFFENCE_VALUES = {"Offence": 1, "No offence": 0, "": 0}  # Empty string maps to 0 (No offence)

# Inverse mappings
INV_OFFENCE_VALUES = {v: k for k, v in OFFENCE_VALUES.items()}
INV_CONTACT_VALUES = {v: k for k, v in CONTACT_VALUES.items()}
INV_BODYPART_VALUES = {v: k for k, v in BODYPART_VALUES.items()}
INV_UPPER_BODYPART_VALUES = {v: k for k, v in UPPER_BODYPART_VALUES.items()}
# INV_LOWER_BODYPART_VALUES = {v: k for k, v in LOWER_BODYPART_VALUES.items()} # Removed
INV_MULTIPLE_FOULS_VALUES = {v: k for k, v in MULTIPLE_FOULS_VALUES.items()}
INV_TRY_TO_PLAY_VALUES = {v: k for k, v in TRY_TO_PLAY_VALUES.items()}
INV_TOUCH_BALL_VALUES = {v: k for k, v in TOUCH_BALL_VALUES.items()}
INV_HANDBALL_VALUES = {v: k for k, v in HANDBALL_VALUES.items()}
INV_HANDBALL_OFFENCE_VALUES = {v: k for k, v in HANDBALL_OFFENCE_VALUES.items()}

# ================================
# COMPREHENSIVE VIDEO AUGMENTATION CLASSES FOR CLASS IMBALANCE
# ================================

class TemporalJitter(torch.nn.Module):
    """Randomly jitter temporal sampling to create variations"""
    def __init__(self, max_jitter=2):
        super().__init__()
        self.max_jitter = max_jitter
    
    def forward(self, clip):
        # Convert to float32 first if it's uint8
        original_dtype = clip.dtype
        if clip.dtype == torch.uint8:
            clip = clip.float() / 255.0
            
        C, T, H, W = clip.shape
        if T <= self.max_jitter * 2:
            return clip
        
        # Randomly shift start position by up to max_jitter frames
        jitter = random.randint(-self.max_jitter, self.max_jitter)
        start_idx = max(0, min(jitter, T - T))  # Ensure we don't go out of bounds
        end_idx = min(T, T + jitter)
        
        if start_idx >= end_idx:
            return clip
        
        # If we have fewer frames than original, pad with repetition
        jittered_clip = clip[:, start_idx:end_idx, :, :]
        if jittered_clip.shape[1] < T:
            # Repeat last frame to maintain temporal dimension
            padding_needed = T - jittered_clip.shape[1]
            last_frame = jittered_clip[:, -1:, :, :].repeat(1, padding_needed, 1, 1)
            jittered_clip = torch.cat([jittered_clip, last_frame], dim=1)
        
        return jittered_clip

class RandomTemporalReverse(torch.nn.Module):
    """Randomly reverse the temporal order of frames"""
    def __init__(self, prob=0.3):
        super().__init__()
        self.prob = prob
    
    def forward(self, clip):
        if random.random() < self.prob:
            return torch.flip(clip, dims=[1])  # Flip temporal dimension
        return clip

class RandomFrameDropout(torch.nn.Module):
    """Randomly drop frames and repeat others to maintain temporal dimension"""
    def __init__(self, dropout_prob=0.1, max_consecutive=2):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.max_consecutive = max_consecutive
    
    def forward(self, clip):
        # Convert to float32 first if it's uint8
        original_dtype = clip.dtype
        if clip.dtype == torch.uint8:
            clip = clip.float() / 255.0
            
        C, T, H, W = clip.shape
        if T <= 4:  # Don't apply to very short clips
            return clip
        
        # Create dropout mask
        keep_mask = torch.rand(T) > self.dropout_prob
        
        # Ensure we keep at least half the frames
        if keep_mask.sum() < T // 2:
            return clip
        
        # Get indices of frames to keep
        keep_indices = torch.where(keep_mask)[0]
        
        # Create new clip by sampling kept frames
        new_clip = clip[:, keep_indices, :, :]
        
        # If we dropped frames, repeat some to maintain temporal dimension
        while new_clip.shape[1] < T:
            # Randomly repeat one of the existing frames
            repeat_idx = random.randint(0, new_clip.shape[1] - 1)
            repeat_frame = new_clip[:, repeat_idx:repeat_idx+1, :, :]
            new_clip = torch.cat([new_clip, repeat_frame], dim=1)
        
        # Trim to exact length if we overshot
        new_clip = new_clip[:, :T, :, :]
        return new_clip

class RandomBrightnessContrast(torch.nn.Module):
    """Apply random brightness and contrast changes per frame"""
    def __init__(self, brightness_range=0.2, contrast_range=0.2, prob=0.7):
        super().__init__()
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.prob = prob
    
    def forward(self, clip):
        if random.random() > self.prob:
            return clip
        
        # DEBUG: Check input
        if torch.isnan(clip).any():
            print(f"🚨 RandomBrightnessContrast: NaN in INPUT! Count: {torch.isnan(clip).sum().item()}")
            return clip
        
        # Convert to float32 first if it's uint8
        original_dtype = clip.dtype
        if clip.dtype == torch.uint8:
            clip = clip.float() / 255.0
            
            # DEBUG: Check after conversion
            if torch.isnan(clip).any():
                print(f"🚨 RandomBrightnessContrast: NaN after uint8->float conversion!")
                return clip
        
        C, T, H, W = clip.shape
        
        # Apply different brightness/contrast to each frame for more variation
        for t in range(T):
            if random.random() < 0.8:  # 80% chance per frame
                # Random brightness adjustment - clamp factor for safety
                brightness_factor = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
                brightness_factor = max(0.1, min(3.0, brightness_factor))  # Safe range
                
                pre_brightness = clip[:, t, :, :].clone()
                clip[:, t, :, :] = torch.clamp(clip[:, t, :, :] * brightness_factor, 0, 1)
                
                # DEBUG: Check after brightness
                if torch.isnan(clip[:, t, :, :]).any():
                    print(f"🚨 RandomBrightnessContrast: NaN after brightness on frame {t}!")
                    print(f"   Brightness factor: {brightness_factor}")
                    print(f"   Pre-brightness range: {pre_brightness.min():.3f} to {pre_brightness.max():.3f}")
                    clip[:, t, :, :] = pre_brightness  # Revert
                    continue
                
                # Random contrast adjustment - clamp factor for safety
                contrast_factor = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
                contrast_factor = max(0.1, min(3.0, contrast_factor))  # Safe range
                mean = clip[:, t, :, :].mean(dim=(1, 2), keepdim=True)
                
                # Ensure mean is not NaN (shouldn't happen, but safety first)
                if torch.isnan(mean).any():
                    print(f"🚨 RandomBrightnessContrast: NaN in mean calculation on frame {t}!")
                    continue  # Skip this frame if mean is NaN
                
                pre_contrast = clip[:, t, :, :].clone()
                clip[:, t, :, :] = torch.clamp(mean + contrast_factor * (clip[:, t, :, :] - mean), 0, 1)
                
                # DEBUG: Check after contrast
                if torch.isnan(clip[:, t, :, :]).any():
                    print(f"🚨 RandomBrightnessContrast: NaN after contrast on frame {t}!")
                    print(f"   Contrast factor: {contrast_factor}")
                    print(f"   Mean: {mean.mean().item():.3f}")
                    clip[:, t, :, :] = pre_contrast  # Revert
        
        # Final safety check
        if torch.isnan(clip).any():
            print(f"🚨 RandomBrightnessContrast: NaN in FINAL output! Count: {torch.isnan(clip).sum().item()}")
            clip = torch.where(torch.isnan(clip), torch.zeros_like(clip), clip)
        
        return clip

class RandomSpatialCrop(torch.nn.Module):
    """Random spatial cropping with automatic resize back to original size"""
    def __init__(self, crop_scale_range=(0.8, 1.0), prob=0.6):
        super().__init__()
        self.crop_scale_range = crop_scale_range
        self.prob = prob
    
    def forward(self, clip):
        if random.random() > self.prob:
            return clip
        
        # DEBUG: Check input
        if torch.isnan(clip).any():
            print(f"🚨 RandomSpatialCrop: NaN in INPUT! Count: {torch.isnan(clip).sum().item()}")
            return clip
            
        # Convert to float32 first if it's uint8
        original_dtype = clip.dtype
        if clip.dtype == torch.uint8:
            clip = clip.float() / 255.0
            
            # DEBUG: Check after conversion
            if torch.isnan(clip).any():
                print(f"🚨 RandomSpatialCrop: NaN after uint8->float conversion!")
                return clip
        
        C, T, H, W = clip.shape
        
        # Random crop scale
        scale = random.uniform(*self.crop_scale_range)
        crop_h = max(int(H * scale), 1)  # Ensure at least 1 pixel
        crop_w = max(int(W * scale), 1)  # Ensure at least 1 pixel
        
        # Random crop position
        top = random.randint(0, max(0, H - crop_h))
        left = random.randint(0, max(0, W - crop_w))
        
        # Crop all frames
        cropped_clip = clip[:, :, top:top+crop_h, left:left+crop_w]
        
        # DEBUG: Check after cropping
        if torch.isnan(cropped_clip).any():
            print(f"🚨 RandomSpatialCrop: NaN after CROPPING!")
            print(f"   Crop params: scale={scale:.3f}, crop_h={crop_h}, crop_w={crop_w}")
            print(f"   Crop position: top={top}, left={left}")
            return clip
        
        # Resize back to original size using proper interpolation
        resized_frames = []
        for t in range(T):
            frame = cropped_clip[:, t]  # [C, H_crop, W_crop]
            
            # DEBUG: Check frame before interpolation
            if torch.isnan(frame).any():
                print(f"🚨 RandomSpatialCrop: NaN in frame {t} before interpolation!")
                resized_frames.append(clip[:, t])  # Use original frame
                continue
            
            resized_frame = torch.nn.functional.interpolate(
                frame.unsqueeze(0),  # [1, C, H_crop, W_crop]
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # [C, H, W]
            
            # DEBUG: Check after interpolation
            if torch.isnan(resized_frame).any():
                print(f"🚨 RandomSpatialCrop: NaN after INTERPOLATION on frame {t}!")
                print(f"   Input frame shape: {frame.shape}")
                print(f"   Input frame range: {frame.min():.3f} to {frame.max():.3f}")
                print(f"   Target size: ({H}, {W})")
                resized_frames.append(clip[:, t])  # Use original frame
                continue
            
            resized_frames.append(resized_frame)
        
        resized_clip = torch.stack(resized_frames, dim=1)  # [C, T, H, W]
        
        # Safety check for NaN
        if torch.isnan(resized_clip).any():
            print(f"🚨 RandomSpatialCrop: NaN in FINAL output! Count: {torch.isnan(resized_clip).sum().item()}")
            return clip
        
        return resized_clip

class RandomHorizontalFlip(torch.nn.Module):
    """Horizontal flip for sports videos"""
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob
    
    def forward(self, clip):
        # Convert to float32 first if it's uint8
        original_dtype = clip.dtype
        if clip.dtype == torch.uint8:
            clip = clip.float() / 255.0
            
        if random.random() < self.prob:
            return torch.flip(clip, dims=[3])  # Flip width dimension
        return clip

class RandomGaussianNoise(torch.nn.Module):
    """Add random Gaussian noise"""
    def __init__(self, std_range=(0.01, 0.05), prob=0.4):
        super().__init__()
        self.std_range = std_range
        self.prob = prob
    
    def forward(self, clip):
        if random.random() > self.prob:
            return clip
        
        # DEBUG: Check input
        if torch.isnan(clip).any():
            print(f"🚨 RandomGaussianNoise: NaN in INPUT! Count: {torch.isnan(clip).sum().item()}")
            return clip
        
        # Convert to float32 first if it's uint8
        if clip.dtype == torch.uint8:
            clip = clip.float() / 255.0
            
            # DEBUG: Check after conversion
            if torch.isnan(clip).any():
                print(f"🚨 RandomGaussianNoise: NaN after uint8->float conversion!")
                return clip
        
        noise_std = random.uniform(*self.std_range)
        # Clamp noise std to safe range
        noise_std = max(0.001, min(0.2, noise_std))
        
        noise = torch.randn_like(clip) * noise_std
        
        # Safety check for NaN in noise (shouldn't happen, but be safe)
        if torch.isnan(noise).any():
            print(f"🚨 RandomGaussianNoise: NaN in NOISE generation! Count: {torch.isnan(noise).sum().item()}")
            print(f"   Noise std: {noise_std}")
            noise = torch.where(torch.isnan(noise), torch.zeros_like(noise), noise)
        
        result = torch.clamp(clip + noise, 0, 1)
        
        # Final safety check
        if torch.isnan(result).any():
            print(f"🚨 RandomGaussianNoise: NaN in FINAL result! Count: {torch.isnan(result).sum().item()}")
            print(f"   Input range: {clip.min():.3f} to {clip.max():.3f}")
            print(f"   Noise range: {noise.min():.3f} to {noise.max():.3f}")
            return clip
        
        return result

class SeverityAwareAugmentation(torch.nn.Module):
    """Apply stronger augmentation to minority severity classes"""
    def __init__(self, severity_label, heavy_aug_classes=[2, 3, 4, 5]):
        super().__init__()
        self.severity_label = severity_label
        self.heavy_aug_classes = heavy_aug_classes
        
        # Light augmentation for majority class (severity 1.0)
        self.light_aug = transforms.Compose([
            RandomHorizontalFlip(prob=0.5),
            TemporalJitter(max_jitter=1),
            RandomBrightnessContrast(brightness_range=0.15, contrast_range=0.15, prob=0.5),
        ])
        
        # Heavy augmentation for minority classes (severity 2.0+)
        self.heavy_aug = transforms.Compose([
            RandomHorizontalFlip(prob=0.6),
            RandomFrameDropout(dropout_prob=0.15, max_consecutive=2),
            TemporalJitter(max_jitter=2),
            RandomBrightnessContrast(brightness_range=0.25, contrast_range=0.25, prob=0.8),
            RandomSpatialCrop(crop_scale_range=(0.75, 1.0), prob=0.7),
            RandomGaussianNoise(std_range=(0.01, 0.04), prob=0.5),
        ])
    
    def forward(self, clip):
        # [NaN-origin] Step 3: Check SeverityAwareAugmentation separately
        clip_before = clip.clone()
        
        if self.severity_label in self.heavy_aug_classes:
            clip_result = self.heavy_aug(clip)
        else:
            clip_result = self.light_aug(clip)
        
        # Check if augmentation introduced NaN
        if torch.isnan(clip_result).any() and not torch.isnan(clip_before).any():
            logger.error(f"[NaN-origin] SeverityAwareAugmentation augmentation pipeline for severity {self.severity_label}")
            raise RuntimeError("NaN introduced by severity-aware augmentation")
        
        return clip_result

class ClassBalancedSampler(torch.utils.data.Sampler):
    """Custom sampler that balances severity classes by oversampling minority classes"""
    def __init__(self, dataset, oversample_factor=3.0):
        self.dataset = dataset
        self.oversample_factor = oversample_factor
        
        # Count samples per severity class
        self.class_counts = defaultdict(int)
        self.class_indices = defaultdict(list)
        
        for idx, action in enumerate(dataset.actions):
            severity_label = action['label_severity']
            self.class_counts[severity_label] += 1
            self.class_indices[severity_label].append(idx)
        
                # Calculate sampling weights
        max_count = 1
        if self.class_counts: # Ensure class_counts is not empty
            max_count = max(self.class_counts.values())
        
        self.sampling_weights = {}
        
        for class_id, count in self.class_counts.items():
            if count == 0: # Avoid division by zero if a class has no samples (should not happen with defaultdict)
                self.sampling_weights[class_id] = 0
                continue

            # Example: if severity 1.0 is majority, its label might be 1.
            # Adjust this condition based on your actual majority class label.
            # Assuming SEVERITY_LABELS maps "1.0" to 1, and this is majority.
            if class_id == SEVERITY_LABELS.get("1.0", 1):  # Majority class
                self.sampling_weights[class_id] = 1.0
            else:  # Minority classes
                self.sampling_weights[class_id] = min(oversample_factor, max_count / count)
        
        print(f"Class distribution: {dict(self.class_counts)}")
        print(f"Sampling weights: {self.sampling_weights}")
        
        # Calculate total samples per epoch
        self.samples_per_epoch = 0
        for class_id, count in self.class_counts.items():
            if class_id in self.sampling_weights: # Ensure class_id has a weight
                self.samples_per_epoch += int(count * self.sampling_weights[class_id])
    
    def __iter__(self):
        indices = []
        
        for class_id, class_indices in self.class_indices.items():
            weight = self.sampling_weights[class_id]
            num_samples = int(len(class_indices) * weight)
            
            # Oversample by randomly selecting with replacement
            if weight > 1.0:
                sampled_indices = random.choices(class_indices, k=num_samples)
            else:
                sampled_indices = class_indices[:num_samples]
            
            indices.extend(sampled_indices)
        
        # Shuffle all indices
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return self.samples_per_epoch

class ActionBalancedSampler(torch.utils.data.Sampler):
    """Custom sampler that balances action type classes by oversampling minority classes"""
    def __init__(self, dataset, oversample_factor=3.0):
        self.dataset = dataset
        self.oversample_factor = oversample_factor
        
        # Count samples per action type class
        self.class_counts = defaultdict(int)
        self.class_indices = defaultdict(list)
        
        for idx, action in enumerate(dataset.actions):
            action_label = action['label_type']
            self.class_counts[action_label] += 1
            self.class_indices[action_label].append(idx)
        
        # Calculate sampling weights
        max_count = 1
        if self.class_counts: # Ensure class_counts is not empty
            max_count = max(self.class_counts.values())
        
        self.sampling_weights = {}
        
        for class_id, count in self.class_counts.items():
            if count == 0: # Avoid division by zero if a class has no samples (should not happen with defaultdict)
                self.sampling_weights[class_id] = 0
                continue

            # Standing tackling (Act_8) is the majority class
            if class_id == ACTION_TYPE_LABELS.get("Standing tackling", 8):  # Majority class
                self.sampling_weights[class_id] = 1.0
            else:  # Minority classes
                self.sampling_weights[class_id] = min(oversample_factor, max_count / count)
        
        print(f"Action class distribution: {dict(self.class_counts)}")
        print(f"Action sampling weights: {self.sampling_weights}")
        
        # Calculate total samples per epoch
        self.samples_per_epoch = 0
        for class_id, count in self.class_counts.items():
            if class_id in self.sampling_weights: # Ensure class_id has a weight
                self.samples_per_epoch += int(count * self.sampling_weights[class_id])
    
    def __iter__(self):
        indices = []
        
        for class_id, class_indices in self.class_indices.items():
            weight = self.sampling_weights[class_id]
            num_samples = int(len(class_indices) * weight)
            
            # Oversample by randomly selecting with replacement
            if weight > 1.0:
                sampled_indices = random.choices(class_indices, k=num_samples)
            else:
                sampled_indices = class_indices[:num_samples]
            
            indices.extend(sampled_indices)
        
        # Shuffle all indices
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return self.samples_per_epoch

class AlternatingSampler(torch.utils.data.Sampler):
    """Alternates between severity and action balanced sampling per epoch"""
    def __init__(self, dataset, severity_oversample_factor=3.0, action_oversample_factor=3.0):
        self.dataset = dataset
        self.current_epoch = 0
        
        # Create both samplers
        self.severity_sampler = ClassBalancedSampler(dataset, severity_oversample_factor)
        self.action_sampler = ActionBalancedSampler(dataset, action_oversample_factor)
        
        print(f"AlternatingSampler initialized:")
        print(f"  - Severity sampler: {len(self.severity_sampler)} samples per epoch")
        print(f"  - Action sampler: {len(self.action_sampler)} samples per epoch")
    
    def set_epoch(self, epoch):
        """Update the current epoch to determine which sampler to use."""
        self.current_epoch = epoch
        if epoch % 10 == 0:  # Log every 10 epochs to avoid spam
            sampler_type = "Severity" if epoch % 2 == 0 else "Action"
            print(f"Epoch {epoch}: Using {sampler_type} balanced sampling")
    
    def __iter__(self):
        # Alternate between severity and action balancing per epoch
        if self.current_epoch % 2 == 0:
            # Even epochs: balance by severity
            return iter(self.severity_sampler)
        else:
            # Odd epochs: balance by action type
            return iter(self.action_sampler)
    
    def __len__(self):
        # Return the current sampler's length
        if self.current_epoch % 2 == 0:
            return len(self.severity_sampler)
        else:
            return len(self.action_sampler)

class SoccerNetMVFoulDataset(Dataset):
    def __init__(self,
                 dataset_path: str, 
                 split: str, 
                 annotation_file_name: str = "annotations.json",
                 frames_per_clip: int = 16,
                 target_fps: int = 17,
                 start_frame: int = 67,
                 end_frame: int = 82,
                 load_all_views: bool = True,
                 max_views_to_load: int = None,
                 views_indices: list[int] = None,
                 transform=None,
                 target_height: int = 224,
                 target_width: int = 224,
                 use_severity_aware_aug: bool = True,
                 clips_per_video: int = 1,
                 clip_sampling: str = 'uniform'):
        """
        Args:
            dataset_path (str): Path to the root of the SoccerNet MVFoul dataset (e.g., /path/to/SoccerNet_data/mvfouls).
                                This directory should contain split folders (train, valid, test).
            split (str): Dataset split, one of ['train', 'valid', 'test'].
            annotation_file_name (str): Name of the JSON file containing annotations, expected inside each split folder.
                                      E.g., <dataset_path>/<split>/<annotation_file_name>.
                                      Each entry should contain:
                                      - 'action_id': A unique identifier for the action.
                                      - 'video_files': A list of relative paths (from split_dir) to video files for different views.
                                      - 'labels': {'severity': 'No Offence', 'type': 'Tackle'}
                                      - 'start_frame' (optional): Start frame for the clip.
                                      - 'end_frame' (optional): End frame for the clip.
                                      - 'original_fps' (optional): FPS of the source video if resampling is needed.
            frames_per_clip (int): Number of frames to sample for each video clip.
            target_fps (int): Desired frames per second for the output clip.
            start_frame (int): Start frame index for foul-centered extraction (default: 67, 8 frames before foul at frame 75).
            end_frame (int): End frame index for foul-centered extraction (default: 82, 7 frames after foul at frame 75).
            load_all_views (bool): If True (default), loads all available views for each action.
            max_views_to_load (int): Optional limit on number of views to load. If None (default), loads all available views.
            views_indices (list[int]): Specific indices of views to load from the 'video_files' list in annotations.
                                       If provided, overrides load_all_views and max_views_to_load.
            transform: PyTorch transforms to be applied to each clip.
            target_height (int): Target height for dummy tensors if video loading fails.
            target_width (int): Target width for dummy tensors if video loading fails.
            use_severity_aware_aug (bool): If True, applies severity-aware augmentations for class imbalance during training.
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.split_dir = self.dataset_path / self.split
        self.annotation_path = self.split_dir / annotation_file_name

        self.frames_per_clip = frames_per_clip
        self.target_fps = float(target_fps) # Ensure target_fps is float for division
        self.start_frame = start_frame
        self.end_frame = end_frame

        self.load_all_views = load_all_views
        self.max_views_to_load = max_views_to_load
        self.views_indices = views_indices

        self.transform = transform
        self.target_height = target_height
        self.target_width = target_width
        self.use_severity_aware_aug = use_severity_aware_aug
        self.clips_per_video = clips_per_video
        self.clip_sampling = clip_sampling

        # Validate frame range
        expected_frames = end_frame - start_frame + 1
        if expected_frames != frames_per_clip:
            print(f"Warning: Frame range ({start_frame}-{end_frame}) gives {expected_frames} frames, but frames_per_clip is {frames_per_clip}. "
                  f"Will sample {frames_per_clip} frames from the range.")

        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_path}. "
                                    "Please ensure it exists or use SoccerNet's tools to generate/locate it.")
        
        raw_annotations_data = {}
        try:
            with open(self.annotation_path, 'r') as f:
                raw_annotations_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {self.annotation_path}: {e}")

        # Build vocabularies for all categorical features
        self.contact_vocab, self.num_contact_classes = self._build_vocab(raw_annotations_data, CONTACT_FIELD)
        self.bodypart_vocab, self.num_bodypart_classes = self._build_vocab(raw_annotations_data, BODYPART_FIELD)
        self.upper_bodypart_vocab, self.num_upper_bodypart_classes = self._build_vocab(raw_annotations_data, UPPER_BODYPART_FIELD)
        # self.lower_bodypart_vocab, self.num_lower_bodypart_classes = self._build_vocab(raw_annotations_data, LOWER_BODYPART_FIELD) # Removed
        self.multiple_fouls_vocab, self.num_multiple_fouls_classes = self._build_vocab(raw_annotations_data, MULTIPLE_FOULS_FIELD)
        self.try_to_play_vocab, self.num_try_to_play_classes = self._build_vocab(raw_annotations_data, TRY_TO_PLAY_FIELD)
        self.touch_ball_vocab, self.num_touch_ball_classes = self._build_vocab(raw_annotations_data, TOUCH_BALL_FIELD)
        self.handball_vocab, self.num_handball_classes = self._build_vocab(raw_annotations_data, HANDBALL_FIELD)
        self.handball_offence_vocab, self.num_handball_offence_classes = self._build_vocab(raw_annotations_data, HANDBALL_OFFENCE_FIELD)
        
        print(f"Built '{CONTACT_FIELD}' vocab ({self.num_contact_classes} classes): {self.contact_vocab}")
        print(f"Built '{BODYPART_FIELD}' vocab ({self.num_bodypart_classes} classes): {self.bodypart_vocab}")
        print(f"Built '{UPPER_BODYPART_FIELD}' vocab ({self.num_upper_bodypart_classes} classes): {self.upper_bodypart_vocab}")
        # print(f"Built '{LOWER_BODYPART_FIELD}' vocab ({self.num_lower_bodypart_classes} classes): {self.lower_bodypart_vocab}") # Removed
        print(f"Built '{MULTIPLE_FOULS_FIELD}' vocab ({self.num_multiple_fouls_classes} classes): {self.multiple_fouls_vocab}")
        print(f"Built '{TRY_TO_PLAY_FIELD}' vocab ({self.num_try_to_play_classes} classes): {self.try_to_play_vocab}")
        print(f"Built '{TOUCH_BALL_FIELD}' vocab ({self.num_touch_ball_classes} classes): {self.touch_ball_vocab}")
        print(f"Built '{HANDBALL_FIELD}' vocab ({self.num_handball_classes} classes): {self.handball_vocab}")
        print(f"Built '{HANDBALL_OFFENCE_FIELD}' vocab ({self.num_handball_offence_classes} classes): {self.handball_offence_vocab}")

        print(f"Dataset configured for foul-centered extraction: frames {start_frame}-{end_frame} ({expected_frames} frames)")

        self.actions = self._process_annotations(raw_annotations_data)

        if not self.actions:
            print(f"Warning: No actions loaded from {self.annotation_path} after processing. Check the annotation file format and content.")

        # Manual shuffling for better data mixing
        if self.split == 'train':
            # Set a different seed for shuffling to ensure thorough data mixing
            original_random_state = random.getstate()
            random.seed(42 + len(self.actions))  # Use dataset size as additional entropy
            random.shuffle(self.actions)
            random.setstate(original_random_state)  # Restore original random state
            print(f"Manually shuffled {len(self.actions)} training actions for better data mixing")

    def _build_vocab(self, all_actions_data: dict, field_name: str, unknown_token: str = UNKNOWN_TOKEN):
        unique_values = set()
        actions_dict = all_actions_data.get("Actions", {})
        if not isinstance(actions_dict, dict):
            print(f"Warning: 'Actions' key not found or not a dict in annotation data while building vocab for '{field_name}'.")
            # Fallback to just unknown token if no actions found
            vocab = {unknown_token: 0}
            return vocab, len(vocab)

        for action_details in actions_dict.values():
            if isinstance(action_details, dict):
                value = action_details.get(field_name)
                if value is not None and isinstance(value, str): # Include empty strings
                    unique_values.add(value)
                
                # Also check in clips for replay speed
                if field_name == REPLAY_SPEED_FIELD and "Clips" in action_details:
                    for clip in action_details["Clips"]:
                        if isinstance(clip, dict) and REPLAY_SPEED_FIELD in clip:
                            replay_speed = str(clip.get(REPLAY_SPEED_FIELD))
                            if replay_speed is not None:
                                unique_values.add(replay_speed)
        
        sorted_values = sorted(list(unique_values))
        vocab = {unknown_token: 0} # UNK token gets index 0
        for i, value in enumerate(sorted_values):
            vocab[value] = i + 1 # Actual values start from 1
        return vocab, len(vocab)

    def _process_annotations(self, annotations_data: dict):
        processed_actions = []

        actions_dict = annotations_data.get("Actions")
        if not isinstance(actions_dict, dict):
            print(f"Warning: 'Actions' key not found in {self.annotation_path} or is not a dictionary. No actions will be loaded.")
            return []

        for action_id_str, action_details in actions_dict.items():
            if not isinstance(action_details, dict):
                continue

            # --- Severity Label (map empty to class 0, others to their explicit classes) ---
            json_severity_val = action_details.get("Severity", "")  # Default to empty string if missing
            
            # Map all values (including empty) through the label mapping
            if json_severity_val in SEVERITY_LABELS:
                numerical_severity = SEVERITY_LABELS[json_severity_val]
                if json_severity_val == "":
                    print(f"Info: Empty severity for action {action_id_str}, mapped to class 0")
            else:
                print(f"Warning: Unknown 'Severity' value '{json_severity_val}' for action {action_id_str}. Mapping to empty class 0.")
                numerical_severity = 0  # Map unknown values to empty class

            # --- Action Type Label (map empty to class 0, others to their explicit classes) ---
            json_action_class = action_details.get("Action class", "")
            
            # Map all values (including empty) through the label mapping
            if json_action_class in ACTION_TYPE_LABELS:
                numerical_action_type = ACTION_TYPE_LABELS[json_action_class]
                if json_action_class == "":
                    print(f"Info: Empty action class for action {action_id_str}, mapped to class 0")
            else:
                print(f"Warning: Unknown 'Action class' value '{json_action_class}' for action {action_id_str}. Mapping to empty class 0.")
                numerical_action_type = 0  # Map unknown values to empty class

            # --- Video Files ---
            clips_info_list = action_details.get("Clips", [])
            if not isinstance(clips_info_list, list) or not clips_info_list:
                # print(f"Warning: No 'Clips' found or 'Clips' is not a list for action {action_id_str}. Skipping.")
                continue
            
            video_files_relative = []
            # Path prefix to remove, specific to split. e.g. "Dataset/Train/"
            path_prefix_to_strip = f"Dataset/{self.split.capitalize()}/" 
            
            # Extract clip-specific information
            clip_replay_speeds = []
            
            for clip_info in clips_info_list:
                raw_url = clip_info.get("Url")
                if raw_url:
                    # Strip the "Dataset/Train/" or similar prefix
                    if raw_url.startswith(path_prefix_to_strip):
                        processed_url = raw_url[len(path_prefix_to_strip):]
                    else:
                        # print(f"Warning: Clip URL '{raw_url}' for action {action_id_str} does not start with expected prefix '{path_prefix_to_strip}'. Using as is, but might be wrong.")
                        processed_url = raw_url
                    
                    # Assume .mp4 extension if not present. Adjust if your files are .mkv or other.
                    if not Path(processed_url).suffix:
                        video_files_relative.append(processed_url + ".mp4")
                    else:
                        video_files_relative.append(processed_url)
                
                # Get replay speed for this clip
                replay_speed = clip_info.get(REPLAY_SPEED_FIELD, None)
                clip_replay_speeds.append(str(replay_speed) if replay_speed is not None else UNKNOWN_TOKEN)
            
            if not video_files_relative:
                # print(f"Warning: No valid video URLs extracted for action {action_id_str} after processing. Skipping.")
                continue

            # --- Original FPS (if available in annotation at the action level) ---
            annotated_original_fps_val = action_details.get("original_fps")
            final_original_fps = None 
            if annotated_original_fps_val is not None:
                try:
                    fps_val = float(annotated_original_fps_val)
                    if fps_val > 0:
                        final_original_fps = fps_val
                    else:
                        print(f"Warning: Invalid original_fps value \'{fps_val}\' (must be > 0) for action {action_id_str} at action level. Will be ignored.")
                except ValueError:
                    print(f"Warning: Non-float original_fps value \'{annotated_original_fps_val}\' for action {action_id_str} at action level. Will be ignored.")

            # --- Process all categorical features using standard mappings ---
            
            # Offence (keep empty values as they are without forcing defaults)
            offence_str = action_details.get("Offence", "")  # Default to empty string
            
            # Keep empty values as empty - don't force any defaults
            if offence_str == "":
                print(f"Info: Empty offence value for action {action_id_str}, keeping as empty")
            
            # Map to indices, using 0 for empty (which can represent "unknown" or "no classification")
            if offence_str in OFFENCE_VALUES:
                offence_idx = OFFENCE_VALUES[offence_str]
            elif offence_str == "":
                offence_idx = 0  # Map empty to 0 (can represent "No offence" or "unknown")
            else:
                print(f"Warning: Unknown offence value '{offence_str}' for action {action_id_str}, mapping to 0")
                offence_idx = 0
            
            # Contact
            contact_str = action_details.get(CONTACT_FIELD, "")  # Default to empty string
            contact_idx = self.contact_vocab.get(contact_str, self.contact_vocab[UNKNOWN_TOKEN])
            # Standard mapping for consistent numerical values
            if contact_str in CONTACT_VALUES:
                contact_standard_idx = CONTACT_VALUES[contact_str]
            else:
                contact_standard_idx = 0 # Default to "Without contact"
            
            # Bodypart
            bodypart_str = action_details.get(BODYPART_FIELD, "")  # Default to empty string
            bodypart_idx = self.bodypart_vocab.get(bodypart_str, self.bodypart_vocab[UNKNOWN_TOKEN])
            # Standard mapping for consistent numerical values
            if bodypart_str in BODYPART_VALUES:
                bodypart_standard_idx = BODYPART_VALUES[bodypart_str]
            else:
                bodypart_standard_idx = 0 # Default to unknown
            
            # Upper body part (only applicable when Bodypart is "Upper body")
            upper_bodypart_str = action_details.get(UPPER_BODYPART_FIELD, "")  # Default to empty string
            upper_bodypart_idx = self.upper_bodypart_vocab.get(upper_bodypart_str, self.upper_bodypart_vocab[UNKNOWN_TOKEN])
            # Standard mapping
            if upper_bodypart_str in UPPER_BODYPART_VALUES:
                upper_bodypart_standard_idx = UPPER_BODYPART_VALUES[upper_bodypart_str]
            else:
                upper_bodypart_standard_idx = 0 # Default to unknown
                
            # Multiple fouls
            multiple_fouls_str = action_details.get(MULTIPLE_FOULS_FIELD, "")  # Default to empty string
            multiple_fouls_idx = self.multiple_fouls_vocab.get(multiple_fouls_str, self.multiple_fouls_vocab[UNKNOWN_TOKEN])
            # Standard mapping
            if multiple_fouls_str in MULTIPLE_FOULS_VALUES:
                multiple_fouls_standard_idx = MULTIPLE_FOULS_VALUES[multiple_fouls_str]
            else:
                multiple_fouls_standard_idx = 0 # Default to "No"
            
            # Try to play
            try_to_play_str = action_details.get(TRY_TO_PLAY_FIELD, "")  # Default to empty string
            try_to_play_idx = self.try_to_play_vocab.get(try_to_play_str, self.try_to_play_vocab[UNKNOWN_TOKEN])
            # Standard mapping
            if try_to_play_str in TRY_TO_PLAY_VALUES:
                try_to_play_standard_idx = TRY_TO_PLAY_VALUES[try_to_play_str]
            else:
                try_to_play_standard_idx = 0 # Default to "No"
            
            # Touch ball
            touch_ball_str = action_details.get(TOUCH_BALL_FIELD, "")  # Default to empty string
            touch_ball_idx = self.touch_ball_vocab.get(touch_ball_str, self.touch_ball_vocab[UNKNOWN_TOKEN])
            # Standard mapping
            if touch_ball_str in TOUCH_BALL_VALUES:
                touch_ball_standard_idx = TOUCH_BALL_VALUES[touch_ball_str]
            else:
                touch_ball_standard_idx = 0 # Default to "No"
            
            # Handball
            handball_str = action_details.get(HANDBALL_FIELD, "No handball")  # Default to "No handball"
            handball_idx = self.handball_vocab.get(handball_str, self.handball_vocab[UNKNOWN_TOKEN])
            # Standard mapping
            if handball_str in HANDBALL_VALUES:
                handball_standard_idx = HANDBALL_VALUES[handball_str]
            else:
                handball_standard_idx = 0 # Default to "No handball"
            
            # Handball offence
            handball_offence_str = action_details.get(HANDBALL_OFFENCE_FIELD, "")  # Default to empty string
            handball_offence_idx = self.handball_offence_vocab.get(handball_offence_str, self.handball_offence_vocab[UNKNOWN_TOKEN])
            # Standard mapping
            if handball_offence_str in HANDBALL_OFFENCE_VALUES:
                handball_offence_standard_idx = HANDBALL_OFFENCE_VALUES[handball_offence_str]
            else:
                handball_offence_standard_idx = 0 # Default to "No"
            
            processed_actions.append({
                "action_id": action_id_str,
                "video_files_relative": video_files_relative,
                "label_severity": numerical_severity,
                "label_type": numerical_action_type,
                "original_fps_from_annotation": final_original_fps,
                
                # Original vocab indices (for compatibility with existing code)
                "contact_idx": contact_idx,
                "bodypart_idx": bodypart_idx,
                "upper_bodypart_idx": upper_bodypart_idx,
                "multiple_fouls_idx": multiple_fouls_idx,
                "try_to_play_idx": try_to_play_idx,
                "touch_ball_idx": touch_ball_idx,
                "handball_idx": handball_idx,
                "handball_offence_idx": handball_offence_idx,
                
                # Standard indices (new)
                "offence_standard_idx": offence_idx,
                "contact_standard_idx": contact_standard_idx,
                "bodypart_standard_idx": bodypart_standard_idx,
                "upper_bodypart_standard_idx": upper_bodypart_standard_idx,
                "multiple_fouls_standard_idx": multiple_fouls_standard_idx,
                "try_to_play_standard_idx": try_to_play_standard_idx,
                "touch_ball_standard_idx": touch_ball_standard_idx,
                "handball_standard_idx": handball_standard_idx,
                "handball_offence_standard_idx": handball_offence_standard_idx,
                
                # Additional data
                "clip_replay_speeds": clip_replay_speeds
            })
            
        if not processed_actions:
            print(f"Warning: No actions were successfully processed from {self.annotation_path}. Check mappings and file paths.")
            
        return processed_actions
        
    def __len__(self):
        return len(self.actions)

    def _get_video_fps(self, video_path_str: str, default_fps: float) -> float:
        """Helper function to robustly get video FPS."""
        try:
            ts_data = read_video_timestamps(video_path_str, pts=[]) # Pass pts=[] to read metadata
            
            video_fps = None
            if len(ts_data) == 3 and isinstance(ts_data[2], dict): # (video_pts, audio_pts, meta)
                video_fps = ts_data[2].get('video_fps')
            elif len(ts_data) >= 2: # Check for (video_pts, video_fps) or (video_pts, audio_pts, video_fps, audio_fps)
                # Heuristic: video_fps is often the last numeric fps-like value if meta is not present
                if isinstance(ts_data[-1], (float, int)) and 5 < ts_data[-1] < 120: # Plausible FPS range
                    video_fps = ts_data[-1]
                # Or if (video_pts, video_fps) structure
                elif len(ts_data) == 2 and isinstance(ts_data[1], (float, int)) and 5 < ts_data[1] < 120 :
                    video_fps = ts_data[1]
                # Or if (vpts, apts, vfps, afps)
                elif len(ts_data) == 4 and isinstance(ts_data[2], (float, int)) and 5 < ts_data[2] < 120:
                     video_fps = ts_data[2]


            if video_fps is not None and video_fps > 0:
                return float(video_fps)
            # print(f"Warning: Could not determine FPS for {video_path_str} from metadata. Using default {default_fps}.")
            return float(default_fps)
        except Exception as e:
            # print(f"Error getting FPS for {video_path_str}: {e}. Using default {default_fps}.")
            return float(default_fps)

    def _get_video_clip(self, video_path_str: str, action_info: dict, start_frame_override: int = None):
        """
        Load and process a video clip with comprehensive NaN debugging.
        """
        video_path = Path(video_path_str)
        
        if not video_path.exists():
            print(f"Video file not found: {video_path}")
            return None
        
        try:
            # Get video info first
            video_fps = self._get_video_fps(str(video_path), default_fps=25.0)
            
            # Calculate frame indices
            if start_frame_override is not None:
                start_frame = start_frame_override
            else:
                start_frame = self.start_frame
            
            end_frame = start_frame + self.frames_per_clip
            
            # Load video with torchvision
            video_tensor, audio_tensor, info = read_video(
                str(video_path),
                start_pts=start_frame / video_fps,
                end_pts=end_frame / video_fps,
                pts_unit='sec'
            )
            
            # DEBUG: Check immediately after loading
            if torch.isnan(video_tensor).any():
                nan_count = torch.isnan(video_tensor).sum().item()
                print(f"🚨 NaN detected IMMEDIATELY after video loading!")
                print(f"   File: {video_path}")
                print(f"   Action ID: {action_info.get('action_id', 'unknown')}")
                print(f"   NaN count: {nan_count}")
                print(f"   Tensor shape: {video_tensor.shape}")
                print(f"   Tensor dtype: {video_tensor.dtype}")
                print(f"   Min/Max values: {video_tensor.min():.3f} / {video_tensor.max():.3f}")
                
                # Check if entire tensor is NaN
                total_elements = video_tensor.numel()
                if nan_count == total_elements:
                    print(f"   ❌ ENTIRE TENSOR IS NaN - corrupted video file!")
                    return None
                else:
                    print(f"   ⚠️  Partial NaN: {nan_count}/{total_elements} elements")
            
            # Check for other problematic values
            if torch.isinf(video_tensor).any():
                inf_count = torch.isinf(video_tensor).sum().item()
                print(f"🚨 Infinite values detected after video loading!")
                print(f"   File: {video_path}")
                print(f"   Inf count: {inf_count}")
            
            # Validate tensor properties
            if video_tensor.numel() == 0:
                print(f"❌ Empty video tensor for {video_path}")
                return None
            
            # Convert to float and normalize if needed
            if video_tensor.dtype == torch.uint8:
                # Removed verbose logging: print(f"🔄 Converting uint8 to float32 for {video_path}")
                video_tensor = video_tensor.float() / 255.0
                
                # DEBUG: Check after conversion
                if torch.isnan(video_tensor).any():
                    print(f"🚨 NaN appeared AFTER uint8->float conversion!")
                    print(f"   This suggests overflow/underflow during conversion")
            
            # Resize to target dimensions
            original_shape = video_tensor.shape
            if len(video_tensor.shape) == 4:  # [T, H, W, C]
                video_tensor = video_tensor.permute(3, 0, 1, 2)  # [C, T, H, W]
            
            C, T, H, W = video_tensor.shape
            
            if H != self.target_height or W != self.target_width:
                # Removed verbose logging: print(f"🔄 Resizing from {H}x{W} to {self.target_height}x{self.target_width}")
                
                # Resize each frame individually to avoid memory issues
                resized_frames = []
                for t in range(T):
                    frame = video_tensor[:, t, :, :]  # [C, H, W]
                    resized_frame = torch.nn.functional.interpolate(
                        frame.unsqueeze(0),  # [1, C, H, W]
                        size=(self.target_height, self.target_width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # [C, H, W]
                    
                    # DEBUG: Check after each frame resize
                    if torch.isnan(resized_frame).any():
                        print(f"🚨 NaN appeared after resizing frame {t}!")
                        print(f"   Original frame shape: {frame.shape}")
                        print(f"   Original frame range: {frame.min():.3f} to {frame.max():.3f}")
                        print(f"   Resized frame shape: {resized_frame.shape}")
                        # Replace NaN in this frame
                        resized_frame = torch.where(torch.isnan(resized_frame), torch.zeros_like(resized_frame), resized_frame)
                    
                    resized_frames.append(resized_frame)
                
                video_tensor = torch.stack(resized_frames, dim=1)  # [C, T, H, W]
                
                # DEBUG: Check after all resizing
                if torch.isnan(video_tensor).any():
                    print(f"🚨 NaN detected AFTER resizing operation!")
                    nan_count = torch.isnan(video_tensor).sum().item()
                    print(f"   NaN count after resize: {nan_count}")
            
            # Apply augmentations if present
            if self.transform is not None:
                # Removed verbose logging: print statements for shape, dtype, range
                
                # Check if this is the expected 4D format [C, T, H, W]
                if len(video_tensor.shape) != 4:
                    print(f"🚨 UNEXPECTED TENSOR SHAPE! Expected 4D [C, T, H, W], got {len(video_tensor.shape)}D: {video_tensor.shape}")
                    
                    # Try to fix the shape if possible
                    if len(video_tensor.shape) == 5 and video_tensor.shape[0] == 1:
                        # Remove batch dimension if present: [1, C, T, H, W] -> [C, T, H, W]
                        video_tensor = video_tensor.squeeze(0)
                        print(f"   Fixed by removing batch dimension: {video_tensor.shape}")
                    elif len(video_tensor.shape) == 3:
                        # Add missing dimension if needed
                        print(f"   ❌ Cannot fix 3D tensor shape automatically")
                        return None
                    else:
                        print(f"   ❌ Cannot fix tensor shape automatically")
                        return None
                
                pre_aug_nan = torch.isnan(video_tensor).any()
                
                video_tensor = self.transform(video_tensor)
                
                # DEBUG: Check after augmentation
                post_aug_nan = torch.isnan(video_tensor).any()
                if post_aug_nan and not pre_aug_nan:
                    print(f"🚨 NaN appeared DURING AUGMENTATION!")
                    print(f"   This indicates a bug in the augmentation pipeline")
                    nan_count = torch.isnan(video_tensor).sum().item()
                    print(f"   NaN count after augmentation: {nan_count}")
                    
                    # Try to identify which augmentation caused it
                    if hasattr(self.transform, 'transforms'):
                        print(f"   Augmentation pipeline: {[type(t).__name__ for t in self.transform.transforms]}")
            
            # Final validation
            if torch.isnan(video_tensor).any():
                nan_count = torch.isnan(video_tensor).sum().item()
                total_elements = video_tensor.numel()
                print(f"🚨 FINAL NaN CHECK FAILED!")
                print(f"   File: {video_path}")
                print(f"   Final NaN count: {nan_count}/{total_elements}")
                print(f"   Final shape: {video_tensor.shape}")
                print(f"   Final range: {video_tensor.min():.3f} to {video_tensor.max():.3f}")
                
                # Replace NaN values for safety but log the issue
                video_tensor = torch.where(torch.isnan(video_tensor), torch.zeros_like(video_tensor), video_tensor)
                print(f"   ⚠️  Replaced NaN with zeros as fallback")
            
            # Ensure proper range
            video_tensor = torch.clamp(video_tensor, 0.0, 1.0)
            
            # Removed verbose logging: print(f"✅ Successfully loaded clip: {video_path.name} -> {video_tensor.shape}")
            return video_tensor
            
        except Exception as e:
            print(f"❌ Error loading video {video_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _pick_start_frames(self, total_frames: int, clip_length: int) -> list[int]:
        """Pick start frames for multi-clip sampling."""
        if self.clips_per_video == 1:
            # Single clip - use existing logic centered around foul
            return [self.start_frame]
        
        # Calculate available range for sampling clips
        max_start_frame = min(total_frames - clip_length, self.end_frame - clip_length + 1)
        min_start_frame = max(0, self.start_frame)
        
        if max_start_frame <= min_start_frame:
            # Not enough frames, return single clip at start_frame
            return [max(0, min(self.start_frame, total_frames - clip_length))]
        
        if self.clip_sampling == 'uniform':
            # Evenly spaced clips across the available range
            if self.clips_per_video == 1:
                return [min_start_frame]
            
            stride = (max_start_frame - min_start_frame) / (self.clips_per_video - 1)
            starts = [min_start_frame + int(stride * i) for i in range(self.clips_per_video)]
            return starts
        else:  # random sampling
            import random
            starts = sorted(random.sample(
                range(min_start_frame, max_start_frame + 1), 
                min(self.clips_per_video, max_start_frame - min_start_frame + 1)
            ))
            # Pad with duplicates if needed
            while len(starts) < self.clips_per_video:
                starts.append(starts[-1])
            return starts

    def __getitem__(self, idx):
        action_info = self.actions[idx]
        
        all_action_video_files_relative = action_info["video_files_relative"]
        selected_video_paths_relative = []

        if self.views_indices is not None:
            # Use specific view indices if provided
            for i in self.views_indices:
                if 0 <= i < len(all_action_video_files_relative):
                    selected_video_paths_relative.append(all_action_video_files_relative[i])
        elif self.load_all_views:
            # Load all available views (default behavior)
            if self.max_views_to_load is not None:
                selected_video_paths_relative = all_action_video_files_relative[:self.max_views_to_load]
            else:
                selected_video_paths_relative = all_action_video_files_relative  # Use all available views
        else: 
            # Fallback: use max_views_to_load or default to 2
            limit = self.max_views_to_load if self.max_views_to_load is not None else 2
            selected_video_paths_relative = all_action_video_files_relative[:limit]
        
        if not selected_video_paths_relative: # Should not happen if _process_annotations filters properly
            # print(f"Warning: No views selected for action {action_info['action_id']}. Returning dummy.")
            num_expected_views = 1 # Fallback to 1
            if self.views_indices: num_expected_views = len(self.views_indices)
            elif self.load_all_views or self.max_views_to_load > 0 : num_expected_views = self.max_views_to_load
            
            dummy_clips = torch.zeros((num_expected_views, 3, self.frames_per_clip, self.target_height, self.target_width))
            
            # Return both original and standardized indices
            return {
                "clips": dummy_clips,
                "label_severity": torch.tensor(action_info["label_severity"], dtype=torch.long),
                "label_type": torch.tensor(action_info["label_type"], dtype=torch.long),
                
                # Original indices from vocabularies
                "contact_idx": torch.tensor(action_info["contact_idx"], dtype=torch.long),
                "bodypart_idx": torch.tensor(action_info["bodypart_idx"], dtype=torch.long),
                "upper_bodypart_idx": torch.tensor(action_info["upper_bodypart_idx"], dtype=torch.long),
                "multiple_fouls_idx": torch.tensor(action_info["multiple_fouls_idx"], dtype=torch.long),
                "try_to_play_idx": torch.tensor(action_info["try_to_play_idx"], dtype=torch.long),
                "touch_ball_idx": torch.tensor(action_info["touch_ball_idx"], dtype=torch.long),
                "handball_idx": torch.tensor(action_info["handball_idx"], dtype=torch.long),
                "handball_offence_idx": torch.tensor(action_info["handball_offence_idx"], dtype=torch.long),
                
                # Standardized indices based on fixed mappings
                "offence_standard_idx": torch.tensor(action_info["offence_standard_idx"], dtype=torch.long),
                "contact_standard_idx": torch.tensor(action_info["contact_standard_idx"], dtype=torch.long),
                "bodypart_standard_idx": torch.tensor(action_info["bodypart_standard_idx"], dtype=torch.long),
                "upper_bodypart_standard_idx": torch.tensor(action_info["upper_bodypart_standard_idx"], dtype=torch.long),
                "multiple_fouls_standard_idx": torch.tensor(action_info["multiple_fouls_standard_idx"], dtype=torch.long),
                "try_to_play_standard_idx": torch.tensor(action_info["try_to_play_standard_idx"], dtype=torch.long),
                "touch_ball_standard_idx": torch.tensor(action_info["touch_ball_standard_idx"], dtype=torch.long),
                "handball_standard_idx": torch.tensor(action_info["handball_standard_idx"], dtype=torch.long),
                "handball_offence_standard_idx": torch.tensor(action_info["handball_offence_standard_idx"], dtype=torch.long)
            }

        # Define item-specific augmentation based on severity for training split
        item_specific_aug = None
        if self.use_severity_aware_aug and self.split == 'train':
            severity_label = action_info["label_severity"]
            item_specific_aug = SeverityAwareAugmentation(severity_label)

        # Multi-clip sampling: get start frames first
        # For this we need to estimate total frames - use a rough estimate based on action duration
        estimated_total_frames = max(100, self.end_frame + 20)  # Conservative estimate
        start_frames = self._pick_start_frames(estimated_total_frames, self.frames_per_clip)
        
        clips_for_action = []  # Shape will be [clips_per_video, num_views, C, T, H, W]
        
        for clip_idx, start_frame in enumerate(start_frames):
            clips_for_this_start = []
            
            for video_path_rel_str in selected_video_paths_relative:
                # Construct full video path by combining dataset path with relative path
                full_video_path = self.split_dir / video_path_rel_str
                clip = self._get_video_clip(str(full_video_path), action_info, start_frame_override=start_frame)
                if clip is not None:
                    if item_specific_aug:
                        # [NaN-origin] Step 3: Check before SeverityAwareAugmentation
                        if torch.isnan(clip).any():
                            logger.error(f"[NaN-origin] NaN before SeverityAwareAugmentation – action {action_info['action_id']}")
                            raise RuntimeError("NaN before SeverityAwareAugmentation")
                        clip = item_specific_aug(clip)
                        # [NaN-origin] Step 3: Check after SeverityAwareAugmentation
                        if torch.isnan(clip).any():
                            logger.error(f"[NaN-origin] SeverityAwareAugmentation introduced NaN")
                            raise RuntimeError("NaN introduced by SeverityAwareAugmentation")

                    if self.transform: # Apply general transforms like normalization
                        # [NaN-origin] Step 2: Identify which CPU-side augmentation inserts NaNs
                        clip_before_transforms = clip.clone()
                        for t_idx, t in enumerate(self.transform.transforms):
                            clip_before = clip.clone()
                            clip = t(clip)
                            if torch.isnan(clip).any() and not torch.isnan(clip_before).any():
                                logger.error(f"[NaN-origin] CPU transform {t_idx}:{t.__class__.__name__}")
                                raise RuntimeError("NaN introduced")
                    clips_for_this_start.append(clip)
            
            clips_for_action.append(clips_for_this_start)
        
        # Now organize clips: clips_for_action is [clips_per_video][views_per_clip]
        # We need to restructure to [clips_per_video, views, C, T, H, W]
        num_expected_views = len(selected_video_paths_relative)
        
        # Check if we got any clips at all
        total_clips_loaded = sum(len(clips_for_start) for clips_for_start in clips_for_action)
        
        final_clips_tensor = None
        if total_clips_loaded == 0:
            print(f"Warning: All clips failed to load for action {action_info['action_id']}. Returning dummy tensor.")
            # Create a dummy tensor of shape (clips_per_video, num_expected_views, C, T, H, W)
            # Use small positive values instead of zeros to avoid numerical issues
            final_clips_tensor = torch.full(
                (self.clips_per_video, num_expected_views, 3, self.frames_per_clip, self.target_height, self.target_width),
                fill_value=0.01,  # Small positive value
                dtype=torch.float32
            )
        else:
            # Process each clip set and create the final tensor
            final_clips_list = []
            target_frames = self.frames_per_clip
            
            for clips_for_start in clips_for_action:
                # Handle this clip's views
                if len(clips_for_start) == 0:
                    # No views for this clip - create dummy with small positive values
                    dummy_clips = torch.full(
                        (num_expected_views, 3, target_frames, self.target_height, self.target_width),
                        fill_value=0.01,  # Small positive value
                        dtype=torch.float32
                    )
                    final_clips_list.append(dummy_clips)
                elif len(clips_for_start) < num_expected_views:
                    # Some views missing - pad with dummies
                    standardized_clips = []
                    
                    for clip in clips_for_start:
                        current_frames = clip.shape[1]  # [C, T, H, W]
                        if current_frames != target_frames:
                            if current_frames > target_frames:
                                start_idx = (current_frames - target_frames) // 2
                                clip = clip[:, start_idx:start_idx + target_frames, :, :]
                            else:
                                padding_needed = target_frames - current_frames
                                left_pad = padding_needed // 2
                                right_pad = padding_needed - left_pad
                                left_padding = clip[:, :1, :, :].repeat(1, left_pad, 1, 1)
                                right_padding = clip[:, -1:, :, :].repeat(1, right_pad, 1, 1)
                                clip = torch.cat([left_padding, clip, right_padding], dim=1)
                        standardized_clips.append(clip)
                    
                    # Add dummy clips for missing views
                    dummy_clip = torch.zeros((3, target_frames, self.target_height, self.target_width))
                    for _ in range(num_expected_views - len(clips_for_start)):
                        standardized_clips.append(dummy_clip)
                    
                    final_clips_list.append(torch.stack(standardized_clips))
                else:
                    # All views present - standardize and stack
                    standardized_clips = []
                    for clip in clips_for_start:
                        current_frames = clip.shape[1]  # [C, T, H, W]
                        if current_frames != target_frames:
                            if current_frames > target_frames:
                                start_idx = (current_frames - target_frames) // 2
                                clip = clip[:, start_idx:start_idx + target_frames, :, :]
                            else:
                                padding_needed = target_frames - current_frames
                                left_pad = padding_needed // 2
                                right_pad = padding_needed - left_pad
                                left_padding = clip[:, :1, :, :].repeat(1, left_pad, 1, 1)
                                right_padding = clip[:, -1:, :, :].repeat(1, right_pad, 1, 1)
                                clip = torch.cat([left_padding, clip, right_padding], dim=1)
                        standardized_clips.append(clip)
                    
                    final_clips_list.append(torch.stack(standardized_clips))
            
            # Stack all clips: [clips_per_video, views, C, T, H, W]
            final_clips_tensor = torch.stack(final_clips_list)

        # Critical: Check for NaN values and replace them
        if torch.isnan(final_clips_tensor).any():
            nan_count = torch.isnan(final_clips_tensor).sum().item()
            print(f"Warning: {nan_count} NaN values detected in final clips tensor for action {action_info['action_id']}")
            # Replace NaN with small positive values
            final_clips_tensor = torch.where(
                torch.isnan(final_clips_tensor), 
                torch.full_like(final_clips_tensor, 0.01), 
                final_clips_tensor
            )
            print(f"Replaced {nan_count} NaN values with 0.01")
        
        # Check for infinite values
        if torch.isinf(final_clips_tensor).any():
            inf_count = torch.isinf(final_clips_tensor).sum().item()
            print(f"Warning: {inf_count} infinite values detected in final clips tensor for action {action_info['action_id']}")
            # Replace inf with clamped values
            final_clips_tensor = torch.where(
                torch.isinf(final_clips_tensor), 
                torch.clamp(final_clips_tensor, min=0.0, max=1.0),
                final_clips_tensor
            )
            print(f"Clamped {inf_count} infinite values")
        
        # Ensure values are in valid range [0, 1]
        final_clips_tensor = torch.clamp(final_clips_tensor, min=0.0, max=1.0)

        # Return both original and standardized indices in a dictionary
        return {
            "clips": final_clips_tensor,
            "label_severity": torch.tensor(action_info["label_severity"], dtype=torch.long),
            "label_type": torch.tensor(action_info["label_type"], dtype=torch.long),
            
            # Original indices from vocabularies
            "contact_idx": torch.tensor(action_info["contact_idx"], dtype=torch.long),
            "bodypart_idx": torch.tensor(action_info["bodypart_idx"], dtype=torch.long),
            "upper_bodypart_idx": torch.tensor(action_info["upper_bodypart_idx"], dtype=torch.long),
            "multiple_fouls_idx": torch.tensor(action_info["multiple_fouls_idx"], dtype=torch.long),
            "try_to_play_idx": torch.tensor(action_info["try_to_play_idx"], dtype=torch.long),
            "touch_ball_idx": torch.tensor(action_info["touch_ball_idx"], dtype=torch.long),
            "handball_idx": torch.tensor(action_info["handball_idx"], dtype=torch.long),
            "handball_offence_idx": torch.tensor(action_info["handball_offence_idx"], dtype=torch.long),
            
            # Standardized indices based on fixed mappings
            "offence_standard_idx": torch.tensor(action_info["offence_standard_idx"], dtype=torch.long),
            "contact_standard_idx": torch.tensor(action_info["contact_standard_idx"], dtype=torch.long),
            "bodypart_standard_idx": torch.tensor(action_info["bodypart_standard_idx"], dtype=torch.long),
            "upper_bodypart_standard_idx": torch.tensor(action_info["upper_bodypart_standard_idx"], dtype=torch.long),
            "multiple_fouls_standard_idx": torch.tensor(action_info["multiple_fouls_standard_idx"], dtype=torch.long),
            "try_to_play_standard_idx": torch.tensor(action_info["try_to_play_standard_idx"], dtype=torch.long),
            "touch_ball_standard_idx": torch.tensor(action_info["touch_ball_standard_idx"], dtype=torch.long),
            "handball_standard_idx": torch.tensor(action_info["handball_standard_idx"], dtype=torch.long),
            "handball_offence_standard_idx": torch.tensor(action_info["handball_offence_standard_idx"], dtype=torch.long)
        }

def variable_views_collate_fn(batch):
    """
    Custom collate function to handle batches with variable numbers of views.
    
    Args:
        batch: List of dictionaries from dataset __getitem__
        
    Returns:
        Dictionary with batched tensors, handling variable view counts
    """
    if not batch:
        return {}
    
    # Separate clips from other features
    clips_list = [item["clips"] for item in batch]
    
    # Get all other keys (excluding clips)
    other_keys = [key for key in batch[0].keys() if key != "clips"]
    
    # Stack other features normally (they should all have same batch dimension)
    batched_data = {}
    for key in other_keys:
        batched_data[key] = torch.stack([item[key] for item in batch])
    
    # Handle variable views for clips
    # Each item in clips_list has shape (clips_per_video, num_views, C, T, H, W)
    # We need to handle the case where num_views (second dimension) varies
    
    # Check the dimensions to determine the format
    first_clip = clips_list[0]
    if first_clip.dim() == 6:  # [clips_per_video, num_views, C, T, H, W]
        # Get max views from the second dimension (num_views)
        max_views = max(clips.shape[1] for clips in clips_list)
        batch_size = len(clips_list)
        clips_per_video = first_clip.shape[0]
        
        if max_views == 1 or all(clips.shape[1] == clips_list[0].shape[1] for clips in clips_list):
            # All items have same number of views, can stack normally
            batched_data["clips"] = torch.stack(clips_list)
        else:
            # Variable number of views - create padded tensor
            # Get dimensions from first item
            _, _, C, T, H, W = first_clip.shape
            
            # Create padded tensor: (batch_size, clips_per_video, max_views, C, T, H, W)
            # Use zeros for padding - we'll handle masking properly
            padded_clips = torch.zeros(
                (batch_size, clips_per_video, max_views, C, T, H, W), 
                dtype=first_clip.dtype
            )
            
            for i, clips in enumerate(clips_list):
                num_views = clips.shape[1]
                padded_clips[i, :, :num_views] = clips
            
            batched_data["clips"] = padded_clips
            
            # Also create a mask indicating which views are real vs padded
            view_mask = torch.zeros(batch_size, clips_per_video, max_views, dtype=torch.bool)
            for i, clips in enumerate(clips_list):
                num_views = clips.shape[1]
                view_mask[i, :, :num_views] = True
            
            batched_data["view_mask"] = view_mask
            
            # [NaN-origin] Step 4: Verify collate & padding step
            if torch.isnan(padded_clips).any():
                logger.error(f"[NaN-origin] collate_fn padding step")
                raise RuntimeError("NaN introduced in collate_fn padding step")
            
    elif first_clip.dim() == 5:  # [num_views, C, T, H, W] - legacy format
        # Get max views from the first dimension (num_views)
        max_views = max(clips.shape[0] for clips in clips_list)
        batch_size = len(clips_list)
        
        if max_views == 1 or all(clips.shape[0] == clips_list[0].shape[0] for clips in clips_list):
            # All items have same number of views, can stack normally
            batched_data["clips"] = torch.stack(clips_list)
        else:
            # Variable number of views - create padded tensor
            # Get dimensions from first item
            _, C, T, H, W = first_clip.shape
            
            # Create padded tensor: (batch_size, max_views, C, T, H, W)
            # Use zeros for padding - we'll handle masking properly
            padded_clips = torch.zeros(
                (batch_size, max_views, C, T, H, W), 
                dtype=first_clip.dtype
            )
            
            for i, clips in enumerate(clips_list):
                num_views = clips.shape[0]
                padded_clips[i, :num_views] = clips
            
            batched_data["clips"] = padded_clips
            
            # Also create a mask indicating which views are real vs padded
            view_mask = torch.zeros(batch_size, max_views, dtype=torch.bool)
            for i, clips in enumerate(clips_list):
                num_views = clips.shape[0]
                view_mask[i, :num_views] = True
            
            batched_data["view_mask"] = view_mask
            
            # [NaN-origin] Step 4: Verify collate & padding step (legacy format)
            if torch.isnan(padded_clips).any():
                logger.error(f"[NaN-origin] collate_fn padding step (legacy format)")
                raise RuntimeError("NaN introduced in collate_fn padding step (legacy format)")
    
    else:
        raise ValueError(f"Unexpected clip tensor dimensions: {first_clip.shape}. Expected 5D or 6D tensor.")
    
    # Final safety check for NaN values in the batched clips
    if "clips" in batched_data:
        clips = batched_data["clips"]
        if torch.isnan(clips).any():
            nan_count = torch.isnan(clips).sum().item()
            print(f"ERROR: {nan_count} NaN values detected in batched clips! Replacing with zeros.")
            batched_data["clips"] = torch.where(
                torch.isnan(clips), 
                torch.zeros_like(clips), 
                clips
            )
        
        # Clamp to valid range
        batched_data["clips"] = torch.clamp(batched_data["clips"], min=0.0, max=1.0)
    
    return batched_data

# Add more extreme augmentations for very small datasets
class RandomRotation(torch.nn.Module):
    """Apply small random rotations to frames"""
    def __init__(self, max_angle=10, prob=0.4):
        super().__init__()
        self.max_angle = max_angle
        self.prob = prob
    
    def forward(self, clip):
        if random.random() > self.prob:
            return clip
        
        C, T, H, W = clip.shape
        angle = random.uniform(-self.max_angle, self.max_angle)
        
        # Apply rotation to each frame
        rotated_frames = []
        for t in range(T):
            frame = clip[:, t]  # [C, H, W]
            # Convert to PIL format for rotation, then back to tensor
            frame_pil = transforms.ToPILImage()(frame)
            rotated_pil = transforms.functional.rotate(frame_pil, angle)
            rotated_tensor = transforms.ToTensor()(rotated_pil)
            rotated_frames.append(rotated_tensor)
        
        return torch.stack(rotated_frames, dim=1)

class RandomElasticDeformation(torch.nn.Module):
    """Apply subtle elastic deformations for more data variety"""
    def __init__(self, alpha=50, sigma=5, prob=0.3):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.prob = prob
    
    def forward(self, clip):
        if random.random() > self.prob:
            return clip
        
        # Check if scipy is available for proper implementation
        if not SCIPY_AVAILABLE:
            # Fallback: apply simple random noise instead of elastic deformation
            C, T, H, W = clip.shape
            noise = torch.randn_like(clip) * 0.01  # Small amount of noise
            return torch.clamp(clip + noise, 0, 1)
        
        # Simple implementation using random displacement fields
        C, T, H, W = clip.shape
        
        # Create random displacement field
        dx = np.random.randn(H, W) * self.alpha
        dy = np.random.randn(H, W) * self.alpha
        
        # Smooth the displacement field
        dx = gaussian_filter(dx, self.sigma)
        dy = gaussian_filter(dy, self.sigma)
        
        # Apply to each frame
        deformed_frames = []
        for t in range(T):
            frame = clip[:, t].numpy()  # [C, H, W]
            
            # Create coordinate grids
            x, y = np.meshgrid(np.arange(W), np.arange(H))
            
            # Clamp indices to be within bounds
            x_new = np.clip(x + dx, 0, W-1).astype(np.float32)
            y_new = np.clip(y + dy, 0, H-1).astype(np.float32)
            
            # Apply deformation to each channel
            deformed_frame = np.zeros_like(frame)
            for c in range(C):
                # Use scipy's map_coordinates for smoother interpolation
                coords = np.array([y_new.flatten(), x_new.flatten()])
                deformed_channel = map_coordinates(frame[c], coords, order=1, mode='reflect')
                deformed_frame[c] = deformed_channel.reshape(H, W)
            
            deformed_frames.append(torch.from_numpy(deformed_frame))
        
        return torch.stack(deformed_frames, dim=1)

class RandomMixup(torch.nn.Module):
    """Apply mixup augmentation between random frames within the same clip"""
    def __init__(self, alpha=0.2, prob=0.3):
        super().__init__()
        self.alpha = alpha
        self.prob = prob
    
    def forward(self, clip):
        if random.random() > self.prob:
            return clip
        
        C, T, H, W = clip.shape
        if T < 2:
            return clip
        
        # Select two random frames
        idx1, idx2 = random.sample(range(T), 2)
        
        # Generate mixup coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Apply mixup
        mixed_clip = clip.clone()
        mixed_frame = lam * clip[:, idx1] + (1 - lam) * clip[:, idx2]
        
        # Replace one of the frames with the mixed version
        mixed_clip[:, idx1] = mixed_frame
        
        return mixed_clip

class RandomCutout(torch.nn.Module):
    """Randomly mask out rectangular regions"""
    def __init__(self, max_holes=3, max_height=20, max_width=20, prob=0.4):
        super().__init__()
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.prob = prob
    
    def forward(self, clip):
        if random.random() > self.prob:
            return clip
        
        C, T, H, W = clip.shape
        cutout_clip = clip.clone()
        
        # Apply to random subset of frames
        num_frames_to_affect = random.randint(1, max(1, T // 2))
        frames_to_affect = random.sample(range(T), num_frames_to_affect)
        
        for t in frames_to_affect:
            num_holes = random.randint(1, self.max_holes)
            
            for _ in range(num_holes):
                # Random hole size
                hole_height = random.randint(1, min(self.max_height, H // 4))
                hole_width = random.randint(1, min(self.max_width, W // 4))
                
                # Random position
                y = random.randint(0, H - hole_height)
                x = random.randint(0, W - hole_width)
                
                # Apply cutout (set to random noise instead of zeros for more variation)
                cutout_clip[:, t, y:y+hole_height, x:x+hole_width] = torch.randn(C, hole_height, hole_width) * 0.1
        
        return cutout_clip

class RandomTimeWarp(torch.nn.Module):
    """Apply time warping by changing frame sampling"""
    def __init__(self, warp_factor=0.2, prob=0.3):
        super().__init__()
        self.warp_factor = warp_factor
        self.prob = prob
    
    def forward(self, clip):
        if random.random() > self.prob:
            return clip
        
        C, T, H, W = clip.shape
        if T <= 4:  # Don't warp very short clips
            return clip
        
        # Create non-uniform time sampling
        warp = random.uniform(-self.warp_factor, self.warp_factor)
        
        # Generate warped indices
        original_indices = torch.linspace(0, T-1, T)
        center = T // 2
        
        # Apply warping around center
        warped_indices = original_indices.clone()
        for i in range(T):
            distance_from_center = abs(i - center) / center
            warp_amount = warp * distance_from_center
            warped_indices[i] = torch.clamp(
                original_indices[i] + warp_amount * (i - center),
                0, T-1
            )
        
        # Sample frames according to warped indices
        warped_indices = warped_indices.long()
        warped_clip = clip[:, warped_indices]
        
        return warped_clip

class VariableLengthAugmentation(torch.nn.Module):
    """
    Foul-aware variable-length augmentation that ensures the action frame is always included.
    
    This augmentation:
    1. GUARANTEES the foul frame is always within the sampled window
    2. Varies the position of the foul within the new clip length
    3. Maintains label consistency (foul is always present in the clip)
    4. STANDARDIZES output to target_frames for consistent tensor shapes
    """
    def __init__(self, 
                 min_frames=12, 
                 max_frames=24, 
                 action_position_variance=0.3,
                 prob=0.3,
                 foul_frame_index=8,
                 target_frames=16):  # Target output length for standardization
        super().__init__()
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.action_position_variance = action_position_variance
        self.prob = prob
        self.foul_frame_index = foul_frame_index  # Frame 75 is at index 8 in 16-frame clip (67-82)
        self.target_frames = target_frames
    
    def forward(self, clip):
        if random.random() > self.prob:
            return clip
            
        # Convert to float32 first if it's uint8
        original_dtype = clip.dtype
        if clip.dtype == torch.uint8:
            clip = clip.float() / 255.0
            
        C, T, H, W = clip.shape
        
        # Randomly choose new clip length
        new_length = random.randint(self.min_frames, self.max_frames)
        
        if new_length >= T:
            # If new length is longer, pad with repeated frames
            if new_length > T:
                # Pad by repeating boundary frames
                padding_needed = new_length - T
                left_pad = padding_needed // 2
                right_pad = padding_needed - left_pad
                
                # Repeat first and last frames
                left_padding = clip[:, :1, :, :].repeat(1, left_pad, 1, 1)
                right_padding = clip[:, -1:, :, :].repeat(1, right_pad, 1, 1)
                
                clip = torch.cat([left_padding, clip, right_padding], dim=1)
            return clip
        else:
            # CRITICAL: Ensure foul frame is always included in the sampled window
            
            # Determine where we want the foul to be in the new clip
            # Add some variance but ensure it's within bounds
            ideal_foul_position = new_length // 2  # Default to center
            max_shift = int(self.action_position_variance * new_length)
            position_shift = random.randint(-max_shift, max_shift)
            target_foul_position = max(0, min(ideal_foul_position + position_shift, new_length - 1))
            
            # Calculate where to start sampling from original clip
            # We want original_foul_index to end up at target_foul_position
            start_frame = self.foul_frame_index - target_foul_position
            
            # Ensure we don't go out of bounds
            start_frame = max(0, min(start_frame, T - new_length))
            end_frame = start_frame + new_length
            
            # Double-check that foul frame is included
            foul_in_window = (start_frame <= self.foul_frame_index < end_frame)
            if not foul_in_window:
                # Fallback: center the window around the foul frame
                start_frame = max(0, min(self.foul_frame_index - new_length // 2, T - new_length))
                end_frame = start_frame + new_length
            
            # Extract the subset
            clip = clip[:, start_frame:end_frame, :, :]
            
            # VALIDATION: Verify foul frame is included (for debugging)
            original_foul_in_new_clip = (start_frame <= self.foul_frame_index < end_frame)
            if not original_foul_in_new_clip:
                # This should never happen with our logic above, but safety check
                raise ValueError(f"Foul frame {self.foul_frame_index} not in window [{start_frame}:{end_frame}]")
        
        # CRITICAL: Standardize to target_frames to ensure consistent tensor shapes
        current_frames = clip.shape[1]
        if current_frames != self.target_frames:
            if current_frames > self.target_frames:
                # Crop: center crop to target length
                start_idx = (current_frames - self.target_frames) // 2
                clip = clip[:, start_idx:start_idx + self.target_frames, :, :]
            else:
                # Pad: repeat boundary frames
                padding_needed = self.target_frames - current_frames
                left_pad = padding_needed // 2
                right_pad = padding_needed - left_pad
                
                left_padding = clip[:, :1, :, :].repeat(1, left_pad, 1, 1)
                right_padding = clip[:, -1:, :, :].repeat(1, right_pad, 1, 1)
                clip = torch.cat([left_padding, clip, right_padding], dim=1)
        
        return clip

class MultiScaleTemporalAugmentation(torch.nn.Module):
    """
    Multi-scale temporal augmentation that samples frames at different rates
    to simulate different video speeds and temporal resolutions.
    
    This helps the model handle:
    1. Videos recorded at different frame rates
    2. Different temporal sampling strategies during inference
    3. Actions that happen at different speeds
    4. STANDARDIZES output to target_frames for consistent tensor shapes
    """
    def __init__(self, 
                 scale_factors=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                 prob=0.4,
                 target_frames=16):
        super().__init__()
        self.scale_factors = scale_factors
        self.prob = prob
        self.target_frames = target_frames
    
    def forward(self, clip):
        if random.random() > self.prob:
            return clip
            
        C, T, H, W = clip.shape
        
        # Randomly choose temporal scale factor
        scale = random.choice(self.scale_factors)
        
        if scale == 1.0:
            return clip
            
        # Calculate new temporal dimension
        new_T = max(4, int(T * scale))  # Ensure minimum 4 frames
        
        if scale > 1.0:
            # Slower motion: interpolate to create more frames
            # Use temporal interpolation
            indices = torch.linspace(0, T - 1, new_T)
            indices_floor = indices.floor().long()
            indices_ceil = indices.ceil().long()
            weights = indices - indices_floor.float()
            
            # Clamp indices to valid range
            indices_floor = torch.clamp(indices_floor, 0, T - 1)
            indices_ceil = torch.clamp(indices_ceil, 0, T - 1)
            
            # Linear interpolation between frames
            interpolated_clip = []
            for i, (floor_idx, ceil_idx, weight) in enumerate(zip(indices_floor, indices_ceil, weights)):
                if floor_idx == ceil_idx:
                    frame = clip[:, floor_idx, :, :]
                else:
                    frame = (1 - weight) * clip[:, floor_idx, :, :] + weight * clip[:, ceil_idx, :, :]
                interpolated_clip.append(frame.unsqueeze(1))
            
            clip = torch.cat(interpolated_clip, dim=1)
            
            # If we have more frames than target, subsample back to target length
            if new_T > T:
                indices = torch.linspace(0, new_T - 1, T).long()
                clip = clip[:, indices, :, :]
                
        else:
            # Faster motion: subsample frames
            indices = torch.linspace(0, T - 1, new_T).long()
            clip = clip[:, indices, :, :]
            
            # If we have fewer frames than target, repeat frames to match length
            if new_T < T:
                repeat_factor = T // new_T
                remainder = T % new_T
                
                repeated_clip = clip.repeat(1, repeat_factor, 1, 1)
                if remainder > 0:
                    extra_frames = clip[:, :remainder, :, :]
                    repeated_clip = torch.cat([repeated_clip, extra_frames], dim=1)
                clip = repeated_clip
        
        # CRITICAL: Standardize to target_frames to ensure consistent tensor shapes
        current_frames = clip.shape[1]
        if current_frames != self.target_frames:
            if current_frames > self.target_frames:
                # Crop: center crop to target length
                start_idx = (current_frames - self.target_frames) // 2
                clip = clip[:, start_idx:start_idx + self.target_frames, :, :]
            else:
                # Pad: repeat boundary frames
                padding_needed = self.target_frames - current_frames
                left_pad = padding_needed // 2
                right_pad = padding_needed - left_pad
                
                left_padding = clip[:, :1, :, :].repeat(1, left_pad, 1, 1)
                right_padding = clip[:, -1:, :, :].repeat(1, right_pad, 1, 1)
                clip = torch.cat([left_padding, clip, right_padding], dim=1)
        
        return clip

# ================================
# ADVANCED AUGMENTATION CLASSES
# ================================

class MultiScaleCrop(torch.nn.Module):
    """Multi-scale spatial cropping with random size selection"""
    def __init__(self, sizes=[224, 256, 288], prob=0.5, target_size=224):
        super().__init__()
        self.sizes = sizes
        self.prob = prob
        self.target_size = target_size
    
    def forward(self, clip):
        if random.random() > self.prob:
            return clip
            
        # Convert to float32 first if it's uint8
        original_dtype = clip.dtype
        if clip.dtype == torch.uint8:
            clip = clip.float() / 255.0
        
        C, T, H, W = clip.shape
        
        # Randomly select a scale
        scale_size = random.choice(self.sizes)
        
        # Calculate crop size maintaining aspect ratio
        if H > W:
            crop_h = int(scale_size * H / W)
            crop_w = scale_size
        else:
            crop_h = scale_size
            crop_w = int(scale_size * W / H)
        
        # Ensure crop size doesn't exceed original dimensions
        crop_h = min(crop_h, H)
        crop_w = min(crop_w, W)
        
        # Random crop position
        top = random.randint(0, max(0, H - crop_h))
        left = random.randint(0, max(0, W - crop_w))
        
        # Crop all frames
        cropped_clip = clip[:, :, top:top+crop_h, left:left+crop_w]
        
        # Resize to target size
        resized_frames = []
        for t in range(T):
            frame = cropped_clip[:, t]  # [C, H, W]
            resized_frame = torch.nn.functional.interpolate(
                frame.unsqueeze(0), size=(self.target_size, self.target_size), 
                mode='bilinear', align_corners=False
            ).squeeze(0)
            resized_frames.append(resized_frame)
        
        return torch.stack(resized_frames, dim=1)


class StrongColorJitter(torch.nn.Module):
    """Enhanced color jittering with hue, saturation, brightness, and contrast"""
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, prob=0.8):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob
    
    def forward(self, clip):
        if random.random() > self.prob:
            return clip
            
        # Convert to float32 first if it's uint8
        original_dtype = clip.dtype
        if clip.dtype == torch.uint8:
            clip = clip.float() / 255.0
        
        C, T, H, W = clip.shape
        
        # Generate random factors
        brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)
        
        # Apply transformations
        for t in range(T):
            frame = clip[:, t]  # [C, H, W]
            
            # Brightness
            frame = frame * brightness_factor
            
            # Contrast
            mean = frame.mean(dim=[1, 2], keepdim=True)
            frame = (frame - mean) * contrast_factor + mean
            
            # Convert to HSV for saturation and hue adjustments
            if C == 3:  # RGB
                # Simple approximation for HSV transformation
                # Saturation adjustment
                gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
                frame = frame * saturation_factor + gray.unsqueeze(0) * (1 - saturation_factor)
                
                # Hue shift (simplified)
                if abs(hue_factor) > 0.01:
                    cos_h = torch.cos(torch.tensor(hue_factor * 2 * 3.14159))
                    sin_h = torch.sin(torch.tensor(hue_factor * 2 * 3.14159))
                    
                    # Rotation matrix for hue shift
                    r = frame[0]
                    g = frame[1]
                    b = frame[2]
                    
                    frame[0] = r * cos_h - g * sin_h
                    frame[1] = r * sin_h + g * cos_h
            
            # Clamp values
            frame = torch.clamp(frame, 0, 1)
            clip[:, t] = frame
        
        return clip


class VideoRandAugment(torch.nn.Module):
    """RandAugment implementation for video data"""
    def __init__(self, n=2, m=10, prob=0.5):
        super().__init__()
        self.n = n  # Number of augmentations to apply
        self.m = m  # Magnitude (1-30)
        self.prob = prob
        
        # Define augmentation operations
        self.ops = [
            'autocontrast', 'equalize', 'invert', 'rotate', 'solarize',
            'color', 'posterize', 'brightness', 'contrast', 'sharpness'
        ]
    
    def forward(self, clip):
        if random.random() > self.prob:
            return clip
            
        # Convert to float32 first if it's uint8
        original_dtype = clip.dtype
        if clip.dtype == torch.uint8:
            clip = clip.float() / 255.0
        
        C, T, H, W = clip.shape
        
        # Select random operations
        selected_ops = random.sample(self.ops, min(self.n, len(self.ops)))
        
        for op in selected_ops:
            clip = self._apply_op(clip, op)
        
        return torch.clamp(clip, 0, 1)
    
    def _apply_op(self, clip, op):
        """Apply a single RandAugment operation"""
        C, T, H, W = clip.shape
        magnitude = self.m / 30.0  # Normalize to [0, 1]
        
        if op == 'autocontrast':
            for t in range(T):
                frame = clip[:, t]
                min_val = frame.min()
                max_val = frame.max()
                # Fix division by zero that causes NaN
                denominator = max_val - min_val
                if denominator > 1e-8:  # Only apply if there's sufficient contrast
                    clip[:, t] = (frame - min_val) / denominator
                # If denominator is too small, skip this frame (leave unchanged)
        
        elif op == 'brightness':
            factor = 1.0 + magnitude * random.uniform(-0.5, 0.5)
            # Clamp factor to prevent extreme values
            factor = torch.clamp(torch.tensor(factor), 0.1, 3.0).item()
            clip = clip * factor
        
        elif op == 'contrast':
            factor = 1.0 + magnitude * random.uniform(-0.5, 0.5)
            # Clamp factor to prevent extreme values
            factor = torch.clamp(torch.tensor(factor), 0.1, 3.0).item()
            mean = clip.mean(dim=[2, 3], keepdim=True)
            # Ensure we don't create extreme values
            clip = torch.clamp((clip - mean) * factor + mean, 0.0, 1.0)
        
        elif op == 'rotate':
            angle = magnitude * random.uniform(-30, 30)
            # Simple rotation approximation (could be improved)
            pass  # Skip for now to avoid complex implementation
        
        elif op == 'solarize':
            threshold = 1.0 - magnitude
            clip = torch.where(clip > threshold, 1.0 - clip, clip)
        
        elif op == 'color':
            factor = 1.0 + magnitude * random.uniform(-0.5, 0.5)
            # Clamp factor to prevent extreme values
            factor = torch.clamp(torch.tensor(factor), 0.1, 3.0).item()
            if C == 3:
                gray = 0.299 * clip[0:1] + 0.587 * clip[1:2] + 0.114 * clip[2:3]
                clip = clip * factor + gray * (1 - factor)
        
        elif op == 'equalize':
            # Add histogram equalization (simple version)
            for t in range(T):
                frame = clip[:, t]
                # Simple histogram equalization approximation
                frame_flat = frame.view(C, -1)
                frame_sorted, indices = torch.sort(frame_flat, dim=1)
                # Create equalized values
                eq_values = torch.linspace(0, 1, frame_flat.size(1), device=frame.device)
                eq_frame = torch.gather(eq_values.unsqueeze(0).expand(C, -1), 1, indices.argsort(dim=1))
                clip[:, t] = eq_frame.view(C, H, W)
        
        elif op == 'invert':
            clip = 1.0 - clip
        
        elif op == 'posterize':
            # Reduce the number of color levels
            bits = max(1, int(8 - magnitude * 4))  # 1-8 bits
            levels = 2 ** bits
            clip = torch.round(clip * (levels - 1)) / (levels - 1)
            
        elif op == 'sharpness':
            # Simple sharpening approximation
            factor = magnitude * 2.0  # 0-2 range
            if factor > 0.1:
                for t in range(T):
                    frame = clip[:, t]
                    # Simple sharpening kernel approximation
                    blurred = torch.nn.functional.avg_pool2d(
                        frame.unsqueeze(0), kernel_size=3, stride=1, padding=1
                    ).squeeze(0)
                    sharpened = frame + factor * (frame - blurred)
                    clip[:, t] = torch.clamp(sharpened, 0.0, 1.0)
        
        # Always clamp output to valid range and check for NaN
        clip = torch.clamp(clip, 0.0, 1.0)
        
        # Safety check: replace any NaN values with zeros
        if torch.isnan(clip).any():
            logger.warning(f"NaN detected after {op} augmentation, replacing with zeros")
            clip = torch.where(torch.isnan(clip), torch.zeros_like(clip), clip)
        
        return clip


class VideoMixUp(torch.nn.Module):
    """MixUp augmentation for video data"""
    def __init__(self, alpha=0.2, prob=0.5):
        super().__init__()
        self.alpha = alpha
        self.prob = prob
    
    def forward(self, clips, labels=None):
        """
        Apply MixUp to a batch of clips
        Args:
            clips: [B, C, T, H, W] or [B, V, C, T, H, W]
            labels: [B, ...] optional labels for mixing
        """
        if random.random() > self.prob or self.alpha <= 0:
            return clips, labels
        
        batch_size = clips.size(0)
        if batch_size < 2:
            return clips, labels
        
        # Generate mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Generate random permutation
        indices = torch.randperm(batch_size)
        
        # Mix clips
        mixed_clips = lam * clips + (1 - lam) * clips[indices]
        
        if labels is not None:
            # Return both original and mixed labels for loss calculation
            return mixed_clips, (labels, labels[indices], lam)
        
        return mixed_clips, None


class VideoCutMix(torch.nn.Module):
    """CutMix augmentation for video data"""
    def __init__(self, alpha=1.0, prob=0.5):
        super().__init__()
        self.alpha = alpha
        self.prob = prob
    
    def forward(self, clips, labels=None):
        """
        Apply CutMix to a batch of clips
        Args:
            clips: [B, C, T, H, W] or [B, V, C, T, H, W]
            labels: [B, ...] optional labels for mixing
        """
        if random.random() > self.prob or self.alpha <= 0:
            return clips, labels
        
        batch_size = clips.size(0)
        if batch_size < 2:
            return clips, labels
        
        # Get dimensions
        if clips.dim() == 6:  # Multi-view
            B, V, C, T, H, W = clips.shape
        else:  # Single view
            B, C, T, H, W = clips.shape
            V = 1
        
        # Generate mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Generate random permutation
        indices = torch.randperm(batch_size)
        
        # Calculate bounding box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        
        # Random position
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_clips = clips.clone()
        if clips.dim() == 6:  # Multi-view
            mixed_clips[:, :, :, :, bby1:bby2, bbx1:bbx2] = clips[indices][:, :, :, :, bby1:bby2, bbx1:bbx2]
        else:  # Single view
            mixed_clips[:, :, :, bby1:bby2, bbx1:bbx2] = clips[indices][:, :, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        if labels is not None:
            return mixed_clips, (labels, labels[indices], lam)
        
        return mixed_clips, None


class TemporalMixUp(torch.nn.Module):
    """Temporal MixUp that mixes frames from different time steps"""
    def __init__(self, alpha=0.4, prob=0.3):
        super().__init__()
        self.alpha = alpha
        self.prob = prob
    
    def forward(self, clip):
        if random.random() > self.prob:
            return clip
        
        C, T, H, W = clip.shape
        if T < 3:
            return clip
        
        # Select temporal mixing range
        mix_length = random.randint(2, min(T, 4))
        start_idx = random.randint(0, T - mix_length)
        
        # Generate mixing coefficients
        lambdas = np.random.beta(self.alpha, self.alpha, size=mix_length)
        
        # Apply temporal mixing
        mixed_clip = clip.clone()
        for i in range(mix_length - 1):
            idx1 = start_idx + i
            idx2 = start_idx + i + 1
            lam = lambdas[i]
            
            mixed_clip[:, idx1] = lam * clip[:, idx1] + (1 - lam) * clip[:, idx2]
        
        return mixed_clip

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
import numpy as np
from typing import Dict, List, Union, Tuple
import logging

from .config import ModelConfig
from .embedding_manager import EmbeddingManager
from .view_aggregator import ViewAggregator
from .validator import InputValidator, ValidationError

logger = logging.getLogger(__name__)

class VideoAugmentation(nn.Module):
    """
    Comprehensive video augmentation for addressing class imbalance and small dataset issues.
    Applies various temporal and spatial augmentations during training.
    """
    
    def __init__(self, severity_weights: Dict[float, float] = None, training: bool = True):
        super().__init__()
        self.training = training
        self.severity_weights = severity_weights or {1.0: 1.0, 2.0: 3.0, 3.0: 5.0, 4.0: 7.0}
        
        # Spatial augmentation parameters
        self.spatial_prob = 0.7
        self.temporal_prob = 0.6
        self.intensity_prob = 0.5
        
    def forward(self, clips: torch.Tensor, severity: torch.Tensor = None) -> torch.Tensor:
        """
        Apply augmentations based on severity class (more aggressive for minority classes).
        
        Args:
            clips: [B, N, C, T, H, W] or [B, C, T, H, W]
            severity: [B] severity labels for adaptive augmentation
            
        Returns:
            Augmented clips
        """
        if not self.training:
            return clips
            
        batch_size = clips.size(0)
        augmented_clips = clips.clone()
        
        for i in range(batch_size):
            # Get augmentation intensity based on severity (more for minority classes)
            aug_intensity = 1.0
            if severity is not None:
                sev_val = severity[i].item()
                aug_intensity = self.severity_weights.get(sev_val, 1.0)
            
            # Apply augmentations with adaptive intensity
            if clips.dim() == 6:  # [B, N, C, T, H, W] - multi-view
                for view in range(clips.size(1)):
                    augmented_clips[i, view] = self._apply_augmentations(
                        clips[i, view], aug_intensity
                    )
            else:  # [B, C, T, H, W] - single view
                augmented_clips[i] = self._apply_augmentations(clips[i], aug_intensity)
                
        return augmented_clips
    
    def _apply_augmentations(self, clip: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply individual augmentations to a single clip."""
        # Adjust probabilities based on intensity
        spatial_prob = min(0.9, self.spatial_prob * intensity)
        temporal_prob = min(0.8, self.temporal_prob * intensity)
        intensity_prob = min(0.7, self.intensity_prob * intensity)
        
        # Temporal augmentations
        if random.random() < temporal_prob:
            clip = self._temporal_augment(clip, intensity)
            
        # Spatial augmentations  
        if random.random() < spatial_prob:
            clip = self._spatial_augment(clip, intensity)
            
        # Intensity/color augmentations
        if random.random() < intensity_prob:
            clip = self._intensity_augment(clip, intensity)
            
        return clip
    
    def _temporal_augment(self, clip: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply temporal augmentations."""
        C, T, H, W = clip.shape
        
        # Random temporal cropping (keep at least 80% of frames)
        if random.random() < 0.4 * intensity:
            min_frames = max(int(T * 0.8), T - 8)
            new_T = random.randint(min_frames, T)
            start_idx = random.randint(0, T - new_T)
            clip = clip[:, start_idx:start_idx + new_T]
            
            # Interpolate back to original temporal size
            clip = F.resize(clip.permute(1, 0, 2, 3), (T, H, W)).permute(1, 0, 2, 3)
        
        # Temporal reversal (for minority classes)
        if random.random() < 0.2 * intensity:
            clip = torch.flip(clip, [1])  # Flip temporal dimension
            
        # Frame dropping and duplication
        if random.random() < 0.3 * intensity:
            # Randomly drop 1-2 frames and duplicate others
            drop_frames = random.randint(1, min(2, T // 4))
            keep_indices = list(range(T))
            for _ in range(drop_frames):
                if len(keep_indices) > T // 2:
                    keep_indices.remove(random.choice(keep_indices))
            
            # Add duplicates to maintain temporal size
            while len(keep_indices) < T:
                keep_indices.append(random.choice(keep_indices))
                
            keep_indices.sort()
            clip = clip[:, keep_indices[:T]]
        
        return clip
    
    def _spatial_augment(self, clip: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply spatial augmentations."""
        C, T, H, W = clip.shape
        
        # Random horizontal flip
        if random.random() < 0.5:
            clip = torch.flip(clip, [3])  # Flip width dimension
        
        # Random rotation (more aggressive for minority classes)
        if random.random() < 0.3 * intensity:
            angle = random.uniform(-10 * intensity, 10 * intensity)
            clip = self._rotate_clip(clip, angle)
        
        # Random scaling and cropping
        if random.random() < 0.6 * intensity:
            scale_factor = random.uniform(0.85, 1.15)
            clip = self._scale_and_crop(clip, scale_factor)
        
        # Random erasing (more aggressive for minority classes)
        if random.random() < 0.25 * intensity:
            clip = self._random_erase(clip, intensity)
            
        return clip
    
    def _intensity_augment(self, clip: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply intensity and color augmentations."""
        # Brightness adjustment
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.7, 1.3)
            clip = torch.clamp(clip * brightness_factor, 0, 1)
        
        # Contrast adjustment  
        if random.random() < 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            mean = clip.mean(dim=[2, 3], keepdim=True)
            clip = torch.clamp((clip - mean) * contrast_factor + mean, 0, 1)
        
        # Saturation adjustment (if RGB)
        if clip.size(0) == 3 and random.random() < 0.4:
            # Convert to grayscale and blend
            gray = 0.299 * clip[0] + 0.587 * clip[1] + 0.114 * clip[2]
            saturation_factor = random.uniform(0.5, 1.5)
            clip = torch.clamp(
                clip * saturation_factor + gray.unsqueeze(0) * (1 - saturation_factor), 
                0, 1
            )
            
        # Gaussian noise (subtle)
        if random.random() < 0.3 * intensity:
            noise_std = 0.01 * intensity
            noise = torch.randn_like(clip) * noise_std
            clip = torch.clamp(clip + noise, 0, 1)
            
        return clip
    
    def _rotate_clip(self, clip: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate entire video clip."""
        C, T, H, W = clip.shape
        rotated_frames = []
        
        for t in range(T):
            frame = clip[:, t]  # [C, H, W]
            # Convert to PIL format, rotate, convert back
            frame_pil = transforms.ToPILImage()(frame)
            rotated_pil = transforms.functional.rotate(frame_pil, angle)
            rotated_tensor = transforms.ToTensor()(rotated_pil)
            rotated_frames.append(rotated_tensor)
            
        return torch.stack(rotated_frames, dim=1)
    
    def _scale_and_crop(self, clip: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """Scale and randomly crop video clip."""
        C, T, H, W = clip.shape
        
        # Resize with scale factor
        new_H, new_W = int(H * scale_factor), int(W * scale_factor)
        scaled_clip = F.resize(clip.permute(1, 0, 2, 3), (new_H, new_W)).permute(1, 0, 2, 3)
        
        # Random crop back to original size
        if scale_factor > 1.0:
            # Crop from larger image
            crop_H, crop_W = H, W
            start_H = random.randint(0, new_H - crop_H)
            start_W = random.randint(0, new_W - crop_W)
            clip = scaled_clip[:, :, start_H:start_H + crop_H, start_W:start_W + crop_W]
        else:
            # Pad smaller image
            pad_H = (H - new_H) // 2
            pad_W = (W - new_W) // 2
            clip = F.pad(scaled_clip, [pad_W, pad_W, pad_H, pad_H])
            
        return clip
    
    def _random_erase(self, clip: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply random erasing to video clip."""
        C, T, H, W = clip.shape
        
        # Erase parameters (more aggressive for minority classes)
        erase_prob = 0.3 * intensity
        area_ratio = random.uniform(0.02, 0.1 * intensity)
        aspect_ratio = random.uniform(0.3, 3.3)
        
        if random.random() < erase_prob:
            area = H * W * area_ratio
            h = int(round(np.sqrt(area * aspect_ratio)))
            w = int(round(np.sqrt(area / aspect_ratio)))
            
            if h < H and w < W:
                x1 = random.randint(0, H - h)
                y1 = random.randint(0, W - w)
                
                # Apply to all frames
                clip[:, :, x1:x1 + h, y1:y1 + w] = torch.randn_like(
                    clip[:, :, x1:x1 + h, y1:y1 + w]
                ) * 0.1
                
        return clip

class ResNet3DBackbone(nn.Module):
    """3D ResNet backbone for video feature extraction."""
    
    def __init__(self, model_name='r2plus1d_18', pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet3D
        if model_name == 'resnet3d_18':
            self.backbone = models.video.r3d_18(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'mc3_18':
            self.backbone = models.video.mc3_18(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'r2plus1d_18':
            self.backbone = models.video.r2plus1d_18(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet3d_50':
            # resnet3d_50 doesn't exist in torchvision, fall back to best available model
            print(f"Warning: resnet3d_50 not available in torchvision. Using r2plus1d_18 (better accuracy) instead.")
            self.backbone = models.video.r2plus1d_18(pretrained=pretrained)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported model: {model_name}. Available options: 'resnet3d_18', 'mc3_18', 'r2plus1d_18'")
        
        # Remove final classification layer
        self.backbone.fc = nn.Identity()
        
        # Stronger dropout for small dataset (increased from 0.3 to 0.4)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W] video tensor
        Returns:
            features: [B, feature_dim] global video features
        """
        features = self.backbone(x)
        features = self.dropout(features)
        return features

class MultiTaskMultiViewResNet3D(nn.Module):
    """
    Multi-task, multi-view ResNet3D model for sports action classification.
    
    This model processes multiple video views and categorical features to predict:
    1. Severity of the action
    2. Type of the action
    """
    
    def __init__(
        self,
        num_severity: int,
        num_action_type: int,
        vocab_sizes: Dict[str, int],
        backbone_name: str = 'r2plus1d_18',
        config: ModelConfig = None,
        use_augmentation: bool = True,
        severity_weights: Dict[float, float] = None
    ):
        """
        Initialize the multi-task multi-view ResNet3D model.
        
        Args:
            num_severity: Number of classes for severity prediction
            num_action_type: Number of classes for action type prediction
            vocab_sizes: Dictionary mapping feature names to vocabulary sizes
            backbone_name: ResNet3D variant ('resnet3d_18', 'mc3_18', 'r2plus1d_18')
            config: Model configuration object
            use_augmentation: Whether to apply adaptive augmentation for class imbalance
            severity_weights: Augmentation intensity weights for each severity class
        """
        super().__init__()
        
        # Use default config if none provided
        self.config = config if config is not None else ModelConfig()
        self.num_severity = num_severity
        self.num_action_type = num_action_type
        self.backbone_name = backbone_name
        self.use_augmentation = use_augmentation
        
        # Validate vocabulary sizes
        self._validate_vocab_sizes(vocab_sizes)
        self.vocab_sizes = vocab_sizes
        
        # Initialize augmentation for addressing class imbalance
        if self.use_augmentation:
            self.video_augmentation = VideoAugmentation(
                severity_weights=severity_weights,
                training=True
            )
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"Initialized MultiTaskMultiViewResNet3D with {backbone_name}, "
                   f"{num_severity} severity classes and {num_action_type} action type classes. "
                   f"Augmentation: {'enabled' if use_augmentation else 'disabled'}")
    
    def _validate_vocab_sizes(self, vocab_sizes: Dict[str, int]) -> None:
        """Validate that all required vocabulary sizes are provided."""
        required_vocab_keys = [
            'contact', 'bodypart', 'upper_bodypart', 'lower_bodypart',
            'multiple_fouls', 'try_to_play', 'touch_ball', 'handball', 'handball_offence'
        ]
        
        missing_keys = [key for key in required_vocab_keys if key not in vocab_sizes]
        if missing_keys:
            raise ValueError(f"Missing vocabulary sizes for: {missing_keys}")
        
        # Validate that all sizes are positive integers
        for key, size in vocab_sizes.items():
            if not isinstance(size, int) or size <= 0:
                raise ValueError(f"Vocabulary size for '{key}' must be a positive integer, got {size}")
    
    def _initialize_components(self) -> None:
        """Initialize all model components."""
        try:
            # Load ResNet3D backbone
            self.backbone = ResNet3DBackbone(self.backbone_name, pretrained=True)
            self.video_feature_dim = self.backbone.feature_dim
            
            # Initialize embedding manager (reuse existing)
            self.embedding_manager = EmbeddingManager(self.config, self.vocab_sizes)
            
            # Initialize view aggregator (reuse existing) 
            self.view_aggregator = ViewAggregator(self.config, self.video_feature_dim)
            
            # Initialize input validator (reuse existing)
            self.input_validator = InputValidator(self.config, self.vocab_sizes)
            
            # Calculate combined feature dimension
            combined_feature_dim = self.video_feature_dim + self.config.get_total_embedding_dim()
            
            # Create classification heads with stronger regularization for small dataset
            self.severity_head = nn.Sequential(
                nn.Linear(combined_feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.6),  # Increased from 0.5 for small dataset
                nn.Linear(256, self.num_severity)
            )
            
            self.action_type_head = nn.Sequential(
                nn.Linear(combined_feature_dim, 256), 
                nn.ReLU(),
                nn.Dropout(0.6),  # Increased from 0.5 for small dataset
                nn.Linear(256, self.num_action_type)
            )
            
            logger.info(f"ResNet3D components initialized successfully. "
                       f"Combined feature dim: {combined_feature_dim}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model components: {str(e)}") from e
    
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            batch_data: Dictionary containing:
                - clips: Video tensor(s) [B, N, C, T, H, W] or List[[N_i, C, T, H, W]]
                - view_mask: Optional boolean mask [B, N] for valid views
                - severity: Optional severity labels [B] for adaptive augmentation
                - All categorical feature indices
        
        Returns:
            Tuple of (severity_logits, action_type_logits)
        """
        try:
            # Validate inputs
            batch_size = self.input_validator.validate_batch_data(batch_data)
            
            # Apply adaptive augmentation based on severity (more for minority classes)
            clips = batch_data["clips"]
            if self.use_augmentation and self.training:
                severity_labels = batch_data.get("severity", None)
                clips = self.video_augmentation(clips, severity_labels)
                # Update batch_data with augmented clips
                batch_data = {**batch_data, "clips": clips}
            
            # Process video features
            video_features = self._process_video_features(batch_data)
            
            # Process categorical features
            categorical_features = self._process_categorical_features(batch_data)
            
            # Combine features
            combined_features = torch.cat([video_features, categorical_features], dim=1)
            
            # Generate predictions
            severity_logits = self.severity_head(combined_features)
            action_type_logits = self.action_type_head(combined_features)
            
            return severity_logits, action_type_logits
            
        except (ValidationError, RuntimeError) as e:
            # Re-raise known errors
            raise e
        except Exception as e:
            # Wrap unexpected errors
            raise RuntimeError(f"Forward pass failed: {str(e)}") from e
    
    def _process_video_features(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process video clips to extract aggregated features."""
        clips = batch_data["clips"]
        view_mask = batch_data.get("view_mask", None)
        
        if isinstance(clips, list):
            # Handle variable-length views
            features, view_mask = self.view_aggregator.process_variable_views(clips, self.backbone)
        else:
            # Handle standard tensor input
            batch_size, max_views = clips.shape[:2]
            
            # Reshape and process through backbone
            clips_flat = clips.view(-1, *clips.shape[2:])  # [B*N, C, T, H, W]
            features_flat = self.backbone(clips_flat)
            
            # Reshape back to [B, N, feature_dim]
            features = features_flat.view(batch_size, max_views, -1)
        
        # Aggregate views
        aggregated_features = self.view_aggregator.aggregate_views(features, view_mask)
        
        return aggregated_features
    
    def _process_categorical_features(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process categorical features through embeddings."""
        original_embedded, standard_embedded = self.embedding_manager.embed_features(batch_data)
        
        # Combine original and standard embeddings
        categorical_features = torch.cat([original_embedded, standard_embedded], dim=1)
        
        return categorical_features
    
    @classmethod
    def create_model(
        cls,
        num_severity: int,
        num_action_type: int,
        vocab_sizes: Dict[str, int],
        backbone_name: str = 'r2plus1d_18',
        use_attention_aggregation: bool = True,
        use_augmentation: bool = True,
        severity_weights: Dict[float, float] = None,
        **config_kwargs
    ) -> 'MultiTaskMultiViewResNet3D':
        """
        Factory method to create a configured model instance.
        
        Args:
            num_severity: Number of severity classes
            num_action_type: Number of action type classes  
            vocab_sizes: Dictionary mapping feature names to vocabulary sizes
            backbone_name: ResNet3D variant ('resnet3d_18', 'mc3_18', 'r2plus1d_18')
            use_attention_aggregation: Whether to use attention-based view aggregation
            use_augmentation: Whether to apply adaptive augmentation for class imbalance
            severity_weights: Custom augmentation weights for severity classes
            **config_kwargs: Additional configuration parameters
            
        Returns:
            Configured MultiTaskMultiViewResNet3D instance
        """
        # Create configuration
        config = ModelConfig(
            use_attention_aggregation=use_attention_aggregation,
            **config_kwargs
        )
        
        # Default severity weights for class imbalance (higher weights for minority classes)
        if severity_weights is None:
            severity_weights = {
                1.0: 1.0,   # Majority class - normal augmentation
                2.0: 2.5,   # 2.5x more aggressive augmentation
                3.0: 4.0,   # 4x more aggressive augmentation  
                4.0: 6.0,   # 6x more aggressive augmentation
                5.0: 8.0    # 8x more aggressive augmentation (if exists)
            }
        
        # Create and return model
        return cls(
            num_severity=num_severity,
            num_action_type=num_action_type,
            vocab_sizes=vocab_sizes,
            backbone_name=backbone_name,
            config=config,
            use_augmentation=use_augmentation,
            severity_weights=severity_weights
        )
    
    def train(self, mode: bool = True):
        """Override train method to handle augmentation state."""
        super().train(mode)
        if hasattr(self, 'video_augmentation'):
            self.video_augmentation.training = mode
        return self
    
    def eval(self):
        """Override eval method to disable augmentation."""
        return self.train(False)
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information for logging/debugging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'backbone_name': self.backbone_name,
            'video_feature_dim': self.video_feature_dim,
            'total_embedding_dim': self.config.get_total_embedding_dim(),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_severity_classes': self.num_severity,
            'num_action_type_classes': self.num_action_type,
            'augmentation_enabled': self.use_augmentation,
            'severity_weights': getattr(self, 'video_augmentation', {}).get('severity_weights', None) if hasattr(self, 'video_augmentation') else None
        } 
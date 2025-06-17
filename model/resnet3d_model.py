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
    Enhanced video augmentation for addressing class imbalance and small dataset issues.
    Applies various temporal and spatial augmentations during training with configurable intensity.
    """
    
    def __init__(self, severity_weights: Dict[float, float] = None, training: bool = True, enabled: bool = True, conservative: bool = False):
        super().__init__()
        self.training = training
        self.enabled = enabled  # Global enable/disable flag for debugging
        self.conservative = conservative  # Conservative mode for safety
        self.severity_weights = severity_weights or {1.0: 1.0, 2.0: 3.0, 3.0: 5.0, 4.0: 7.0}
        
        # Augmentation parameters - more aggressive by default
        self.spatial_prob = 0.6 if conservative else 0.8
        self.temporal_prob = 0.5 if conservative else 0.7
        self.intensity_prob = 0.4 if conservative else 0.6
        
    def forward(self, clips: torch.Tensor, severity: torch.Tensor = None) -> torch.Tensor:
        """
        Apply augmentations based on severity class (more aggressive for minority classes).
        
        Args:
            clips: [B, N, C, T, H, W] or [B, C, T, H, W]
            severity: [B] severity labels for adaptive augmentation
            
        Returns:
            Augmented clips
        """
        # Quick exit if augmentation is disabled or in eval mode
        if not self.training or not self.enabled:
            return clips
            
        batch_size = clips.size(0)
        # Create a copy to avoid in-place modifications
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
                    original_clip = clips[i, view].clone()
                    augmented_clip = self._apply_augmentations(original_clip, aug_intensity)
                    # Ensure dimensions match before assignment
                    if augmented_clip.shape == original_clip.shape:
                        augmented_clips[i, view] = augmented_clip
                    else:
                        # If dimensions don't match, resize to match original
                        C, T, H, W = original_clip.shape
                        augmented_clips[i, view] = torch.nn.functional.interpolate(
                            augmented_clip.view(C*T, H, W).unsqueeze(0),
                            size=(H, W), mode='bilinear', align_corners=False
                        ).squeeze(0).view(C, T, H, W)
            else:  # [B, C, T, H, W] - single view
                original_clip = clips[i].clone()
                augmented_clip = self._apply_augmentations(original_clip, aug_intensity)
                # Ensure dimensions match before assignment
                if augmented_clip.shape == original_clip.shape:
                    augmented_clips[i] = augmented_clip
                else:
                    # If dimensions don't match, resize to match original
                    C, T, H, W = original_clip.shape
                    augmented_clips[i] = torch.nn.functional.interpolate(
                        augmented_clip.view(C*T, H, W).unsqueeze(0),
                        size=(H, W), mode='bilinear', align_corners=False
                    ).squeeze(0).view(C, T, H, W)
                
        return augmented_clips
    
    def _apply_augmentations(self, clip: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply individual augmentations to a single clip."""
        # Store original dimensions
        original_shape = clip.shape
        
        # Adjust probabilities based on intensity
        spatial_prob = min(0.95, self.spatial_prob * intensity)
        temporal_prob = min(0.85, self.temporal_prob * intensity)
        intensity_prob = min(0.8, self.intensity_prob * intensity)
        
        # Apply augmentations - less conservative unless in conservative mode
        prob_reduction = 0.5 if self.conservative else 0.8
        
        # Temporal augmentations
        if random.random() < temporal_prob * prob_reduction:
            clip = self._temporal_augment(clip, intensity)
            
        # Spatial augmentations
        if random.random() < spatial_prob * prob_reduction:
            clip = self._spatial_augment(clip, intensity)
            
        # Intensity/color augmentations (these are always safe)
        if random.random() < intensity_prob:
            clip = self._intensity_augment(clip, intensity)
        
        # Ensure output has same shape as input
        if clip.shape != original_shape:
            C, T, H, W = original_shape
            clip = torch.nn.functional.interpolate(
                clip.view(C*T, -1, clip.shape[-1]).unsqueeze(0),
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(0).view(C, T, H, W)
            
        return clip
    
    def _temporal_augment(self, clip: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply temporal augmentations with improved variety."""
        C, T, H, W = clip.shape
        
        # Temporal reversal
        if random.random() < 0.4 * intensity:
            clip = torch.flip(clip, [1])  # Flip temporal dimension
        
        # Frame shuffling (more aggressive)
        if random.random() < 0.3 * intensity and not self.conservative:
            if T > 4:
                # Shuffle middle frames, optionally keep boundaries
                if random.random() < 0.7:  # Keep first and last
                    middle_indices = list(range(1, T-1))
                    random.shuffle(middle_indices)
                    new_indices = [0] + middle_indices + [T-1]
                else:  # Shuffle all frames
                    new_indices = list(range(T))
                    random.shuffle(new_indices)
                clip = clip[:, new_indices]
        
        # Temporal dropout (drop random frames and repeat others)
        if random.random() < 0.25 * intensity and not self.conservative:
            if T > 6:
                num_drops = random.randint(1, min(3, T//4))
                drop_indices = random.sample(range(T), num_drops)
                keep_indices = [i for i in range(T) if i not in drop_indices]
                
                # Repeat some kept frames to maintain temporal dimension
                while len(keep_indices) < T:
                    keep_indices.append(random.choice(keep_indices))
                
                clip = clip[:, keep_indices[:T]]
        
        return clip
    
    def _spatial_augment(self, clip: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply spatial augmentations with improved variety."""
        C, T, H, W = clip.shape
        
        # Random horizontal flip
        if random.random() < 0.6:
            clip = torch.flip(clip, [3])  # Flip width dimension
        
        # More aggressive random cropping
        if random.random() < 0.5 * intensity:
            # More varied crop ratios
            crop_ratio = random.uniform(0.75 if self.conservative else 0.7, 1.0)
            crop_h = max(int(H * crop_ratio), H - 20)
            crop_w = max(int(W * crop_ratio), W - 20)
            
            if crop_h < H or crop_w < W:
                top = random.randint(0, max(0, H - crop_h))
                left = random.randint(0, max(0, W - crop_w))
                
                # Crop and resize back
                cropped = clip[:, :, top:top+crop_h, left:left+crop_w]
                # Resize back to original size frame by frame
                resized_frames = []
                for t in range(T):
                    frame = cropped[:, t]  # [C, H, W]
                    resized_frame = torch.nn.functional.interpolate(
                        frame.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
                    ).squeeze(0)
                    resized_frames.append(resized_frame)
                clip = torch.stack(resized_frames, dim=1)
        
        # Random rotation (small angles)
        if random.random() < 0.2 * intensity and not self.conservative:
            angle = random.uniform(-5, 5) * intensity  # Small rotation
            # Simple rotation approximation using affine transform
            cos_angle = torch.cos(torch.tensor(angle * 3.14159 / 180))
            sin_angle = torch.sin(torch.tensor(angle * 3.14159 / 180))
            
            # Apply rotation to random subset of frames
            num_frames_to_rotate = random.randint(1, max(1, T//2))
            frames_to_rotate = random.sample(range(T), num_frames_to_rotate)
            
            for t in frames_to_rotate:
                # Simple shear approximation for rotation
                if random.random() < 0.5:
                    clip[:, t] = torch.roll(clip[:, t], shifts=int(sin_angle * 2), dims=1)
        
        return clip
    
    def _intensity_augment(self, clip: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply intensity and color augmentations with enhanced variety."""
        # Brightness adjustment (more varied)
        if random.random() < 0.6:
            brightness_factor = random.uniform(0.7, 1.3) if not self.conservative else random.uniform(0.8, 1.2)
            clip = torch.clamp(clip * brightness_factor, 0, 1)
        
        # Contrast adjustment (more varied)
        if random.random() < 0.6:
            contrast_factor = random.uniform(0.8, 1.2) if not self.conservative else random.uniform(0.9, 1.1)
            mean = clip.mean(dim=[2, 3], keepdim=True)
            clip = torch.clamp((clip - mean) * contrast_factor + mean, 0, 1)
        
        # Saturation-like adjustment
        if random.random() < 0.4 * intensity:
            saturation_factor = random.uniform(0.8, 1.2)
            if clip.size(0) == 3:  # RGB
                gray = 0.299 * clip[0:1] + 0.587 * clip[1:2] + 0.114 * clip[2:3]
                clip = clip * saturation_factor + gray * (1 - saturation_factor)
        
        # Gaussian noise (adaptive)
        if random.random() < 0.4 * intensity:
            noise_std = (0.01 if self.conservative else 0.015) * intensity
            noise = torch.randn_like(clip) * noise_std
            clip = torch.clamp(clip + noise, 0, 1)
        
        # Gamma correction
        if random.random() < 0.3 * intensity:
            gamma = random.uniform(0.8, 1.2)
            clip = torch.clamp(torch.pow(clip, gamma), 0, 1)
            
        return clip
    
    def enable_conservative_mode(self):
        """Enable conservative augmentation mode."""
        self.conservative = True
        
    def disable_conservative_mode(self):
        """Disable conservative augmentation mode."""
        self.conservative = False
    
    def disable(self):
        """Disable augmentation for debugging."""
        self.enabled = False
        
    def enable(self):
        """Re-enable augmentation."""
        self.enabled = True

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
        severity_weights: Dict[float, float] = None,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the multi-task multi-view ResNet3D model.
        
        Args:
            num_severity: Number of severity classes
            num_action_type: Number of action type classes
            vocab_sizes: Dictionary mapping feature names to vocabulary sizes
            backbone_name: ResNet3D variant ('resnet3d_18', 'mc3_18', 'r2plus1d_18')
            config: Model configuration
            use_augmentation: Whether to apply adaptive augmentation for class imbalance
            severity_weights: Custom augmentation weights for severity classes
            dropout_rate: Dropout rate for classification heads (default: 0.1)
        """
        super().__init__()
        
        # Store configuration
        self.num_severity = num_severity
        self.num_action_type = num_action_type
        self.vocab_sizes = vocab_sizes
        self.backbone_name = backbone_name  # Store backbone name for get_model_info
        self.config = config or ModelConfig()
        self.use_augmentation = use_augmentation
        self.severity_weights = severity_weights
        self.dropout_rate = dropout_rate
        
        # Validate inputs
        self._validate_vocab_sizes(vocab_sizes)
        
        # Initialize all components
        self._initialize_components(backbone_name)
        
        logger.info(f"Initialized MultiTaskMultiViewResNet3D with {backbone_name}, "
                   f"{num_severity} severity classes and {num_action_type} action type classes. "
                   f"Augmentation: {'enabled' if use_augmentation else 'disabled'}, "
                   f"dropout_rate: {dropout_rate}")
    
    def _validate_vocab_sizes(self, vocab_sizes: Dict[str, int]) -> None:
        """Validate that all required vocabulary sizes are provided."""
        required_vocab_keys = [
            'contact', 'bodypart', 'upper_bodypart', 
            'multiple_fouls', 'try_to_play', 'touch_ball', 'handball', 'handball_offence'
        ]
        
        missing_keys = [key for key in required_vocab_keys if key not in vocab_sizes]
        if missing_keys:
            raise ValueError(f"Missing vocabulary sizes for: {missing_keys}")
        
        # Validate that all sizes are positive integers
        for key, size in vocab_sizes.items():
            if not isinstance(size, int) or size <= 0:
                raise ValueError(f"Vocabulary size for '{key}' must be a positive integer, got {size}")
    
    def _initialize_components(self, backbone_name: str) -> None:
        """Initialize all model components."""
        try:
            # Initialize video backbone
            self.backbone = ResNet3DBackbone(backbone_name)
            self.video_feature_dim = self.backbone.feature_dim
            
            # Initialize augmentation for addressing class imbalance
            if self.use_augmentation:
                # Use provided severity weights or defaults
                if self.severity_weights is None:
                    self.severity_weights = {
                        1.0: 1.0,   # Majority class - normal augmentation
                        2.0: 2.5,   # 2.5x more aggressive augmentation
                        3.0: 4.0,   # 4x more aggressive augmentation  
                        4.0: 6.0,   # 6x more aggressive augmentation
                        5.0: 8.0    # 8x more aggressive augmentation (if exists)
                    }
                
                self.video_augmentation = VideoAugmentation(
                    severity_weights=self.severity_weights,
                    training=True,
                    enabled=True
                )
            
            # Initialize embedding manager (reuse existing)
            self.embedding_manager = EmbeddingManager(self.config, self.vocab_sizes)
            
            # Initialize view aggregator (reuse existing) 
            self.view_aggregator = ViewAggregator(self.config, self.video_feature_dim)
            
            # Initialize input validator (reuse existing)
            self.input_validator = InputValidator(self.config, self.vocab_sizes)
            
            # Calculate combined feature dimension
            combined_feature_dim = self.video_feature_dim + self.config.get_total_embedding_dim()
            
            # Create classification heads with configurable dropout
            self.severity_head = nn.Sequential(
                nn.Linear(combined_feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),  # Use configurable dropout rate
                nn.Linear(256, self.num_severity)
            )
            
            self.action_type_head = nn.Sequential(
                nn.Linear(combined_feature_dim, 256), 
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),  # Use configurable dropout rate
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
    
    def _process_video_features(self, batch_data):
        """Process video clips through backbone."""
        clips = batch_data["clips"]
        
        # Handle different tensor dimensions
        if clips.ndim == 7:  # [B, clips_per_video, num_views, C, T, H, W]
            batch_size, clips_per_video, num_views, channels, frames, height, width = clips.shape
            # Reshape to [B * clips_per_video * num_views, C, T, H, W] for backbone processing
            clips_flat = clips.view(-1, channels, frames, height, width)
        elif clips.ndim == 6:  # [B, num_views, C, T, H, W]
            batch_size, num_views, channels, frames, height, width = clips.shape
            clips_per_video = 1
            # Reshape to [B * num_views, C, T, H, W] for backbone processing
            clips_flat = clips.view(-1, channels, frames, height, width)
        else:
            raise ValueError(f"Unexpected clips tensor shape: {clips.shape}")
        
        # Process through backbone
        features_flat = self.backbone(clips_flat)
        
        # Reshape back to batch format
        if clips.ndim == 7:
            # Reshape back to [B, clips_per_video, num_views, feature_dim]
            features = features_flat.view(batch_size, clips_per_video, num_views, -1)
            # Average across clips_per_video and num_views
            features = features.mean(dim=(1, 2))  # [B, feature_dim]
        else:
            # Reshape back to [B, num_views, feature_dim]
            features = features_flat.view(batch_size, num_views, -1)
            # Average across views
            features = features.mean(dim=1)  # [B, feature_dim]
        
        return features
    
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
        disable_in_model_augmentation: bool = False,
        severity_weights: Dict[float, float] = None,
        dropout_rate: float = 0.1,
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
            disable_in_model_augmentation: Whether to disable in-model augmentation
            severity_weights: Custom augmentation weights for severity classes
            dropout_rate: Dropout rate for classification heads (default: 0.1)
            **config_kwargs: Additional configuration parameters
            
        Returns:
            Configured MultiTaskMultiViewResNet3D instance
        """
        final_model_config: ModelConfig
        
        # Check if a ModelConfig instance was passed directly via config_kwargs
        passed_config_instance = None
        if 'config' in config_kwargs and isinstance(config_kwargs['config'], ModelConfig):
            passed_config_instance = config_kwargs.pop('config') # Use and remove it from kwargs

        if passed_config_instance is not None:
            final_model_config = passed_config_instance
            # If use_attention_aggregation in create_model's signature differs from the one in the passed config,
            # the one from the passed config object will implicitly be used.
            # We can add a log if they differ and other config_kwargs are present, as they'd be ignored.
            if final_model_config.use_attention_aggregation != use_attention_aggregation:
                logger.info(
                    f"Using 'use_attention_aggregation={final_model_config.use_attention_aggregation}' "
                    f"from provided ModelConfig object, not the default/passed "
                    f"'use_attention_aggregation={use_attention_aggregation}' in create_model signature."
                )
            if config_kwargs: # If any kwargs remain after popping 'config'
                logger.warning(
                    f"A ModelConfig object was passed directly. Other keyword arguments "
                    f"{list(config_kwargs.keys())} also present in create_model call are being ignored "
                    f"for ModelConfig creation, as the provided ModelConfig object is used directly."
                )
        else:
            # No ModelConfig instance was passed, so create one using
            # use_attention_aggregation from the signature and any other relevant config_kwargs.
            # At this point, config_kwargs should not contain 'config'.
            final_model_config = ModelConfig(
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
        
        # Override use_augmentation if disable_in_model_augmentation is True
        if disable_in_model_augmentation:
            use_augmentation = False
            logger.info("🚫 In-model augmentation disabled via disable_in_model_augmentation flag")
        
        # Create and return model
        model = cls(
            num_severity=num_severity,
            num_action_type=num_action_type,
            vocab_sizes=vocab_sizes,
            backbone_name=backbone_name,
            config=final_model_config, # Use the resolved config
            use_augmentation=use_augmentation,
            severity_weights=severity_weights,
            dropout_rate=dropout_rate
        )
        
        # If we have augmentation but it should be disabled, disable it
        if hasattr(model, 'video_augmentation') and disable_in_model_augmentation:
            model.video_augmentation.disable()
            logger.info("🚫 VideoAugmentation disabled for debugging")
        
        return model
    
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
        
        # Safely get severity weights
        severity_weights = None
        if hasattr(self, 'video_augmentation') and hasattr(self.video_augmentation, 'severity_weights'):
            severity_weights = self.video_augmentation.severity_weights
        
        return {
            'backbone_name': self.backbone_name,
            'video_feature_dim': self.video_feature_dim,
            'total_embedding_dim': self.config.get_total_embedding_dim(),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_severity_classes': self.num_severity,
            'num_action_type_classes': self.num_action_type,
            'augmentation_enabled': self.use_augmentation,
            'severity_weights': severity_weights
        } 
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
    
    def __init__(self, severity_weights: Dict[float, float] = None, training: bool = True, enabled: bool = True):
        super().__init__()
        self.training = training
        self.enabled = enabled  # Global enable/disable flag for debugging
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
        spatial_prob = min(0.9, self.spatial_prob * intensity)
        temporal_prob = min(0.8, self.temporal_prob * intensity)
        intensity_prob = min(0.7, self.intensity_prob * intensity)
        
        # Apply lighter augmentations to avoid dimension issues
        
        # Temporal augmentations (safer ones)
        if random.random() < temporal_prob * 0.5:  # Reduced probability
            clip = self._safe_temporal_augment(clip, intensity)
            
        # Spatial augmentations (safer ones)
        if random.random() < spatial_prob * 0.5:  # Reduced probability
            clip = self._safe_spatial_augment(clip, intensity)
            
        # Intensity/color augmentations (these are safe)
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
    
    def _safe_temporal_augment(self, clip: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply safe temporal augmentations that preserve dimensions."""
        C, T, H, W = clip.shape
        
        # Temporal reversal (safe)
        if random.random() < 0.3 * intensity:
            clip = torch.flip(clip, [1])  # Flip temporal dimension
        
        # Frame shuffling (safe - maintains frame count)
        if random.random() < 0.2 * intensity:
            # Shuffle middle frames only, keep first and last
            if T > 4:
                middle_indices = list(range(1, T-1))
                random.shuffle(middle_indices)
                new_indices = [0] + middle_indices + [T-1]
                clip = clip[:, new_indices]
        
        return clip
    
    def _safe_spatial_augment(self, clip: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply safe spatial augmentations that preserve dimensions."""
        C, T, H, W = clip.shape
        
        # Random horizontal flip (safe)
        if random.random() < 0.5:
            clip = torch.flip(clip, [3])  # Flip width dimension
        
        # Small random cropping (safe - maintains output size)
        if random.random() < 0.4 * intensity:
            # Crop a slightly larger region and resize back
            crop_ratio = random.uniform(0.9, 1.0)  # Very conservative cropping
            crop_h = max(int(H * crop_ratio), H - 4)  # At most 4 pixels off
            crop_w = max(int(W * crop_ratio), W - 4)
            
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
        
        return clip
    
    def _intensity_augment(self, clip: torch.Tensor, intensity: float) -> torch.Tensor:
        """Apply intensity and color augmentations (these are always safe)."""
        # Brightness adjustment
        if random.random() < 0.5:
            brightness_factor = random.uniform(0.8, 1.2)  # More conservative
            clip = torch.clamp(clip * brightness_factor, 0, 1)
        
        # Contrast adjustment  
        if random.random() < 0.5:
            contrast_factor = random.uniform(0.9, 1.1)  # More conservative
            mean = clip.mean(dim=[2, 3], keepdim=True)
            clip = torch.clamp((clip - mean) * contrast_factor + mean, 0, 1)
        
        # Gaussian noise (subtle)
        if random.random() < 0.3 * intensity:
            noise_std = 0.005 * intensity  # Reduced noise
            noise = torch.randn_like(clip) * noise_std
            clip = torch.clamp(clip + noise, 0, 1)
            
        return clip
    
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
                training=True,
                enabled=True  # Will be controlled by disable_in_model_augmentation
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
        disable_in_model_augmentation: bool = False,
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
            disable_in_model_augmentation: Whether to disable in-model augmentation
            severity_weights: Custom augmentation weights for severity classes
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
            severity_weights=severity_weights
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
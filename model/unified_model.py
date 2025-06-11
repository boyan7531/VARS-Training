import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import logging

from .resnet3d_model import MultiTaskMultiViewResNet3D, ResNet3DBackbone
from .config import ModelConfig
from .loader import ModelLoader, ModelLoadingError
from .embedding_manager import EmbeddingManager
from .view_aggregator import ViewAggregator
from .validator import InputValidator, ValidationError

logger = logging.getLogger(__name__)


class MultiTaskMultiViewMViT(nn.Module):
    """
    Multi-task, multi-view MViTv2 model for sports action classification.
    
    This model processes multiple video views and categorical features to predict:
    1. Severity of the action
    2. Type of the action
    """
    
    def __init__(
        self,
        num_severity: int,
        num_action_type: int,
        vocab_sizes: Dict[str, int],
        config: ModelConfig = None,
        use_augmentation: bool = True,
        severity_weights: Dict[float, float] = None
    ):
        super().__init__()
        
        # Use default config if none provided
        self.config = config if config is not None else ModelConfig()
        self.num_severity = num_severity
        self.num_action_type = num_action_type
        self.use_augmentation = use_augmentation
        
        # Validate vocabulary sizes
        self._validate_vocab_sizes(vocab_sizes)
        self.vocab_sizes = vocab_sizes
        
        # Initialize augmentation for addressing class imbalance
        if self.use_augmentation:
            from .resnet3d_model import VideoAugmentation
            self.video_augmentation = VideoAugmentation(
                severity_weights=severity_weights,
                training=True,
                enabled=True
            )
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"Initialized MultiTaskMultiViewMViT with {self.config.pretrained_model_name}, "
                   f"{num_severity} severity classes and {num_action_type} action type classes. "
                   f"Augmentation: {'enabled' if use_augmentation else 'disabled'}")
    
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
    
    def _initialize_components(self) -> None:
        """Initialize all model components."""
        try:
            # Load MViT backbone using the loader
            model_loader = ModelLoader(self.config)
            self.backbone, self.video_feature_dim = model_loader.load_base_model()
            
            # Initialize embedding manager
            self.embedding_manager = EmbeddingManager(self.config, self.vocab_sizes)
            
            # Initialize view aggregator
            self.view_aggregator = ViewAggregator(self.config, self.video_feature_dim)
            
            # Initialize input validator
            self.input_validator = InputValidator(self.config, self.vocab_sizes)
            
            # Calculate combined feature dimension
            combined_feature_dim = self.video_feature_dim + self.config.get_total_embedding_dim()
            
            # Create classification heads with strong regularization for small dataset
            self.severity_head = nn.Sequential(
                nn.Linear(combined_feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.6),
                nn.Linear(256, self.num_severity)
            )
            
            self.action_type_head = nn.Sequential(
                nn.Linear(combined_feature_dim, 256), 
                nn.ReLU(),
                nn.Dropout(0.6),
                nn.Linear(256, self.num_action_type)
            )
            
            logger.info(f"MViT components initialized successfully. "
                       f"Combined feature dim: {combined_feature_dim}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MViT components: {str(e)}") from e
    
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
            # Handle standard tensor input - reshape for MViT
            batch_size, max_views = clips.shape[:2]
            
            # Reshape and process through backbone
            clips_flat = clips.view(-1, *clips.shape[2:])  # [B*N, C, T, H, W]
            
            # MViT expects [B, C, T, H, W] format
            features_flat = self.backbone(clips_flat)
            
            # Handle MViT output format
            if isinstance(features_flat, torch.Tensor):
                if features_flat.ndim == 3:  # [B, seq_len, feature_dim]
                    # Take CLS token (first token) or mean pool
                    features_flat = features_flat[:, 0]  # CLS token
                elif features_flat.ndim == 2:  # [B, feature_dim]
                    pass  # Already in correct format
                else:
                    raise ValueError(f"Unexpected MViT output shape: {features_flat.shape}")
            
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
            'backbone_name': self.config.pretrained_model_name,
            'video_feature_dim': self.video_feature_dim,
            'total_embedding_dim': self.config.get_total_embedding_dim(),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_severity_classes': self.num_severity,
            'num_action_type_classes': self.num_action_type,
            'augmentation_enabled': self.use_augmentation,
            'severity_weights': severity_weights
        }


def create_unified_model(
    backbone_type: str,
    num_severity: int,
    num_action_type: int,
    vocab_sizes: Dict[str, int],
    backbone_name: str = None,
    use_attention_aggregation: bool = True,
    use_augmentation: bool = True,
    disable_in_model_augmentation: bool = False,
    severity_weights: Dict[float, float] = None,
    **config_kwargs
) -> nn.Module:
    """
    Factory function to create either ResNet3D or MViTv2 model.
    
    Args:
        backbone_type: Either 'resnet3d' or 'mvit'
        num_severity: Number of severity classes
        num_action_type: Number of action type classes  
        vocab_sizes: Dictionary mapping feature names to vocabulary sizes
        backbone_name: Specific model name (e.g., 'r2plus1d_18' for ResNet3D or 'mvit_base_16x4' for MViT)
        use_attention_aggregation: Whether to use attention-based view aggregation
        use_augmentation: Whether to apply adaptive augmentation for class imbalance
        disable_in_model_augmentation: Whether to disable in-model augmentation
        severity_weights: Custom augmentation weights for severity classes
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured model instance (either MultiTaskMultiViewResNet3D or MultiTaskMultiViewMViT)
    """
    
    if backbone_type.lower() == 'resnet3d':
        # Set default backbone_name for ResNet3D if not provided
        if backbone_name is None:
            backbone_name = 'r2plus1d_18'
        
        logger.info(f"Creating ResNet3D model with backbone: {backbone_name}")
        
        return MultiTaskMultiViewResNet3D.create_model(
            num_severity=num_severity,
            num_action_type=num_action_type,
            vocab_sizes=vocab_sizes,
            backbone_name=backbone_name,
            use_attention_aggregation=use_attention_aggregation,
            use_augmentation=use_augmentation,
            disable_in_model_augmentation=disable_in_model_augmentation,
            severity_weights=severity_weights,
            **config_kwargs
        )
    
    elif backbone_type.lower() == 'mvit':
        # Set default backbone_name for MViT if not provided
        if backbone_name is None:
            backbone_name = 'mvit_base_16x4'
        
        logger.info(f"Creating MViT model with backbone: {backbone_name}")
        
        # Create MViT-specific config
        mvit_config = ModelConfig(
            use_attention_aggregation=use_attention_aggregation,
            pretrained_model_name=backbone_name,
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
        
        # Create MViT model
        model = MultiTaskMultiViewMViT(
            num_severity=num_severity,
            num_action_type=num_action_type,
            vocab_sizes=vocab_sizes,
            config=mvit_config,
            use_augmentation=use_augmentation,
            severity_weights=severity_weights
        )
        
        # If we have augmentation but it should be disabled, disable it
        if hasattr(model, 'video_augmentation') and disable_in_model_augmentation:
            model.video_augmentation.disable()
            logger.info("🚫 VideoAugmentation disabled for debugging")
        
        return model
    
    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}. Choose 'resnet3d' or 'mvit'")

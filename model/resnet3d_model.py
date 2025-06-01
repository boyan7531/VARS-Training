import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Union, Tuple
import logging

from .config import ModelConfig
from .embedding_manager import EmbeddingManager
from .view_aggregator import ViewAggregator
from .validator import InputValidator, ValidationError

logger = logging.getLogger(__name__)

class ResNet3DBackbone(nn.Module):
    """3D ResNet backbone for video feature extraction."""
    
    def __init__(self, model_name='resnet3d_18', pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet3D
        if model_name == 'resnet3d_18':
            self.backbone = models.video.r3d_18(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet3d_50':
            self.backbone = models.video.r3d_50(pretrained=pretrained)  
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Remove final classification layer
        self.backbone.fc = nn.Identity()
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
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
        backbone_name: str = 'resnet3d_18',
        config: ModelConfig = None
    ):
        """
        Initialize the multi-task multi-view ResNet3D model.
        
        Args:
            num_severity: Number of classes for severity prediction
            num_action_type: Number of classes for action type prediction
            vocab_sizes: Dictionary mapping feature names to vocabulary sizes
            backbone_name: ResNet3D variant ('resnet3d_18' or 'resnet3d_50')
            config: Model configuration object
        """
        super().__init__()
        
        # Use default config if none provided
        self.config = config if config is not None else ModelConfig()
        self.num_severity = num_severity
        self.num_action_type = num_action_type
        self.backbone_name = backbone_name
        
        # Validate vocabulary sizes
        self._validate_vocab_sizes(vocab_sizes)
        self.vocab_sizes = vocab_sizes
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"Initialized MultiTaskMultiViewResNet3D with {backbone_name}, "
                   f"{num_severity} severity classes and {num_action_type} action type classes")
    
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
            
            # Create classification heads with regularization
            self.severity_head = nn.Sequential(
                nn.Linear(combined_feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, self.num_severity)
            )
            
            self.action_type_head = nn.Sequential(
                nn.Linear(combined_feature_dim, 256), 
                nn.ReLU(),
                nn.Dropout(0.5),
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
                - All categorical feature indices
        
        Returns:
            Tuple of (severity_logits, action_type_logits)
        """
        try:
            # Validate inputs
            batch_size = self.input_validator.validate_batch_data(batch_data)
            
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
        backbone_name: str = 'resnet3d_18',
        use_attention_aggregation: bool = True,
        **config_kwargs
    ) -> 'MultiTaskMultiViewResNet3D':
        """
        Factory method to create a ResNet3D model with custom configuration.
        
        Args:
            num_severity: Number of severity classes
            num_action_type: Number of action type classes
            vocab_sizes: Dictionary of vocabulary sizes
            backbone_name: ResNet3D variant ('resnet3d_18' or 'resnet3d_50')
            use_attention_aggregation: Whether to use attention for view aggregation
            **config_kwargs: Additional configuration parameters
        
        Returns:
            Configured MultiTaskMultiViewResNet3D model
        """
        # Create configuration
        config = ModelConfig(
            use_attention_aggregation=use_attention_aggregation,
            **config_kwargs
        )
        
        # Create and return model
        return cls(
            num_severity=num_severity,
            num_action_type=num_action_type,
            vocab_sizes=vocab_sizes,
            backbone_name=backbone_name,
            config=config
        )
    
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
            'num_action_type_classes': self.num_action_type
        } 
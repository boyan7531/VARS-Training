import torch
import torch.nn as nn
from typing import Dict, List, Union, Tuple
import logging

# Import our modular components
from .config import ModelConfig
from .loader import ModelLoader, ModelLoadingError
from .embedding_manager import EmbeddingManager
from .view_aggregator import ViewAggregator
from .validator import InputValidator, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiTaskMultiViewMViT(nn.Module):
    """
    Multi-task, multi-view MViT model for sports action classification.
    
    This model processes multiple video views and categorical features to predict:
    1. Severity of the action
    2. Type of the action
    """
    
    def __init__(
        self,
        num_severity: int,
        num_action_type: int,
        vocab_sizes: Dict[str, int],
        config: ModelConfig = None
    ):
        """
        Initialize the multi-task multi-view model.
        
        Args:
            num_severity: Number of classes for severity prediction
            num_action_type: Number of classes for action type prediction
            vocab_sizes: Dictionary mapping feature names to vocabulary sizes
            config: Model configuration object
        """
        super().__init__()
        
        # Use default config if none provided
        self.config = config if config is not None else ModelConfig()
        self.num_severity = num_severity
        self.num_action_type = num_action_type
        
        # Validate vocabulary sizes
        self._validate_vocab_sizes(vocab_sizes)
        self.vocab_sizes = vocab_sizes
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"Initialized MultiTaskMultiViewMViT with {num_severity} severity classes "
                   f"and {num_action_type} action type classes")
    
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
            # Load base model
            model_loader = ModelLoader(self.config)
            self.base_model, self.mvit_feature_dim = model_loader.load_base_model()
            
            # Initialize embedding manager
            self.embedding_manager = EmbeddingManager(self.config, self.vocab_sizes)
            
            # Initialize view aggregator
            self.view_aggregator = ViewAggregator(self.config, self.mvit_feature_dim)
            
            # Initialize input validator
            self.input_validator = InputValidator(self.config, self.vocab_sizes)
            
            # Calculate combined feature dimension
            combined_feature_dim = self.mvit_feature_dim + self.config.get_total_embedding_dim()
            
            # Create classification heads
            self.severity_head = nn.Linear(combined_feature_dim, self.num_severity)
            self.action_type_head = nn.Linear(combined_feature_dim, self.num_action_type)
            
            logger.info(f"Model components initialized successfully. "
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
            features, view_mask = self.view_aggregator.process_variable_views(clips, self.base_model)
        else:
            # Handle standard tensor input
            batch_size, max_views = clips.shape[:2]
            
            # Reshape and process through base model
            clips_flat = clips.view(-1, *clips.shape[2:])  # [B*N, C, T, H, W]
            features_flat = self.base_model(clips_flat)
            
            # Handle different output formats
            if features_flat.ndim == 3:
                features_flat = features_flat[:, 0]  # Use class token
            
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
        pretrained_model_name: str = 'mvit_base_16x4',
        use_attention_aggregation: bool = True,
        **config_kwargs
    ) -> 'MultiTaskMultiViewMViT':
        """
        Factory method to create a model with custom configuration.
        
        Args:
            num_severity: Number of severity classes
            num_action_type: Number of action type classes
            vocab_sizes: Dictionary of vocabulary sizes
            pretrained_model_name: Name of pretrained MViT model
            use_attention_aggregation: Whether to use attention for view aggregation
            **config_kwargs: Additional configuration parameters
        
        Returns:
            Initialized model instance
        """
        config = ModelConfig(
            pretrained_model_name=pretrained_model_name,
            use_attention_aggregation=use_attention_aggregation,
            **config_kwargs
        )
        
        return cls(
            num_severity=num_severity,
            num_action_type=num_action_type,
            vocab_sizes=vocab_sizes,
            config=config
        )
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the model configuration."""
        return {
            'num_severity_classes': self.num_severity,
            'num_action_type_classes': self.num_action_type,
            'vocab_sizes': self.vocab_sizes,
            'mvit_feature_dim': self.mvit_feature_dim,
            'total_embedding_dim': self.config.get_total_embedding_dim(),
            'pretrained_model_name': self.config.pretrained_model_name,
            'use_attention_aggregation': self.config.use_attention_aggregation,
            'input_dimensions': {
                'channels': self.config.input_channels,
                'frames': self.config.input_frames,
                'height': self.config.input_height,
                'width': self.config.input_width
            }
        }
    
    def enable_attention_debug(self, enable: bool = True):
        """Enable/disable attention weight debugging."""
        if hasattr(self.view_aggregator, 'attention_net'):
            self.view_aggregator.debug_attention = enable
            logger.info(f"Attention debugging {'enabled' if enable else 'disabled'}")
        else:
            logger.warning("Model is not using attention aggregation")

# Backward compatibility aliases
def create_multitask_multiview_mvit(*args, **kwargs):
    """Legacy function for backward compatibility."""
    logger.warning("create_multitask_multiview_mvit is deprecated. Use MultiTaskMultiViewMViT.create_model instead.")
    return MultiTaskMultiViewMViT.create_model(*args, **kwargs) 
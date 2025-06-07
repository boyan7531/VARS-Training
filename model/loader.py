import torch
import torch.nn as nn
import torch.hub
from torch.hub import load_state_dict_from_url
import logging
from typing import Tuple
from .config import ModelConfig
from .resnet3d_model import MultiTaskMultiViewResNet3D

logger = logging.getLogger(__name__)

class ModelLoadingError(Exception):
    """Custom exception for model loading errors."""
    pass

class ModelLoader:
    """Handles loading and initialization of pretrained MViT models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def load_base_model(self) -> Tuple[nn.Module, int]:
        """
        Load and initialize the base MViT model.
        
        Returns:
            Tuple of (base_model, feature_dimension)
            
        Raises:
            ModelLoadingError: If model loading fails
        """
        try:
            # Try to load with pretrained weights first
            base_model = self._load_pretrained_model()
            if base_model is None:
                # Fall back to non-pretrained if pretrained fails
                base_model = self._initialize_model_architecture()
                
            feature_dim = self._extract_feature_dimension(base_model)
            self._replace_head_with_identity(base_model)
            
            logger.info(f"Successfully loaded {self.config.pretrained_model_name} with feature_dim: {feature_dim}")
            return base_model, feature_dim
            
        except Exception as e:
            raise ModelLoadingError(f"Failed to load base model: {str(e)}") from e
    
    def _load_pretrained_model(self) -> nn.Module:
        """Load model with pretrained weights using PyTorch Hub."""
        logger.info(f"Loading pretrained model: {self.config.pretrained_model_name}")
        
        try:
            # Use PyTorch Hub's built-in pretrained loading
            model = torch.hub.load(
                'facebookresearch/pytorchvideo', 
                model=self.config.pretrained_model_name, 
                pretrained=True,
                verbose=False  # Reduce download verbosity
            )
            logger.info("Successfully loaded pretrained model")
            return model
            
        except Exception as e:
            logger.warning(f"Failed to load pretrained model: {e}")
            logger.info("Falling back to non-pretrained model")
            return None
    
    def _initialize_model_architecture(self) -> nn.Module:
        """Initialize the model architecture without pretrained weights."""
        logger.info(f"Initializing model architecture: {self.config.pretrained_model_name}")
        
        try:
            model = torch.hub.load(
                'facebookresearch/pytorchvideo', 
                model=self.config.pretrained_model_name, 
                pretrained=False, 
                head_num_classes=400,
                verbose=False
            )
            logger.info("Successfully initialized model architecture")
            return model
            
        except AssertionError as e:
            if "kwargs" in str(e):
                logger.warning("Retrying initialization without head_num_classes parameter")
                try:
                    model = torch.hub.load(
                        'facebookresearch/pytorchvideo', 
                        model=self.config.pretrained_model_name, 
                        pretrained=False,
                        verbose=False
                    )
                    logger.info("Successfully initialized model architecture (fallback)")
                    return model
                except Exception as e2:
                    raise ModelLoadingError(f"Fallback initialization failed: {e2}") from e2
            else:
                raise ModelLoadingError(f"Architecture initialization failed: {e}") from e
        except Exception as e:
            raise ModelLoadingError(f"Architecture initialization failed: {e}") from e
    
    def _extract_feature_dimension(self, model: nn.Module) -> int:
        """Extract feature dimension from the model head."""
        if not hasattr(model, 'head') or model.head is None:
            logger.warning(f"No head found, using default feature_dim: {self.config.default_mvit_feature_dim}")
            return self.config.default_mvit_feature_dim
        
        # Handle different head types
        if isinstance(model.head, nn.Linear):
            return model.head.in_features
        
        if hasattr(model.head, 'proj') and isinstance(model.head.proj, nn.Linear):
            return model.head.proj.in_features
        
        # Search for last linear layer in head
        for layer in reversed(list(model.head.modules())):
            if isinstance(layer, nn.Linear):
                return layer.in_features
        
        logger.warning(f"Could not determine feature_dim, using default: {self.config.default_mvit_feature_dim}")
        return self.config.default_mvit_feature_dim
    
    def _replace_head_with_identity(self, model: nn.Module) -> None:
        """Replace the model head with Identity layer."""
        model.head = nn.Identity()
        logger.info("Replaced model head with nn.Identity")

def create_model(args, vocab_sizes=None, device=None, num_gpus=None):
    """
    Create a MultiTaskMultiViewResNet3D model with Sports-1M pretrained weights.
    
    Args:
        args: Command-line arguments.
        vocab_sizes: Dictionary of vocabulary sizes for categorical features.
        device: Device to place the model on.
        num_gpus: Number of GPUs to use for DataParallel (if > 1).
        
    Returns:
        model: The initialized model
    """
    try:
        logger.info(f"Creating model with backbone {args.backbone_name}")
        logger.info(f"Using Sports-1M pretrained weights for improved transfer learning")
        
        # Create the base model
        model = MultiTaskMultiViewResNet3D.create_model(
            num_severity=6,  # Number of severity classes including None (0-5)
            num_action_type=10,  # Number of action type classes including None (0-9)
            vocab_sizes=vocab_sizes,
            backbone_name=args.backbone_name,
            use_attention_aggregation=args.attention_aggregation,
            use_augmentation=not args.disable_in_model_augmentation,
            max_views=args.max_views,
            dropout_rate=args.dropout_rate,
        )
        
        # Apply DataParallel if using multiple GPUs
        if num_gpus is not None and num_gpus > 1 and not args.force_batch_size:
            model = torch.nn.DataParallel(model)
            logger.info(f"Using DataParallel across {num_gpus} GPUs")
        
        # Move model to device
        if device is not None:
            model = model.to(device)
            logger.info(f"Model moved to {device}")
        
        # Log total number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.1%})")
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise 
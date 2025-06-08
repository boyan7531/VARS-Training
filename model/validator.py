import torch
from typing import Dict, List, Union
import logging
from .config import ModelConfig

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for input validation errors."""
    pass

class InputValidator:
    """Handles validation of input data for the model."""
    
    def __init__(self, config: ModelConfig, vocab_sizes: Dict[str, int]):
        self.config = config
        self.vocab_sizes = vocab_sizes
    
    def validate_batch_data(self, batch_data: Dict[str, torch.Tensor]) -> int:
        """
        Validate batch data and return batch size.
        
        Args:
            batch_data: Dictionary containing model inputs
            
        Returns:
            batch_size: Validated batch size
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Check required keys
            self._check_required_keys(batch_data)
            
            # Validate clips and get batch size
            batch_size = self._validate_clips(batch_data["clips"])
            
            # Validate feature tensors
            self._validate_feature_tensors(batch_data, batch_size)
            
            logger.debug(f"Input validation passed for batch_size: {batch_size}")
            return batch_size
            
        except Exception as e:
            raise ValidationError(f"Input validation failed: {str(e)}") from e
    
    def _check_required_keys(self, batch_data: Dict[str, torch.Tensor]) -> None:
        """Check that all required keys are present."""
        missing_keys = [key for key in self.config.required_feature_keys if key not in batch_data]
        if missing_keys:
            raise ValidationError(f"Missing required keys: {missing_keys}")
    
    def _validate_clips(self, clips: Union[torch.Tensor, List[torch.Tensor]]) -> int:
        """Validate video clips and return batch size."""
        if isinstance(clips, list):
            return self._validate_variable_clips(clips)
        else:
            return self._validate_tensor_clips(clips)
    
    def _validate_variable_clips(self, clips: List[torch.Tensor]) -> int:
        """Validate variable-length clips list."""
        if not clips:
            raise ValidationError("Empty clips list")
        
        batch_size = len(clips)
        
        for i, clip in enumerate(clips):
            if not isinstance(clip, torch.Tensor):
                raise ValidationError(f"Clip {i} is not a tensor: {type(clip)}")
            
            if clip.ndim != 5:  # [num_views, C, T, H, W]
                raise ValidationError(f"Clip {i} should have 5 dimensions, got {clip.ndim}")
            
            # Validate dimensions are reasonable (more flexible)
            num_views, channels, frames, height, width = clip.shape
            
            if channels != self.config.input_channels:
                raise ValidationError(f"Clip {i} has {channels} channels, expected {self.config.input_channels}")
            
            if frames != self.config.input_frames:
                raise ValidationError(f"Clip {i} has {frames} frames, expected {self.config.input_frames}")
            
            if height <= 0 or width <= 0:
                raise ValidationError(f"Clip {i} has invalid dimensions: height={height}, width={width}")
            
            if num_views <= 0:
                raise ValidationError(f"Clip {i} has {num_views} views, expected > 0")
        
        return batch_size
    
    def _validate_tensor_clips(self, clips: torch.Tensor) -> int:
        """Validate tensor clips (more flexible on spatial dimensions)."""
        if not isinstance(clips, torch.Tensor):
            raise ValidationError(f"Expected clips to be torch.Tensor, got {type(clips)}")
        
        if clips.ndim != 6:  # [B, num_views, C, T, H, W]
            raise ValidationError(f"Clips tensor should have 6 dimensions, got {clips.ndim}")
        
        batch_size, num_views, channels, frames, height, width = clips.shape
        
        # Validate specific dimensions that must match
        if channels != self.config.input_channels:
            raise ValidationError(f"Clips have {channels} channels, expected {self.config.input_channels}")
        
        if frames != self.config.input_frames:
            raise ValidationError(f"Clips have {frames} frames, expected {self.config.input_frames}")
        
        # Validate dimensions are reasonable (flexible on spatial size)
        if height <= 0 or width <= 0:
            raise ValidationError(f"Invalid spatial dimensions: height={height}, width={width}")
        
        if batch_size <= 0:
            raise ValidationError(f"Invalid batch size: {batch_size}")
        
        if num_views <= 0:
            raise ValidationError(f"Invalid number of views: {num_views}")
        
        return batch_size
    
    def _validate_feature_tensors(self, batch_data: Dict[str, torch.Tensor], batch_size: int) -> None:
        """Validate all feature tensors."""
        # Define feature vocabulary mappings
        feature_vocab_mapping = {
            # Original vocabulary features
            "contact_idx": self.vocab_sizes['contact'],
            "bodypart_idx": self.vocab_sizes['bodypart'],
            "upper_bodypart_idx": self.vocab_sizes['upper_bodypart'],
            # "lower_bodypart_idx": self.vocab_sizes['lower_bodypart'],
            "multiple_fouls_idx": self.vocab_sizes['multiple_fouls'],
            "try_to_play_idx": self.vocab_sizes['try_to_play'],
            "touch_ball_idx": self.vocab_sizes['touch_ball'],
            "handball_idx": self.vocab_sizes['handball'],
            "handball_offence_idx": self.vocab_sizes['handball_offence'],
            
            # Standard vocabulary features
            "offence_standard_idx": self.config.num_offence_classes,
            "contact_standard_idx": self.config.num_contact_standard_classes,
            "bodypart_standard_idx": self.config.num_bodypart_standard_classes,
            "upper_bodypart_standard_idx": self.config.num_upper_bodypart_standard_classes,
            # "lower_bodypart_standard_idx": self.config.num_lower_bodypart_standard_classes,
            "multiple_fouls_standard_idx": self.config.num_multiple_fouls_standard_classes,
            "try_to_play_standard_idx": self.config.num_try_to_play_standard_classes,
            "touch_ball_standard_idx": self.config.num_touch_ball_standard_classes,
            "handball_standard_idx": self.config.num_handball_standard_classes,
            "handball_offence_standard_idx": self.config.num_handball_offence_standard_classes
        }
        
        for feature_name, vocab_size in feature_vocab_mapping.items():
            if feature_name not in batch_data:
                continue  # Already checked in _check_required_keys
            
            feature_tensor = batch_data[feature_name]
            
            # Check tensor type
            if not isinstance(feature_tensor, torch.Tensor):
                raise ValidationError(f"Feature '{feature_name}' should be torch.Tensor, got {type(feature_tensor)}")
            
            # Check tensor shape
            if feature_tensor.shape != (batch_size,):
                raise ValidationError(
                    f"Feature '{feature_name}' should have shape ({batch_size},), got {feature_tensor.shape}"
                )
            
            # Check value range
            if torch.any(feature_tensor < 0) or torch.any(feature_tensor >= vocab_size):
                min_val = torch.min(feature_tensor).item()
                max_val = torch.max(feature_tensor).item()
                raise ValidationError(
                    f"Feature '{feature_name}' values should be in range [0, {vocab_size-1}], "
                    f"got range [{min_val}, {max_val}]"
                ) 
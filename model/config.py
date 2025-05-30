from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    """Configuration class for MultiTaskMultiViewMViT model."""
    
    # Video input dimensions
    input_channels: int = 3
    input_frames: int = 16
    input_height: int = 224
    input_width: int = 224
    
    # Embedding dimensions
    contact_embedding_dim: int = 16
    bodypart_embedding_dim: int = 32
    upper_bodypart_embedding_dim: int = 16
    lower_bodypart_embedding_dim: int = 16
    multiple_fouls_embedding_dim: int = 8
    try_to_play_embedding_dim: int = 8
    touch_ball_embedding_dim: int = 8
    handball_embedding_dim: int = 8
    handball_offence_embedding_dim: int = 8
    offence_embedding_dim: int = 8
    
    # Standard vocabulary sizes (fixed) - Updated to match actual dataset values
    num_offence_classes: int = 3  # "No offence": 0, "Offence": 1, "Between": 2
    num_contact_standard_classes: int = 2  # "Without contact": 0, "With contact": 1
    num_bodypart_standard_classes: int = 3  # "": 0, "Upper body": 1, "Under body": 2
    num_upper_bodypart_standard_classes: int = 3  # "": 0, "Use of shoulder": 1, "Use of arms": 2
    num_lower_bodypart_standard_classes: int = 4  # "": 0, "Use of leg": 1, "Use of knee": 2, "Use of foot": 3
    num_multiple_fouls_standard_classes: int = 2  # "": 0, "Yes": 1
    num_try_to_play_standard_classes: int = 2  # "": 0, "No": 0, "Yes": 1
    num_touch_ball_standard_classes: int = 3  # "": 0, "No": 0, "Yes": 1, "Maybe": 2
    num_handball_standard_classes: int = 2  # "No handball": 0, "Handball": 1
    num_handball_offence_standard_classes: int = 2  # "": 0, "No offence": 0, "Offence": 1
    
    # Model settings
    pretrained_model_name: str = 'mvit_base_16x4'
    use_attention_aggregation: bool = True
    attention_hidden_dim: int = 128
    default_mvit_feature_dim: int = 768  # Fallback if detection fails
    
    def get_total_embedding_dim(self) -> int:
        """Calculate total embedding dimension for all features."""
        return (
            self.contact_embedding_dim +
            self.bodypart_embedding_dim +
            self.upper_bodypart_embedding_dim +
            self.lower_bodypart_embedding_dim +
            self.multiple_fouls_embedding_dim +
            self.try_to_play_embedding_dim +
            self.touch_ball_embedding_dim +
            self.handball_embedding_dim +
            self.handball_offence_embedding_dim +
            self.offence_embedding_dim +
            self.contact_embedding_dim +  # Standard embeddings reuse same dims
            self.bodypart_embedding_dim +
            self.upper_bodypart_embedding_dim +
            self.lower_bodypart_embedding_dim +
            self.multiple_fouls_embedding_dim +
            self.try_to_play_embedding_dim +
            self.touch_ball_embedding_dim +
            self.handball_embedding_dim +
            self.handball_offence_embedding_dim
        )
    
    @property
    def required_feature_keys(self) -> list:
        """Return list of required feature keys in batch data."""
        return [
            "clips",
            # Original vocabulary indices
            "contact_idx", "bodypart_idx", "upper_bodypart_idx", "lower_bodypart_idx",
            "multiple_fouls_idx", "try_to_play_idx", "touch_ball_idx", 
            "handball_idx", "handball_offence_idx",
            # Standard indices
            "offence_standard_idx", "contact_standard_idx", "bodypart_standard_idx",
            "upper_bodypart_standard_idx", "lower_bodypart_standard_idx",
            "multiple_fouls_standard_idx", "try_to_play_standard_idx", 
            "touch_ball_standard_idx", "handball_standard_idx", "handball_offence_standard_idx"
        ] 
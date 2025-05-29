import torch
import torch.nn as nn
from typing import Dict, Tuple
from .config import ModelConfig

class EmbeddingManager(nn.Module):
    """Manages all categorical feature embeddings for the model."""
    
    def __init__(self, config: ModelConfig, vocab_sizes: Dict[str, int]):
        super().__init__()
        self.config = config
        self.vocab_sizes = vocab_sizes
        
        # Create embedding layers
        self.original_embeddings = self._create_original_embeddings()
        self.standard_embeddings = self._create_standard_embeddings()
        
    def _create_original_embeddings(self) -> nn.ModuleDict:
        """Create embeddings for original vocabulary indices."""
        embeddings = nn.ModuleDict({
            'contact': nn.Embedding(
                self.vocab_sizes['contact'], 
                self.config.contact_embedding_dim
            ),
            'bodypart': nn.Embedding(
                self.vocab_sizes['bodypart'], 
                self.config.bodypart_embedding_dim
            ),
            'upper_bodypart': nn.Embedding(
                self.vocab_sizes['upper_bodypart'], 
                self.config.upper_bodypart_embedding_dim
            ),
            'lower_bodypart': nn.Embedding(
                self.vocab_sizes['lower_bodypart'], 
                self.config.lower_bodypart_embedding_dim
            ),
            'multiple_fouls': nn.Embedding(
                self.vocab_sizes['multiple_fouls'], 
                self.config.multiple_fouls_embedding_dim
            ),
            'try_to_play': nn.Embedding(
                self.vocab_sizes['try_to_play'], 
                self.config.try_to_play_embedding_dim
            ),
            'touch_ball': nn.Embedding(
                self.vocab_sizes['touch_ball'], 
                self.config.touch_ball_embedding_dim
            ),
            'handball': nn.Embedding(
                self.vocab_sizes['handball'], 
                self.config.handball_embedding_dim
            ),
            'handball_offence': nn.Embedding(
                self.vocab_sizes['handball_offence'], 
                self.config.handball_offence_embedding_dim
            ),
        })
        return embeddings
    
    def _create_standard_embeddings(self) -> nn.ModuleDict:
        """Create embeddings for standardized indices."""
        embeddings = nn.ModuleDict({
            'offence': nn.Embedding(
                self.config.num_offence_classes, 
                self.config.offence_embedding_dim
            ),
            'contact': nn.Embedding(
                self.config.num_contact_standard_classes, 
                self.config.contact_embedding_dim
            ),
            'bodypart': nn.Embedding(
                self.config.num_bodypart_standard_classes, 
                self.config.bodypart_embedding_dim
            ),
            'upper_bodypart': nn.Embedding(
                self.config.num_upper_bodypart_standard_classes, 
                self.config.upper_bodypart_embedding_dim
            ),
            'lower_bodypart': nn.Embedding(
                self.config.num_lower_bodypart_standard_classes, 
                self.config.lower_bodypart_embedding_dim
            ),
            'multiple_fouls': nn.Embedding(
                self.config.num_multiple_fouls_standard_classes, 
                self.config.multiple_fouls_embedding_dim
            ),
            'try_to_play': nn.Embedding(
                self.config.num_try_to_play_standard_classes, 
                self.config.try_to_play_embedding_dim
            ),
            'touch_ball': nn.Embedding(
                self.config.num_touch_ball_standard_classes, 
                self.config.touch_ball_embedding_dim
            ),
            'handball': nn.Embedding(
                self.config.num_handball_standard_classes, 
                self.config.handball_embedding_dim
            ),
            'handball_offence': nn.Embedding(
                self.config.num_handball_offence_standard_classes, 
                self.config.handball_offence_embedding_dim
            ),
        })
        return embeddings
    
    def embed_features(self, batch_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embed all categorical features from batch data.
        
        Returns:
            Tuple of (original_embeddings, standard_embeddings) concatenated tensors
        """
        # Original embeddings
        orig_embeds = []
        feature_names = ['contact', 'bodypart', 'upper_bodypart', 'lower_bodypart',
                        'multiple_fouls', 'try_to_play', 'touch_ball', 'handball', 'handball_offence']
        
        for feature in feature_names:
            idx_key = f"{feature}_idx"
            if idx_key not in batch_data:
                raise KeyError(f"Missing feature key: {idx_key}")
            embed = self.original_embeddings[feature](batch_data[idx_key])
            orig_embeds.append(embed)
        
        # Standard embeddings
        std_embeds = []
        std_feature_names = ['offence', 'contact', 'bodypart', 'upper_bodypart', 'lower_bodypart',
                            'multiple_fouls', 'try_to_play', 'touch_ball', 'handball', 'handball_offence']
        
        for feature in std_feature_names:
            idx_key = f"{feature}_standard_idx"
            if idx_key not in batch_data:
                raise KeyError(f"Missing standard feature key: {idx_key}")
            embed = self.standard_embeddings[feature](batch_data[idx_key])
            std_embeds.append(embed)
        
        # Concatenate embeddings
        original_embedded = torch.cat(orig_embeds, dim=1)
        standard_embedded = torch.cat(std_embeds, dim=1)
        
        return original_embedded, standard_embedded 
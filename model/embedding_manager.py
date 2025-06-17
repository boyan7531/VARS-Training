import torch
import torch.nn as nn
from typing import Dict, Tuple
from .config import ModelConfig

class EmbeddingManager(nn.Module):
    """Simplified embedding manager - now returns zero embeddings since we only use video features."""
    
    def __init__(self, config: ModelConfig, vocab_sizes: Dict[str, int] = None):
        super().__init__()
        self.config = config
        # No longer need to store vocab_sizes or create embeddings
        
    def embed_features(self, batch_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return zero tensors since we no longer use categorical features.
        
        Returns:
            Tuple of (original_embedded, standard_embedded) - both zero tensors
        """
        # Get batch size from video clips
        clips = batch_data.get("clips")
        if isinstance(clips, list):
            batch_size = len(clips)
        else:
            batch_size = clips.shape[0]
        
        device = clips.device if hasattr(clips, 'device') else torch.device('cpu')
        
        # Return two zero tensors as expected by the calling code
        zero_tensor = torch.zeros(batch_size, 0, device=device)
        return zero_tensor, zero_tensor  # (original_embedded, standard_embedded) 
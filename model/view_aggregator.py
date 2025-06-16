import torch
import torch.nn as nn
from typing import Union, List, Optional, Tuple
from .config import ModelConfig

class TransformerViewAggregator(nn.Module):
    """Transformer-based view aggregator using self-attention."""
    
    def __init__(self, feature_dim: int, max_views: int, agg_heads: int = 2, agg_layers: int = 1):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_views = max_views
        self.agg_heads = agg_heads
        self.agg_layers = agg_layers
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        
        # Positional embeddings for views (max_views + 1 for CLS token)
        self.pos_embeddings = nn.Parameter(torch.randn(1, max_views + 1, feature_dim) * 0.02)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=agg_heads,
            dim_feedforward=4 * feature_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=agg_layers)
        
        # Layer norm for output
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, features: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, max_views, feature_dim]
            view_mask: [B, max_views] boolean mask for valid views
            
        Returns:
            aggregated_features: [B, feature_dim]
        """
        batch_size, max_views, feature_dim = features.shape
        device = features.device
        
        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, feature_dim]
        
        # Concatenate CLS token with features
        sequence = torch.cat([cls_tokens, features], dim=1)  # [B, max_views+1, feature_dim]
        
        # Add positional embeddings
        sequence = sequence + self.pos_embeddings[:, :max_views+1, :]
        
        # Create attention mask: CLS token is always valid, then view_mask
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        full_mask = torch.cat([cls_mask, view_mask], dim=1)  # [B, max_views+1]
        
        # Create key padding mask (True for positions to ignore)
        key_padding_mask = ~full_mask  # [B, max_views+1]
        
        # Apply transformer encoder
        encoded = self.transformer_encoder(sequence, src_key_padding_mask=key_padding_mask)
        
        # Extract CLS token output
        cls_output = encoded[:, 0]  # [B, feature_dim]
        
        # Apply layer norm
        aggregated_features = self.layer_norm(cls_output)
        
        return aggregated_features


class MoEViewAggregator(nn.Module):
    """Simple Mixture of Experts view aggregator for future experiments."""
    
    def __init__(self, feature_dim: int, num_experts: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, features: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, max_views, feature_dim]
            view_mask: [B, max_views] boolean mask for valid views
            
        Returns:
            aggregated_features: [B, feature_dim]
        """
        batch_size, max_views, feature_dim = features.shape
        
        # Apply mask and compute mean over valid views
        expanded_mask = view_mask.unsqueeze(-1).expand(-1, -1, feature_dim).float()
        masked_features = features * expanded_mask
        valid_view_counts = view_mask.sum(dim=1, keepdim=True).expand(-1, feature_dim).float()
        valid_view_counts = torch.clamp(valid_view_counts, min=1.0)
        
        # Mean pooled features
        mean_features = torch.sum(masked_features, dim=1) / valid_view_counts  # [B, feature_dim]
        
        # Compute expert weights
        expert_weights = self.gate(mean_features)  # [B, num_experts]
        
        # Apply experts
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(mean_features)
            expert_outputs.append(expert_output)
        
        # Stack and weighted sum
        expert_stack = torch.stack(expert_outputs, dim=2)  # [B, feature_dim, num_experts]
        expert_weights = expert_weights.unsqueeze(1)  # [B, 1, num_experts]
        
        # Weighted combination
        aggregated_features = torch.sum(expert_stack * expert_weights, dim=2)  # [B, feature_dim]
        
        return aggregated_features


class ViewAggregator(nn.Module):
    """Handles aggregation of multiple views using attention or mean pooling."""
    
    def __init__(self, config: ModelConfig, feature_dim: int):
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim
        self.aggregator_type = config.aggregator_type
        
        if self.aggregator_type == 'transformer':
            self.aggregator = TransformerViewAggregator(
                feature_dim=feature_dim,
                max_views=config.max_views,
                agg_heads=config.agg_heads,
                agg_layers=config.agg_layers
            )
        elif self.aggregator_type == 'moe':
            self.aggregator = MoEViewAggregator(
                feature_dim=feature_dim,
                num_experts=2  # Simple 2-expert setup
            )
        else:  # 'mlp' or fallback
            self.use_attention = config.use_attention_aggregation
            if self.use_attention:
                self.attention_net = nn.Sequential(
                    nn.Linear(feature_dim, config.attention_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.attention_hidden_dim, 1)
                )
    
    def aggregate_views(
        self, 
        features: torch.Tensor, 
        view_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate features from multiple views.
        
        Args:
            features: Features tensor of shape [B, max_views, feature_dim]
            view_mask: Boolean mask of shape [B, max_views] indicating valid views
            
        Returns:
            Aggregated features of shape [B, feature_dim]
        """
        batch_size, max_views, feature_dim = features.shape
        
        # Create default mask if not provided
        if view_mask is None:
            view_mask = torch.ones(batch_size, max_views, dtype=torch.bool, device=features.device)
        
        if self.aggregator_type == 'transformer':
            return self.aggregator(features, view_mask)
        elif self.aggregator_type == 'moe':
            return self.aggregator(features, view_mask)
        else:  # 'mlp' or fallback
            if self.use_attention:
                return self._attention_aggregate(features, view_mask)
            else:
                return self._mean_aggregate(features, view_mask)
    
    def _attention_aggregate(self, features: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        """Aggregate using attention mechanism."""
        batch_size, max_views, feature_dim = features.shape
        
        # Calculate attention scores
        features_flat = features.reshape(batch_size * max_views, -1)
        attention_scores = self.attention_net(features_flat)
        attention_scores = attention_scores.view(batch_size, max_views, 1)
        
        # Apply mask to set scores of invalid views to -inf
        mask_expanded = view_mask.unsqueeze(-1)
        masked_scores = attention_scores.masked_fill(~mask_expanded, -1e4)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(masked_scores, dim=1)
        
        # Apply attention weights and sum
        weighted_features = features * attention_weights
        aggregated_features = torch.sum(weighted_features, dim=1)
        
        return aggregated_features
    
    def _mean_aggregate(self, features: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        """Aggregate using mean pooling over valid views."""
        batch_size, max_views, feature_dim = features.shape
        
        # Apply mask and compute mean
        expanded_mask = view_mask.unsqueeze(-1).expand(-1, -1, feature_dim).float()
        masked_features = features * expanded_mask
        valid_view_counts = view_mask.sum(dim=1, keepdim=True).expand(-1, feature_dim).float()
        
        # Avoid division by zero
        valid_view_counts = torch.clamp(valid_view_counts, min=1.0)
        aggregated_features = torch.sum(masked_features, dim=1) / valid_view_counts
        
        return aggregated_features
    
    def process_variable_views(
        self,
        view_list: List[torch.Tensor],
        base_model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process list of tensors with different view counts.
        
        Args:
            view_list: List of tensors, each of shape [num_views_i, C, T, H, W]
            base_model: The base model to process individual views
            
        Returns:
            Tuple of (padded_features, view_mask)
        """
        batch_size = len(view_list)
        processed_views = []
        
        # Process each sample's views
        for sample_clips in view_list:
            num_views_i = sample_clips.shape[0]
            
            # Process through base model
            sample_output = base_model(sample_clips)
            
            # Extract features (handle different output formats)
            if sample_output.ndim == 3:
                sample_features = sample_output[:, 0]  # Class token
            elif sample_output.ndim == 2:
                sample_features = sample_output
            else:
                raise ValueError(f"Unexpected output dimensions: {sample_output.ndim}")
            
            processed_views.append(sample_features)
        
        # Get max views for padding
        max_views = max(features.shape[0] for features in processed_views)
        
        # Create view mask
        view_mask = torch.zeros(batch_size, max_views, dtype=torch.bool, device=view_list[0].device)
        
        # Pad sequences
        padded_features = []
        for i, sample_features in enumerate(processed_views):
            num_views_i = sample_features.shape[0]
            feature_dim = sample_features.shape[1]
            
            # Update mask
            view_mask[i, :num_views_i] = True
            
            # Pad if necessary
            if num_views_i < max_views:
                padding = torch.zeros(
                    max_views - num_views_i, 
                    feature_dim,
                    dtype=sample_features.dtype,
                    device=sample_features.device
                )
                padded_sample = torch.cat([sample_features, padding], dim=0)
            else:
                padded_sample = sample_features
            
            padded_features.append(padded_sample)
        
        # Stack all padded features
        features_tensor = torch.stack(padded_features, dim=0)
        
        return features_tensor, view_mask 
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import logging
import os
import torch.utils.checkpoint as checkpoint

from .resnet3d_model import MultiTaskMultiViewResNet3D, ResNet3DBackbone
from .config import ModelConfig
from .loader import ModelLoader, ModelLoadingError
from .embedding_manager import EmbeddingManager
from .view_aggregator import ViewAggregator
from .validator import InputValidator, ValidationError

logger = logging.getLogger(__name__)


def _configure_attention_kernels_for_checkpointing(use_gradient_checkpointing: bool) -> None:
    """
    Configure PyTorch attention kernels to avoid metadata mismatch errors with gradient checkpointing.
    
    The issue: FlashAttention can produce tensors with different shapes/strides between the forward
    pass and recompute pass during gradient checkpointing, causing CheckpointError.
    
    The fix: When gradient checkpointing is enabled, we disable FlashAttention and use the more
    stable 'efficient' or 'math' kernels that don't have this issue.
    
    Args:
        use_gradient_checkpointing: Whether gradient checkpointing is enabled
    """
    if not use_gradient_checkpointing:
        return
    
    # Check for user override
    user_kernel = os.environ.get('PYTORCH_CUDA_SDP_KERNEL', '').lower()
    if user_kernel:
        logger.info(f"🔧 User override detected: PYTORCH_CUDA_SDP_KERNEL={user_kernel}")
        return
    
    # Check if we have the required backends
    if not hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        logger.warning("⚠️  torch.backends.cuda.enable_flash_sdp not available, skipping kernel configuration")
        return
    
    try:
        # Disable FlashAttention to prevent metadata mismatch during checkpointing
        original_flash_enabled = torch.backends.cuda.flash_sdp_enabled()
        
        if original_flash_enabled:
            torch.backends.cuda.enable_flash_sdp(False)
            logger.info("🚀 Gradient Checkpointing Optimization:")
            logger.info("   ✅ Gradient checkpointing enabled")
            logger.info("   🔧 FlashAttention disabled (prevents metadata mismatch)")
            logger.info("   📊 Using efficient/math attention kernels for stability")
            logger.info("   💡 Tip: ~5-10% slower but enables large batch training")
            logger.info("   🔄 Override with PYTORCH_CUDA_SDP_KERNEL=flash if needed")
        else:
            logger.info("🚀 Gradient Checkpointing Optimization:")
            logger.info("   ✅ Gradient checkpointing enabled")
            logger.info("   ✅ FlashAttention already disabled")
            
    except Exception as e:
        logger.warning(f"⚠️  Failed to configure attention kernels: {e}")
        logger.warning("   Gradient checkpointing may encounter metadata mismatch errors")


class OptimizedMViTProcessor(nn.Module):
    """
    Optimized processor for MViT with memory-efficient view handling.
    Processes views sequentially to avoid memory fragmentation and improve GPU utilization.
    """
    
    def __init__(self, backbone, use_gradient_checkpointing=False):
        super().__init__()
        self.backbone = backbone
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
    def forward(self, clips, view_mask=None):
        """
        Process clips with optimized memory usage and sequential view processing.
        
        Args:
            clips: Either [B, N, C, T, H, W] tensor or list of variable-length views
            view_mask: Optional mask for valid views
            
        Returns:
            features: [B, feature_dim] aggregated features
            actual_view_mask: Updated view mask
        """
        if isinstance(clips, list):
            return self._process_variable_views(clips)
        else:
            return self._process_tensor_views(clips, view_mask)
    
    def _process_tensor_views(self, clips, view_mask):
        """Process standard tensor input with sequential view processing."""
        batch_size, max_views = clips.shape[:2]
        device = clips.device
        
        logger.debug(f"Processing tensor views: batch_size={batch_size}, max_views={max_views}, clips.shape={clips.shape}")
        
        # Pre-allocate output tensors for better memory management
        # Get feature dimension from a single forward pass
        with torch.no_grad():
            sample_clip = clips[0, 0].unsqueeze(0)  # [1, C, T, H, W]
            logger.debug(f"Sample clip shape: {sample_clip.shape}")
            sample_features = self._forward_single_view(sample_clip)
            logger.debug(f"Sample features shape: {sample_features.shape}")
            if isinstance(sample_features, torch.Tensor):
                if sample_features.ndim == 3:  # [1, seq_len, feature_dim]
                    feature_dim = sample_features.shape[-1]
                    logger.debug(f"3D output detected, feature_dim={feature_dim}")
                elif sample_features.ndim == 2:  # [1, feature_dim]
                    feature_dim = sample_features.shape[-1]
                    logger.debug(f"2D output detected, feature_dim={feature_dim}")
                else:
                    raise ValueError(f"Unexpected backbone output shape: {sample_features.shape}")
            else:
                raise ValueError("Backbone output must be a tensor")
        
        # Pre-allocate feature tensor
        all_features = torch.zeros(batch_size, max_views, feature_dim, 
                                 device=device, dtype=clips.dtype)
        logger.debug(f"Pre-allocated all_features shape: {all_features.shape}")
        
        # Create view mask if not provided
        if view_mask is None:
            view_mask = torch.ones(batch_size, max_views, dtype=torch.bool, device=device)
        
        # Process views sequentially to avoid memory fragmentation
        for view_idx in range(max_views):
            # Get current view for all batches
            current_view = clips[:, view_idx]  # [B, C, T, H, W]
            
            # Only process if any batch has a valid view at this index
            if view_mask[:, view_idx].any():
                # Extract features for this view across all batches
                if self.use_gradient_checkpointing and self.training:
                    # PyTorch's activation checkpointing requires at least one input
                    # that has `requires_grad=True`. The video tensor coming from the
                    # dataloader does **not** need gradients, therefore we attach a
                    # single dummy tensor that does. This unlocks true activation
                    # recomputation without keeping the (huge) intermediate video
                    # activations in memory.

                    dummy = torch.ones(1, device=current_view.device, requires_grad=True)
                    view_features = checkpoint.checkpoint(
                        lambda _dummy, x: self._forward_single_view(x),
                        dummy,
                        current_view,
                        use_reentrant=False,
                    )
                else:
                    view_features = self._forward_single_view(current_view)
                
                logger.debug(f"View {view_idx} features shape before processing: {view_features.shape}")
                
                # Handle different output formats efficiently
                if view_features.ndim == 3:  # [B, seq_len, feature_dim]
                    # Extract CLS token (first token) efficiently
                    view_features = view_features[:, 0]  # [B, feature_dim]
                    logger.debug(f"View {view_idx} features shape after CLS extraction: {view_features.shape}")
                
                # Store features for valid views only
                valid_mask = view_mask[:, view_idx]
                logger.debug(f"Valid mask for view {view_idx}: {valid_mask.sum().item()}/{len(valid_mask)} samples")
                all_features[valid_mask, view_idx] = view_features[valid_mask]
            
            # Clear intermediate tensors to free memory
            if view_idx % 2 == 0:  # Clean up every 2 views
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.debug(f"Final all_features shape: {all_features.shape}")
        return all_features, view_mask
    
    def _process_variable_views(self, clips_list):
        """Process variable-length views efficiently."""
        batch_size = len(clips_list)
        device = clips_list[0].device if clips_list else torch.device('cpu')
        
        # Determine max views and feature dimension
        max_views = max(len(clips) for clips in clips_list) if clips_list else 0
        
        if max_views == 0:
            return torch.zeros(batch_size, 0, device=device), torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        
        # Get feature dimension from first clip
        with torch.no_grad():
            sample_features = self._forward_single_view(clips_list[0][:1])
            if sample_features.ndim == 3:
                feature_dim = sample_features.shape[-1]
            else:
                feature_dim = sample_features.shape[-1]
        
        # Pre-allocate tensors
        all_features = torch.zeros(batch_size, max_views, feature_dim, 
                                 device=device, dtype=clips_list[0].dtype)
        view_mask = torch.zeros(batch_size, max_views, dtype=torch.bool, device=device)
        
        # Process each batch's views
        for batch_idx, clips in enumerate(clips_list):
            num_views = len(clips)
            if num_views > 0:
                # Process all views for this batch at once for efficiency
                batch_clips = torch.stack(clips)  # [N, C, T, H, W]
                
                if self.use_gradient_checkpointing and self.training:
                    dummy = torch.ones(1, device=batch_clips.device, requires_grad=True)
                    batch_features = checkpoint.checkpoint(
                        lambda _dummy, x: self._forward_single_view(x),
                        dummy,
                        batch_clips,
                        use_reentrant=False,
                    )
                else:
                    batch_features = self._forward_single_view(batch_clips)
                
                if batch_features.ndim == 3:  # [N, seq_len, feature_dim]
                    batch_features = batch_features[:, 0]  # Extract CLS tokens
                
                all_features[batch_idx, :num_views] = batch_features
                view_mask[batch_idx, :num_views] = True
        
        return all_features, view_mask
    
    def _forward_single_view(self, clips):
        """Forward pass through backbone with optimized memory usage."""
        # Ensure proper input format for MViT
        return self.backbone(clips)


class MultiTaskMultiViewMViT(nn.Module):
    """
    Multi-task, multi-view MViTv2 model for sports action classification.
    
    This model processes multiple video views and categorical features to predict:
    1. Severity of the action
    2. Type of the action
    
    Optimized for improved GPU utilization and memory efficiency.
    """
    
    def __init__(
        self,
        num_severity: int,
        num_action_type: int,
        vocab_sizes: Dict[str, int],
        config: ModelConfig = None,
        use_augmentation: bool = True,
        severity_weights: Dict[float, float] = None,
        use_gradient_checkpointing: bool = False,
        enable_memory_optimization: bool = True,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the optimized MViT model with video-only capabilities.
        
        Args:
            num_severity: Number of severity classes
            num_action_type: Number of action type classes
            vocab_sizes: Dictionary mapping feature names to vocabulary sizes (now optional)
            config: Model configuration
            use_augmentation: Whether to apply adaptive augmentation for class imbalance
            severity_weights: Custom augmentation weights for severity classes
            use_gradient_checkpointing: Whether to enable gradient checkpointing for memory efficiency
            enable_memory_optimization: Whether to enable memory optimization features
            dropout_rate: Dropout rate for classification heads (default: 0.1)
        """
        super().__init__()
        
        # Store configuration
        self.num_severity = num_severity
        self.num_action_type = num_action_type
        self.vocab_sizes = vocab_sizes or {}
        self.config = config or ModelConfig()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.enable_memory_optimization = enable_memory_optimization
        self.use_augmentation = use_augmentation
        self.severity_weights = severity_weights
        self.dropout_rate = dropout_rate
        
        # Validate inputs
        self._validate_vocab_sizes(vocab_sizes)
        
        # Initialize all components
        self._initialize_components()
        
        logger.info(f"Optimized MViT model initialized with dropout_rate={dropout_rate}")
    
    def _validate_vocab_sizes(self, vocab_sizes: Dict[str, int]) -> None:
        """Validate vocabulary sizes - now optional since we only use video features."""
        # No validation needed since we don't use categorical features anymore
        pass
    
    def _initialize_components(self) -> None:
        """Initialize all model components."""
        try:
            # Configure attention kernels for gradient checkpointing compatibility
            _configure_attention_kernels_for_checkpointing(self.use_gradient_checkpointing)
            
            # Load MViT backbone using the loader
            model_loader = ModelLoader(self.config)
            backbone, self.video_feature_dim = model_loader.load_base_model()
            
            # Wrap backbone with optimized processor
            self.mvit_processor = OptimizedMViTProcessor(
                backbone, 
                use_gradient_checkpointing=self.use_gradient_checkpointing
            )
            
            # Initialize augmentation for addressing class imbalance
            if self.use_augmentation:
                from .resnet3d_model import VideoAugmentation
                # Use default severity weights if none provided
                severity_weights = getattr(self, 'severity_weights', None)
                if severity_weights is None:
                    severity_weights = {
                        1.0: 1.0,   # Majority class - normal augmentation
                        2.0: 2.5,   # 2.5x more aggressive augmentation
                        3.0: 4.0,   # 4x more aggressive augmentation  
                        4.0: 6.0,   # 6x more aggressive augmentation
                        5.0: 8.0    # 8x more aggressive augmentation (if exists)
                    }
                
                self.video_augmentation = VideoAugmentation(
                    severity_weights=severity_weights,
                    training=True,
                    enabled=True
                )
            
            # Initialize embedding manager (simplified for video-only)
            self.embedding_manager = EmbeddingManager(self.config, self.vocab_sizes)
            
            # Initialize view aggregator with memory optimization
            self.view_aggregator = ViewAggregator(self.config, self.video_feature_dim)
            
            # Initialize input validator (simplified for video-only)
            self.input_validator = InputValidator(self.config, self.vocab_sizes)
            
            # Calculate combined feature dimension (now just video features)
            combined_feature_dim = self.video_feature_dim  # No categorical features anymore
            
            # Create classification heads with strong regularization for small dataset
            # Use efficient activation and initialization
            self.severity_head = self._create_optimized_head(combined_feature_dim, self.num_severity)
            self.action_type_head = self._create_optimized_head(combined_feature_dim, self.num_action_type)
            
            logger.info(f"Optimized MViT components initialized successfully. "
                       f"Combined feature dim: {combined_feature_dim}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize optimized MViT components: {str(e)}") from e
    
    def _create_optimized_head(self, input_dim: int, num_classes: int) -> nn.Module:
        """Create optimized classification head with LayerNorm for transformer feature stabilization."""
        head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),  # More efficient than ReLU for transformers
            nn.Dropout(self.dropout_rate),  # Use configurable dropout rate
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights properly for better training stability
        for module in head.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.constant_(module.weight, 1.0)
                torch.nn.init.constant_(module.bias, 0)
        
        return head
    
    def forward(self, batch_data: Dict[str, torch.Tensor], return_view_logits: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized forward pass through the model - now video-only.
        
        Args:
            batch_data: Dictionary containing:
                - clips: Video tensor(s) [B, N, C, T, H, W] or List[[N_i, C, T, H, W]]
                - view_mask: Optional boolean mask [B, N] for valid views
                - severity: Optional severity labels [B] for adaptive augmentation
            return_view_logits: Whether to return per-view logits for view consistency loss
        
        Returns:
            Tuple of (severity_logits, action_type_logits) or 
            Dict with aggregated and per-view logits if return_view_logits=True
        """
        try:
            # Get clips and handle multi-clip format
            clips = batch_data["clips"]
            if isinstance(clips, list):
                batch_size = len(clips)
                clips_per_video = 1  # Variable list format doesn't support multi-clip yet
            else:
                # Check if we have multi-clip format: [B, C, V, 3, T, H, W]
                if clips.dim() == 7:
                    batch_size, clips_per_video = clips.shape[:2]
                    # Flatten clips and views: [B*C, V, 3, T, H, W]
                    clips = clips.view(batch_size * clips_per_video, *clips.shape[2:])
                else:
                    # Standard format: [B, V, 3, T, H, W]
                    batch_size = clips.shape[0]
                    clips_per_video = 1
            
            # Apply adaptive augmentation based on severity (more for minority classes)
            if self.use_augmentation and self.training:
                severity_labels = batch_data.get("severity", None)
                clips = self.video_augmentation(clips, severity_labels)
                # Update batch_data with augmented clips
                batch_data = {**batch_data, "clips": clips}
            
            # Process video features with optimized pipeline
            if return_view_logits:
                video_features, view_mask, per_view_features = self._process_video_features_optimized(batch_data, preserve_per_view=True)
                logger.debug(f"Video features shape: {video_features.shape}")
                logger.debug(f"Per-view features shape: {per_view_features.shape}")
            else:
                video_features = self._process_video_features_optimized(batch_data)
                logger.debug(f"Video features shape: {video_features.shape}")
            
            # Use video features directly (no categorical features anymore)
            combined_features = video_features
            logger.debug(f"Final features shape: {combined_features.shape}")
            
            # Generate predictions with potential checkpointing for large models
            if self.use_gradient_checkpointing and self.training:
                severity_logits = checkpoint.checkpoint(
                    self.severity_head, combined_features, use_reentrant=False
                )
                action_type_logits = checkpoint.checkpoint(
                    self.action_type_head, combined_features, use_reentrant=False
                )
            else:
                severity_logits = self.severity_head(combined_features)
                action_type_logits = self.action_type_head(combined_features)
            
            # Handle multi-clip averaging if we flattened clips earlier
            if clips_per_video > 1:
                # Reshape back to [B, C, num_classes] and average over clips
                severity_logits = severity_logits.view(batch_size, clips_per_video, -1).mean(dim=1)
                action_type_logits = action_type_logits.view(batch_size, clips_per_video, -1).mean(dim=1)
            
            # Generate per-view logits if requested
            if return_view_logits:
                sev_logits_v, act_logits_v = self._logits_per_view(per_view_features)
                logger.debug(f"Per-view severity logits shape: {sev_logits_v.shape}")
                logger.debug(f"Per-view action logits shape: {act_logits_v.shape}")
                
                # Clean up intermediate tensors if memory optimization is enabled
                if self.enable_memory_optimization:
                    del video_features, combined_features, per_view_features
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                return {
                    'severity_logits': severity_logits,
                    'action_logits': action_type_logits,
                    'sev_logits_v': sev_logits_v,
                    'act_logits_v': act_logits_v,
                    'view_mask': view_mask
                }
            else:
                # Clean up intermediate tensors if memory optimization is enabled
                if self.enable_memory_optimization:
                    del video_features, combined_features
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                return severity_logits, action_type_logits
            
        except RuntimeError as e:
            # Re-raise known errors
            raise e
        except Exception as e:
            # Wrap unexpected errors with more context
            logger.error(f"Forward pass failed with error: {str(e)}")
            if 'video_features' in locals():
                logger.error(f"Video features shape: {video_features.shape if 'video_features' in locals() else 'Not computed'}")
            raise RuntimeError(f"Video-only forward pass failed: {str(e)}") from e
    
    def _process_video_features_optimized(self, batch_data: Dict[str, torch.Tensor], preserve_per_view: bool = False):
        """Process video clips with optimized memory usage and sequential processing."""
        clips = batch_data["clips"]
        view_mask = batch_data.get("view_mask", None)
        
        logger.debug(f"Input clips shape: {clips.shape if hasattr(clips, 'shape') else f'List of {len(clips)} items'}")
        
        # Use optimized processor for efficient view handling
        features, updated_view_mask = self.mvit_processor(clips, view_mask)
        logger.debug(f"MViT processor output features shape: {features.shape}")
        logger.debug(f"Updated view mask shape: {updated_view_mask.shape}")
        
        # Aggregate views efficiently
        aggregated_features = self.view_aggregator.aggregate_views(features, updated_view_mask)
        logger.debug(f"Aggregated features shape: {aggregated_features.shape}")
        
        if preserve_per_view:
            return aggregated_features, updated_view_mask, features
        else:
            return aggregated_features
    
    def _logits_per_view(self, features):
        """
        Apply classification heads to per-view features.
        
        Args:
            features: [B, V, d] per-view features
            
        Returns:
            sev_logits_v: [B, V, 6] severity logits per view
            act_logits_v: [B, V, 10] action logits per view
        """
        batch_size, num_views, feature_dim = features.shape
        
        # Reshape to [B*V, d] for batch processing
        features_flat = features.view(-1, feature_dim)
        
        # Apply heads
        if self.use_gradient_checkpointing and self.training:
            sev_logits_flat = checkpoint.checkpoint(
                self.severity_head, features_flat, use_reentrant=False
            )
            act_logits_flat = checkpoint.checkpoint(
                self.action_type_head, features_flat, use_reentrant=False
            )
        else:
            sev_logits_flat = self.severity_head(features_flat)
            act_logits_flat = self.action_type_head(features_flat)
        
        # Reshape back to [B, V, num_classes]
        sev_logits_v = sev_logits_flat.view(batch_size, num_views, -1)
        act_logits_v = act_logits_flat.view(batch_size, num_views, -1)
        
        return sev_logits_v, act_logits_v
    
    def _process_categorical_features(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process categorical features through embeddings - now returns empty tensor."""
        # Return empty categorical features since we only use video features
        return self.embedding_manager.embed_features(batch_data)
    
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
            'gradient_checkpointing_enabled': self.use_gradient_checkpointing,
            'memory_optimization_enabled': self.enable_memory_optimization,
            'severity_weights': severity_weights
        }


def create_unified_model(
    backbone_type: str,
    num_severity: int,
    num_action_type: int,
    vocab_sizes: Dict[str, int] = None,
    backbone_name: str = None,
    use_attention_aggregation: bool = True,
    use_augmentation: bool = True,
    disable_in_model_augmentation: bool = False,
    severity_weights: Dict[float, float] = None,
    enable_gradient_checkpointing: bool = False,
    enable_memory_optimization: bool = True,
    dropout_rate: float = 0.1,
    # New aggregator configuration
    aggregator_type: str = 'transformer',
    max_views: int = 8,
    agg_heads: int = 2,
    agg_layers: int = 1,
    **config_kwargs
) -> nn.Module:
    """
    Factory function to create either ResNet3D or MViTv2 model - now video-only.
    
    Args:
        backbone_type: Either 'resnet3d' or 'mvit'
        num_severity: Number of severity classes
        num_action_type: Number of action type classes  
        vocab_sizes: Optional dictionary (not used anymore since we only use video features)
        backbone_name: Specific model name (e.g., 'r2plus1d_18' for ResNet3D or 'mvit_base_16x4' for MViT)
        use_attention_aggregation: Whether to use attention-based view aggregation
        use_augmentation: Whether to apply adaptive augmentation for class imbalance
        disable_in_model_augmentation: Whether to disable in-model augmentation
        severity_weights: Custom augmentation weights for severity classes
        enable_gradient_checkpointing: Whether to enable gradient checkpointing for memory efficiency
        enable_memory_optimization: Whether to enable memory optimization features
        dropout_rate: Dropout rate for classification heads (default: 0.1)
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
            vocab_sizes=vocab_sizes or {},
            backbone_name=backbone_name,
            use_attention_aggregation=use_attention_aggregation,
            use_augmentation=use_augmentation,
            disable_in_model_augmentation=disable_in_model_augmentation,
            severity_weights=severity_weights,
            dropout_rate=dropout_rate,
            **config_kwargs
        )
    
    elif backbone_type.lower() == 'mvit':
        # Set default backbone_name for MViT if not provided
        if backbone_name is None:
            backbone_name = 'mvit_base_16x4'
        
        logger.info(f"Creating optimized MViT model with backbone: {backbone_name}")
        
        # Create MViT-specific config
        mvit_config = ModelConfig(
            use_attention_aggregation=use_attention_aggregation,
            pretrained_model_name=backbone_name,
            aggregator_type=aggregator_type,
            max_views=max_views,
            agg_heads=agg_heads,
            agg_layers=agg_layers,
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
        
        # Create optimized MViT model
        model = MultiTaskMultiViewMViT(
            num_severity=num_severity,
            num_action_type=num_action_type,
            vocab_sizes=vocab_sizes or {},
            config=mvit_config,
            use_augmentation=use_augmentation,
            severity_weights=severity_weights,
            use_gradient_checkpointing=enable_gradient_checkpointing,
            enable_memory_optimization=enable_memory_optimization,
            dropout_rate=dropout_rate
        )
        
        # If we have augmentation but it should be disabled, disable it
        if hasattr(model, 'video_augmentation') and disable_in_model_augmentation:
            model.video_augmentation.disable()
            logger.info("🚫 VideoAugmentation disabled for debugging")
        
        logger.info(f"✅ Optimized MViT model created with gradient checkpointing: {enable_gradient_checkpointing}, "
                   f"memory optimization: {enable_memory_optimization}, dropout_rate: {dropout_rate}")
        
        return model
    
    else:
        raise ValueError(f"Unsupported backbone_type: {backbone_type}. Choose 'resnet3d' or 'mvit'")

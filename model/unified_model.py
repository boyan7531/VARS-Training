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
    
    def __init__(self, backbone, use_gradient_checkpointing=False, use_mixed_precision=False, expected_max_views=None):
        super().__init__()
        self.backbone = backbone
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_mixed_precision = use_mixed_precision
        self.expected_max_views = expected_max_views
        self._debug_silent_errors = False  # Enable for debugging silent backbone errors
        
    def enable_debug_mode(self, enabled=True):
        """Enable debugging for silent backbone errors."""
        self._debug_silent_errors = enabled
        logger.info(f"Debug mode for silent backbone errors: {'enabled' if enabled else 'disabled'}")
    
    def stabilize_after_unfreezing(self):
        """Stabilize model after gradual unfreezing by ensuring proper parameter states."""
        try:
            # Ensure all parameters are properly initialized and contiguous
            for name, param in self.backbone.named_parameters():
                if param.requires_grad and not param.is_contiguous():
                    param.data = param.data.contiguous()
            
            # Clear any cached computations
            if hasattr(self.backbone, 'clear_cache'):
                self.backbone.clear_cache()
            
            logger.debug("Model stabilized after unfreezing")
        except Exception as e:
            logger.warning(f"Failed to stabilize model after unfreezing: {e}")
        
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
        # Handle different clip formats
        if clips.dim() == 7:  # [B, clips_per_video, actual_views, C, T, H, W]
            batch_size, clips_per_video, actual_views = clips.shape[:3]
            
            # Handle view dimension mismatch - pad or truncate to expected max_views
            # Note: max_views is determined by the model configuration, not the data
            expected_max_views = getattr(self, 'expected_max_views', actual_views)
            if hasattr(self, 'expected_max_views') and expected_max_views != actual_views:
                logger.debug(f"Adjusting clips from {actual_views} to {expected_max_views} views")
                if actual_views < expected_max_views:
                    # Pad with zeros
                    pad_size = expected_max_views - actual_views
                    padding = torch.zeros(batch_size, clips_per_video, pad_size, *clips.shape[3:], 
                                        device=clips.device, dtype=clips.dtype)
                    clips = torch.cat([clips, padding], dim=2)
                elif actual_views > expected_max_views:
                    # Truncate
                    clips = clips[:, :, :expected_max_views]
                max_views = expected_max_views
            else:
                max_views = actual_views
                
            # Flatten clips and views: [B*clips_per_video, max_views, C, T, H, W]
            clips = clips.view(batch_size * clips_per_video, max_views, *clips.shape[3:])
            
            # Handle view_mask flattening if provided
            view_mask_processed = False
            if view_mask is not None:
                if view_mask.dim() == 3:  # [B, clips_per_video, actual_views]
                    # Handle dimension mismatch before flattening
                    actual_views_mask = view_mask.shape[2]
                    if actual_views_mask != max_views:
                        logger.debug(f"Adjusting view_mask from {actual_views_mask} to {max_views} views")
                        if actual_views_mask < max_views:
                            # Pad with False (invalid views)
                            padding = torch.zeros(batch_size, clips_per_video, max_views - actual_views_mask, 
                                                dtype=torch.bool, device=view_mask.device)
                            view_mask = torch.cat([view_mask, padding], dim=2)
                        elif actual_views_mask > max_views:
                            # Truncate
                            view_mask = view_mask[:, :, :max_views]
                    # Now flatten: [B, clips_per_video, max_views] -> [B*clips_per_video, max_views]
                    view_mask = view_mask.view(batch_size * clips_per_video, max_views)
                    view_mask_processed = True
                # If view_mask is 2D, assume it applies to all clips
            
            effective_batch_size = batch_size * clips_per_video
        elif clips.dim() == 6:  # [B, max_views, C, T, H, W]
            batch_size, max_views = clips.shape[:2]
            clips_per_video = 1
            effective_batch_size = batch_size
            view_mask_processed = False
            
            # Handle view_mask dimensions
            if view_mask is not None and view_mask.dim() == 3:
                # If we have 3D view_mask but 2D clips, squeeze the clips dimension
                if view_mask.shape[1] == 1:  # [B, 1, max_views]
                    view_mask = view_mask.squeeze(1)  # [B, max_views]
                    view_mask_processed = True
        elif clips.dim() == 5:  # [B, C, T, H, W] - single view case
            batch_size = clips.shape[0]
            max_views = 1
            clips_per_video = 1
            effective_batch_size = batch_size
            view_mask_processed = False
            
            # Create view_mask for single view if not provided
            if view_mask is None:
                view_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=clips.device)
                view_mask_processed = True
            elif view_mask.dim() == 2 and view_mask.shape[1] > 1:
                # If we have multi-view mask but single view clips, take first column
                view_mask = view_mask[:, :1]
                view_mask_processed = True
        else:
            raise ValueError(f"Unexpected clips tensor dimensions: {clips.dim()}. Expected 5D [B,C,T,H,W], 6D [B,V,C,T,H,W], or 7D [B,clips_per_video,V,C,T,H,W]")
        
        device = clips.device
        
        logger.debug(f"Processing tensor views: effective_batch_size={effective_batch_size}, max_views={max_views}, clips.shape={clips.shape}")
        
        # Pre-allocate output tensors for better memory management
        # Get feature dimension from a single forward pass
        with torch.no_grad():
            # Handle different tensor dimensions correctly
            if clips.dim() == 6:  # [B, max_views, C, T, H, W]
                sample_clip = clips[0, 0].unsqueeze(0)  # [1, C, T, H, W]
            elif clips.dim() == 5:  # [B, C, T, H, W] - single view case
                sample_clip = clips[0].unsqueeze(0)  # [1, C, T, H, W]
            else:
                raise ValueError(f"Unexpected clips tensor dimensions: {clips.dim()}")
                
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
        all_features = torch.zeros(effective_batch_size, max_views, feature_dim, 
                                 device=device, dtype=clips.dtype)
        logger.debug(f"Pre-allocated all_features shape: {all_features.shape}")
        
        # Create view mask if not provided
        if view_mask is None:
            view_mask = torch.ones(effective_batch_size, max_views, dtype=torch.bool, device=device)
        elif not view_mask_processed:
            # Ensure view_mask matches effective_batch_size after clip flattening (only if not already processed)
            if view_mask.shape[0] != effective_batch_size or (view_mask.dim() > 1 and view_mask.shape[-1] != max_views):
                logger.debug(f"Reshaping view_mask from {view_mask.shape} to match effective_batch_size={effective_batch_size}, max_views={max_views}")
                logger.debug(f"Condition check: clips_per_video={clips_per_video}, view_mask.dim()={view_mask.dim()}, view_mask.shape={view_mask.shape}")
                if clips_per_video > 1 and view_mask.dim() == 3:
                    # Handle 3D view_mask: [B, C, V_actual] -> [B*C, max_views]
                    actual_views = view_mask.shape[2]
                    if view_mask.shape[:2] == (batch_size, clips_per_video):
                        # Reshape to [B*C, V_actual]
                        view_mask_flat = view_mask.view(effective_batch_size, actual_views)
                        # Pad or truncate to max_views
                        if actual_views < max_views:
                            # Pad with False (invalid views)
                            padding = torch.zeros(effective_batch_size, max_views - actual_views, dtype=torch.bool, device=device)
                            view_mask = torch.cat([view_mask_flat, padding], dim=1)
                        elif actual_views > max_views:
                            # Truncate to max_views
                            view_mask = view_mask_flat[:, :max_views]
                        else:
                            view_mask = view_mask_flat
                    else:
                        logger.warning(f"3D view_mask shape {view_mask.shape} doesn't match expected [{batch_size}, {clips_per_video}, *]. Using default mask.")
                        view_mask = torch.ones(effective_batch_size, max_views, dtype=torch.bool, device=device)
                elif clips_per_video > 1 and view_mask.shape[0] == batch_size:
                    # Handle 2D view_mask: [B, V_actual] -> [B*C, max_views] by repeating
                    actual_views = view_mask.shape[1] if view_mask.dim() > 1 else 1
                    if actual_views < max_views:
                        # Pad the 2D mask first
                        padding = torch.zeros(batch_size, max_views - actual_views, dtype=torch.bool, device=device)
                        view_mask_padded = torch.cat([view_mask, padding], dim=1)
                    elif actual_views > max_views:
                        # Truncate the 2D mask first
                        view_mask_padded = view_mask[:, :max_views]
                    else:
                        view_mask_padded = view_mask
                    # Now repeat for clips
                    view_mask = view_mask_padded.unsqueeze(1).repeat(1, clips_per_video, 1).view(effective_batch_size, max_views)
                elif view_mask.shape[0] != effective_batch_size:
                    # Create default mask if dimensions don't match
                    logger.warning(f"view_mask shape mismatch: got {view_mask.shape}, expected [{effective_batch_size}, {max_views}]. Using default mask.")
                    view_mask = torch.ones(effective_batch_size, max_views, dtype=torch.bool, device=device)
        
        # Process views sequentially to avoid memory fragmentation
        for view_idx in range(max_views):
            # Get current view for all batches - handle different tensor dimensions
            if clips.dim() == 6:  # [B, max_views, C, T, H, W]
                current_view = clips[:, view_idx]  # [effective_batch_size, C, T, H, W]
            elif clips.dim() == 5:  # [B, C, T, H, W] - single view case
                if view_idx == 0:
                    current_view = clips  # [effective_batch_size, C, T, H, W]
                else:
                    # For single view case, only process view_idx 0
                    continue
            else:
                raise ValueError(f"Unexpected clips tensor dimensions: {clips.dim()}")
            
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
                if view_features.ndim == 3:  # [effective_batch_size, seq_len, feature_dim]
                    # Extract CLS token (first token) efficiently
                    view_features = view_features[:, 0]  # [effective_batch_size, feature_dim]
                    logger.debug(f"View {view_idx} features shape after CLS extraction: {view_features.shape}")
                
                # Store features for valid views only
                valid_mask = view_mask[:, view_idx]
                logger.debug(f"Valid mask for view {view_idx}: {valid_mask.sum().item()}/{len(valid_mask)} samples")
                all_features[valid_mask, view_idx] = view_features[valid_mask]
            
            # Clear intermediate tensors to free memory
            if view_idx % 2 == 0:  # Clean up every 2 views
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.debug(f"Final all_features shape: {all_features.shape}")
        
        # If we had multi-clip format, we might need to reshape back
        if clips_per_video > 1:
            # Reshape features back to [B, clips_per_video, max_views, feature_dim]
            all_features = all_features.view(batch_size, clips_per_video, max_views, feature_dim)
            # Average over clips dimension
            all_features = all_features.mean(dim=1)  # [B, max_views, feature_dim]
            
            # Also reshape view_mask back if needed
            if view_mask.dim() == 2:
                view_mask = view_mask.view(batch_size, clips_per_video, max_views)
                # Use logical OR to combine clip masks (a view is valid if valid in any clip)
                view_mask = view_mask.any(dim=1)  # [B, max_views]
        
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
        """Forward pass through backbone with optimized memory usage and NaN protection."""
        # Add input validation
        if torch.isnan(clips).any():
            nan_count = torch.isnan(clips).sum().item()
            logger.debug(f"NaN detected in input clips! Shape: {clips.shape}, Count: {nan_count}")
            # Replace NaN with zeros as a fallback
            clips = torch.where(torch.isnan(clips), torch.zeros_like(clips), clips)
            logger.debug("Replaced NaN input values with zeros")
        
        # Check for padded views (zeros or very small values)
        is_padded = (clips.abs() < 1e-4).all(dim=(1,2,3,4)) if clips.dim() == 5 else (clips.abs() < 1e-4).all()
        if is_padded.any() if torch.is_tensor(is_padded) else is_padded:
            logger.debug(f"Detected padded views in input clips")
            # For padded views, return zero features to be masked out later
            if torch.is_tensor(is_padded):
                # Multiple clips, some are padded
                num_clips = clips.shape[0]
                # Process non-padded clips normally
                non_padded_mask = ~is_padded
                if non_padded_mask.any():
                    non_padded_clips = clips[non_padded_mask]
                    # Process the non-padded clips
                    non_padded_clips = torch.clamp(non_padded_clips, min=-10.0, max=10.0)
                    # Continue with normal processing for non-padded clips
                    clips = clips.clone()
                    clips[non_padded_mask] = non_padded_clips
                    # Set padded clips to valid range but will be masked later
                    clips[is_padded] = torch.zeros_like(clips[is_padded])
            else:
                # Single clip is padded, return zero features
                feature_dim = 768  # Default MViT feature dimension
                return torch.zeros(1, feature_dim, device=clips.device, dtype=clips.dtype)
        else:
            # Clamp input values to prevent extreme values
            clips = torch.clamp(clips, min=-10.0, max=10.0)
        
        # Ensure proper input format for MViT (expects BCTHW)
        if clips.dim() == 5:  # [N, C, T, H, W] - multiple clips
            clips_to_process = clips
        elif clips.dim() == 4:  # [C, T, H, W] - single clip
            clips_to_process = clips.unsqueeze(0)  # Add batch dimension
        else:
            raise ValueError(f"Unexpected input dimensions: {clips.shape}")
        
        try:
            # Ensure tensor is contiguous and on the correct device
            if not clips_to_process.is_contiguous():
                clips_to_process = clips_to_process.contiguous()
                if self._debug_silent_errors:
                    logger.debug("Made input tensor contiguous before backbone forward")
            
            # Validate tensor properties before backbone forward
            if self._debug_silent_errors:
                logger.debug(f"Backbone input: shape={clips_to_process.shape}, "
                           f"device={clips_to_process.device}, dtype={clips_to_process.dtype}, "
                           f"contiguous={clips_to_process.is_contiguous()}")
            
            # Forward pass through MViT backbone
            with torch.amp.autocast('cuda', enabled=self.use_mixed_precision):
                if self.use_gradient_checkpointing and self.training:
                    # Use gradient checkpointing to save memory
                    dummy = torch.ones(1, device=clips_to_process.device, requires_grad=True)
                    features = checkpoint.checkpoint(
                        lambda _dummy, x: self.backbone(x),
                        dummy,
                        clips_to_process,
                        use_reentrant=False,
                    )
                else:
                    features = self.backbone(clips_to_process)
            
            # Handle different output formats
            if isinstance(features, dict):
                # Some models return dict with 'features' key
                features = features.get('features', features.get('last_hidden_state', list(features.values())[0]))
            
            if features.dim() == 3:  # [B, seq_len, feature_dim] - transformer output
                features = features[:, 0]  # Extract CLS token features
            elif features.dim() == 4:  # [B, feature_dim, T, 1] - some conv models
                features = features.mean(dim=(2, 3))  # Global average pooling
            elif features.dim() == 2:  # [B, feature_dim] - already pooled
                pass  # Use as-is
            else:
                logger.warning(f"Unexpected feature dimensions: {features.shape}, applying adaptive pooling")
                # Adaptive handling for unexpected dimensions
                while features.dim() > 2:
                    features = features.mean(dim=-1)
            
            # Ensure output is 2D: [B, feature_dim]
            if features.dim() != 2:
                raise ValueError(f"Failed to process features to 2D tensor, got shape: {features.shape}")
            
            # Final NaN check
            if torch.isnan(features).any():
                nan_count = torch.isnan(features).sum().item()
                logger.debug(f"NaN detected in backbone output! Shape: {features.shape}, Count: {nan_count}")
                # Replace NaN with zeros
                features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
                logger.debug("Replaced NaN backbone features with zeros")
            
            return features
            
        except Exception as e:
            # Get comprehensive error information
            error_msg = str(e) if str(e) else ""
            error_type = type(e).__name__
            error_repr = repr(e)
            
            # Try to get more context about the error
            import traceback
            error_traceback = traceback.format_exc()
            
            # Categorize errors for better logging
            is_expected_error = any(pattern in error_msg.lower() for pattern in [
                'expected input',     # Channel dimension mismatches
                'size mismatch',      # Tensor size issues
                'out of bounds',      # Index errors
                'invalid argument',   # Invalid tensor operations
                'dimension',          # Dimension-related errors
                'shape',              # Shape-related errors
                'cuda error',         # CUDA memory issues
                'out of memory'       # Memory issues
            ])
            
            # Handle different types of errors appropriately
            if error_msg and error_msg.strip():
                # Non-empty error message
                if is_expected_error:
                    logger.debug(f"Handled expected backbone error: {error_msg}")
                else:
                    logger.warning(f"Unexpected backbone error: {error_msg} (type: {error_type})")
                    logger.debug(f"Full traceback: {error_traceback}")
            elif error_type in ['RuntimeError', 'ValueError', 'TypeError', '_StopRecomputationError']:
                # Known error types with empty messages - likely CUDA/tensor issues
                # _StopRecomputationError is normal PyTorch gradient checkpointing control flow
                if error_type == '_StopRecomputationError':
                    logger.debug(f"Handled {error_type} in backbone (normal gradient checkpointing control flow)")
                else:
                    logger.debug(f"Handled silent {error_type} in backbone (likely tensor/CUDA issue)")
                # Add more debugging for silent errors during epoch 3+
                if hasattr(self, '_debug_silent_errors'):
                    logger.debug(f"Silent {error_type} details: repr={error_repr}")
                    logger.debug(f"Input tensor shape: {clips_to_process.shape if clips_to_process is not None else 'None'}")
                    logger.debug(f"Input tensor device: {clips_to_process.device if clips_to_process is not None else 'None'}")
                
                # For RuntimeError during training (common during gradual unfreezing), try stabilization
                if error_type == 'RuntimeError' and self.training:
                    logger.debug("Attempting model stabilization for RuntimeError during training")
                    try:
                        # Ensure model is in consistent state
                        self.stabilize_after_unfreezing()
                        
                        # Try forward pass once more with stabilized model
                        with torch.amp.autocast('cuda', enabled=self.use_mixed_precision):
                            if self.use_gradient_checkpointing and self.training:
                                features = torch.utils.checkpoint.checkpoint(self.backbone, clips_to_process, use_reentrant=False)
                            else:
                                features = self.backbone(clips_to_process)
                        
                        # If successful, log recovery
                        logger.debug("Successfully recovered from RuntimeError with model stabilization")
                        
                        # Check for NaN in recovered features
                        if torch.isnan(features).any():
                            nan_count = torch.isnan(features).sum().item()
                            logger.debug(f"NaN detected in recovered backbone output! Shape: {features.shape}, Count: {nan_count}")
                            features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
                            logger.debug("Replaced NaN in recovered features with zeros")
                        
                        return features
                        
                    except Exception as recovery_e:
                        logger.debug(f"Model stabilization recovery failed: {recovery_e}")
                        # Fall through to return zero features
            else:
                # Unknown error type - provide better debugging information
                debug_info = []
                if error_repr and error_repr.strip() and error_repr != error_type + "()":
                    debug_info.append(f"repr: {error_repr}")
                if clips_to_process is not None:
                    debug_info.append(f"input_shape: {clips_to_process.shape}")
                    debug_info.append(f"input_device: {clips_to_process.device}")
                    debug_info.append(f"input_dtype: {clips_to_process.dtype}")
                
                debug_str = " | ".join(debug_info) if debug_info else "no additional info"
                logger.warning(f"Unknown backbone error type '{error_type}' with empty message ({debug_str})")
                logger.debug(f"Full traceback: {error_traceback}")
                
            # Return zero features as fallback
            feature_dim = 768  # Default feature dimension
            batch_size = clips_to_process.shape[0] if clips_to_process.dim() > 4 else 1
            return torch.zeros(batch_size, feature_dim, device=clips_to_process.device, dtype=clips_to_process.dtype)


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
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                use_mixed_precision=getattr(self.config, 'use_mixed_precision', False),
                expected_max_views=getattr(self.config, 'max_views', None)
            )
            
            # Initialize augmentation for addressing class imbalance
            if self.use_augmentation:
                from .resnet3d_model import VideoAugmentation
                # Use default severity weights if none provided
                severity_weights = getattr(self, 'severity_weights', None)
                if severity_weights is None:
                    severity_weights = {
                        1.0: 1.0,   # Majority class - normal augmentation
                        2.0: 1.3,   # Reduced from 2.5x to 1.3x more aggressive augmentation
                        3.0: 1.6,   # Reduced from 4.0x to 1.6x more aggressive augmentation  
                        4.0: 2.0,   # Reduced from 6.0x to 2.0x more aggressive augmentation
                        5.0: 2.5    # Reduced from 8.0x to 2.5x more aggressive augmentation
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
            # Input normalization to stabilize transformer features
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),  # More efficient than ReLU for transformers
            nn.Dropout(self.dropout_rate),  # Use configurable dropout rate
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(self.dropout_rate * 0.5),  # Reduced dropout for second layer
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights properly for better training stability with smaller scale
        for module in head.modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization scale to prevent exploding gradients
                torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
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
            
            # Add NaN protection before classification heads
            if torch.isnan(combined_features).any():
                logger.error(f"NaN detected in combined features before classification!")
                logger.error(f"Combined features NaN count: {torch.isnan(combined_features).sum().item()}")
                # Replace NaN with small random values
                nan_mask = torch.isnan(combined_features)
                replacement_values = torch.randn_like(combined_features) * 0.01
                combined_features = torch.where(nan_mask, replacement_values, combined_features)
                logger.warning("Replaced NaN combined features with small random values")
            
            # Clamp features before classification heads
            combined_features = torch.clamp(combined_features, min=-10.0, max=10.0)
            
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
            
            # Final NaN protection for logits
            if torch.isnan(severity_logits).any():
                logger.error(f"NaN detected in severity logits! Replacing with zeros.")
                severity_logits = torch.where(torch.isnan(severity_logits), 
                                            torch.zeros_like(severity_logits), severity_logits)
            
            if torch.isnan(action_type_logits).any():
                logger.error(f"NaN detected in action logits! Replacing with zeros.")
                action_type_logits = torch.where(torch.isnan(action_type_logits), 
                                               torch.zeros_like(action_type_logits), action_type_logits)
            
            # Clamp final logits to reasonable range
            severity_logits = torch.clamp(severity_logits, min=-20.0, max=20.0)
            action_type_logits = torch.clamp(action_type_logits, min=-20.0, max=20.0)
            
            # If the clip dimension has not been averaged out yet (i.e., logits were
            # produced per-clip), they will have batch_size * clips_per_video rows.
            # Only then do we reshape and average; otherwise the features were already
            # collapsed earlier and logits have shape [batch_size, num_classes].
            expected_rows = batch_size * clips_per_video
            if clips_per_video > 1 and severity_logits.shape[0] == expected_rows:
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
        """Get comprehensive model information."""
        try:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            # Get backbone-specific info
            backbone_params = sum(p.numel() for p in self.mvit_processor.backbone.parameters())
            backbone_trainable = sum(p.numel() for p in self.mvit_processor.backbone.parameters() if p.requires_grad)
            
            info = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'backbone_parameters': backbone_params,
                'backbone_trainable': backbone_trainable,
                'backbone_frozen_ratio': (backbone_params - backbone_trainable) / backbone_params if backbone_params > 0 else 0,
                'video_feature_dim': self.video_feature_dim,
                'num_severity': self.num_severity,
                'num_action_type': self.num_action_type,
                'use_gradient_checkpointing': self.use_gradient_checkpointing,
                'enable_memory_optimization': self.enable_memory_optimization,
                'dropout_rate': self.dropout_rate
            }
            
            return info
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {'error': str(e)}
    
    def enable_backbone_debug_mode(self, enabled=True):
        """Enable debug mode for backbone processor to help diagnose gradual unfreezing issues."""
        if hasattr(self, 'mvit_processor'):
            self.mvit_processor.enable_debug_mode(enabled)
            logger.info(f"Backbone debug mode: {'enabled' if enabled else 'disabled'}")
        else:
            logger.warning("No mvit_processor found - cannot enable debug mode")
    
    def stabilize_after_gradual_unfreezing(self):
        """Stabilize model after gradual unfreezing to prevent temporary instabilities."""
        try:
            if hasattr(self, 'mvit_processor'):
                self.mvit_processor.stabilize_after_unfreezing()
                logger.info("Model stabilized after gradual unfreezing")
            else:
                logger.warning("No mvit_processor found - cannot stabilize")
        except Exception as e:
            logger.warning(f"Failed to stabilize model after gradual unfreezing: {e}")


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
                2.0: 1.3,   # Reduced from 2.5x to 1.3x more aggressive augmentation
                3.0: 1.6,   # Reduced from 4.0x to 1.6x more aggressive augmentation  
                4.0: 2.0,   # Reduced from 6.0x to 2.0x more aggressive augmentation
                5.0: 2.5    # Reduced from 8.0x to 2.5x more aggressive augmentation
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

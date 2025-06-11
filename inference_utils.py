"""
Inference utilities for handling variable-length videos and domain shift between training and inference.

This module provides tools to bridge the gap between:
- Training: Fixed 16-frame clips centered on foul moments
- Inference: Variable-length videos where foul timing is unknown
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class SlidingWindowInference:
    """
    Sliding window inference for variable-length videos.
    
    Handles the domain shift between training (fixed clips) and inference (variable length)
    by running the model on overlapping windows and aggregating predictions.
    """
    
    def __init__(self, 
                 model,
                 window_size: int = 16,
                 stride: int = 8,
                 aggregation_method: str = "max_confidence",
                 confidence_threshold: float = 0.5,
                 device: str = "cuda"):
        """
        Args:
            model: Trained model for inference
            window_size: Size of sliding window (should match training clip length)
            stride: Step size between windows (smaller = more overlap)
            aggregation_method: How to combine predictions ("max_confidence", "ensemble", "weighted")
            confidence_threshold: Minimum confidence for positive detection
            device: Device to run inference on
        """
        self.model = model
        self.window_size = window_size
        self.stride = stride
        self.aggregation_method = aggregation_method
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        self.model.eval()
        self.model.to(device)
    
    def __call__(self, video: torch.Tensor) -> Dict:
        """
        Run sliding window inference on a variable-length video.
        
        Args:
            video: Input video tensor of shape (C, T, H, W) where T can be any length
            
        Returns:
            Dictionary containing:
            - 'prediction': Final aggregated prediction
            - 'confidence': Confidence score
            - 'window_predictions': List of all window predictions
            - 'foul_timestamp': Estimated timestamp of foul (if detected)
        """
        if video.dim() != 4:
            raise ValueError(f"Expected video tensor of shape (C, T, H, W), got {video.shape}")
        
        C, T, H, W = video.shape
        
        if T < self.window_size:
            # Video is shorter than window size, pad it
            logger.warning(f"Video length ({T}) shorter than window size ({self.window_size}). Padding.")
            padding_needed = self.window_size - T
            # Repeat last frame to maintain temporal structure
            last_frame = video[:, -1:, :, :].repeat(1, padding_needed, 1, 1)
            video = torch.cat([video, last_frame], dim=1)
            T = self.window_size
        
        # Generate sliding windows
        windows = []
        window_positions = []
        
        for start_idx in range(0, T - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            window = video[:, start_idx:end_idx, :, :]
            windows.append(window)
            window_positions.append((start_idx, end_idx))
        
        # If we didn't cover the end of the video, add a final window
        if len(windows) == 0 or window_positions[-1][1] < T:
            final_start = max(0, T - self.window_size)
            final_window = video[:, final_start:T, :, :]
            if final_window.shape[1] < self.window_size:
                # Pad if necessary
                padding = self.window_size - final_window.shape[1]
                pad_frames = final_window[:, -1:, :, :].repeat(1, padding, 1, 1)
                final_window = torch.cat([final_window, pad_frames], dim=1)
            windows.append(final_window)
            window_positions.append((final_start, T))
        
        # Run inference on all windows
        window_predictions = []
        window_confidences = []
        
        with torch.no_grad():
            for i, window in enumerate(windows):
                # Add batch dimension and move to device
                window_batch = window.unsqueeze(0).to(self.device)
                
                # Forward pass
                prediction = self.model(window_batch)
                
                # Extract prediction and confidence
                if isinstance(prediction, dict):
                    # Multi-task model
                    severity_pred = prediction.get('severity', prediction.get('label_severity', None))
                    if severity_pred is not None:
                        confidence = F.softmax(severity_pred, dim=-1).max().item()
                        pred_class = severity_pred.argmax(dim=-1).item()
                    else:
                        confidence = 0.5
                        pred_class = 0
                    window_predictions.append(prediction)
                else:
                    # Single output
                    confidence = F.softmax(prediction, dim=-1).max().item()
                    pred_class = prediction.argmax(dim=-1).item()
                    window_predictions.append(prediction)
                
                window_confidences.append(confidence)
        
        # Aggregate predictions
        aggregated_result = self._aggregate_predictions(
            window_predictions, 
            window_confidences, 
            window_positions
        )
        
        return aggregated_result
    
    def _aggregate_predictions(self, 
                             predictions: List,
                             confidences: List[float],
                             positions: List[Tuple[int, int]]) -> Dict:
        """Aggregate predictions from multiple windows."""
        
        if self.aggregation_method == "max_confidence":
            # Use prediction from window with highest confidence
            best_idx = np.argmax(confidences)
            best_prediction = predictions[best_idx]
            best_confidence = confidences[best_idx]
            best_position = positions[best_idx]
            
            # Estimate foul timestamp (center of best window)
            foul_timestamp = (best_position[0] + best_position[1]) / 2
            
            return {
                'prediction': best_prediction,
                'confidence': best_confidence,
                'window_predictions': predictions,
                'window_confidences': confidences,
                'foul_timestamp': foul_timestamp,
                'num_windows': len(predictions)
            }
            
        elif self.aggregation_method == "ensemble":
            # Average predictions across all windows
            return self._ensemble_predictions(predictions, confidences, positions)
            
        elif self.aggregation_method == "weighted":
            # Weighted average based on confidence
            return self._weighted_predictions(predictions, confidences, positions)
            
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _ensemble_predictions(self, predictions, confidences, positions):
        """Ensemble predictions by averaging."""
        if isinstance(predictions[0], dict):
            # Multi-task model - ensemble each output separately
            ensemble_pred = {}
            for key in predictions[0].keys():
                if key in predictions[0] and torch.is_tensor(predictions[0][key]):
                    stacked = torch.stack([pred[key] for pred in predictions])
                    ensemble_pred[key] = stacked.mean(dim=0)
            
            # Use severity prediction for confidence if available
            if 'severity' in ensemble_pred or 'label_severity' in ensemble_pred:
                severity_key = 'severity' if 'severity' in ensemble_pred else 'label_severity'
                avg_confidence = F.softmax(ensemble_pred[severity_key], dim=-1).max().item()
            else:
                avg_confidence = np.mean(confidences)
        else:
            # Single output
            stacked = torch.stack(predictions)
            ensemble_pred = stacked.mean(dim=0)
            avg_confidence = F.softmax(ensemble_pred, dim=-1).max().item()
        
        return {
            'prediction': ensemble_pred,
            'confidence': avg_confidence,
            'window_predictions': predictions,
            'window_confidences': confidences,
            'foul_timestamp': None,  # No single timestamp for ensemble
            'num_windows': len(predictions)
        }
    
    def _weighted_predictions(self, predictions, confidences, positions):
        """Weighted average based on confidence scores."""
        weights = np.array(confidences)
        weights = weights / weights.sum()  # Normalize
        
        if isinstance(predictions[0], dict):
            # Multi-task model
            weighted_pred = {}
            for key in predictions[0].keys():
                if key in predictions[0] and torch.is_tensor(predictions[0][key]):
                    weighted_sum = torch.zeros_like(predictions[0][key])
                    for pred, weight in zip(predictions, weights):
                        weighted_sum += pred[key] * weight
                    weighted_pred[key] = weighted_sum
                    
            # Calculate weighted confidence
            if 'severity' in weighted_pred or 'label_severity' in weighted_pred:
                severity_key = 'severity' if 'severity' in weighted_pred else 'label_severity'
                weighted_confidence = F.softmax(weighted_pred[severity_key], dim=-1).max().item()
            else:
                weighted_confidence = np.average(confidences, weights=weights)
        else:
            # Single output
            weighted_pred = torch.zeros_like(predictions[0])
            for pred, weight in zip(predictions, weights):
                weighted_pred += pred * weight
            weighted_confidence = F.softmax(weighted_pred, dim=-1).max().item()
        
        # Estimate foul timestamp based on weighted average of positions
        position_centers = [(pos[0] + pos[1]) / 2 for pos in positions]
        weighted_timestamp = np.average(position_centers, weights=weights)
        
        return {
            'prediction': weighted_pred,
            'confidence': weighted_confidence,
            'window_predictions': predictions,
            'window_confidences': confidences,
            'foul_timestamp': weighted_timestamp,
            'num_windows': len(predictions)
        }


class AdaptiveWindowInference:
    """
    Adaptive sliding window that adjusts window size and stride based on video characteristics.
    """
    
    def __init__(self, 
                 model,
                 base_window_size: int = 16,
                 min_window_size: int = 8,
                 max_window_size: int = 32,
                 device: str = "cuda"):
        self.model = model
        self.base_window_size = base_window_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.device = device
        
        self.model.eval()
        self.model.to(device)
    
    def __call__(self, video: torch.Tensor) -> Dict:
        """
        Adaptive inference that adjusts to video characteristics.
        """
        C, T, H, W = video.shape
        
        # Adapt window size based on video length
        if T < 30:
            # Short video: use smaller windows with high overlap
            window_size = min(self.base_window_size, T)
            stride = max(1, window_size // 4)
        elif T > 300:
            # Long video: use larger windows with less overlap
            window_size = min(self.max_window_size, self.base_window_size * 2)
            stride = window_size // 2
        else:
            # Medium video: use standard settings
            window_size = self.base_window_size
            stride = window_size // 2
        
        # Use sliding window inference with adapted parameters
        sliding_inference = SlidingWindowInference(
            model=self.model,
            window_size=window_size,
            stride=stride,
            device=self.device
        )
        
        return sliding_inference(video)


def create_inference_pipeline(model, inference_type: str = "sliding_window", **kwargs):
    """
    Factory function to create appropriate inference pipeline.
    
    Args:
        model: Trained model
        inference_type: Type of inference ("sliding_window", "adaptive")
        **kwargs: Additional arguments for inference pipeline
    """
    if inference_type == "sliding_window":
        return SlidingWindowInference(model, **kwargs)
    elif inference_type == "adaptive":
        return AdaptiveWindowInference(model, **kwargs)
    else:
        raise ValueError(f"Unknown inference type: {inference_type}") 
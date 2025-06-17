"""
PyTorch Lightning DataModule for Multi-Task Multi-View ResNet3D training.

This module wraps the existing data loading logic in PyTorch Lightning
while preserving all custom features like class-balanced sampling, 
progressive balancing, and optimized data loading.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import logging
import torch
import numpy as np
from collections import Counter

# Import existing data components
from .data import create_datasets, create_dataloaders

logger = logging.getLogger(__name__)


class VideoDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for video classification.
    
    This module wraps the existing sophisticated data loading logic
    while providing Lightning's standardized interface for distributed training
    and automatic data handling.
    """
    
    def __init__(
        self,
        args: Any,
        **kwargs
    ):
        """
        Initialize the DataModule.
        
        Args:
            args: Training arguments from config
        """
        super().__init__()
        
        # Store configuration
        self.args = args
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['args'])
        
        # Initialize datasets and dataloaders as None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.train_dataloader_obj = None
        self.val_dataloader_obj = None
        self.test_dataloader_obj = None
        
        # Initialize class weights as None - will be computed during setup
        self.severity_class_weights = None
        self.action_class_weights = None
        
        logger.info("Initialized VideoDataModule")
    
    def prepare_data(self):
        """
        Download or prepare data (called only once per node).
        
        This is called only once per node and is a good place to download
        data or do any one-time setup that doesn't require access to the data.
        """
        # In our case, we assume data is already available
        # This could be extended to include data download logic
        logger.info("Preparing data (no-op - assuming data is available)")
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for the given stage.
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        logger.info(f"Setting up DataModule for stage: {stage}")
        
        if stage == 'fit' or stage is None:
            # Create datasets using existing logic
            if self.train_dataset is None or self.val_dataset is None:
                logger.info("Creating train and validation datasets...")
                self.train_dataset, self.val_dataset = create_datasets(self.args)
                logger.info(f"Datasets created: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}")
                
                # Compute class weights from training dataset
                self._compute_class_weights()
            
            # Create dataloaders using existing logic
            if self.train_dataloader_obj is None or self.val_dataloader_obj is None:
                logger.info("Creating train and validation dataloaders...")
                self.train_dataloader_obj, self.val_dataloader_obj = create_dataloaders(
                    self.args, self.train_dataset, self.val_dataset
                )
                logger.info("Dataloaders created successfully")
        
        if stage == 'validate':
            # Only need validation dataset
            if self.val_dataset is None:
                logger.info("Creating validation dataset...")
                _, self.val_dataset = create_datasets(self.args)
                logger.info(f"Validation dataset created: {len(self.val_dataset)}")
            
            if self.val_dataloader_obj is None:
                logger.info("Creating validation dataloader...")
                _, self.val_dataloader_obj = create_dataloaders(
                    self.args, None, self.val_dataset
                )
                logger.info("Validation dataloader created")
        
        if stage == 'test':
            # For test, we can reuse validation dataset or create a separate test dataset
            if self.test_dataset is None:
                logger.info("Creating test dataset (using validation dataset)...")
                _, self.test_dataset = create_datasets(self.args)
                logger.info(f"Test dataset created: {len(self.test_dataset)}")
            
            if self.test_dataloader_obj is None:
                logger.info("Creating test dataloader...")
                _, self.test_dataloader_obj = create_dataloaders(
                    self.args, None, self.test_dataset
                )
                logger.info("Test dataloader created")
        
        if stage == 'predict':
            # For prediction, we can reuse validation dataset
            if self.val_dataset is None:
                logger.info("Creating prediction dataset (using validation dataset)...")
                _, self.val_dataset = create_datasets(self.args)
                logger.info(f"Prediction dataset created: {len(self.val_dataset)}")
    
    def _compute_class_weights(self):
        """
        Compute class weights using the effective-number-of-samples formula.
        
        The effective number of samples formula helps handle class imbalance by
        computing weights that account for the diminishing returns of additional
        samples from the same class.
        
        Formula: E_n = (1 - β^n) / (1 - β)
        where β is typically 0.9999 for large datasets or 0.99 for smaller ones.
        """
        if self.train_dataset is None:
            logger.warning("Training dataset not available for class weight computation")
            return
        
        logger.info("Computing automatic class weights using effective-number-of-samples formula...")
        
        # Extract labels from training dataset
        severity_labels = []
        action_labels = []
        
        for action in self.train_dataset.actions:
            severity_labels.append(action['label_severity'])
            action_labels.append(action['label_type'])
        
        # Count samples per class
        severity_counts = Counter(severity_labels)
        action_counts = Counter(action_labels)
        
        # Import the rank checking function
        from .training_utils import is_main_process
        
        # Log class distributions
        if is_main_process():
            logger.info("Class distributions in training set:")
        
        if is_main_process():
            logger.info(f"Severity classes: {dict(severity_counts)}")
            logger.info(f"Action classes: {dict(action_counts)}")
        
        # Compute effective number of samples
        # Use β=0.99 for smaller datasets (< 10k samples), β=0.9999 for larger ones
        total_samples = len(self.train_dataset)
        beta = 0.99 if total_samples < 10000 else 0.9999
        
        if is_main_process():
            logger.info(f"Using beta={beta} for effective-number-of-samples computation (dataset size: {total_samples})")
        
        # Compute severity class weights
        severity_weights = {}
        for class_id, count in severity_counts.items():
            if count > 0:
                effective_num = (1.0 - beta ** count) / (1.0 - beta)
                severity_weights[class_id] = 1.0 / effective_num
            else:
                severity_weights[class_id] = 0.0
        
        # Compute action class weights
        action_weights = {}
        for class_id, count in action_counts.items():
            if count > 0:
                effective_num = (1.0 - beta ** count) / (1.0 - beta)
                action_weights[class_id] = 1.0 / effective_num
            else:
                action_weights[class_id] = 0.0
        
        # Normalize weights so they sum to number of classes
        def normalize_weights(weights_dict):
            if not weights_dict:
                return {}
            
            weight_values = list(weights_dict.values())
            weight_sum = sum(weight_values)
            num_classes = len(weights_dict)
            
            if weight_sum > 0:
                normalization_factor = num_classes / weight_sum
                return {k: v * normalization_factor for k, v in weights_dict.items()}
            else:
                return {k: 1.0 for k in weights_dict.keys()}
        
        severity_weights = normalize_weights(severity_weights)
        action_weights = normalize_weights(action_weights)
        
        # Convert to tensors for easy use in loss functions
        # Get the maximum class ID to determine tensor size
        max_severity_class = max(severity_weights.keys()) if severity_weights else 0
        max_action_class = max(action_weights.keys()) if action_weights else 0
        
        # Create weight tensors (index 0 will be for class 0, etc.)
        self.severity_class_weights = torch.ones(max_severity_class + 1)
        self.action_class_weights = torch.ones(max_action_class + 1)
        
        for class_id, weight in severity_weights.items():
            self.severity_class_weights[class_id] = weight
        
        for class_id, weight in action_weights.items():
            self.action_class_weights[class_id] = weight
        
        # Import the rank checking function
        from .training_utils import is_main_process
        
        # Log computed weights (only from main process)
        if is_main_process():
            logger.info("Computed class weights:")
            logger.info(f"Severity weights: {severity_weights}")
            logger.info(f"Action weights: {action_weights}")
            
            # Log weight ratios for analysis
            if len(severity_weights) > 1:
                severity_weight_values = list(severity_weights.values())
                severity_ratio = max(severity_weight_values) / min(severity_weight_values)
                logger.info(f"Severity weight ratio (max/min): {severity_ratio:.2f}")
            
            if len(action_weights) > 1:
                action_weight_values = list(action_weights.values())
                action_ratio = max(action_weight_values) / min(action_weight_values)
                logger.info(f"Action weight ratio (max/min): {action_ratio:.2f}")
    
    def get_class_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get computed class weights for use in loss functions.
        
        Returns:
            Dictionary containing 'severity' and 'action' class weight tensors
        """
        if self.severity_class_weights is None or self.action_class_weights is None:
            logger.warning("Class weights not computed yet. Call setup() first.")
            return {}
        
        return {
            'severity': self.severity_class_weights,
            'action': self.action_class_weights
        }
    
    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        if self.train_dataloader_obj is None:
            raise RuntimeError("Training dataloader not initialized. Call setup('fit') first.")
        
        logger.debug("Returning training dataloader")
        return self.train_dataloader_obj
    
    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        if self.val_dataloader_obj is None:
            raise RuntimeError("Validation dataloader not initialized. Call setup('fit' or 'validate') first.")
        
        logger.debug("Returning validation dataloader")
        return self.val_dataloader_obj
    
    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        if self.test_dataloader_obj is None:
            raise RuntimeError("Test dataloader not initialized. Call setup('test') first.")
        
        logger.debug("Returning test dataloader")
        return self.test_dataloader_obj
    
    def predict_dataloader(self) -> DataLoader:
        """Return the prediction dataloader."""
        if self.val_dataloader_obj is None:
            raise RuntimeError("Prediction dataloader not initialized. Call setup('predict') first.")
        
        logger.debug("Returning prediction dataloader")
        return self.val_dataloader_obj
    
    def teardown(self, stage: Optional[str] = None):
        """
        Clean up after training/validation/testing.
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        logger.info(f"Tearing down DataModule for stage: {stage}")
        # Cleanup if needed - our current implementation doesn't require special cleanup
    
    def get_vocab_sizes(self) -> Optional[Dict[str, int]]:
        """
        Get vocabulary sizes from the dataset.
        
        Returns:
            Dictionary of vocabulary sizes or None if not available
        """
        if self.train_dataset is not None:
            # Extract vocabulary sizes from dataset attributes
            try:
                vocab_sizes = {
                    'contact': getattr(self.train_dataset, 'num_contact_classes', 3),
                    'bodypart': getattr(self.train_dataset, 'num_bodypart_classes', 4),
                    'upper_bodypart': getattr(self.train_dataset, 'num_upper_bodypart_classes', 5),
                    'multiple_fouls': getattr(self.train_dataset, 'num_multiple_fouls_classes', 5),
                    'try_to_play': getattr(self.train_dataset, 'num_try_to_play_classes', 4),
                    'touch_ball': getattr(self.train_dataset, 'num_touch_ball_classes', 5),
                    'handball': getattr(self.train_dataset, 'num_handball_classes', 3),
                    'handball_offence': getattr(self.train_dataset, 'num_handball_offence_classes', 4),
                }
                return vocab_sizes
            except Exception as e:
                logger.warning(f"Could not extract vocabulary sizes from dataset: {e}")
                return None
        elif self.val_dataset is not None:
            # Try validation dataset
            try:
                vocab_sizes = {
                    'contact': getattr(self.val_dataset, 'num_contact_classes', 3),
                    'bodypart': getattr(self.val_dataset, 'num_bodypart_classes', 4),
                    'upper_bodypart': getattr(self.val_dataset, 'num_upper_bodypart_classes', 5),
                    'multiple_fouls': getattr(self.val_dataset, 'num_multiple_fouls_classes', 5),
                    'try_to_play': getattr(self.val_dataset, 'num_try_to_play_classes', 4),
                    'touch_ball': getattr(self.val_dataset, 'num_touch_ball_classes', 5),
                    'handball': getattr(self.val_dataset, 'num_handball_classes', 3),
                    'handball_offence': getattr(self.val_dataset, 'num_handball_offence_classes', 4),
                }
                return vocab_sizes
            except Exception as e:
                logger.warning(f"Could not extract vocabulary sizes from validation dataset: {e}")
                return None
        else:
            logger.warning("No vocabulary sizes available from datasets")
            return None
    
    def get_class_distribution(self) -> Optional[Dict[str, Any]]:
        """
        Get class distribution information from the training dataset.
        
        Returns:
            Dictionary with class distribution info or None if not available
        """
        if self.train_dataset is not None and hasattr(self.train_dataset, 'get_class_distribution'):
            return self.train_dataset.get_class_distribution()
        else:
            logger.warning("No class distribution information available from training dataset")
            return None
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset information.
        
        Returns:
            Dictionary with dataset information
        """
        info = {
            'train_size': len(self.train_dataset) if self.train_dataset is not None else 0,
            'val_size': len(self.val_dataset) if self.val_dataset is not None else 0,
            'test_size': len(self.test_dataset) if self.test_dataset is not None else 0,
            'vocab_sizes': self.get_vocab_sizes(),
            'class_distribution': self.get_class_distribution(),
            'class_weights': self.get_class_weights()
        }
        
        return info 
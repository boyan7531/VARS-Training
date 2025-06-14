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
            'class_distribution': self.get_class_distribution()
        }
        
        return info 
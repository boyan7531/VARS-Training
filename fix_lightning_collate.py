#!/usr/bin/env python3
"""
Fix for Lightning DataModule collate function issue.
This patches the VideoDataModule to ensure the collate function is preserved.
"""

import torch
from torch.utils.data import DataLoader
from dataset import variable_views_collate_fn

def patch_lightning_datamodule():
    """Patch the VideoDataModule to fix collate function issue."""
    
    # Import the module
    from training.lightning_datamodule import VideoDataModule
    
    # Store original methods
    original_train_dataloader = VideoDataModule.train_dataloader
    original_val_dataloader = VideoDataModule.val_dataloader
    
    def patched_train_dataloader(self):
        """Patched train_dataloader that ensures collate_fn is used."""
        # Get the original dataloader
        dataloader = original_train_dataloader(self)
        
        # Check if it has the correct collate_fn (handle OptimizedDataLoader)
        collate_fn = getattr(dataloader, 'collate_fn', None)
        
        # Special handling for OptimizedDataLoader
        if 'OptimizedDataLoader' in str(type(dataloader)):
            print(f"INFO: OptimizedDataLoader detected - bypassing collate_fn patch")
            print(f"OptimizedDataLoader should handle collate_fn internally")
            return dataloader
        
        if collate_fn != variable_views_collate_fn:
            print(f"WARNING: Train DataLoader collate_fn mismatch!")
            print(f"Expected: {variable_views_collate_fn}")
            print(f"Got: {collate_fn}")
            print(f"DataLoader type: {type(dataloader)}")
            
            # Create a new DataLoader with correct collate_fn
            return DataLoader(
                dataset=dataloader.dataset,
                batch_size=dataloader.batch_size,
                shuffle=False,  # Sampler handles shuffling
                sampler=dataloader.sampler,
                batch_sampler=dataloader.batch_sampler,
                num_workers=dataloader.num_workers,
                collate_fn=variable_views_collate_fn,  # Force correct collate
                pin_memory=dataloader.pin_memory,
                drop_last=dataloader.drop_last,
                timeout=dataloader.timeout,
                worker_init_fn=dataloader.worker_init_fn,
                multiprocessing_context=dataloader.multiprocessing_context,
                generator=dataloader.generator,
                prefetch_factor=dataloader.prefetch_factor,
                persistent_workers=dataloader.persistent_workers,
            )
        
        return dataloader
    
    def patched_val_dataloader(self):
        """Patched val_dataloader that ensures collate_fn is used."""
        # Get the original dataloader
        dataloader = original_val_dataloader(self)
        
        # Check if it has the correct collate_fn (handle OptimizedDataLoader)
        collate_fn = getattr(dataloader, 'collate_fn', None)
        
        # Special handling for OptimizedDataLoader
        if 'OptimizedDataLoader' in str(type(dataloader)):
            print(f"INFO: OptimizedDataLoader detected - bypassing collate_fn patch")
            print(f"OptimizedDataLoader should handle collate_fn internally")
            return dataloader
        
        if collate_fn != variable_views_collate_fn:
            print(f"WARNING: Val DataLoader collate_fn mismatch!")
            print(f"Expected: {variable_views_collate_fn}")
            print(f"Got: {collate_fn}")
            print(f"DataLoader type: {type(dataloader)}")
            
            # Create a new DataLoader with correct collate_fn
            return DataLoader(
                dataset=dataloader.dataset,
                batch_size=dataloader.batch_size,
                shuffle=False,
                sampler=dataloader.sampler,
                batch_sampler=dataloader.batch_sampler,
                num_workers=dataloader.num_workers,
                collate_fn=variable_views_collate_fn,  # Force correct collate
                pin_memory=dataloader.pin_memory,
                drop_last=dataloader.drop_last,
                timeout=dataloader.timeout,
                worker_init_fn=dataloader.worker_init_fn,
                multiprocessing_context=dataloader.multiprocessing_context,
                generator=dataloader.generator,
                prefetch_factor=dataloader.prefetch_factor,
                persistent_workers=dataloader.persistent_workers,
            )
        
        return dataloader
    
    # Apply patches
    VideoDataModule.train_dataloader = patched_train_dataloader
    VideoDataModule.val_dataloader = patched_val_dataloader
    
    print("✅ Applied Lightning DataModule collate_fn patch")

if __name__ == "__main__":
    patch_lightning_datamodule()
    print("Patch applied. Import this module before training.") 
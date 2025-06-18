#!/usr/bin/env python3
"""
Debug script to diagnose the batch structure issue.
Run this to see what your DataLoader is actually returning.
"""

import torch
from torch.utils.data import DataLoader

# Import your dataset and collate function
try:
    from dataset import SoccerNetMVFoulDataset, variable_views_collate_fn
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)

class MinimalArgs:
    """Minimal args object for testing."""
    def __init__(self):
        self.mvfouls_path = "./mvfouls"
        self.batch_size = 2
        self.num_workers = 0

def debug_batch_structure():
    """Debug what the DataLoader returns."""
    
    # Create minimal config
    args = MinimalArgs()
    
    # Set minimal parameters for testing
    args.mvfouls_path = "./mvfouls"  # Adjust path as needed
    args.batch_size = 2
    args.num_workers = 0  # Use 0 to avoid multiprocessing issues
    
    try:
        # Create dataset
        print("Creating dataset...")
        dataset = SoccerNetMVFoulDataset(
            dataset_path=args.mvfouls_path,
            split="train",
            frames_per_clip=16,
            target_fps=25,
            start_frame=67,
            end_frame=82,
            load_all_views=True,
            target_height=224,
            target_width=224,
            use_severity_aware_aug=False,  # Disable for debugging
            clips_per_video=1
        )
        print(f"Dataset created with {len(dataset)} samples")
        
        # Test single sample
        print("\n=== Testing single sample ===")
        sample = dataset[0]
        print(f"Sample type: {type(sample)}")
        print(f"Sample keys: {sample.keys() if isinstance(sample, dict) else 'Not a dict!'}")
        
        # Test collate function
        print("\n=== Testing collate function ===")
        batch_list = [sample]
        collated = variable_views_collate_fn(batch_list)
        print(f"Collated type: {type(collated)}")
        print(f"Collated keys: {collated.keys() if isinstance(collated, dict) else 'Not a dict!'}")
        
        if isinstance(collated, dict) and "clips" in collated:
            print(f"Clips shape: {collated['clips'].shape}")
        
        # Test DataLoader without collate_fn
        print("\n=== Testing DataLoader WITHOUT collate_fn ===")
        loader_default = DataLoader(dataset, batch_size=2, shuffle=False)
        batch_default = next(iter(loader_default))
        print(f"Default batch type: {type(batch_default)}")
        if isinstance(batch_default, (list, tuple)):
            print(f"Default batch length: {len(batch_default)}")
            print(f"Default batch[0] type: {type(batch_default[0])}")
        
        # Test DataLoader with collate_fn
        print("\n=== Testing DataLoader WITH collate_fn ===")
        loader_custom = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False,
            collate_fn=variable_views_collate_fn
        )
        batch_custom = next(iter(loader_custom))
        print(f"Custom batch type: {type(batch_custom)}")
        print(f"Custom batch keys: {batch_custom.keys() if isinstance(batch_custom, dict) else 'Not a dict!'}")
        
        if isinstance(batch_custom, dict):
            for key, value in batch_custom.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
        
        print("\n✅ Diagnosis complete!")
        print("If you see 'Custom batch keys: dict_keys([...])' then your collate function works.")
        print("If you see 'Not a dict!' then there's a collate function issue.")
        
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_batch_structure() 
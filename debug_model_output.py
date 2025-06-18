#!/usr/bin/env python3
"""
Debug script to check what the model returns.
"""

import torch
from dataset import SoccerNetMVFoulDataset, variable_views_collate_fn
from torch.utils.data import DataLoader

def test_model_output():
    """Test what the model actually returns."""
    
    # Create minimal dataset and dataloader
    class MinimalArgs:
        def __init__(self):
            self.mvfouls_path = "./mvfouls"
            self.batch_size = 2
            self.num_workers = 0
    
    args = MinimalArgs()
    
    try:
        # Create dataset
        dataset = SoccerNetMVFoulDataset(args.mvfouls_path, split='train')
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size,
            collate_fn=variable_views_collate_fn,
            num_workers=0
        )
        
        # Get a batch
        batch = next(iter(dataloader))
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Clips shape: {batch['clips'].shape}")
        
        # Create a minimal model
        from model.resnet3d_model import MultiTaskMultiViewResNet3D
        from model import ModelConfig
        
        vocab_sizes = {
            'contact': 3, 'bodypart': 4, 'upper_bodypart': 5,
            'multiple_fouls': 5, 'try_to_play': 4, 'touch_ball': 5,
            'handball': 3, 'handball_offence': 4
        }
        
        config = ModelConfig()  # Use default config
        
        model = MultiTaskMultiViewResNet3D(config, vocab_sizes)
        model.eval()
        
        # Test model output
        print("\n=== Testing model output ===")
        with torch.no_grad():
            output = model(batch)
            print(f"Model output type: {type(output)}")
            print(f"Model output length: {len(output) if hasattr(output, '__len__') else 'N/A'}")
            
            if isinstance(output, tuple):
                print(f"Tuple elements:")
                for i, elem in enumerate(output):
                    print(f"  [{i}]: {type(elem)} - {elem.shape if hasattr(elem, 'shape') else elem}")
            elif isinstance(output, dict):
                print(f"Dict keys: {list(output.keys())}")
                for key, value in output.items():
                    print(f"  {key}: {type(value)} - {value.shape if hasattr(value, 'shape') else value}")
            else:
                print(f"Unexpected output type: {type(output)}")
        
        # Test with return_view_logits=True if supported
        print("\n=== Testing with return_view_logits=True ===")
        try:
            with torch.no_grad():
                output = model(batch, return_view_logits=True)
                print(f"Model output type: {type(output)}")
                if isinstance(output, dict):
                    print(f"Dict keys: {list(output.keys())}")
                    for key, value in output.items():
                        print(f"  {key}: {type(value)} - {value.shape if hasattr(value, 'shape') else value}")
        except Exception as e:
            print(f"Model doesn't support return_view_logits=True: {e}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_output() 
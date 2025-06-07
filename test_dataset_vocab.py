import torch
from dataset import SoccerNetMVFoulDataset, SEVERITY_LABELS, ACTION_TYPE_LABELS

print("Testing dataset loading...")

try:
    # Load dataset
    train_dataset = SoccerNetMVFoulDataset(
        dataset_path="mvfouls",
        split="train"
    )
    
    print(f"Dataset loaded successfully with {len(train_dataset.actions)} actions")
    
    # Print vocab sizes
    vocab_sizes = {
        'contact': train_dataset.num_contact_classes,
        'bodypart': train_dataset.num_bodypart_classes,
        'upper_bodypart': train_dataset.num_upper_bodypart_classes,
        'lower_bodypart': train_dataset.num_lower_bodypart_classes,
        'multiple_fouls': train_dataset.num_multiple_fouls_classes,
        'try_to_play': train_dataset.num_try_to_play_classes,
        'touch_ball': train_dataset.num_touch_ball_classes,
        'handball': train_dataset.num_handball_classes,
        'handball_offence': train_dataset.num_handball_offence_classes
    }
    
    print("Vocabulary sizes for model:")
    for key, size in vocab_sizes.items():
        print(f"  {key}: {size}")
    
    # Print first few action items
    print("\nSample actions:")
    for i, action in enumerate(train_dataset.actions[:3]):
        print(f"Action {i}:")
        for k, v in action.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: Tensor of shape {v.shape}")
            else:
                print(f"  {k}: {v}")
        print()
        
except Exception as e:
    print(f"Error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc() 
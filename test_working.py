import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torchvision.transforms as transforms
from collections import defaultdict

# Import model and dataset classes
from model import MultiTaskMultiViewResNet3D, ModelConfig
from dataset import SoccerNetMVFoulDataset, variable_views_collate_fn

def load_test_model(checkpoint_path='best_model_epoch_2.pth'):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract vocab sizes from the model state dict shapes
    state_dict = checkpoint['model_state_dict']
    
    vocab_sizes = {}
    embedding_keys = {
        'contact': 'embedding_manager.original_embeddings.contact.weight',
        'bodypart': 'embedding_manager.original_embeddings.bodypart.weight', 
        'upper_bodypart': 'embedding_manager.original_embeddings.upper_bodypart.weight',
        'lower_bodypart': 'embedding_manager.original_embeddings.lower_bodypart.weight',
        'multiple_fouls': 'embedding_manager.original_embeddings.multiple_fouls.weight',
        'try_to_play': 'embedding_manager.original_embeddings.try_to_play.weight',
        'touch_ball': 'embedding_manager.original_embeddings.touch_ball.weight',
        'handball': 'embedding_manager.original_embeddings.handball.weight',
        'handball_offence': 'embedding_manager.original_embeddings.handball_offence.weight'
    }
    
    for field, weight_key in embedding_keys.items():
        if weight_key in state_dict:
            vocab_sizes[field] = state_dict[weight_key].shape[0]
    
    print(f"Extracted vocab sizes: {vocab_sizes}")
    
    # Create model configuration
    model_config = ModelConfig(
        use_attention_aggregation=True,
        input_frames=16,
        input_height=224,
        input_width=398  # ResNet3D supports rectangular inputs
    )
    
    # Initialize model
    model = MultiTaskMultiViewResNet3D(
        num_severity=5,
        num_action_type=9,
        vocab_sizes=vocab_sizes,
        backbone_name='resnet3d_18',
        config=model_config
    )
    
    # Load model weights
    # Remove 'module.' prefix if present (DataParallel)
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")
    return model, vocab_sizes

def clamp_vocab_indices(sample, vocab_sizes):
    """Clamp vocabulary indices to valid ranges to prevent validation errors."""
    clamped_sample = sample.copy()
    
    # Mapping of feature names to their vocab sizes
    vocab_mappings = {
        'contact_idx': vocab_sizes['contact'],
        'bodypart_idx': vocab_sizes['bodypart'],
        'upper_bodypart_idx': vocab_sizes['upper_bodypart'],
        'lower_bodypart_idx': vocab_sizes['lower_bodypart'],
        'multiple_fouls_idx': vocab_sizes['multiple_fouls'],
        'try_to_play_idx': vocab_sizes['try_to_play'],
        'touch_ball_idx': vocab_sizes['touch_ball'],
        'handball_idx': vocab_sizes['handball'],
        'handball_offence_idx': vocab_sizes['handball_offence']
    }
    
    for feature_key, max_vocab_size in vocab_mappings.items():
        if feature_key in sample:
            original_value = sample[feature_key].item()
            clamped_value = min(original_value, max_vocab_size - 1)  # Clamp to valid range [0, max_size-1]
            if original_value != clamped_value:
                print(f"Warning: Clamped {feature_key} from {original_value} to {clamped_value} (vocab size: {max_vocab_size})")
            clamped_sample[feature_key] = torch.tensor(clamped_value, dtype=torch.long)
    
    return clamped_sample

def create_test_dataset():
    """Create test dataset for the first example."""
    print("Creating test dataset...")
    
    # Define transforms to match training data preprocessing
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float() / 255.0),  # Convert to float and normalize
        transforms.Resize((224, 224)),  # Resize to 224x224
    ])
    
    dataset = SoccerNetMVFoulDataset(
        dataset_path='mvfouls',
        split='test',
        annotation_file_name='annotations.json',
        frames_per_clip=16,
        target_fps=17,
        start_frame=67,
        end_frame=82,
        load_all_views=True,
        target_height=224,
        target_width=224,
        transform=transform
    )
    
    print(f"Test dataset created with {len(dataset)} samples")
    return dataset

def get_class_names():
    """Get readable class names for predictions."""
    severity_names = {
        0: "1.0 (Lowest)",
        1: "2.0 (Low)", 
        2: "3.0 (Medium)",
        3: "4.0 (High)",
        4: "5.0 (Highest/Red card)"
    }
    
    action_names = {
        0: "Challenge",
        1: "Dive",
        2: "Don't know", 
        3: "Elbowing",
        4: "High leg",
        5: "Holding",
        6: "Pushing",
        7: "Standing tackling",
        8: "Tackling"
    }
    
    return severity_names, action_names

def show_ground_truth(sample):
    """Show ground truth labels for comparison."""
    print("\n" + "="*50)
    print("GROUND TRUTH LABELS")
    print("="*50)
    
    # Main labels - use correct key names
    severity_names, action_names = get_class_names()
    sev_gt = sample['label_severity'].item()
    act_gt = sample['label_type'].item()
    
    print(f"Severity: {severity_names[sev_gt]} (class {sev_gt})")
    print(f"Action Type: {action_names[act_gt]} (class {act_gt})")
    
    # Auxiliary features - use correct key names
    print(f"\nAuxiliary Features:")
    print(f"Contact: {sample['contact_idx'].item()}")
    print(f"Bodypart: {sample['bodypart_idx'].item()}")
    print(f"Upper bodypart: {sample['upper_bodypart_idx'].item()}")
    print(f"Lower bodypart: {sample['lower_bodypart_idx'].item()}")
    print(f"Multiple fouls: {sample['multiple_fouls_idx'].item()}")
    print(f"Try to play: {sample['try_to_play_idx'].item()}")
    print(f"Touch ball: {sample['touch_ball_idx'].item()}")
    print(f"Handball: {sample['handball_idx'].item()}")
    print(f"Handball offence: {sample['handball_offence_idx'].item()}")

def run_inference(model, sample, vocab_sizes):
    """Run model inference and show predictions."""
    print("\n" + "="*50)
    print("MODEL PREDICTIONS")
    print("="*50)
    
    # Clamp vocabulary indices to prevent validation errors
    clamped_sample = clamp_vocab_indices(sample, vocab_sizes)
    
    with torch.no_grad():
        # Prepare batch data (add batch dimension)
        batch_data = {}
        for key, value in clamped_sample.items():
            if key == 'clips':
                if isinstance(value, list):
                    # Video already transformed (normalized and resized)
                    batch_data[key] = [v.unsqueeze(0) for v in value]  # Add batch dim
                else:
                    # Video already transformed (normalized and resized)
                    batch_data[key] = value.unsqueeze(0)  # Add batch dimension
            else:
                batch_data[key] = value.unsqueeze(0)  # Add batch dimension
        
        # Run inference
        severity_logits, action_logits = model(batch_data)
        
        # Convert to probabilities
        severity_probs = F.softmax(severity_logits, dim=1)
        action_probs = F.softmax(action_logits, dim=1)
        
        # Get predictions
        severity_pred = torch.argmax(severity_probs, dim=1).item()
        action_pred = torch.argmax(action_probs, dim=1).item()
        
        # Get class names
        severity_names, action_names = get_class_names()
        
        # Show severity predictions
        print("SEVERITY PREDICTION:")
        print(f"Predicted: {severity_names[severity_pred]} (class {severity_pred})")
        print(f"Confidence: {severity_probs[0, severity_pred].item():.3f}")
        print("\nAll severity probabilities:")
        for i, prob in enumerate(severity_probs[0]):
            print(f"  {severity_names[i]}: {prob.item():.3f}")
        
        # Show action predictions  
        print(f"\nACTION TYPE PREDICTION:")
        print(f"Predicted: {action_names[action_pred]} (class {action_pred})")
        print(f"Confidence: {action_probs[0, action_pred].item():.3f}")
        print("\nAll action type probabilities:")
        for i, prob in enumerate(action_probs[0]):
            print(f"  {action_names[i]}: {prob.item():.3f}")
        
        return severity_pred, action_pred, severity_probs[0], action_probs[0]

def main():
    """Main function to run inference test."""
    try:
        # Load model
        model, vocab_sizes = load_test_model()
        
        # Create test dataset
        test_dataset = create_test_dataset()
        
        # Get first test sample
        print(f"\nLoading first test sample (action_0)...")
        sample = test_dataset[0]  # First test example
        
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Video clips shape: {sample['clips'][0].shape if isinstance(sample['clips'], list) else sample['clips'].shape}")
        print(f"Number of views: {len(sample['clips']) if isinstance(sample['clips'], list) else 'Single tensor'}")
        
        # Show ground truth
        show_ground_truth(sample)
        
        # Run inference
        severity_pred, action_pred, sev_probs, act_probs = run_inference(model, sample, vocab_sizes)
        
        # Compare with ground truth
        print("\n" + "="*50)
        print("PREDICTION SUMMARY")
        print("="*50)
        
        severity_names, action_names = get_class_names()
        sev_gt = sample['label_severity'].item()
        act_gt = sample['label_type'].item()
        
        sev_correct = "✓" if severity_pred == sev_gt else "✗"
        act_correct = "✓" if action_pred == act_gt else "✗"
        
        print(f"Severity: {sev_correct} GT: {severity_names[sev_gt]} | Pred: {severity_names[severity_pred]} (conf: {sev_probs[severity_pred]:.3f})")
        print(f"Action:   {act_correct} GT: {action_names[act_gt]} | Pred: {action_names[action_pred]} (conf: {act_probs[action_pred]:.3f})")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

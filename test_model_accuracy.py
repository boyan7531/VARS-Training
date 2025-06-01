import torch
import torch.nn.functional as F
import numpy as np
import warnings
import time
from tqdm import tqdm
warnings.filterwarnings("ignore")
import torchvision.transforms as transforms

# Import model and dataset classes
from model import MultiTaskMultiViewResNet3D, ModelConfig
from dataset import SoccerNetMVFoulDataset, variable_views_collate_fn

def load_test_model(checkpoint_path='best_model_epoch_2.pth'):
    """Load the trained ResNet3D model from checkpoint."""
    print(f"Loading ResNet3D model from {checkpoint_path}...")
    
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
    
    # Create model configuration for ResNet3D
    model_config = ModelConfig(
        use_attention_aggregation=True,
        input_frames=16,
        input_height=224,
        input_width=398  # ResNet3D supports rectangular inputs
    )
    
    # Initialize ResNet3D model
    model = MultiTaskMultiViewResNet3D(
        num_severity=5,
        num_action_type=9,
        vocab_sizes=vocab_sizes,
        backbone_name='resnet3d_18',  # Default to ResNet3D-18
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
            clamped_sample[feature_key] = torch.tensor(clamped_value, dtype=torch.long)
    
    return clamped_sample

def create_test_dataset():
    """Create test dataset with proper transforms."""
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

def run_single_inference(model, sample, vocab_sizes):
    """Run inference on a single sample."""
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
        
        # Get predictions
        severity_pred = torch.argmax(severity_logits, dim=1).item()
        action_pred = torch.argmax(action_logits, dim=1).item()
        
        # Get ground truth
        severity_gt = sample['label_severity'].item()
        action_gt = sample['label_type'].item()
        
        # Get probabilities for confidence
        severity_probs = F.softmax(severity_logits, dim=1)
        action_probs = F.softmax(action_logits, dim=1)
        
        severity_confidence = severity_probs[0, severity_pred].item()
        action_confidence = action_probs[0, action_pred].item()
        
        return {
            'severity_pred': severity_pred,
            'severity_gt': severity_gt,
            'severity_correct': severity_pred == severity_gt,
            'severity_confidence': severity_confidence,
            'action_pred': action_pred,
            'action_gt': action_gt,
            'action_correct': action_pred == action_gt,
            'action_confidence': action_confidence
        }

def calculate_metrics(results):
    """Calculate comprehensive metrics from results."""
    total_samples = len(results)
    
    # Basic accuracies
    severity_correct = sum(r['severity_correct'] for r in results)
    action_correct = sum(r['action_correct'] for r in results)
    both_correct = sum(r['severity_correct'] and r['action_correct'] for r in results)
    
    severity_accuracy = severity_correct / total_samples
    action_accuracy = action_correct / total_samples
    combined_accuracy = both_correct / total_samples
    
    # Confidence statistics
    severity_confidences = [r['severity_confidence'] for r in results]
    action_confidences = [r['action_confidence'] for r in results]
    
    # Per-class accuracy
    severity_per_class = {}
    action_per_class = {}
    
    severity_names, action_names = get_class_names()
    
    for i in range(5):  # 5 severity classes
        class_results = [r for r in results if r['severity_gt'] == i]
        if class_results:
            severity_per_class[i] = {
                'accuracy': sum(r['severity_correct'] for r in class_results) / len(class_results),
                'count': len(class_results),
                'name': severity_names[i]
            }
    
    for i in range(9):  # 9 action classes
        class_results = [r for r in results if r['action_gt'] == i]
        if class_results:
            action_per_class[i] = {
                'accuracy': sum(r['action_correct'] for r in class_results) / len(class_results),
                'count': len(class_results),
                'name': action_names[i]
            }
    
    return {
        'total_samples': total_samples,
        'severity_accuracy': severity_accuracy,
        'action_accuracy': action_accuracy,
        'combined_accuracy': combined_accuracy,
        'severity_avg_confidence': np.mean(severity_confidences),
        'action_avg_confidence': np.mean(action_confidences),
        'severity_per_class': severity_per_class,
        'action_per_class': action_per_class
    }

def print_detailed_results(metrics):
    """Print comprehensive results."""
    print("\n" + "="*70)
    print("MODEL ACCURACY EVALUATION RESULTS")
    print("="*70)
    
    print(f"Total test samples: {metrics['total_samples']}")
    print(f"\nOverall Accuracy:")
    print(f"  Severity:  {metrics['severity_accuracy']:.3f} ({metrics['severity_accuracy']*100:.1f}%)")
    print(f"  Action:    {metrics['action_accuracy']:.3f} ({metrics['action_accuracy']*100:.1f}%)")
    print(f"  Combined:  {metrics['combined_accuracy']:.3f} ({metrics['combined_accuracy']*100:.1f}%)")
    
    print(f"\nAverage Confidence:")
    print(f"  Severity:  {metrics['severity_avg_confidence']:.3f}")
    print(f"  Action:    {metrics['action_avg_confidence']:.3f}")
    
    print(f"\nPer-Class Severity Accuracy:")
    for class_id, data in metrics['severity_per_class'].items():
        print(f"  {data['name']}: {data['accuracy']:.3f} ({data['accuracy']*100:.1f}%) [{data['count']} samples]")
    
    print(f"\nPer-Class Action Accuracy:")
    for class_id, data in metrics['action_per_class'].items():
        print(f"  {data['name']}: {data['accuracy']:.3f} ({data['accuracy']*100:.1f}%) [{data['count']} samples]")

def test_model_accuracy(max_samples=None):
    """Test model accuracy on the test set."""
    print("Starting comprehensive model accuracy evaluation...")
    
    # Load model and dataset
    model, vocab_sizes = load_test_model()
    test_dataset = create_test_dataset()
    
    # Determine number of samples to test
    total_samples = len(test_dataset)
    if max_samples is not None:
        num_samples = min(max_samples, total_samples)
        print(f"Testing on {num_samples}/{total_samples} samples...")
    else:
        num_samples = total_samples
        print(f"Testing on all {num_samples} samples...")
    
    # Run inference on all samples
    results = []
    start_time = time.time()
    
    for i in tqdm(range(num_samples), desc="Running inference"):
        try:
            sample = test_dataset[i]
            result = run_single_inference(model, sample, vocab_sizes)
            result['sample_id'] = i
            results.append(result)
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            continue
    
    elapsed_time = time.time() - start_time
    
    if not results:
        print("No successful predictions made!")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print results
    print_detailed_results(metrics)
    
    print(f"\nTesting completed in {elapsed_time:.2f} seconds")
    print(f"Average time per sample: {elapsed_time/len(results):.3f} seconds")
    
    return results, metrics

if __name__ == "__main__":
    # Test on first 50 samples for quick evaluation, or all samples
    # Change max_samples=None to test on full dataset
    results, metrics = test_model_accuracy(max_samples=50) 
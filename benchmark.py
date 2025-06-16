#!/usr/bin/env python3
"""
Benchmark Script for VARS Model Testing
Generates predictions on test set in the required JSON format.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop
from pathlib import Path
import argparse
import logging
import json
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Imports from our other files
from dataset import SoccerNetMVFoulDataset, variable_views_collate_fn
from model import MultiTaskMultiViewResNet3D, ModelConfig
from pytorchvideo.transforms import ShortSideScale, Normalize as VideoNormalize

# Define transforms locally
class ConvertToFloatAndScale(torch.nn.Module):
    """Converts a uint8 video tensor (C, T, H, W) from [0, 255] to float32 [0, 1]."""
    def __call__(self, clip_cthw_uint8):
        if clip_cthw_uint8.dtype != torch.uint8:
            return clip_cthw_uint8
        return clip_cthw_uint8.float() / 255.0

class PerFrameCenterCrop(torch.nn.Module):
    """Applies CenterCrop to each frame of a (C, T, H, W) video tensor."""
    def __init__(self, size):
        super().__init__()
        self.cropper = CenterCrop(size)

    def forward(self, clip_cthw):
        clip_tchw = clip_cthw.permute(1, 0, 2, 3)
        cropped_frames = [self.cropper(frame) for frame in clip_tchw]
        cropped_clip_tchw = torch.stack(cropped_frames)
        return cropped_clip_tchw.permute(1, 0, 2, 3)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Label mappings
SEVERITY_LABELS = ["", "1.0", "2.0", "3.0", "4.0", "5.0"]
ACTION_TYPE_LABELS = ["", "Challenge", "Dive", "Dont know", "Elbowing", "High leg", 
                     "Holding", "Pushing", "Standing tackling", "Tackling"]

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark VARS Model on Test Set')
    parser.add_argument('--checkpoint_path', type=str, default='best_model_epoch_20.pth', 
                        help='Path to model checkpoint')
    parser.add_argument('--dataset_root', type=str, default="/workspace/VARS-Training/", 
                        help='Root directory containing the mvfouls folder')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'challenge'], 
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--backbone_type', type=str, default='resnet3d', choices=['resnet3d', 'mvit'],
                        help='Backbone type: resnet3d or mvit')
    parser.add_argument('--backbone_name', type=str, default='r2plus1d_18', 
                        help="Backbone model name (e.g., 'r2plus1d_18' for ResNet3D or 'mvit_base_16x4' for MViT)")
    parser.add_argument('--frames_per_clip', type=int, default=16, help='Number of frames per clip')
    parser.add_argument('--target_fps', type=int, default=15, help='Target FPS for clips')
    parser.add_argument('--start_frame', type=int, default=67, help='Start frame index')
    parser.add_argument('--end_frame', type=int, default=82, help='End frame index')
    parser.add_argument('--img_height', type=int, default=224, help='Target image height')
    parser.add_argument('--img_width', type=int, default=398, help='Target image width')
    parser.add_argument('--max_views', type=int, default=None, help='Optional limit on max views')
    parser.add_argument('--attention_aggregation', action='store_true', default=True, 
                        help='Use attention for view aggregation')
    parser.add_argument('--output_file', type=str, default='benchmark_results.json', 
                        help='Output JSON file name')
    
    args = parser.parse_args()
    
    # Auto-detect backbone type from checkpoint if not specified
    if args.backbone_type == 'resnet3d' and 'mvit' in args.checkpoint_path.lower():
        args.backbone_type = 'mvit'
        args.backbone_name = 'mvit_base_16x4'  # Default MViT model
        logger.info("Auto-detected MViT model from checkpoint path")
    
    # Adjust dimensions for MViT models (they require square inputs)
    if args.backbone_type == 'mvit':
        args.img_width = args.img_height  # Force square dimensions for MViT
        logger.info(f"Adjusted dimensions for MViT: {args.img_height}x{args.img_width}")
    
    # Construct the specific mvfouls path from the root
    if not args.dataset_root:
        raise ValueError("Please provide the --dataset_root argument.")
    args.mvfouls_path = str(Path(args.dataset_root) / "mvfouls")
    
    return args

def load_model_checkpoint(checkpoint_path, device):
    """Load model from checkpoint and extract vocab sizes"""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract metadata
    metrics = checkpoint.get('metrics', {})
    epoch = checkpoint.get('epoch', 'unknown')
    has_ema = checkpoint.get('has_ema', False)
    
    logger.info(f"Checkpoint from epoch: {epoch}")
    if 'best_val_acc' in metrics:
        logger.info(f"Best validation accuracy: {metrics['best_val_acc']:.4f}")
    
    # Use EMA weights if available (they typically perform better)
    if has_ema and 'ema_state_dict' in checkpoint:
        logger.info("✅ Using EMA weights for benchmark (better performance)")
        state_dict = checkpoint['ema_state_dict']
    else:
        logger.info("Using online model weights for benchmark")
        state_dict = checkpoint['model_state_dict']
    
    vocab_sizes = {}
    
    # Check if embeddings exist in the state dict
    embedding_keys = {
        'contact': ['embedding_manager.original_embeddings.contact.weight', 'module.embedding_manager.original_embeddings.contact.weight'],
        'bodypart': ['embedding_manager.original_embeddings.bodypart.weight', 'module.embedding_manager.original_embeddings.bodypart.weight'],
        'upper_bodypart': ['embedding_manager.original_embeddings.upper_bodypart.weight', 'module.embedding_manager.original_embeddings.upper_bodypart.weight'],
        'multiple_fouls': ['embedding_manager.original_embeddings.multiple_fouls.weight', 'module.embedding_manager.original_embeddings.multiple_fouls.weight'],
        'try_to_play': ['embedding_manager.original_embeddings.try_to_play.weight', 'module.embedding_manager.original_embeddings.try_to_play.weight'],
        'touch_ball': ['embedding_manager.original_embeddings.touch_ball.weight', 'module.embedding_manager.original_embeddings.touch_ball.weight'],
        'handball': ['embedding_manager.original_embeddings.handball.weight', 'module.embedding_manager.original_embeddings.handball.weight'],
        'handball_offence': ['embedding_manager.original_embeddings.handball_offence.weight', 'module.embedding_manager.original_embeddings.handball_offence.weight']
    }
    
    # Try to extract vocab sizes from model state dict
    for field, possible_keys in embedding_keys.items():
        found = False
        for weight_key in possible_keys:
            if weight_key in state_dict:
                vocab_sizes[field] = state_dict[weight_key].shape[0]
                found = True
                break
        
        if not found:
            logger.warning(f"Could not find embedding for {field} in checkpoint")
            # Use reasonable defaults based on common vocab sizes
            default_sizes = {
                'contact': 3, 'bodypart': 4, 'upper_bodypart': 3, 'multiple_fouls': 2, 'try_to_play': 3, 'touch_ball': 4, 
                'handball': 3, 'handball_offence': 3
            }
            vocab_sizes[field] = default_sizes.get(field, 3)
            logger.warning(f"Using default vocab size {vocab_sizes[field]} for {field}")
    
    logger.info(f"Extracted vocab sizes from checkpoint: {vocab_sizes}")
    
    return checkpoint, metrics, epoch, vocab_sizes

def predict_offence_from_severity(severity_logits):
    """
    Determine offence and severity based on model predictions.
    Class 0 (empty) = No offence, Classes 1-5 = Offence with severity.
    """
    # Apply softmax to get probabilities
    severity_probs = torch.softmax(severity_logits, dim=1)
    
    # Get predicted class
    _, severity_pred = torch.max(severity_probs, dim=1)
    
    # Class 0 = No offence, Classes 1-5 = Offence
    has_offence = severity_pred > 0
    
    return has_offence, severity_pred

def clamp_vocab_indices(batch_data, vocab_sizes):
    """Clamp vocabulary indices to valid ranges to prevent validation errors."""
    # Mapping of feature names to their vocab sizes
    vocab_mappings = {
        'contact_idx': vocab_sizes['contact'],
        'bodypart_idx': vocab_sizes['bodypart'],
        'upper_bodypart_idx': vocab_sizes['upper_bodypart'],
        'multiple_fouls_idx': vocab_sizes['multiple_fouls'],
        'try_to_play_idx': vocab_sizes['try_to_play'],
        'touch_ball_idx': vocab_sizes['touch_ball'],
        'handball_idx': vocab_sizes['handball'],
        'handball_offence_idx': vocab_sizes['handball_offence']
    }
    
    for feature_key, max_vocab_size in vocab_mappings.items():
        if feature_key in batch_data and isinstance(batch_data[feature_key], torch.Tensor):
            original_tensor = batch_data[feature_key]
            # Clamp to valid range [0, max_size-1]
            clamped_tensor = torch.clamp(original_tensor, 0, max_vocab_size - 1)
            
            # Check if any values were clamped
            if not torch.equal(original_tensor, clamped_tensor):
                num_clamped = (original_tensor != clamped_tensor).sum().item()
                logger.warning(f"Clamped {num_clamped} indices in {feature_key} to fit vocab size {max_vocab_size}")
            
            batch_data[feature_key] = clamped_tensor
    
    return batch_data

def run_benchmark(model, dataloader, device, vocab_sizes):
    """Run model inference and generate predictions"""
    model.eval()
    
    all_predictions = {}
    action_counter = 0
    
    # Add detailed diagnostics storage
    all_severity_probs = []
    all_severity_preds = []
    all_action_probs = []
    all_action_preds = []
    
    logger.info("Starting benchmark evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing batches")):
            try:
                # Move data to device
                for key in batch_data:
                    if isinstance(batch_data[key], torch.Tensor):
                        batch_data[key] = batch_data[key].to(device, non_blocking=True)
                
                # Clamp vocabulary indices to prevent out-of-bounds errors
                batch_data = clamp_vocab_indices(batch_data, vocab_sizes)
                
                # Forward pass
                sev_logits, act_logits = model(batch_data)
                
                # Get action type predictions
                act_probs = torch.softmax(act_logits, dim=1)
                _, act_pred = torch.max(act_probs, dim=1)
                
                # Determine offence and severity
                severity_probs = torch.softmax(sev_logits, dim=1)
                _, sev_pred = torch.max(severity_probs, dim=1)
                has_offence = sev_pred > 0
                
                # Store all predictions for analysis
                all_severity_probs.append(severity_probs.cpu())
                all_severity_preds.append(sev_pred.cpu())
                all_action_probs.append(act_probs.cpu())
                all_action_preds.append(act_pred.cpu())
                
                # Debug logging for first few batches
                if batch_idx < 3:
                    logger.info(f"Batch {batch_idx} sample predictions:")
                    for i in range(min(2, sev_logits.size(0))):
                        logger.info(f"  Sample {i}: sev_pred={sev_pred[i].item()}, has_offence={has_offence[i].item()}")
                        logger.info(f"    Severity probs: {severity_probs[i].cpu().numpy()}")
                        logger.info(f"    Action probs: {act_probs[i].cpu().numpy()}")
                
                # Process each sample in the batch
                batch_size = sev_logits.size(0)
                for i in range(batch_size):
                    action_id = str(action_counter)
                    
                    # Get predictions for this sample
                    action_class = ACTION_TYPE_LABELS[act_pred[i].item()]
                    offence = "Offence" if has_offence[i].item() else "No offence"
                    severity = SEVERITY_LABELS[sev_pred[i].item()] if has_offence[i].item() else ""
                    
                    # Store prediction with confidence scores
                    all_predictions[action_id] = {
                        "Action class": action_class,
                        "Offence": offence,
                        "Severity": severity,
                        "Severity_confidence": float(severity_probs[i, sev_pred[i]].item()),
                        "Action_confidence": float(act_probs[i, act_pred[i]].item())
                    }
                    
                    action_counter += 1
                
                # Log progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches, {action_counter} actions")
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                logger.warning("Skipping this batch and continuing...")
                continue
    
    logger.info(f"Benchmark completed! Processed {action_counter} actions total.")
    
    # Combine all predictions for analysis
    if all_severity_probs:
        all_severity_probs = torch.cat(all_severity_probs, dim=0)
        all_severity_preds = torch.cat(all_severity_preds, dim=0)
        all_action_probs = torch.cat(all_action_probs, dim=0)
        all_action_preds = torch.cat(all_action_preds, dim=0)
        
        # Run diagnostics
        analyze_prediction_bias(all_severity_probs, all_severity_preds, all_action_probs, all_action_preds, model)
    
    return all_predictions

def analyze_prediction_bias(severity_probs, severity_preds, action_probs, action_preds, model):
    """Analyze potential biases in model predictions"""
    logger.info("="*60)
    logger.info("DETAILED PREDICTION ANALYSIS")
    logger.info("="*60)
    
    # Overall statistics
    total_samples = len(severity_preds)
    logger.info(f"Total samples analyzed: {total_samples}")
    
    # Severity class distribution
    sev_class_counts = {}
    for i in range(len(SEVERITY_LABELS)):
        count = (severity_preds == i).sum().item()
        percentage = 100 * count / total_samples
        sev_class_counts[i] = count
        logger.info(f"Severity class {i} ({SEVERITY_LABELS[i]}): {count} samples ({percentage:.1f}%)")
    
    # Check for high confidence in majority class
    sev_confidences = torch.gather(severity_probs, 1, severity_preds.unsqueeze(1)).squeeze(1)
    high_conf_mask = sev_confidences > 0.9
    high_conf_count = high_conf_mask.sum().item()
    
    if high_conf_count > 0:
        logger.info(f"High confidence predictions (>90%): {high_conf_count} ({100*high_conf_count/total_samples:.1f}%)")
        
        # Check which classes have high confidence
        for i in range(len(SEVERITY_LABELS)):
            class_high_conf = (severity_preds == i) & high_conf_mask
            count = class_high_conf.sum().item()
            if count > 0:
                logger.info(f"  Class {i} ({SEVERITY_LABELS[i]}) high confidence: {count} ({100*count/high_conf_count:.1f}% of high conf)")
    
    # Analyze severity distribution details
    logger.info("\nSeverity probability distribution statistics:")
    mean_probs = severity_probs.mean(dim=0)
    logger.info(f"Mean probabilities across all samples:")
    for i, prob in enumerate(mean_probs):
        logger.info(f"  Class {i} ({SEVERITY_LABELS[i]}): {prob:.4f}")
    
    # Check for separation between top predictions
    top_probs, top_classes = torch.topk(severity_probs, k=2, dim=1)
    confidence_gaps = top_probs[:, 0] - top_probs[:, 1]
    mean_gap = confidence_gaps.mean().item()
    logger.info(f"\nMean confidence gap between top 2 severity predictions: {mean_gap:.4f}")
    
    # Add class confusion analysis
    logger.info("\nClass confusion analysis:")
    for i in range(len(SEVERITY_LABELS)):
        if sev_class_counts[i] == 0:
            continue
            
        # For samples predicted as class i, what was the second highest prediction?
        class_i_mask = severity_preds == i
        class_i_probs = severity_probs[class_i_mask]
        
        if len(class_i_probs) > 0:
            # Zero out the predicted class to find the second highest
            second_probs = class_i_probs.clone()
            second_probs[:, i] = 0
            _, second_highest = torch.max(second_probs, dim=1)
            
            # Count occurrences of each second highest class
            second_counts = {}
            for j in range(len(SEVERITY_LABELS)):
                count = (second_highest == j).sum().item()
                if count > 0:
                    percentage = 100 * count / len(second_highest)
                    second_counts[j] = (count, percentage)
            
            logger.info(f"For samples predicted as class {i} ({SEVERITY_LABELS[i]}):")
            for j, (count, percentage) in sorted(second_counts.items(), key=lambda x: x[1][0], reverse=True):
                logger.info(f"  Second highest prediction: class {j} ({SEVERITY_LABELS[j]}) - {count} samples ({percentage:.1f}%)")
    
    logger.info("\nFurther diagnostics:")
    # Check if model is making very certain predictions
    avg_entropy = -torch.sum(severity_probs * torch.log(severity_probs + 1e-10), dim=1).mean().item()
    max_entropy = -torch.log(torch.tensor(1.0/len(SEVERITY_LABELS)))
    entropy_ratio = avg_entropy / max_entropy
    
    logger.info(f"Average prediction entropy: {avg_entropy:.4f} (ratio to max entropy: {entropy_ratio:.4f})")
    if entropy_ratio < 0.5:
        logger.info("INSIGHT: Model is making very confident predictions (low entropy), may indicate overfitting or overconfidence")
    elif entropy_ratio > 0.8:
        logger.info("INSIGHT: Model is making uncertain predictions (high entropy), may be underfitting or confused")
    
    # Check if predictions are balanced or skewed
    predominant_class = severity_preds.mode().values.item()
    predominant_count = (severity_preds == predominant_class).sum().item()
    predominant_ratio = predominant_count / total_samples
    
    logger.info(f"Predominant class: {predominant_class} ({SEVERITY_LABELS[predominant_class]}) with {predominant_ratio*100:.1f}% of predictions")
    if predominant_ratio > 0.7:
        logger.info("INSIGHT: Model predictions are heavily skewed toward one class, may indicate bias or class imbalance issues")
        logger.info("SUGGESTION: Consider checking model weights, specifically the final layer bias terms for the severity head")
        
    # Output detailed suggestions based on findings
    logger.info("\nDIAGNOSTIC SUGGESTIONS:")
    if predominant_ratio > 0.7 and entropy_ratio < 0.5:
        logger.info("1. The model appears to be strongly biased toward one class with high confidence")
        logger.info("   - Verify class balancing during training")
        logger.info("   - Check if class weights were applied correctly")
        logger.info("   - Examine data quality for the underrepresented classes")
        logger.info("   - Try modifying the final layer bias values directly to calibrate predictions")
    elif predominant_ratio > 0.7:
        logger.info("1. The model predictions are skewed toward one class, but with moderate confidence")
        logger.info("   - Review training process for class balancing effectiveness")
        logger.info("   - Consider adjusting class weights or focal loss parameters")
    
    if sev_class_counts.get(1, 0) < sev_class_counts.get(2, 0):
        logger.info("2. Model is predicting more severity 2.0 than 1.0, which is unexpected based on class distribution")
        logger.info("   - Check if class mapping is consistent between training and inference")
        logger.info("   - Verify that label indices match between dataset and benchmark code")
        logger.info("   - Consider examining the model's decision boundary between classes 1 and 2")
        
    logger.info("\nEXAMINING MODEL WEIGHTS (if available):")
    try:
        # If we have access to the model, inspect its weights directly
        if hasattr(model, 'severity_head') and isinstance(model.severity_head[-1], torch.nn.Linear):
            final_layer = model.severity_head[-1]
            weights = final_layer.weight.detach().cpu()
            biases = final_layer.bias.detach().cpu()
            
            logger.info(f"Severity head final layer shape: weights {weights.shape}, biases {biases.shape}")
            logger.info("Final layer biases:")
            for i, bias in enumerate(biases):
                logger.info(f"  Class {i} ({SEVERITY_LABELS[i]}) bias: {bias.item():.4f}")
            
            # Check for bias imbalance
            max_bias_idx = torch.argmax(biases).item()
            min_bias_idx = torch.argmin(biases).item()
            bias_range = biases.max().item() - biases.min().item()
            
            logger.info(f"Bias range: {bias_range:.4f}")
            logger.info(f"Max bias: Class {max_bias_idx} ({SEVERITY_LABELS[max_bias_idx]}): {biases[max_bias_idx].item():.4f}")
            logger.info(f"Min bias: Class {min_bias_idx} ({SEVERITY_LABELS[min_bias_idx]}): {biases[min_bias_idx].item():.4f}")
            
            # If there's a big difference, flag it
            if bias_range > 1.0:
                logger.info("POTENTIAL ISSUE: Large bias difference between classes detected")
                logger.info("This could cause the model to favor certain classes regardless of input")
                
                # If the predominantly predicted class has highest bias
                if max_bias_idx == predominant_class:
                    logger.info("The class with highest bias matches the predominantly predicted class")
                    logger.info("SUGGESTION: Try reducing the bias term for class " + 
                                f"{max_bias_idx} ({SEVERITY_LABELS[max_bias_idx]}) to calibrate predictions")
    except Exception as e:
        logger.warning(f"Could not examine model weights: {e}")

def save_benchmark_results(predictions, output_file, split="test"):
    """Save predictions in the required JSON format"""
    
    # Create the final result structure
    results = {
        "Set": split,
        "Actions": predictions
    }
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Benchmark results saved to: {output_file}")
    
    # Print some statistics
    total_actions = len(predictions)
    offence_count = sum(1 for pred in predictions.values() if pred["Offence"] == "Offence")
    no_offence_count = total_actions - offence_count
    
    # Severity distribution
    severity_dist = {}
    for pred in predictions.values():
        if pred["Severity"]:
            severity = pred["Severity"]
            severity_dist[severity] = severity_dist.get(severity, 0) + 1
    
    # Action class distribution
    action_dist = {}
    for pred in predictions.values():
        action = pred["Action class"]
        action_dist[action] = action_dist.get(action, 0) + 1
    
    logger.info("="*60)
    logger.info("BENCHMARK RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total actions processed: {total_actions}")
    logger.info(f"Offences detected: {offence_count} ({offence_count/total_actions*100:.1f}%)")
    logger.info(f"No offences: {no_offence_count} ({no_offence_count/total_actions*100:.1f}%)")
    logger.info("")
    logger.info("Severity distribution:")
    for severity, count in sorted(severity_dist.items()):
        logger.info(f"  {severity}: {count} ({count/offence_count*100:.1f}% of offences)")
    logger.info("")
    logger.info("Most common action classes:")
    sorted_actions = sorted(action_dist.items(), key=lambda x: x[1], reverse=True)
    for action, count in sorted_actions[:5]:
        logger.info(f"  {action}: {count} ({count/total_actions*100:.1f}%)")

def main():
    args = parse_args()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint, metrics, epoch, vocab_sizes = load_model_checkpoint(args.checkpoint_path, device)
    
    # Create test transforms (deterministic)
    test_transform = Compose([
        ConvertToFloatAndScale(),
        VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ShortSideScale(size=args.img_height),
        PerFrameCenterCrop((args.img_height, args.img_width))
    ])
    
    # Load test dataset
    logger.info(f"Loading {args.split} dataset...")
    try:
        test_dataset = SoccerNetMVFoulDataset(
            dataset_path=args.mvfouls_path,
            split=args.split,
            frames_per_clip=args.frames_per_clip,
            target_fps=args.target_fps,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            max_views_to_load=args.max_views,
            transform=test_transform,
            target_height=args.img_height,
            target_width=args.img_width
        )
        
        logger.info(f"Test samples: {len(test_dataset)}")
        
        if len(test_dataset) == 0:
            logger.error("Test dataset is empty after loading.")
            return
            
    except FileNotFoundError as e:
        logger.error(f"Error loading test dataset: {e}")
        logger.error(f"Make sure the dataset path is correct: {args.mvfouls_path}")
        logger.error(f"And that the {args.split} split exists with annotations.json")
        return
    except Exception as e:
        logger.error(f"Unexpected error loading test dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Important: don't shuffle for consistent ordering
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=variable_views_collate_fn
    )
    
    # Model configuration
    model_config = ModelConfig(
        use_attention_aggregation=args.attention_aggregation,
        input_frames=args.frames_per_clip,
        input_height=args.img_height,
        input_width=args.img_width
    )
    
    # Initialize model using unified interface
    logger.info(f"Initializing {args.backbone_type.upper()} model: {args.backbone_name}")
    try:
        # Import the unified model interface
        from model import create_unified_model
        
        # Create config dict without conflicting parameters
        config_kwargs = {k: v for k, v in model_config.__dict__.items() 
                        if k not in ['use_attention_aggregation', 'pretrained_model_name']}
        
        model = create_unified_model(
            backbone_type=args.backbone_type,
            num_severity=6,  # 6 severity classes: "", 1.0, 2.0, 3.0, 4.0, 5.0
            num_action_type=10,  # 10 action types
            vocab_sizes=vocab_sizes,
            backbone_name=args.backbone_name,
            use_attention_aggregation=args.attention_aggregation,
            use_augmentation=False,  # Disable augmentation for inference
            disable_in_model_augmentation=True,
            dropout_rate=getattr(args, 'dropout_rate', 0.1),
            **config_kwargs
        )
        logger.info("Model initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        logger.error("This might be due to vocab size mismatches or unsupported backbone")
        return
    
    # Load model weights with DataParallel compatibility
    try:
        state_dict = checkpoint['model_state_dict']
        model_state_dict = model.state_dict()
        
        # Handle DataParallel state dict key mismatch
        model_keys = list(model_state_dict.keys())
        checkpoint_keys = list(state_dict.keys())
        
        if len(model_keys) > 0 and len(checkpoint_keys) > 0:
            model_has_module = model_keys[0].startswith('module.')
            checkpoint_has_module = checkpoint_keys[0].startswith('module.')
            
            if model_has_module and not checkpoint_has_module:
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
                logger.info("Added 'module.' prefix to checkpoint keys for DataParallel compatibility")
            elif not model_has_module and checkpoint_has_module:
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                logger.info("Removed 'module.' prefix from checkpoint keys for DataParallel compatibility")
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully from epoch {epoch}")
        
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        logger.error("This might be due to model architecture mismatch or corrupted checkpoint")
        return
    
    # Run benchmark
    predictions = run_benchmark(model, test_loader, device, vocab_sizes)
    
    # Save results
    save_benchmark_results(predictions, args.output_file, args.split)
    
    logger.info("Benchmark completed successfully!")

if __name__ == "__main__":
    main()

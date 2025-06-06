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
    parser.add_argument('--backbone_name', type=str, default='r2plus1d_18', 
                        choices=['resnet3d_18', 'mc3_18', 'r2plus1d_18', 'resnet3d_50'], 
                        help="ResNet3D backbone variant")
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
    
    logger.info(f"Checkpoint from epoch: {epoch}")
    if 'best_val_acc' in metrics:
        logger.info(f"Best validation accuracy: {metrics['best_val_acc']:.4f}")
    
    # Extract vocab sizes from checkpoint state dict
    state_dict = checkpoint['model_state_dict']
    
    vocab_sizes = {}
    
    # Check if embeddings exist in the state dict
    embedding_keys = {
        'contact': ['embedding_manager.original_embeddings.contact.weight', 'module.embedding_manager.original_embeddings.contact.weight'],
        'bodypart': ['embedding_manager.original_embeddings.bodypart.weight', 'module.embedding_manager.original_embeddings.bodypart.weight'],
        'upper_bodypart': ['embedding_manager.original_embeddings.upper_bodypart.weight', 'module.embedding_manager.original_embeddings.upper_bodypart.weight'],
        'lower_bodypart': ['embedding_manager.original_embeddings.lower_bodypart.weight', 'module.embedding_manager.original_embeddings.lower_bodypart.weight'],
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
                'contact': 3, 'bodypart': 4, 'upper_bodypart': 3, 'lower_bodypart': 4,
                'multiple_fouls': 2, 'try_to_play': 3, 'touch_ball': 4, 
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
        'lower_bodypart_idx': vocab_sizes['lower_bodypart'],
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
                _, act_pred = torch.max(act_logits, dim=1)
                
                # Determine offence and severity
                has_offence, sev_pred = predict_offence_from_severity(sev_logits)
                
                # Debug logging for first few batches
                if batch_idx < 3:
                    severity_probs = torch.softmax(sev_logits, dim=1)
                    logger.info(f"Batch {batch_idx} sample predictions:")
                    for i in range(min(2, sev_logits.size(0))):
                        logger.info(f"  Sample {i}: sev_pred={sev_pred[i].item()}, has_offence={has_offence[i].item()}")
                        logger.info(f"    Severity probs: {severity_probs[i].cpu().numpy()}")
                
                # Process each sample in the batch
                batch_size = sev_logits.size(0)
                for i in range(batch_size):
                    action_id = str(action_counter)
                    
                    # Get predictions for this sample
                    action_class = ACTION_TYPE_LABELS[act_pred[i].item()]
                    offence = "Offence" if has_offence[i].item() else "No offence"
                    severity = SEVERITY_LABELS[sev_pred[i].item()] if has_offence[i].item() else ""
                    
                    # Store prediction
                    all_predictions[action_id] = {
                        "Action class": action_class,
                        "Offence": offence,
                        "Severity": severity
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
    return all_predictions

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
    
    # Initialize model
    logger.info(f"Initializing ResNet3D model: {args.backbone_name}")
    try:
        model = MultiTaskMultiViewResNet3D(
            num_severity=6,  # 6 severity classes: "", 1.0, 2.0, 3.0, 4.0, 5.0
            num_action_type=10,  # 10 action types: "", Challenge, Dive, Dont know, Elbowing, High leg, Holding, Pushing, Standing tackling, Tackling
            vocab_sizes=vocab_sizes,
            backbone_name=args.backbone_name,
            config=model_config
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

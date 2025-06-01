#!/usr/bin/env python3
"""
Model Testing Script for VARS (Video Assistant Referee System)
Evaluates trained models on the test dataset with comprehensive metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torchvision.transforms import Compose, CenterCrop
from pathlib import Path
import argparse
import logging
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')
import time
import os

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
        logging.FileHandler('testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Label mappings for interpretability
SEVERITY_LABELS = ["1.0", "2.0", "3.0", "4.0", "5.0"]
ACTION_TYPE_LABELS = ["Challenge", "Dive", "Dont know", "Elbowing", "High leg", 
                     "Holding", "Pushing", "Standing tackling", "Tackling"]

def parse_args():
    parser = argparse.ArgumentParser(description='Test Multi-Task Multi-View ResNet3D for Foul Recognition')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset_root', type=str, default="", help='Root directory containing the mvfouls folder')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'challenge'], help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--backbone_name', type=str, default='resnet3d_18', 
                        choices=['resnet3d_18', 'mc3_18', 'r2plus1d_18', 'resnet3d_50'], 
                        help="ResNet3D backbone variant (r2plus1d_18 recommended for best accuracy)")
    parser.add_argument('--frames_per_clip', type=int, default=16, help='Number of frames per clip')
    parser.add_argument('--target_fps', type=int, default=15, help='Target FPS for clips')
    parser.add_argument('--start_frame', type=int, default=67, help='Start frame index for foul-centered extraction')
    parser.add_argument('--end_frame', type=int, default=82, help='End frame index for foul-centered extraction')
    parser.add_argument('--img_height', type=int, default=224, help='Target image height')
    parser.add_argument('--img_width', type=int, default=398, help='Target image width')
    parser.add_argument('--max_views', type=int, default=None, help='Optional limit on max views per action')
    parser.add_argument('--attention_aggregation', action='store_true', default=True, help='Use attention for view aggregation')
    parser.add_argument('--save_results', action='store_true', help='Save detailed results to file')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision inference')
    
    args = parser.parse_args()
    
    # Construct the specific mvfouls path from the root
    if not args.dataset_root:
        raise ValueError("Please provide the --dataset_root argument.")
    args.mvfouls_path = str(Path(args.dataset_root) / "mvfouls")
    
    return args

def load_model_checkpoint(checkpoint_path, device):
    """Load model from checkpoint and return model with metadata"""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model configuration from checkpoint if available
    metrics = checkpoint.get('metrics', {})
    epoch = checkpoint.get('epoch', 'unknown')
    
    logger.info(f"Checkpoint from epoch: {epoch}")
    if 'best_val_acc' in metrics:
        logger.info(f"Best validation accuracy: {metrics['best_val_acc']:.4f}")
    
    return checkpoint, metrics, epoch

def calculate_detailed_metrics(predictions, labels, class_names):
    """Calculate comprehensive metrics for classification"""
    # Basic accuracy
    accuracy = (predictions == labels).float().mean().item()
    
    # F1 scores
    f1_macro = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='macro', zero_division=0)
    f1_weighted = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='weighted', zero_division=0)
    
    # Per-class metrics
    report = classification_report(
        labels.cpu().numpy(), 
        predictions.cpu().numpy(), 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy())
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': report,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, class_names, title, save_path=None):
    """Plot and optionally save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    plt.close()

def evaluate_model(model, dataloader, device, save_results=False, results_dir=None):
    """Evaluate model on test dataset"""
    model.eval()
    
    all_sev_predictions = []
    all_sev_labels = []
    all_act_predictions = []
    all_act_labels = []
    
    total_samples = 0
    running_sev_loss = 0.0
    running_act_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    logger.info("Starting model evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move data to device
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(device, non_blocking=True)
            
            severity_labels = batch_data["label_severity"]
            action_labels = batch_data["label_type"]
            
            # Forward pass
            sev_logits, act_logits = model(batch_data)
            
            # Calculate losses
            sev_loss = criterion(sev_logits, severity_labels)
            act_loss = criterion(act_logits, action_labels)
            
            running_sev_loss += sev_loss.item() * severity_labels.size(0)
            running_act_loss += act_loss.item() * action_labels.size(0)
            
            # Get predictions
            _, sev_pred = torch.max(sev_logits, 1)
            _, act_pred = torch.max(act_logits, 1)
            
            # Store predictions and labels
            all_sev_predictions.append(sev_pred.cpu())
            all_sev_labels.append(severity_labels.cpu())
            all_act_predictions.append(act_pred.cpu())
            all_act_labels.append(action_labels.cpu())
            
            total_samples += severity_labels.size(0)
    
    # Concatenate all predictions and labels
    all_sev_predictions = torch.cat(all_sev_predictions)
    all_sev_labels = torch.cat(all_sev_labels)
    all_act_predictions = torch.cat(all_act_predictions)
    all_act_labels = torch.cat(all_act_labels)
    
    # Calculate average losses
    avg_sev_loss = running_sev_loss / total_samples
    avg_act_loss = running_act_loss / total_samples
    
    # Calculate detailed metrics
    sev_metrics = calculate_detailed_metrics(all_sev_predictions, all_sev_labels, SEVERITY_LABELS)
    act_metrics = calculate_detailed_metrics(all_act_predictions, all_act_labels, ACTION_TYPE_LABELS)
    
    # Combined accuracy
    combined_accuracy = (sev_metrics['accuracy'] + act_metrics['accuracy']) / 2
    
    results = {
        'total_samples': total_samples,
        'severity': {
            'loss': avg_sev_loss,
            'metrics': sev_metrics
        },
        'action_type': {
            'loss': avg_act_loss,
            'metrics': act_metrics
        },
        'combined_accuracy': combined_accuracy
    }
    
    # Save detailed results if requested
    if save_results and results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        # Save numerical results
        results_file = Path(results_dir) / 'test_results.json'
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'total_samples': total_samples,
                'combined_accuracy': combined_accuracy,
                'severity': {
                    'loss': avg_sev_loss,
                    'accuracy': sev_metrics['accuracy'],
                    'f1_macro': sev_metrics['f1_macro'],
                    'f1_weighted': sev_metrics['f1_weighted']
                },
                'action_type': {
                    'loss': avg_act_loss,
                    'accuracy': act_metrics['accuracy'],
                    'f1_macro': act_metrics['f1_macro'],
                    'f1_weighted': act_metrics['f1_weighted']
                }
            }
            json.dump(json_results, f, indent=2)
        
        # Save confusion matrices
        plot_confusion_matrix(
            sev_metrics['confusion_matrix'], 
            SEVERITY_LABELS, 
            'Severity Classification Confusion Matrix',
            Path(results_dir) / 'severity_confusion_matrix.png'
        )
        
        plot_confusion_matrix(
            act_metrics['confusion_matrix'], 
            ACTION_TYPE_LABELS, 
            'Action Type Classification Confusion Matrix',
            Path(results_dir) / 'action_confusion_matrix.png'
        )
        
        # Save classification reports
        with open(Path(results_dir) / 'severity_classification_report.txt', 'w') as f:
            f.write("Severity Classification Report\n")
            f.write("=" * 50 + "\n")
            f.write(classification_report(all_sev_labels.numpy(), all_sev_predictions.numpy(), 
                                        target_names=SEVERITY_LABELS, zero_division=0))
        
        with open(Path(results_dir) / 'action_classification_report.txt', 'w') as f:
            f.write("Action Type Classification Report\n")
            f.write("=" * 50 + "\n")
            f.write(classification_report(all_act_labels.numpy(), all_act_predictions.numpy(), 
                                        target_names=ACTION_TYPE_LABELS, zero_division=0))
    
    return results

def print_results(results, save_to_file=False, results_dir=None, checkpoint_path=None, epoch=None):
    """Print comprehensive test results and optionally save to file"""
    
    # Create the report content
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("VARS MODEL EVALUATION RESULTS")
    report_lines.append("="*80)
    
    if checkpoint_path:
        report_lines.append(f"Checkpoint: {checkpoint_path}")
    if epoch:
        report_lines.append(f"Model Epoch: {epoch}")
    report_lines.append("")
    
    report_lines.append(f"Total test samples: {results['total_samples']}")
    report_lines.append(f"Combined accuracy: {results['combined_accuracy']:.4f}")
    report_lines.append("")
    
    # Severity results
    sev_metrics = results['severity']['metrics']
    report_lines.append("SEVERITY CLASSIFICATION:")
    report_lines.append(f"  Loss: {results['severity']['loss']:.4f}")
    report_lines.append(f"  Accuracy: {sev_metrics['accuracy']:.4f}")
    report_lines.append(f"  F1 (Macro): {sev_metrics['f1_macro']:.4f}")
    report_lines.append(f"  F1 (Weighted): {sev_metrics['f1_weighted']:.4f}")
    report_lines.append("")
    
    # Action type results
    act_metrics = results['action_type']['metrics']
    report_lines.append("ACTION TYPE CLASSIFICATION:")
    report_lines.append(f"  Loss: {results['action_type']['loss']:.4f}")
    report_lines.append(f"  Accuracy: {act_metrics['accuracy']:.4f}")
    report_lines.append(f"  F1 (Macro): {act_metrics['f1_macro']:.4f}")
    report_lines.append(f"  F1 (Weighted): {act_metrics['f1_weighted']:.4f}")
    report_lines.append("")
    
    # Performance Analysis
    report_lines.append("PERFORMANCE ANALYSIS:")
    if sev_metrics['accuracy'] > 0.7:
        report_lines.append("  ✅ Severity classification: GOOD (>70%)")
    elif sev_metrics['accuracy'] > 0.6:
        report_lines.append("  ⚠️  Severity classification: MODERATE (60-70%)")
    else:
        report_lines.append("  ❌ Severity classification: NEEDS IMPROVEMENT (<60%)")
    
    if act_metrics['accuracy'] > 0.6:
        report_lines.append("  ✅ Action classification: GOOD (>60%)")
    elif act_metrics['accuracy'] > 0.5:
        report_lines.append("  ⚠️  Action classification: MODERATE (50-60%)")
    else:
        report_lines.append("  ❌ Action classification: NEEDS IMPROVEMENT (<50%)")
    
    if results['combined_accuracy'] > 0.65:
        report_lines.append("  🎯 Overall performance: EXCELLENT")
    elif results['combined_accuracy'] > 0.55:
        report_lines.append("  📈 Overall performance: GOOD")
    else:
        report_lines.append("  📉 Overall performance: NEEDS IMPROVEMENT")
    
    report_lines.append("")
    report_lines.append("="*80)
    
    # Print to console
    for line in report_lines:
        logger.info(line)
    
    # Save to file if requested
    if save_to_file and results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        report_file = Path(results_dir) / 'evaluation_summary_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"Summary report saved to: {report_file}")
        
        # Also create a brief summary for quick reference
        brief_summary = [
            "VARS MODEL PERFORMANCE SUMMARY",
            "=" * 40,
            f"Combined Accuracy: {results['combined_accuracy']:.1%}",
            f"Severity Accuracy: {sev_metrics['accuracy']:.1%}",
            f"Action Accuracy: {act_metrics['accuracy']:.1%}",
            f"Test Samples: {results['total_samples']}",
            ""
        ]
        
        if checkpoint_path:
            brief_summary.insert(-1, f"Model: {Path(checkpoint_path).name}")
        if epoch:
            brief_summary.insert(-1, f"Epoch: {epoch}")
        
        brief_file = Path(results_dir) / 'performance_summary.txt'
        with open(brief_file, 'w') as f:
            f.write("\n".join(brief_summary))
        
        logger.info(f"Brief summary saved to: {brief_file}")

def main():
    args = parse_args()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint, metrics, epoch = load_model_checkpoint(args.checkpoint_path, device)
    
    # Create test transforms (deterministic, optimized for ResNet3D)
    test_transform = Compose([
        ConvertToFloatAndScale(),
        VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ShortSideScale(size=args.img_height),
        PerFrameCenterCrop((args.img_height, args.img_width))  # Rectangular crop for ResNet3D
    ])
    
    # Load test dataset
    logger.info("Loading test dataset...")
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
    except FileNotFoundError as e:
        logger.error(f"Error loading test dataset: {e}")
        return
    
    if len(test_dataset) == 0:
        logger.error("Test dataset is empty after loading.")
        return
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=variable_views_collate_fn
    )
    
    # Create vocabulary sizes dictionary
    vocab_sizes = {
        'contact': test_dataset.num_contact_classes,
        'bodypart': test_dataset.num_bodypart_classes,
        'upper_bodypart': test_dataset.num_upper_bodypart_classes,
        'lower_bodypart': test_dataset.num_lower_bodypart_classes,
        'multiple_fouls': test_dataset.num_multiple_fouls_classes,
        'try_to_play': test_dataset.num_try_to_play_classes,
        'touch_ball': test_dataset.num_touch_ball_classes,
        'handball': test_dataset.num_handball_classes,
        'handball_offence': test_dataset.num_handball_offence_classes,
    }
    
    # Model configuration
    model_config = ModelConfig(
        use_attention_aggregation=args.attention_aggregation,
        input_frames=args.frames_per_clip,
        input_height=args.img_height,
        input_width=args.img_width  # ResNet3D supports rectangular inputs
    )
    
    # Initialize model
    logger.info(f"Initializing ResNet3D model: {args.backbone_name}")
    model = MultiTaskMultiViewResNet3D(
        num_severity=5,
        num_action_type=9,
        vocab_sizes=vocab_sizes,
        backbone_name=args.backbone_name,
        config=model_config
    )
    
    # Load model weights with DataParallel compatibility
    state_dict = checkpoint['model_state_dict']
    model_state_dict = model.state_dict()
    
    # Check if we need to add or remove 'module.' prefix
    model_keys = list(model_state_dict.keys())
    checkpoint_keys = list(state_dict.keys())
    
    if len(model_keys) > 0 and len(checkpoint_keys) > 0:
        model_has_module = model_keys[0].startswith('module.')
        checkpoint_has_module = checkpoint_keys[0].startswith('module.')
        
        if model_has_module and not checkpoint_has_module:
            # Model is DataParallel wrapped, checkpoint is not - add 'module.' prefix
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            logger.info("Added 'module.' prefix to checkpoint keys for DataParallel compatibility")
        elif not model_has_module and checkpoint_has_module:
            # Model is not DataParallel wrapped, checkpoint is - remove 'module.' prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            logger.info("Removed 'module.' prefix from checkpoint keys for DataParallel compatibility")
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully from epoch {epoch}")
    
    # Evaluate model
    results = evaluate_model(
        model, 
        test_loader, 
        device, 
        save_results=args.save_results,
        results_dir=args.results_dir if args.save_results else None
    )
    
    # Print results and save to file
    print_results(
        results, 
        save_to_file=args.save_results,
        results_dir=args.results_dir if args.save_results else None,
        checkpoint_path=args.checkpoint_path,
        epoch=epoch
    )
    
    if args.save_results:
        logger.info(f"All results saved to: {args.results_dir}")

if __name__ == "__main__":
    main()

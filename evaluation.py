#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script for VARS Training

This script evaluates a trained model on the test dataset and provides:
- Overall accuracy metrics
- Per-class accuracy and F1 scores
- Confusion matrices
- Detailed prediction analysis
- Class distribution comparison
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import argparse
import logging
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Imports from our codebase
from dataset import SoccerNetMVFoulDataset, variable_views_collate_fn, SEVERITY_LABELS, ACTION_TYPE_LABELS
from model import create_unified_model
from model.config import ModelConfig
from training.data import create_transforms

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained VARS model on test dataset')
    
    # Model and data paths
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Root directory containing the mvfouls folder')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'valid', 'challenge'],
                        help='Dataset split to evaluate on')
    
    # Model configuration
    parser.add_argument('--backbone_type', type=str, default='mvit', choices=['resnet3d', 'mvit'],
                        help='Backbone architecture type')
    parser.add_argument('--backbone_name', type=str, default='mvit_base_16x4',
                        help='Specific backbone model name')
    parser.add_argument('--frames_per_clip', type=int, default=16,
                        help='Number of frames per clip')
    parser.add_argument('--target_fps', type=int, default=15,
                        help='Target FPS for clips')
    parser.add_argument('--start_frame', type=int, default=67,
                        help='Start frame index')
    parser.add_argument('--end_frame', type=int, default=82,
                        help='End frame index')
    parser.add_argument('--img_height', type=int, default=224,
                        help='Target image height')
    parser.add_argument('--img_width', type=int, default=224,
                        help='Target image width')
    parser.add_argument('--max_views', type=int, default=None,
                        help='Maximum number of views to use')
    
    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for DataLoader')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use EMA weights if available in checkpoint')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save individual predictions to file')
    parser.add_argument('--plot_confusion_matrices', action='store_true',
                        help='Generate and save confusion matrix plots')
    
    return parser.parse_args()

def load_model_and_checkpoint(args, device):
    """Load model and checkpoint."""
    logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract metadata
    epoch = checkpoint.get('epoch', 'unknown')
    metrics = checkpoint.get('metrics', {})
    has_ema = checkpoint.get('has_ema', False)
    
    logger.info(f"Checkpoint from epoch: {epoch}")
    if 'best_val_acc' in metrics:
        logger.info(f"Best validation accuracy: {metrics['best_val_acc']:.4f}")
    
    # Create model
    logger.info(f"Creating {args.backbone_type.upper()} model: {args.backbone_name}")
    model = create_unified_model(
        backbone_type=args.backbone_type,
        num_severity=6,  # 6 severity classes
        num_action_type=10,  # 10 action type classes
        vocab_sizes=None,  # Video-only evaluation
        backbone_name=args.backbone_name,
        use_attention_aggregation=True,
        use_augmentation=False,  # No augmentation for evaluation
        disable_in_model_augmentation=True,
        dropout_rate=0.1
    )
    
    # Load weights
    if args.use_ema and has_ema and 'ema_state_dict' in checkpoint:
        logger.info("Using EMA weights for evaluation")
        state_dict = checkpoint['ema_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        raise KeyError("No model weights found in checkpoint")
    
    # Handle DataParallel prefix issues
    model_keys = list(model.state_dict().keys())
    checkpoint_keys = list(state_dict.keys())
    
    if len(model_keys) > 0 and len(checkpoint_keys) > 0:
        model_has_module = model_keys[0].startswith('module.')
        checkpoint_has_module = checkpoint_keys[0].startswith('module.')
        
        if model_has_module and not checkpoint_has_module:
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif not model_has_module and checkpoint_has_module:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
    
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model, checkpoint

def create_test_dataset(args):
    """Create test dataset and dataloader."""
    # Create transforms (no augmentation for evaluation)
    test_transform = create_transforms(args, is_training=False)
    
    # Construct dataset path
    mvfouls_path = str(Path(args.dataset_root) / "mvfouls")
    
    logger.info(f"Loading {args.split} dataset from {mvfouls_path}")
    test_dataset = SoccerNetMVFoulDataset(
        dataset_path=mvfouls_path,
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
    
    logger.info(f"Test dataset loaded: {len(test_dataset)} samples")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Important: don't shuffle for reproducible results
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=variable_views_collate_fn
    )
    
    return test_dataset, test_loader

def evaluate_model(model, test_loader, device):
    """Run model evaluation and collect predictions."""
    logger.info("Starting model evaluation...")
    
    all_sev_preds = []
    all_sev_labels = []
    all_sev_probs = []
    all_act_preds = []
    all_act_labels = []
    all_act_probs = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            try:
                # Move batch to device
                batch_on_device = {
                    "clips": batch["clips"].to(device, non_blocking=True)
                }
                if "view_mask" in batch and isinstance(batch["view_mask"], torch.Tensor):
                    batch_on_device["view_mask"] = batch["view_mask"].to(device, non_blocking=True)
                
                # Forward pass
                sev_logits, act_logits = model(batch_on_device)
                
                # Convert to probabilities and predictions
                sev_probs = F.softmax(sev_logits, dim=1)
                act_probs = F.softmax(act_logits, dim=1)
                
                sev_preds = torch.argmax(sev_logits, dim=1)
                act_preds = torch.argmax(act_logits, dim=1)
                
                # Store results
                all_sev_preds.append(sev_preds.cpu())
                all_sev_labels.append(batch["label_severity"])
                all_sev_probs.append(sev_probs.cpu())
                
                all_act_preds.append(act_preds.cpu())
                all_act_labels.append(batch["label_type"])
                all_act_probs.append(act_probs.cpu())
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Concatenate all results
    all_sev_preds = torch.cat(all_sev_preds, dim=0).numpy()
    all_sev_labels = torch.cat(all_sev_labels, dim=0).numpy()
    all_sev_probs = torch.cat(all_sev_probs, dim=0).numpy()
    
    all_act_preds = torch.cat(all_act_preds, dim=0).numpy()
    all_act_labels = torch.cat(all_act_labels, dim=0).numpy()
    all_act_probs = torch.cat(all_act_probs, dim=0).numpy()
    
    logger.info(f"Evaluation completed on {len(all_sev_preds)} samples")
    
    return {
        'severity': {
            'predictions': all_sev_preds,
            'labels': all_sev_labels,
            'probabilities': all_sev_probs
        },
        'action': {
            'predictions': all_act_preds,
            'labels': all_act_labels,
            'probabilities': all_act_probs
        }
    }

def calculate_metrics(results):
    """Calculate comprehensive evaluation metrics."""
    metrics = {}
    
    # Severity metrics
    sev_preds = results['severity']['predictions']
    sev_labels = results['severity']['labels']
    
    sev_accuracy = np.mean(sev_preds == sev_labels)
    sev_f1_macro = f1_score(sev_labels, sev_preds, average='macro', zero_division=0)
    sev_f1_weighted = f1_score(sev_labels, sev_preds, average='weighted', zero_division=0)
    
    metrics['severity'] = {
        'accuracy': sev_accuracy,
        'f1_macro': sev_f1_macro,
        'f1_weighted': sev_f1_weighted,
        'confusion_matrix': confusion_matrix(sev_labels, sev_preds),
        'classification_report': classification_report(
            sev_labels, sev_preds, 
            target_names=[f"Sev_{i}" for i in range(6)],
            zero_division=0,
            output_dict=True
        )
    }
    
    # Action metrics
    act_preds = results['action']['predictions']
    act_labels = results['action']['labels']
    
    act_accuracy = np.mean(act_preds == act_labels)
    act_f1_macro = f1_score(act_labels, act_preds, average='macro', zero_division=0)
    act_f1_weighted = f1_score(act_labels, act_preds, average='weighted', zero_division=0)
    
    metrics['action'] = {
        'accuracy': act_accuracy,
        'f1_macro': act_f1_macro,
        'f1_weighted': act_f1_weighted,
        'confusion_matrix': confusion_matrix(act_labels, act_preds),
        'classification_report': classification_report(
            act_labels, act_preds,
            target_names=[f"Act_{i}" for i in range(10)],
            zero_division=0,
            output_dict=True
        )
    }
    
    # Combined metrics
    metrics['combined'] = {
        'accuracy': (sev_accuracy + act_accuracy) / 2,
        'f1_macro': (sev_f1_macro + act_f1_macro) / 2,
        'f1_weighted': (sev_f1_weighted + act_f1_weighted) / 2
    }
    
    return metrics

def print_results(metrics):
    """Print evaluation results to console."""
    logger.info("="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    
    # Overall metrics
    logger.info(f"COMBINED METRICS:")
    logger.info(f"  Accuracy: {metrics['combined']['accuracy']:.4f}")
    logger.info(f"  F1 Macro: {metrics['combined']['f1_macro']:.4f}")
    logger.info(f"  F1 Weighted: {metrics['combined']['f1_weighted']:.4f}")
    logger.info("")
    
    # Severity metrics
    logger.info(f"SEVERITY CLASSIFICATION:")
    logger.info(f"  Accuracy: {metrics['severity']['accuracy']:.4f}")
    logger.info(f"  F1 Macro: {metrics['severity']['f1_macro']:.4f}")
    logger.info(f"  F1 Weighted: {metrics['severity']['f1_weighted']:.4f}")
    logger.info("")
    
    # Per-class severity results
    logger.info("Per-class Severity Results:")
    sev_report = metrics['severity']['classification_report']
    for i in range(6):
        class_name = f"Sev_{i}"
        if class_name in sev_report:
            precision = sev_report[class_name]['precision']
            recall = sev_report[class_name]['recall']
            f1 = sev_report[class_name]['f1-score']
            support = sev_report[class_name]['support']
            logger.info(f"  {class_name}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} Support={support}")
    logger.info("")
    
    # Action metrics
    logger.info(f"ACTION CLASSIFICATION:")
    logger.info(f"  Accuracy: {metrics['action']['accuracy']:.4f}")
    logger.info(f"  F1 Macro: {metrics['action']['f1_macro']:.4f}")
    logger.info(f"  F1 Weighted: {metrics['action']['f1_weighted']:.4f}")
    logger.info("")
    
    # Per-class action results (top classes only)
    logger.info("Per-class Action Results (top classes):")
    act_report = metrics['action']['classification_report']
    for i in range(10):
        class_name = f"Act_{i}"
        if class_name in act_report and act_report[class_name]['support'] > 0:
            precision = act_report[class_name]['precision']
            recall = act_report[class_name]['recall']
            f1 = act_report[class_name]['f1-score']
            support = act_report[class_name]['support']
            logger.info(f"  {class_name}: P={precision:.3f} R={recall:.3f} F1={f1:.3f} Support={support}")

def plot_confusion_matrices(metrics, output_dir):
    """Generate and save confusion matrix plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Severity confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['severity']['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"Sev_{i}" for i in range(6)],
                yticklabels=[f"Sev_{i}" for i in range(6)])
    plt.title('Severity Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'severity_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Action confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics['action']['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"Act_{i}" for i in range(10)],
                yticklabels=[f"Act_{i}" for i in range(10)])
    plt.title('Action Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'action_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrices saved to {output_dir}")

def save_results(metrics, results, args, output_dir):
    """Save evaluation results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save metrics summary
    summary = {
        'evaluation_args': vars(args),
        'combined_metrics': metrics['combined'],
        'severity_metrics': {
            'accuracy': metrics['severity']['accuracy'],
            'f1_macro': metrics['severity']['f1_macro'],
            'f1_weighted': metrics['severity']['f1_weighted']
        },
        'action_metrics': {
            'accuracy': metrics['action']['accuracy'],
            'f1_macro': metrics['action']['f1_macro'],
            'f1_weighted': metrics['action']['f1_weighted']
        }
    }
    
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed classification reports
    with open(output_dir / 'severity_classification_report.json', 'w') as f:
        json.dump(metrics['severity']['classification_report'], f, indent=2)
    
    with open(output_dir / 'action_classification_report.json', 'w') as f:
        json.dump(metrics['action']['classification_report'], f, indent=2)
    
    # Save confusion matrices
    np.save(output_dir / 'severity_confusion_matrix.npy', metrics['severity']['confusion_matrix'])
    np.save(output_dir / 'action_confusion_matrix.npy', metrics['action']['confusion_matrix'])
    
    # Save individual predictions if requested
    if args.save_predictions:
        predictions_data = {
            'severity': {
                'predictions': results['severity']['predictions'].tolist(),
                'labels': results['severity']['labels'].tolist(),
                'probabilities': results['severity']['probabilities'].tolist()
            },
            'action': {
                'predictions': results['action']['predictions'].tolist(),
                'labels': results['action']['labels'].tolist(),
                'probabilities': results['action']['probabilities'].tolist()
            }
        }
        
        with open(output_dir / 'predictions.json', 'w') as f:
            json.dump(predictions_data, f, indent=2)
        
        logger.info(f"Individual predictions saved to {output_dir / 'predictions.json'}")
    
    logger.info(f"Evaluation results saved to {output_dir}")

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model, checkpoint = load_model_and_checkpoint(args, device)
    
    # Create test dataset
    test_dataset, test_loader = create_test_dataset(args)
    
    # Run evaluation
    results = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print results
    print_results(metrics)
    
    # Save results
    save_results(metrics, results, args, args.output_dir)
    
    # Generate plots if requested
    if args.plot_confusion_matrices:
        try:
            plot_confusion_matrices(metrics, args.output_dir)
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()

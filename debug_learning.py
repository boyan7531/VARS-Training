#!/usr/bin/env python3
"""
Debug script for analyzing learning issues:
- Loss decreasing but accuracy flat
- Class distribution analysis
- Prediction pattern analysis
- Gradient flow analysis
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Import your modules
from dataset import SoccerNetMVFoulDataset, SEVERITY_LABELS, ACTION_TYPE_LABELS
from model import MultiTaskMultiViewResNet3D, ModelConfig
from torch.utils.data import DataLoader


def analyze_class_distribution(dataset, output_dir="debug_output"):
    """Analyze the distribution of classes in the dataset"""
    Path(output_dir).mkdir(exist_ok=True)
    
    severity_counts = Counter()
    action_counts = Counter()
    
    for action in dataset.actions:
        severity_counts[action['label_severity']] += 1
        action_counts[action['label_type']] += 1
    
    print("=" * 60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Severity distribution
    print("\n📊 SEVERITY CLASS DISTRIBUTION:")
    total_severity = sum(severity_counts.values())
    severity_labels_rev = {v: k for k, v in SEVERITY_LABELS.items()}
    
    for class_id in sorted(severity_counts.keys()):
        count = severity_counts[class_id]
        percentage = (count / total_severity) * 100
        label = severity_labels_rev.get(class_id, f"Unknown({class_id})")
        print(f"  Class {class_id} ({label}): {count:4d} samples ({percentage:5.1f}%)")
    
    # Action type distribution
    print("\n📊 ACTION TYPE CLASS DISTRIBUTION:")
    total_action = sum(action_counts.values())
    action_labels_rev = {v: k for k, v in ACTION_TYPE_LABELS.items()}
    
    for class_id in sorted(action_counts.keys()):
        count = action_counts[class_id]
        percentage = (count / total_action) * 100
        label = action_labels_rev.get(class_id, f"Unknown({class_id})")
        print(f"  Class {class_id} ({label}): {count:4d} samples ({percentage:5.1f}%)")
    
    # Calculate imbalance ratios
    print("\n⚖️  CLASS IMBALANCE ANALYSIS:")
    max_sev = max(severity_counts.values())
    min_sev = min(severity_counts.values())
    print(f"  Severity imbalance ratio: {max_sev/min_sev:.1f}:1")
    
    max_act = max(action_counts.values())
    min_act = min(action_counts.values())
    print(f"  Action type imbalance ratio: {max_act/min_act:.1f}:1")
    
    return severity_counts, action_counts


def analyze_model_predictions(model, dataloader, device, num_batches=10):
    """Analyze what the model is actually predicting"""
    model.eval()
    
    all_sev_preds = []
    all_sev_labels = []
    all_act_preds = []
    all_act_labels = []
    all_sev_logits = []
    all_act_logits = []
    
    print("\n🔍 ANALYZING MODEL PREDICTIONS...")
    
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            if i >= num_batches:
                break
                
            # Move to device
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(device)
            
            # Forward pass
            sev_logits, act_logits = model(batch_data)
            
            # Get predictions
            sev_preds = torch.argmax(sev_logits, dim=1)
            act_preds = torch.argmax(act_logits, dim=1)
            
            # Store results
            all_sev_preds.extend(sev_preds.cpu().numpy())
            all_sev_labels.extend(batch_data["label_severity"].cpu().numpy())
            all_act_preds.extend(act_preds.cpu().numpy())
            all_act_labels.extend(batch_data["label_type"].cpu().numpy())
            all_sev_logits.append(sev_logits.cpu())
            all_act_logits.append(act_logits.cpu())
    
    # Analyze prediction patterns
    print("\n📈 PREDICTION PATTERN ANALYSIS:")
    
    # Severity predictions
    sev_pred_counter = Counter(all_sev_preds)
    sev_label_counter = Counter(all_sev_labels)
    
    print("\n  Severity - True vs Predicted distribution:")
    for class_id in sorted(set(all_sev_labels + all_sev_preds)):
        true_count = sev_label_counter.get(class_id, 0)
        pred_count = sev_pred_counter.get(class_id, 0)
        print(f"    Class {class_id}: True={true_count:3d}, Predicted={pred_count:3d}")
    
    # Action predictions
    act_pred_counter = Counter(all_act_preds)
    act_label_counter = Counter(all_act_labels)
    
    print("\n  Action Type - True vs Predicted distribution:")
    for class_id in sorted(set(all_act_labels + all_act_preds)):
        true_count = act_label_counter.get(class_id, 0)
        pred_count = act_pred_counter.get(class_id, 0)
        print(f"    Class {class_id}: True={true_count:3d}, Predicted={pred_count:3d}")
    
    # Analyze confidence (logit magnitudes)
    all_sev_logits = torch.cat(all_sev_logits, dim=0)
    all_act_logits = torch.cat(all_act_logits, dim=0)
    
    sev_confidence = torch.softmax(all_sev_logits, dim=1).max(dim=1)[0]
    act_confidence = torch.softmax(all_act_logits, dim=1).max(dim=1)[0]
    
    print(f"\n📊 PREDICTION CONFIDENCE:")
    print(f"  Severity - Mean confidence: {sev_confidence.mean():.3f} (std: {sev_confidence.std():.3f})")
    print(f"  Action   - Mean confidence: {act_confidence.mean():.3f} (std: {act_confidence.std():.3f})")
    
    # Check if model is overly confident on wrong predictions
    sev_correct = (torch.tensor(all_sev_preds) == torch.tensor(all_sev_labels))
    act_correct = (torch.tensor(all_act_preds) == torch.tensor(all_act_labels))
    
    if len(sev_correct) > 0:
        correct_sev_conf = sev_confidence[sev_correct].mean() if sev_correct.sum() > 0 else 0
        wrong_sev_conf = sev_confidence[~sev_correct].mean() if (~sev_correct).sum() > 0 else 0
        print(f"  Severity - Correct predictions confidence: {correct_sev_conf:.3f}")
        print(f"  Severity - Wrong predictions confidence: {wrong_sev_conf:.3f}")
    
    if len(act_correct) > 0:
        correct_act_conf = act_confidence[act_correct].mean() if act_correct.sum() > 0 else 0
        wrong_act_conf = act_confidence[~act_correct].mean() if (~act_correct).sum() > 0 else 0
        print(f"  Action   - Correct predictions confidence: {correct_act_conf:.3f}")
        print(f"  Action   - Wrong predictions confidence: {wrong_act_conf:.3f}")
    
    return {
        'sev_preds': all_sev_preds,
        'sev_labels': all_sev_labels,
        'act_preds': all_act_preds,
        'act_labels': all_act_labels,
        'sev_logits': all_sev_logits,
        'act_logits': all_act_logits
    }


def analyze_loss_components(model, dataloader, device, num_batches=10):
    """Analyze individual loss components"""
    model.eval()
    
    sev_losses = []
    act_losses = []
    total_losses = []
    
    print("\n🔍 ANALYZING LOSS COMPONENTS...")
    
    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            if i >= num_batches:
                break
                
            # Move to device
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(device)
            
            # Forward pass
            sev_logits, act_logits = model(batch_data)
            
            # Calculate individual losses
            sev_loss = nn.CrossEntropyLoss()(sev_logits, batch_data["label_severity"])
            act_loss = nn.CrossEntropyLoss()(act_logits, batch_data["label_type"])
            total_loss = sev_loss + act_loss
            
            sev_losses.append(sev_loss.item())
            act_losses.append(act_loss.item())
            total_losses.append(total_loss.item())
    
    print(f"\n📊 LOSS ANALYSIS:")
    print(f"  Severity Loss   - Mean: {np.mean(sev_losses):.4f}, Std: {np.std(sev_losses):.4f}")
    print(f"  Action Loss     - Mean: {np.mean(act_losses):.4f}, Std: {np.std(act_losses):.4f}")
    print(f"  Total Loss      - Mean: {np.mean(total_losses):.4f}, Std: {np.std(total_losses):.4f}")
    print(f"  Severity/Action - Ratio: {np.mean(sev_losses)/np.mean(act_losses):.2f}")
    
    return {
        'sev_losses': sev_losses,
        'act_losses': act_losses,
        'total_losses': total_losses
    }


def check_gradient_flow(model, dataloader, device):
    """Check if gradients are flowing properly"""
    model.train()
    
    # Get one batch
    batch_data = next(iter(dataloader))
    for key in batch_data:
        if isinstance(batch_data[key], torch.Tensor):
            batch_data[key] = batch_data[key].to(device)
    
    # Forward pass
    sev_logits, act_logits = model(batch_data)
    
    # Calculate loss
    sev_loss = nn.CrossEntropyLoss()(sev_logits, batch_data["label_severity"])
    act_loss = nn.CrossEntropyLoss()(act_logits, batch_data["label_type"])
    total_loss = sev_loss + act_loss
    
    # Backward pass
    total_loss.backward()
    
    print("\n🔍 GRADIENT FLOW ANALYSIS:")
    
    # Check gradient magnitudes
    total_norm = 0
    param_count = 0
    zero_grad_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
            if param_norm.item() < 1e-7:
                zero_grad_count += 1
        else:
            print(f"  ⚠️  No gradient for parameter: {name}")
    
    total_norm = total_norm ** (1. / 2)
    
    print(f"  Total gradient norm: {total_norm:.6f}")
    print(f"  Parameters with gradients: {param_count}")
    print(f"  Parameters with near-zero gradients: {zero_grad_count}")
    
    if zero_grad_count > param_count * 0.5:
        print("  ⚠️  WARNING: More than 50% of parameters have near-zero gradients!")
    
    return total_norm


def suggest_fixes(severity_counts, action_counts, prediction_analysis):
    """Suggest potential fixes based on analysis"""
    print("\n🔧 SUGGESTED FIXES:")
    print("=" * 60)
    
    # Check for severe class imbalance
    max_sev = max(severity_counts.values())
    min_sev = min(severity_counts.values())
    
    if max_sev / min_sev > 10:
        print("1. 🚨 SEVERE CLASS IMBALANCE DETECTED:")
        print("   - Try focal loss: --loss_function focal")
        print("   - Increase class weights: --class_weighting_strategy balanced_capped")
        print("   - Use stronger oversampling")
    
    # Check if model predicts only majority class
    sev_preds = prediction_analysis['sev_preds']
    sev_pred_unique = np.unique(sev_preds)
    
    if len(sev_pred_unique) <= 2:
        print("\n2. 🚨 MODEL COLLAPSED TO MAJORITY CLASS:")
        print("   - Try lower learning rate: --learning_rate 0.0001")
        print("   - Use label smoothing: --label_smoothing 0.1")
        print("   - Reduce batch size for more updates")
    
    print("\n3. 📚 GENERAL RECOMMENDATIONS:")
    print("   - Lower learning rate: --learning_rate 0.0001")
    print("   - Try focal loss: --loss_function focal --focal_gamma 2.0")  
    print("   - Add label smoothing: --label_smoothing 0.1")
    print("   - Check your data quality and labels")


def main():
    """Main debugging function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug learning issues')
    parser.add_argument('--train_dataset_path', required=True, help='Path to training dataset')
    parser.add_argument('--model_checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for analysis')
    parser.add_argument('--num_batches', type=int, default=20, help='Number of batches to analyze')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    train_dataset = SoccerNetMVFoulDataset(
        dataset_path=args.train_dataset_path,
        split='train'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # Analyze class distribution
    severity_counts, action_counts = analyze_class_distribution(train_dataset)
    
    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    
    # Get vocab sizes from dataset
    vocab_sizes = {}
    for action in train_dataset.actions[:100]:  # Sample some actions to get vocab
        for key, value in action.items():
            if isinstance(value, str) and key not in ['video_path', 'label_severity', 'label_type']:
                vocab_sizes[key] = vocab_sizes.get(key, 0) + 1
    
    # Create model (you might need to adjust this based on your model structure)
    model = MultiTaskMultiViewResNet3D.create_model(
        num_severity=6,  # 0-5 severity classes
        num_action_type=10,  # 0-9 action classes  
        vocab_sizes=vocab_sizes,
        backbone_name='r2plus1d_18'
    )
    
    # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Analyze model predictions
    prediction_analysis = analyze_model_predictions(model, train_loader, device, args.num_batches)
    
    # Analyze loss components
    loss_analysis = analyze_loss_components(model, train_loader, device, args.num_batches)
    
    # Check gradient flow
    gradient_norm = check_gradient_flow(model, train_loader, device)
    
    # Suggest fixes
    suggest_fixes(severity_counts, action_counts, prediction_analysis)
    
    print("\n" + "=" * 60)
    print("DEBUGGING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
import torch
import argparse
import logging
import os
from model.resnet3d_model import MultiTaskMultiViewResNet3D
import sys

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def load_model_checkpoint(checkpoint_path, device):
    """Load model checkpoint and return model with adjusted weights"""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_state_dict = checkpoint['model_state_dict']
        
        # Get model metadata from checkpoint
        metrics = checkpoint.get('metrics', {})
        epoch = checkpoint.get('epoch', 0)
        vocab_sizes = checkpoint.get('vocab_sizes', {})
        
        # Extract model architecture parameters
        num_severity = model_state_dict['severity_head.3.weight'].size(0)
        num_action_type = model_state_dict['action_type_head.3.weight'].size(0)
        
        logger.info(f"Original model has {num_severity} severity classes and {num_action_type} action type classes")
        
        # Add default vocabulary sizes if not present
        # We only need these for model initialization, not for our bias adjustment
        default_vocab_sizes = {
            'contact': 3,
            'bodypart': 3,
            'upper_bodypart': 3,
            'lower_bodypart': 3,
            'multiple_fouls': 3,
            'try_to_play': 3,
            'touch_ball': 3,
            'handball': 3,
            'handball_offence': 3
        }
        
        # Add any missing vocabulary sizes with defaults
        for key in default_vocab_sizes:
            if key not in vocab_sizes:
                vocab_sizes[key] = default_vocab_sizes[key]
                logger.info(f"Using default value {default_vocab_sizes[key]} for {key}")
        
        # Initialize a new model with the same architecture
        model = MultiTaskMultiViewResNet3D.create_model(
            num_severity=num_severity,
            num_action_type=num_action_type,
            vocab_sizes=vocab_sizes,
            backbone_name='r2plus1d_18'
        )
        
        # Load the state dict
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        
        return model, metrics, epoch, vocab_sizes
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        sys.exit(1)

def adjust_model_bias(model, severity_bias_adjustments):
    """
    Adjust the bias terms in the model's final layer
    
    Args:
        model: The loaded model
        severity_bias_adjustments: Dictionary mapping class indices to bias adjustments
    """
    logger.info("Current bias values in severity head:")
    
    # Get the final layer of the severity head
    final_layer = model.severity_head[-1]
    original_biases = final_layer.bias.clone().detach()
    
    # Log original bias values
    for i, bias in enumerate(original_biases):
        logger.info(f"  Class {i} bias: {bias.item():.4f}")
    
    # Apply adjustments
    with torch.no_grad():
        for class_idx, adjustment in severity_bias_adjustments.items():
            final_layer.bias[class_idx] += adjustment
            logger.info(f"Adjusting class {class_idx} bias by {adjustment:.4f}")
    
    # Log new bias values
    logger.info("New bias values after adjustment:")
    for i, bias in enumerate(final_layer.bias):
        logger.info(f"  Class {i} bias: {bias.item():.4f}")
    
    return model

def save_model(model, original_checkpoint_path, metrics, epoch, vocab_sizes, suffix="_bias_fixed"):
    """Save the adjusted model"""
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(original_checkpoint_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create new filename with suffix
    base_name = os.path.basename(original_checkpoint_path)
    name_parts = os.path.splitext(base_name)
    new_filename = f"{name_parts[0]}{suffix}{name_parts[1]}"
    output_path = os.path.join(output_dir, new_filename)
    
    # Save the model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'epoch': epoch,
        'vocab_sizes': vocab_sizes
    }
    torch.save(checkpoint, output_path)
    logger.info(f"Saved adjusted model to: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Adjust model bias terms to correct class prediction imbalance")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--class1_adjustment', type=float, default=0.4, help='Amount to adjust class 1 bias (positive value increases likelihood)')
    parser.add_argument('--class2_adjustment', type=float, default=-0.2, help='Amount to adjust class 2 bias (negative value decreases likelihood)')
    parser.add_argument('--output_suffix', type=str, default="_bias_fixed", help='Suffix to add to output model filename')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load the model
    model, metrics, epoch, vocab_sizes = load_model_checkpoint(args.checkpoint_path, device)
    
    # Define bias adjustments (positive increases likelihood, negative decreases)
    severity_bias_adjustments = {
        1: args.class1_adjustment,    # Increase bias for class 1.0 (more likely to predict)
        2: args.class2_adjustment,    # Decrease bias for class 2.0 (less likely to predict)
    }
    
    # Apply bias adjustments
    model = adjust_model_bias(model, severity_bias_adjustments)
    
    # Save the adjusted model
    output_path = save_model(model, args.checkpoint_path, metrics, epoch, vocab_sizes, args.output_suffix)
    
    logger.info(f"Model bias adjustment complete. New model saved to: {output_path}")
    logger.info("Run benchmark.py on the new model to verify the bias correction.")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Debug script for validation accuracy issues.

This script provides tools to:
1. Test class weights only mode (recommended fix for overfitting)
2. Analyze validation prediction patterns
3. Compare different class balancing strategies

Usage Examples:
    # Use class weights only (recommended fix)
    python debug_validation_accuracy.py --mode class_weights_only
    
    # Analyze current training run
    python debug_validation_accuracy.py --mode analyze --checkpoint_dir checkpoints
    
    # Compare strategies
    python debug_validation_accuracy.py --mode compare
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from train_lightning import main as train_main
from training.config import parse_args

logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('debug_validation.log')
        ]
    )

def test_class_weights_only():
    """Test training with class weights only mode."""
    logger.info("🧪 Testing CLASS WEIGHTS ONLY mode (recommended fix for stuck validation accuracy)")
    logger.info("=" * 80)
    
    # Create modified args for class weights only
    original_args = sys.argv.copy()
    
    # Override sys.argv to use class weights only mode
    sys.argv = [
        'debug_validation_accuracy.py',
        '--use_class_weights_only',  # Key flag to fix overfitting
        '--loss_function', 'weighted',  # Use weighted loss for class weights
        '--epochs', '5',  # Short test run
        '--batch_size', '8',
        '--backbone_type', 'mvit',
        '--backbone_name', 'mvit_base_16x4',
        '--learning_rate', '1e-4',
        '--scheduler', 'cosine_annealing',
        '--early_stopping_patience', '10',
        '--save_dir', 'debug_class_weights_test'
    ]
    
    logger.info("Configuration:")
    logger.info("  ✅ Class weights only: True")
    logger.info("  ❌ Oversampling: Disabled")
    logger.info("  ❌ Progressive sampling: Disabled") 
    logger.info("  ✅ Loss function: weighted")
    logger.info("  📊 This should reduce overfitting and allow validation accuracy to improve")
    logger.info("")
    
    try:
        train_main()
        logger.info("✅ Class weights only test completed successfully!")
    except Exception as e:
        logger.error(f"❌ Class weights only test failed: {e}")
    finally:
        # Restore original args
        sys.argv = original_args

def analyze_checkpoint_dir(checkpoint_dir):
    """Analyze training logs and checkpoints for validation issues."""
    logger.info(f"🔍 Analyzing checkpoint directory: {checkpoint_dir}")
    
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Look for training logs
    log_files = list(checkpoint_path.glob("*.log"))
    if not log_files:
        log_files = list(Path(".").glob("*.log"))
    
    if log_files:
        logger.info(f"Found {len(log_files)} log files")
        # Analyze the most recent log file
        latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
        analyze_training_log(latest_log)
    else:
        logger.warning("No log files found")
    
    # Look for checkpoints
    checkpoint_files = list(checkpoint_path.glob("*.pth"))
    if checkpoint_files:
        logger.info(f"Found {len(checkpoint_files)} checkpoint files")
        # You could add checkpoint analysis here
    else:
        logger.warning("No checkpoint files found")

def analyze_training_log(log_file):
    """Analyze training log for validation accuracy patterns."""
    logger.info(f"📊 Analyzing training log: {log_file}")
    
    val_accuracies = []
    overfitting_alerts = 0
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Look for validation accuracy
                if 'val_combined_acc=' in line:
                    try:
                        # Extract accuracy value
                        acc_str = line.split('val_combined_acc=')[1].split(',')[0].strip()
                        val_acc = float(acc_str)
                        val_accuracies.append(val_acc)
                    except (IndexError, ValueError):
                        continue
                
                # Count overfitting alerts
                if 'OVERFITTING ALERT' in line:
                    overfitting_alerts += 1
        
        # Analyze patterns
        if val_accuracies:
            logger.info(f"📈 Validation Accuracy Analysis:")
            logger.info(f"   Total epochs tracked: {len(val_accuracies)}")
            logger.info(f"   First accuracy: {val_accuracies[0]:.4f}")
            logger.info(f"   Last accuracy: {val_accuracies[-1]:.4f}")
            logger.info(f"   Max accuracy: {max(val_accuracies):.4f}")
            logger.info(f"   Min accuracy: {min(val_accuracies):.4f}")
            
            # Check if stuck
            if len(set([round(acc, 3) for acc in val_accuracies])) <= 2:
                logger.warning("   ⚠️ VALIDATION ACCURACY APPEARS STUCK!")
                logger.warning("   📋 Recommended fixes:")
                logger.warning("     1. Use: python train_lightning.py --use_class_weights_only")
                logger.warning("     2. Reduce learning rate: --learning_rate 1e-5")
                logger.warning("     3. Increase regularization: --dropout_rate 0.3")
            
            # Check improvement trend
            if len(val_accuracies) >= 5:
                recent_avg = sum(val_accuracies[-5:]) / 5
                early_avg = sum(val_accuracies[:5]) / 5
                if recent_avg <= early_avg + 0.01:
                    logger.warning("   ⚠️ NO SIGNIFICANT IMPROVEMENT in recent epochs")
        
        logger.info(f"🚨 Overfitting alerts: {overfitting_alerts}")
        if overfitting_alerts > 0:
            logger.warning("   High number of overfitting alerts detected!")
            logger.warning("   Strong recommendation: Use --use_class_weights_only")
    
    except Exception as e:
        logger.error(f"Error analyzing log file: {e}")

def compare_strategies():
    """Compare different class balancing strategies."""
    logger.info("⚖️ Comparison of Class Balancing Strategies")
    logger.info("=" * 80)
    
    strategies = [
        {
            'name': 'Class Weights Only (RECOMMENDED)',
            'description': 'Use computed class weights without oversampling',
            'command': '--use_class_weights_only --loss_function weighted',
            'pros': ['Reduces overfitting', 'Stable training', 'Natural validation distribution'],
            'cons': ['May need more epochs', 'Requires good class weight computation']
        },
        {
            'name': 'Progressive Sampling (CURRENT)',
            'description': 'Gradually increase minority class oversampling',
            'command': '--progressive_class_balancing --oversample_factor 2.0',
            'pros': ['Gradual adaptation', 'Can handle severe imbalance'],
            'cons': ['Can cause overfitting', 'Training/validation mismatch']
        },
        {
            'name': 'Fixed Oversampling',
            'description': 'Fixed oversampling throughout training',
            'command': '--use_class_balanced_sampler --oversample_factor 3.0',
            'pros': ['Simple to implement', 'Immediate balance'],
            'cons': ['High overfitting risk', 'Validation mismatch']
        },
        {
            'name': 'No Balancing',
            'description': 'Train on natural class distribution',
            'command': '--disable_class_balancing',
            'pros': ['No overfitting from balancing', 'True data distribution'],
            'cons': ['Poor minority class performance', 'Biased towards majority']
        }
    ]
    
    for i, strategy in enumerate(strategies, 1):
        logger.info(f"{i}. {strategy['name']}")
        logger.info(f"   Description: {strategy['description']}")
        logger.info(f"   Command: {strategy['command']}")
        logger.info(f"   Pros: {', '.join(strategy['pros'])}")
        logger.info(f"   Cons: {', '.join(strategy['cons'])}")
        logger.info("")
    
    logger.info("💡 RECOMMENDATION:")
    logger.info("   For stuck validation accuracy (0.445), try:")
    logger.info("   python train_lightning.py --use_class_weights_only --loss_function weighted")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Debug validation accuracy issues')
    parser.add_argument('--mode', choices=['class_weights_only', 'analyze', 'compare'], 
                       default='class_weights_only',
                       help='Debug mode to run')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory containing checkpoints and logs to analyze')
    
    args = parser.parse_args()
    
    setup_logging()
    
    logger.info("🔧 VALIDATION ACCURACY DEBUG TOOL")
    logger.info("=" * 50)
    
    if args.mode == 'class_weights_only':
        test_class_weights_only()
    elif args.mode == 'analyze':
        analyze_checkpoint_dir(args.checkpoint_dir)
    elif args.mode == 'compare':
        compare_strategies()
    
    logger.info("🏁 Debug session completed")

if __name__ == '__main__':
    main() 
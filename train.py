# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from torchvision.transforms import Compose, CenterCrop
from pathlib import Path
import argparse
import time
import os
import logging
import json
from collections import defaultdict
import numpy as np
import random
from tqdm import tqdm

# Imports from our other files
from dataset import SoccerNetMVFoulDataset, variable_views_collate_fn
from model import MultiTaskMultiViewResNet3D, ModelConfig

# Import transforms directly
from pytorchvideo.transforms import ShortSideScale, Normalize as VideoNormalize

# Define transforms locally instead of importing from test file
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
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration & Hyperparameters ---
def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-Task Multi-View ResNet3D for Foul Recognition')
    parser.add_argument('--dataset_root', type=str, default="", help='Root directory containing the mvfouls folder')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--backbone_name', type=str, default='resnet3d_18', choices=['resnet3d_18', 'resnet3d_50'], help="ResNet3D backbone variant")
    parser.add_argument('--frames_per_clip', type=int, default=16, help='Number of frames per clip')
    parser.add_argument('--target_fps', type=int, default=15, help='Target FPS for clips')
    parser.add_argument('--start_frame', type=int, default=67, help='Start frame index for foul-centered extraction (8 frames before foul at frame 75)')
    parser.add_argument('--end_frame', type=int, default=82, help='End frame index for foul-centered extraction (7 frames after foul at frame 75)')
    parser.add_argument('--img_height', type=int, default=224, help='Target image height')
    parser.add_argument('--img_width', type=int, default=398, help='Target image width (matches original VARS paper)')
    # Note: ResNet3D supports rectangular inputs unlike MViT
    parser.add_argument('--max_views', type=int, default=None, help='Optional limit on max views per action (default: use all available)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Advanced training options
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'onecycle', 'none'], help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    
    # Multi-task loss balancing
    parser.add_argument('--main_task_weights', type=float, nargs=2, default=[3.0, 3.0], 
                       help='Loss weights for [severity, action_type] - main tasks')
    parser.add_argument('--auxiliary_task_weight', type=float, default=0.5,
                       help='Loss weight for all auxiliary tasks (currently not implemented)')
    
    parser.add_argument('--attention_aggregation', action='store_true', default=True, help='Use attention for view aggregation')
    
    # Test run arguments
    parser.add_argument('--test_run', action='store_true', help='Perform a quick test run (1 epoch, few batches, no saving)')
    parser.add_argument('--test_batches', type=int, default=2, help='Number of batches to run in test mode')
    
    # New argument for forcing batch size even with multi-GPU
    parser.add_argument('--force_batch_size', action='store_true', 
                       help='Force specified batch size even with multi-GPU (disable auto-scaling)')
    
    args = parser.parse_args()
    
    # Construct the specific mvfouls path from the root
    if not args.dataset_root:
        raise ValueError("Please provide the --dataset_root argument.")
    args.mvfouls_path = str(Path(args.dataset_root) / "mvfouls")
    
    return args

def set_seed(seed):
    """Sets seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For reproducibility in CuDNN operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_accuracy(outputs, labels):
    """Calculates accuracy for a single task."""
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total if total > 0 else 0

def calculate_f1_score(outputs, labels, num_classes):
    """Calculate F1 score for multi-class classification."""
    from sklearn.metrics import f1_score
    _, predicted = torch.max(outputs.data, 1)
    return f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted', zero_division=0)

def calculate_multitask_loss(sev_logits, act_logits, batch_data, main_weights, aux_weight=0.5):
    """
    Calculate weighted multi-task loss with proper balancing between main and auxiliary tasks.
    
    Args:
        sev_logits: Severity classification logits
        act_logits: Action type classification logits  
        batch_data: Batch data containing labels
        main_weights: [severity_weight, action_weight] for main tasks
        aux_weight: Weight for auxiliary tasks (currently not implemented)
    
    Returns:
        total_loss, loss_sev, loss_act
    """
    # Main task losses with weighting
    loss_sev = nn.CrossEntropyLoss()(sev_logits, batch_data["label_severity"]) * main_weights[0]
    loss_act = nn.CrossEntropyLoss()(act_logits, batch_data["label_type"]) * main_weights[1]
    
    # Future: Add auxiliary task losses here
    # aux_losses = []
    # if "contact_logits" in outputs:
    #     aux_losses.append(nn.CrossEntropyLoss()(outputs["contact_logits"], batch_data["contact_idx"]) * aux_weight)
    # ... add other auxiliary tasks
    
    total_loss = loss_sev + loss_act  # + sum(aux_losses)
    
    return total_loss, loss_sev, loss_act

class EarlyStopping:
    """Early stopping utility."""
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        """Save model weights."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

def train_one_epoch(model, dataloader, criterion_severity, criterion_action, optimizer, device, 
                   scaler=None, max_batches=None, loss_weights=[1.0, 1.0], gradient_clip_norm=1.0):
    model.train()
    running_loss = 0.0
    running_sev_acc = 0.0
    running_act_acc = 0.0
    running_sev_f1 = 0.0
    running_act_f1 = 0.0
    processed_batches = 0

    start_time = time.time()
    
    # Create progress bar
    total_batches = max_batches if max_batches else len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=total_batches, desc="Training", 
                leave=False, ncols=100)
    
    for i, batch_data in pbar:
        if max_batches is not None and (i + 1) > max_batches:
            break 
        
        # Move all tensors in the batch to the device
        for key in batch_data:
            if isinstance(batch_data[key], torch.Tensor):
                batch_data[key] = batch_data[key].to(device, non_blocking=True)
            
        severity_labels = batch_data["label_severity"]
        action_labels = batch_data["label_type"]

        optimizer.zero_grad()

        # Mixed precision forward pass
        if scaler is not None:
            with autocast():
                sev_logits, act_logits = model(batch_data)
                total_loss, loss_sev_weighted, loss_act_weighted = calculate_multitask_loss(
                    sev_logits, act_logits, batch_data, loss_weights
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            sev_logits, act_logits = model(batch_data)
            total_loss, loss_sev_weighted, loss_act_weighted = calculate_multitask_loss(
                sev_logits, act_logits, batch_data, loss_weights
            )

            total_loss.backward()
            
            # Gradient clipping
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            
            optimizer.step()

        # Calculate metrics
        running_loss += total_loss.item() * batch_data["clips"].size(0)
        sev_acc = calculate_accuracy(sev_logits, severity_labels)
        act_acc = calculate_accuracy(act_logits, action_labels)
        sev_f1 = calculate_f1_score(sev_logits, severity_labels, 5)  # 5 severity classes
        act_f1 = calculate_f1_score(act_logits, action_labels, 9)  # 9 action classes
        
        running_sev_acc += sev_acc
        running_act_acc += act_acc
        running_sev_f1 += sev_f1
        running_act_f1 += act_f1
        processed_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss.item():.4f}',
            'Sev_Acc': f'{sev_acc:.3f}',
            'Act_Acc': f'{act_acc:.3f}'
        })

    pbar.close()
    
    num_samples_processed = len(dataloader.dataset) if max_batches is None else processed_batches * dataloader.batch_size 
    epoch_loss = running_loss / num_samples_processed if num_samples_processed > 0 else 0
    epoch_sev_acc = running_sev_acc / processed_batches if processed_batches > 0 else 0
    epoch_act_acc = running_act_acc / processed_batches if processed_batches > 0 else 0
    epoch_sev_f1 = running_sev_f1 / processed_batches if processed_batches > 0 else 0
    epoch_act_f1 = running_act_f1 / processed_batches if processed_batches > 0 else 0
    
    epoch_time = time.time() - start_time
    return epoch_loss, epoch_sev_acc, epoch_act_acc, epoch_sev_f1, epoch_act_f1

def validate_one_epoch(model, dataloader, criterion_severity, criterion_action, device, 
                      max_batches=None, loss_weights=[1.0, 1.0]):
    model.eval()
    running_loss = 0.0
    running_sev_acc = 0.0
    running_act_acc = 0.0
    running_sev_f1 = 0.0
    running_act_f1 = 0.0
    processed_batches = 0

    start_time = time.time()
    
    # Create progress bar
    total_batches = max_batches if max_batches else len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=total_batches, desc="Validation", 
                leave=False, ncols=100)
    
    with torch.no_grad():
        for i, batch_data in pbar:
            if max_batches is not None and (i + 1) > max_batches:
                break
            
            # Move all tensors in the batch to the device
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(device, non_blocking=True)
                
            severity_labels = batch_data["label_severity"]
            action_labels = batch_data["label_type"]

            sev_logits, act_logits = model(batch_data)
            
            total_loss, loss_sev_weighted, loss_act_weighted = calculate_multitask_loss(
                sev_logits, act_logits, batch_data, loss_weights
            )

            running_loss += total_loss.item() * batch_data["clips"].size(0)
            sev_acc = calculate_accuracy(sev_logits, severity_labels)
            act_acc = calculate_accuracy(act_logits, action_labels)
            running_sev_acc += sev_acc
            running_act_acc += act_acc
            running_sev_f1 += calculate_f1_score(sev_logits, severity_labels, 5)  # 5 severity classes
            running_act_f1 += calculate_f1_score(act_logits, action_labels, 9)  # 9 action classes
            processed_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Sev_Acc': f'{sev_acc:.3f}',
                'Act_Acc': f'{act_acc:.3f}'
            })

    pbar.close()

    num_samples_processed = len(dataloader.dataset) if max_batches is None else processed_batches * dataloader.batch_size
    epoch_loss = running_loss / num_samples_processed if num_samples_processed > 0 else 0
    epoch_sev_acc = running_sev_acc / processed_batches if processed_batches > 0 else 0
    epoch_act_acc = running_act_acc / processed_batches if processed_batches > 0 else 0
    epoch_sev_f1 = running_sev_f1 / processed_batches if processed_batches > 0 else 0
    epoch_act_f1 = running_act_f1 / processed_batches if processed_batches > 0 else 0
    
    epoch_time = time.time() - start_time
    return epoch_loss, epoch_sev_acc, epoch_act_acc, epoch_sev_f1, epoch_act_f1

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, filepath):
    """Save training checkpoint."""
    # Handle DataParallel models
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, scaler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    logger.info(f"Checkpoint loaded from {filepath}")
    return checkpoint['epoch'], checkpoint.get('metrics', {})


if __name__ == "__main__":
    args = parse_args()
    
    # Test run setup
    if args.test_run:
        logger.info("=" * 60)
        logger.info(f"PERFORMING TEST RUN (1 Epoch, {args.test_batches} Batches)")
        logger.info("Model checkpoints will NOT be saved.")
        logger.info("=" * 60)
        args.epochs = 1
        num_batches_to_run = args.test_batches
    else:
        num_batches_to_run = None
        
    # Set seed and create directories
    set_seed(args.seed)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Multi-GPU setup
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        logger.info(f"Found {num_gpus} GPUs! Using multi-GPU training.")
        # Adjust batch size for multi-GPU if not explicitly set
        if args.batch_size == 8:  # Default value
            recommended_batch_size = min(32, args.batch_size * num_gpus * 2)
            logger.info(f"Automatically scaling batch size from {args.batch_size} to {recommended_batch_size} for multi-GPU")
            args.batch_size = recommended_batch_size
        
        # Adjust learning rate for larger effective batch size (linear scaling rule)
        if args.lr == 2e-4:  # Default value
            lr_scale = args.batch_size / 8  # Scale from base batch size of 8
            args.lr = args.lr * lr_scale
            logger.info(f"Scaled learning rate to {args.lr:.6f} for larger batch size")
    else:
        logger.info("Using single GPU training.")

    # Mixed precision setup
    scaler = GradScaler() if args.mixed_precision and device.type == 'cuda' else None
    if scaler:
        logger.info("Using mixed precision training")

    # Transforms optimized for ResNet3D (supports rectangular inputs)
    train_transform = Compose([
        ConvertToFloatAndScale(),
        VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ShortSideScale(size=int(args.img_height * 1.1)),  # Slightly larger for cropping
        PerFrameCenterCrop((args.img_height, args.img_width))  # Rectangular crop for ResNet3D
    ])
    
    # Deterministic validation transforms
    val_transform = Compose([
        ConvertToFloatAndScale(),
        VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ShortSideScale(size=args.img_height),
        PerFrameCenterCrop((args.img_height, args.img_width))  # Rectangular crop for ResNet3D
    ])

    # Load datasets
    logger.info("Loading datasets...")
    try:
        train_dataset = SoccerNetMVFoulDataset(
            dataset_path=args.mvfouls_path,
            split='train',
            frames_per_clip=args.frames_per_clip,
            target_fps=args.target_fps,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            max_views_to_load=args.max_views,  # None by default = use all views
            transform=train_transform,
            target_height=args.img_height,
            target_width=args.img_width
        )
        
        val_dataset = SoccerNetMVFoulDataset(
            dataset_path=args.mvfouls_path,
            split='valid', 
            frames_per_clip=args.frames_per_clip,
            target_fps=args.target_fps,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            max_views_to_load=args.max_views,  # None by default = use all views
            transform=val_transform,
            target_height=args.img_height,
            target_width=args.img_width
        )
    except FileNotFoundError as e:
        logger.error(f"Error loading dataset: {e}")
        exit(1)
        
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.error("One or both datasets are empty after loading.")
        exit(1)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=variable_views_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=variable_views_collate_fn
    )
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create vocabulary sizes dictionary for the model
    vocab_sizes = {
        'contact': train_dataset.num_contact_classes,
        'bodypart': train_dataset.num_bodypart_classes,
        'upper_bodypart': train_dataset.num_upper_bodypart_classes,
        'lower_bodypart': train_dataset.num_lower_bodypart_classes,
        'multiple_fouls': train_dataset.num_multiple_fouls_classes,
        'try_to_play': train_dataset.num_try_to_play_classes,
        'touch_ball': train_dataset.num_touch_ball_classes,
        'handball': train_dataset.num_handball_classes,
        'handball_offence': train_dataset.num_handball_offence_classes,
    }

    # Model configuration
    model_config = ModelConfig(
        use_attention_aggregation=args.attention_aggregation,
        input_frames=args.frames_per_clip,
        input_height=args.img_height,
        input_width=args.img_width  # ResNet3D supports rectangular inputs
    )

    # Initialize model with proper configuration
    logger.info(f"Initializing ResNet3D model: {args.backbone_name}")
    model = MultiTaskMultiViewResNet3D(
        num_severity=5,  # 5 severity classes: 1.0, 2.0, 3.0, 4.0, 5.0
        num_action_type=9,  # 9 action types: Challenge, Dive, Dont know, Elbowing, High leg, Holding, Pushing, Standing tackling, Tackling
        vocab_sizes=vocab_sizes,
        backbone_name=args.backbone_name,
        config=model_config
    )
    model.to(device)
    
    # Wrap model with DataParallel for multi-GPU
    if num_gpus > 1:
        model = nn.DataParallel(model)
        logger.info(f"Model wrapped with DataParallel for {num_gpus} GPUs")

    # Log model info
    model_info = model.get_model_info()
    logger.info(f"Model initialized - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Combined feature dimension: {model_info['video_feature_dim'] + model_info['total_embedding_dim']}")

    # Loss functions and optimizer
    criterion_severity = nn.CrossEntropyLoss()
    criterion_action = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    elif args.scheduler == 'onecycle':
        steps_per_epoch = len(train_loader)
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=args.warmup_epochs / args.epochs
        )

    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        logger.info(f"Resuming training from epoch {start_epoch}")

    # Training history
    history = defaultdict(list)
    best_val_acc = 0.0
    best_epoch = -1

    logger.info("Starting Training")
    logger.info(f"Configuration: {vars(args)}")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Training
        train_metrics = train_one_epoch(
            model, train_loader, criterion_severity, criterion_action, optimizer, device,
            scaler=scaler, max_batches=num_batches_to_run, 
            loss_weights=args.main_task_weights, gradient_clip_norm=args.gradient_clip_norm
        )
        
        train_loss, train_sev_acc, train_act_acc, train_sev_f1, train_act_f1 = train_metrics
        
        # Validation
        val_metrics = validate_one_epoch(
            model, val_loader, criterion_severity, criterion_action, device,
            max_batches=num_batches_to_run, loss_weights=args.main_task_weights
        )
        
        val_loss, val_sev_acc, val_act_acc, val_sev_f1, val_act_f1 = val_metrics

        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, OneCycleLR):
                # OneCycleLR updates per batch, handled in training loop if needed
                pass
            else:
                scheduler.step()

        # Log epoch results
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1} Results:")
        logger.info(f"  Train - Loss: {train_loss:.4f}, Sev Acc: {train_sev_acc:.4f}, Act Acc: {train_act_acc:.4f}")
        logger.info(f"  Train - Sev F1: {train_sev_f1:.4f}, Act F1: {train_act_f1:.4f}")
        logger.info(f"  Val   - Loss: {val_loss:.4f}, Sev Acc: {val_sev_acc:.4f}, Act Acc: {val_act_acc:.4f}")
        logger.info(f"  Val   - Sev F1: {val_sev_f1:.4f}, Act F1: {val_act_f1:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")

        # Store history
        history['train_loss'].append(train_loss)
        history['train_sev_acc'].append(train_sev_acc)
        history['train_act_acc'].append(train_act_acc)
        history['val_loss'].append(val_loss)
        history['val_sev_acc'].append(val_sev_acc)
        history['val_act_acc'].append(val_act_acc)

        # Model saving and early stopping
        if not args.test_run:
            current_val_acc = (val_sev_acc + val_act_acc) / 2
            
            # Save best model
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                best_epoch = epoch + 1
                
                metrics = {
                    'epoch': best_epoch,
                    'best_val_acc': best_val_acc,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_sev_acc': train_sev_acc,
                    'train_act_acc': train_act_acc,
                    'val_sev_acc': val_sev_acc,
                    'val_act_acc': val_act_acc
                }
                
                save_path = os.path.join(args.save_dir, f'best_model_epoch_{best_epoch}.pth')
                save_checkpoint(model, optimizer, scheduler, scaler, best_epoch, metrics, save_path)
                logger.info(f"New best model saved! Combined val acc: {best_val_acc:.4f}")

            # Early stopping check
            if early_stopping(current_val_acc, model):
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                metrics = {'epoch': epoch + 1}
                save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, metrics, checkpoint_path)

    # Save training history
    if not args.test_run:
        history_path = os.path.join(args.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            history_serializable = {k: [float(x) for x in v] for k, v in history.items()}
            json.dump(history_serializable, f, indent=2)
        logger.info(f"Training history saved to {history_path}")

    logger.info("\n" + "="*60)
    logger.info("Training Finished!")
    if not args.test_run:
        logger.info(f"Best validation combined accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    else:
        logger.info("Test run completed successfully")
    logger.info("="*60) 
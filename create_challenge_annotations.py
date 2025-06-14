#!/usr/bin/env python3
"""
Create minimal annotations.json for challenge split to enable inference.
This creates dummy annotations that allow the benchmark script to run on challenge data.
"""

import json
from pathlib import Path
import argparse

def create_challenge_annotations(challenge_path, output_file="annotations.json"):
    """Create minimal annotations.json for challenge split."""
    
    challenge_dir = Path(challenge_path)
    if not challenge_dir.exists():
        print(f"Error: Challenge directory {challenge_path} does not exist")
        return False
    
    # Find all action directories
    action_dirs = []
    for action_dir in challenge_dir.iterdir():
        if action_dir.is_dir() and action_dir.name.startswith('action_'):
            action_dirs.append(action_dir)
    
    action_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
    print(f"Found {len(action_dirs)} action directories")
    
    # Create minimal annotations structure
    annotations = {
        "Set": "challenge",
        "Actions": {}
    }
    
    for action_dir in action_dirs:
        action_id = action_dir.name.split('_')[1]  # Extract number from action_X
        
        # Find all video files in this action directory
        video_files = list(action_dir.glob('*.mp4'))
        
        if not video_files:
            print(f"Warning: No video files found in {action_dir}")
            continue
        
        # Create clips list with minimal info
        clips = []
        for video_file in video_files:
            # Create relative path as expected by dataset
            relative_path = f"Dataset/Challenge/{action_dir.name}/{video_file.name}"
            clips.append({
                "Url": relative_path,
                "Replay speed": "1.0"  # Default replay speed
            })
        
        # Create minimal action entry for video-only model
        action_entry = {
            "Severity": "",  # Empty severity (will be predicted)
            "Action class": "",  # Empty action class (will be predicted)
            "Clips": clips
        }
        
        annotations["Actions"][action_id] = action_entry
    
    # Save annotations file
    output_path = challenge_dir / output_file
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Created {output_path} with {len(annotations['Actions'])} actions")
    print("You can now run the benchmark script on the challenge split!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Create minimal annotations.json for challenge split')
    parser.add_argument('--challenge_path', type=str, default='mvfouls/challenge',
                        help='Path to challenge directory')
    parser.add_argument('--output_file', type=str, default='annotations.json',
                        help='Output annotations file name')
    
    args = parser.parse_args()
    
    success = create_challenge_annotations(args.challenge_path, args.output_file)
    
    if success:
        print("\n" + "="*60)
        print("SUCCESS! Now you can run the benchmark on challenge split:")
        print("="*60)
        print("python benchmark.py \\")
        print("  --checkpoint_path './checkpoints/mvit_mean_pooling/best_model_epoch_15.pth' \\")
        print("  --backbone_type mvit \\")
        print("  --backbone_name mvit_base_16x4 \\")
        print("  --dataset_root . \\")
        print("  --batch_size 4 \\")
        print("  --num_workers 8 \\")
        print("  --frames_per_clip 16 \\")
        print("  --target_fps 15 \\")
        print("  --img_height 224 \\")
        print("  --img_width 224 \\")
        print("  --output_file real_challenge.json \\")
        print("  --split challenge")
        print("="*60)

if __name__ == "__main__":
    main() 
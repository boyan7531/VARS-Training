#!/usr/bin/env python3
"""
Dataset diagnostic script to identify missing files and structural issues.

Usage:
    python diagnose_dataset.py --dataset-path mvfouls --split train

This script analyzes the dataset structure and reports:
- Missing video files
- Incorrect paths
- Statistics about data availability
- Suggestions for fixes
"""

import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

def analyze_dataset_structure(dataset_path: str, split: str):
    """Analyze the dataset structure and report issues."""
    print(f"🔍 Analyzing dataset structure...")
    print(f"   Dataset path: {dataset_path}")
    print(f"   Split: {split}")
    
    dataset_root = Path(dataset_path)
    split_dir = dataset_root / split
    annotations_file = split_dir / "annotations.json"
    
    # Check basic structure
    if not dataset_root.exists():
        print(f"❌ Dataset root directory not found: {dataset_root}")
        return False
    
    if not split_dir.exists():
        print(f"❌ Split directory not found: {split_dir}")
        print(f"   Available directories: {list(dataset_root.iterdir())}")
        return False
    
    if not annotations_file.exists():
        print(f"❌ Annotations file not found: {annotations_file}")
        print(f"   Available files in split dir: {list(split_dir.iterdir())}")
        return False
    
    print(f"✅ Basic structure looks good")
    
    # Load and analyze annotations
    print(f"\n📋 Loading annotations from {annotations_file}...")
    try:
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return False
    
    print(f"✅ Annotations loaded successfully")
    
    # Debug: Show the structure of annotations
    print(f"\n🔍 Inspecting annotations structure...")
    print(f"   Type: {type(annotations)}")
    if isinstance(annotations, dict):
        sample_keys = list(annotations.keys())[:3]
        print(f"   Sample keys: {sample_keys}")
        if sample_keys:
            first_key = sample_keys[0]
            first_value = annotations[first_key]
            print(f"   Type of '{first_key}': {type(first_value)}")
            if isinstance(first_value, dict):
                print(f"   Keys in '{first_key}': {list(first_value.keys())}")
            elif isinstance(first_value, str):
                print(f"   Value of '{first_key}': {first_value[:100]}...")
    
    # Analyze video file availability
    print(f"\n🎥 Analyzing video file availability...")
    
    total_actions = 0
    total_clips = 0
    missing_clips = 0
    missing_actions = []
    found_clips = 0
    
    # Handle different annotation structures
    if isinstance(annotations, dict):
        # Check if this is the SoccerNet format with "Actions" key
        if "Actions" in annotations:
            print(f"   📝 Found SoccerNet format with 'Actions' key")
            actions_data = annotations["Actions"]
            print(f"   📊 Number of actions: {len(actions_data)}")
        else:
            print(f"   📝 Using direct format (actions as top-level keys)")
            actions_data = annotations
        
        for action_id, action_data in actions_data.items():
            total_actions += 1
            
            # Handle different data types
            if isinstance(action_data, str):
                print(f"   ⚠️  Action {action_id} has string data instead of dict: {action_data[:50]}...")
                missing_actions.append(action_id)
                continue
            elif not isinstance(action_data, dict):
                print(f"   ⚠️  Action {action_id} has unexpected data type: {type(action_data)}")
                missing_actions.append(action_id)
                continue
            
            clips_info = action_data.get("Clips", [])
            
            if not clips_info:
                missing_actions.append(action_id)
                continue
            
            action_has_videos = False
            for clip_info in clips_info:
                total_clips += 1
                
                if isinstance(clip_info, str):
                    raw_url = clip_info
                elif isinstance(clip_info, dict):
                    raw_url = clip_info.get("Url", "")
                else:
                    print(f"   ⚠️  Unexpected clip info type: {type(clip_info)}")
                    continue
                
                if raw_url:
                    # Process URL as done in dataset.py
                    path_prefix_to_strip = f"Dataset/{split.capitalize()}/"
                    if raw_url.startswith(path_prefix_to_strip):
                        processed_url = raw_url[len(path_prefix_to_strip):]
                    else:
                        processed_url = raw_url
                    
                    # Add .mp4 extension if missing
                    if not Path(processed_url).suffix:
                        processed_url += ".mp4"
                    
                    # Check if file exists
                    video_path = split_dir / processed_url
                    if video_path.exists():
                        found_clips += 1
                        action_has_videos = True
                    else:
                        missing_clips += 1
                        if missing_clips <= 10:  # Only show first 10 for brevity
                            print(f"   ❌ Missing: {video_path}")
            
            if not action_has_videos:
                missing_actions.append(action_id)
    else:
        print(f"   ❌ Unexpected annotations structure: {type(annotations)}")
        return False
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total actions: {total_actions}")
    print(f"   Total clips expected: {total_clips}")
    print(f"   Clips found: {found_clips}")
    print(f"   Clips missing: {missing_clips}")
    print(f"   Actions with no videos: {len(missing_actions)}")
    if total_clips > 0:
        print(f"   Data availability: {found_clips/total_clips*100:.1f}%")
    else:
        print(f"   Data availability: N/A (no clips expected from annotations)")
    
    if missing_clips > 0:
        print(f"\n⚠️  Found {missing_clips} missing video files!")
        print(f"   This explains the 'Video file not found' errors in training")
        
        # Show sample of missing actions
        if missing_actions:
            print(f"\n🚨 Actions with missing videos (sample):")
            for action_id in missing_actions[:20]:  # Show first 20
                print(f"   - action_{action_id}")
            
            if len(missing_actions) > 20:
                print(f"   ... and {len(missing_actions) - 20} more")
    
    return found_clips > 0

def check_file_patterns(dataset_path: str, split: str):
    """Check what file patterns actually exist in the dataset."""
    print(f"\n🔍 Checking actual file patterns in dataset...")
    
    split_dir = Path(dataset_path) / split
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(split_dir.glob(f"**/*{ext}"))
    
    print(f"   Found {len(video_files)} video files")
    
    if video_files:
        print(f"   Sample files:")
        for f in video_files[:10]:
            rel_path = f.relative_to(split_dir)
            print(f"     - {rel_path}")
        
        if len(video_files) > 10:
            print(f"     ... and {len(video_files) - 10} more")
    
    # Check action directories
    action_dirs = [d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith('action_')]
    print(f"   Found {len(action_dirs)} action directories")
    
    if action_dirs:
        print(f"   Sample action directories:")
        for d in action_dirs[:10]:
            files_in_dir = list(d.glob("*"))
            print(f"     - {d.name}: {len(files_in_dir)} files")
        
        if len(action_dirs) > 10:
            print(f"     ... and {len(action_dirs) - 10} more")
    
    return len(video_files) > 0

def suggest_fixes(dataset_path: str, split: str):
    """Suggest potential fixes for dataset issues."""
    print(f"\n💡 Suggested fixes:")
    print(f"")
    print(f"1. **Verify dataset download:**")
    print(f"   - Make sure you've downloaded the complete MVFouls dataset")
    print(f"   - Check if download was interrupted or incomplete")
    print(f"")
    print(f"2. **Check dataset path:**")
    print(f"   - Current path: {Path(dataset_path).absolute()}")
    print(f"   - Make sure this points to the correct dataset location")
    print(f"   - Expected structure:")
    print(f"     {dataset_path}/")
    print(f"     ├── {split}/")
    print(f"     │   ├── annotations.json")
    print(f"     │   ├── action_0001/")
    print(f"     │   │   ├── clip_0.mp4")
    print(f"     │   │   └── clip_1.mp4")
    print(f"     │   └── action_0002/")
    print(f"     │       └── clip_0.mp4")
    print(f"")
    print(f"3. **Extract/unzip files:**")
    print(f"   - If videos are in archives, make sure they're extracted")
    print(f"   - Check for .zip, .tar.gz, .7z files that need extraction")
    print(f"")
    print(f"4. **Check file permissions:**")
    print(f"   - Ensure read permissions on all video files")
    print(f"   - Run: chmod -R 755 {dataset_path}")
    print(f"")
    print(f"5. **Verify file format:**")
    print(f"   - Ensure video files are in .mp4 format")
    print(f"   - Convert if necessary: ffmpeg -i input.mkv output.mp4")

def main():
    parser = argparse.ArgumentParser(description='Diagnose dataset loading issues')
    parser.add_argument('--dataset-path', type=str, default='mvfouls', help='Path to dataset')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to analyze')
    
    args = parser.parse_args()
    
    print("🏥 Dataset Diagnostic Tool")
    print("=" * 50)
    
    # Check dataset structure
    structure_ok = analyze_dataset_structure(args.dataset_path, args.split)
    
    # Check file patterns
    files_found = check_file_patterns(args.dataset_path, args.split)
    
    # Provide suggestions
    suggest_fixes(args.dataset_path, args.split)
    
    print("\n" + "=" * 50)
    if structure_ok and files_found:
        print("✅ Dataset appears to have some video files available")
        print("   Some missing files are normal, but too many will hurt training")
    else:
        print("❌ Dataset has significant issues that need to be resolved")
        print("   Training will not work properly with this dataset state")
    
    return 0 if (structure_ok and files_found) else 1

if __name__ == "__main__":
    sys.exit(main()) 
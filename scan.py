#!/usr/bin/env python3
"""
Dataset Annotation Scanner - Analyzes SoccerNet MVFOUL annotations
to understand the structure and all possible annotation values.
"""
import json
import os
from collections import Counter, defaultdict
import pandas as pd
from pathlib import Path

def load_annotations(dataset_path, split):
    """Load annotations for a specific split."""
    annotation_file = os.path.join(dataset_path, split, "annotations.json")
    if not os.path.exists(annotation_file):
        print(f"❌ Annotation file not found: {annotation_file}")
        return None
    
    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Handle SoccerNet MVFoul annotation structure
        if isinstance(data, dict) and 'Actions' in data:
            # This is the correct SoccerNet structure: {"Set": "train", "Actions": {...}}
            actions_data = data['Actions']
            annotations = []
            
            print(f"📊 Found SoccerNet format with {len(actions_data)} actions in {split}")
            print(f"   Set: {data.get('Set', 'unknown')}")
            print(f"   Number of actions: {data.get('Number of actions', len(actions_data))}")
            
            # Convert actions dict to list of annotations
            for action_id, annotation in actions_data.items():
                if isinstance(annotation, dict):
                    # Add the action_id as a field
                    annotation['action_id'] = action_id
                    annotations.append(annotation)
                else:
                    print(f"⚠️  Unexpected annotation type for action '{action_id}': {type(annotation)}")
            
            print(f"✅ Converted to {len(annotations)} annotations from {split}")
            return annotations
            
        elif isinstance(data, dict):
            # Legacy format: direct dictionary of action_id -> annotation
            annotations = []
            print(f"📊 Loaded dictionary with {len(data)} keys from {split}")
            
            for key, value in data.items():
                if isinstance(value, dict):
                    value['action_id'] = key
                    annotations.append(value)
                else:
                    print(f"⚠️  Unexpected value type for key '{key}': {type(value)}")
            
            print(f"✅ Converted to {len(annotations)} annotations from {split}")
            return annotations
            
        elif isinstance(data, list):
            print(f"✅ Loaded {len(data)} annotations from {split}")
            return data
        else:
            print(f"❌ Unexpected JSON structure: {type(data)}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading {annotation_file}: {e}")
        return None

def analyze_annotation_structure(annotations):
    """Analyze the basic structure of annotations."""
    print("\n" + "="*60)
    print("📊 ANNOTATION STRUCTURE ANALYSIS")
    print("="*60)
    
    if not annotations:
        return
    
    # Get sample annotation to understand structure
    sample = annotations[0] if annotations else {}
    print(f"\n🔍 Sample annotation keys: {list(sample.keys())}")
    
    # Analyze each field
    all_keys = set()
    for ann in annotations:
        all_keys.update(ann.keys())
    
    print(f"\n📋 All unique annotation fields ({len(all_keys)}):")
    for key in sorted(all_keys):
        print(f"   • {key}")
    
    return all_keys

def analyze_field_values(annotations, field_name):
    """Analyze all possible values for a specific field."""
    values = []
    missing_count = 0
    
    for ann in annotations:
        if field_name in ann:
            val = ann[field_name]
            if val is not None and val != "":
                values.append(val)
            else:
                missing_count += 1
        else:
            missing_count += 1
    
    unique_values = list(set(values))
    value_counts = Counter(values)
    
    return {
        'total_samples': len(annotations),
        'present_count': len(values),
        'missing_count': missing_count,
        'unique_values': sorted(unique_values),
        'unique_count': len(unique_values),
        'value_counts': value_counts,
        'most_common': value_counts.most_common()
    }

def print_field_analysis(field_name, analysis):
    """Print detailed analysis for a field."""
    print(f"\n🏷️  FIELD: '{field_name}'")
    print("-" * 50)
    print(f"   Total samples: {analysis['total_samples']}")
    print(f"   Present: {analysis['present_count']}")
    print(f"   Missing: {analysis['missing_count']}")
    print(f"   Unique values: {analysis['unique_count']}")
    
    if analysis['unique_count'] <= 20:
        print(f"   All values: {analysis['unique_values']}")
    else:
        print(f"   Sample values: {analysis['unique_values'][:10]}... (showing first 10)")
    
    print(f"\n   📈 Value distribution:")
    for value, count in analysis['most_common'][:10]:  # Top 10
        percentage = (count / analysis['present_count']) * 100
        print(f"      '{value}': {count} ({percentage:.1f}%)")

def analyze_severity_action_mapping(annotations):
    """Analyze severity and action type mappings."""
    print("\n" + "="*60)
    print("🎯 SEVERITY & ACTION TYPE ANALYSIS")
    print("="*60)
    
    severity_values = []
    action_values = []
    offence_values = []
    
    for ann in annotations:
        if 'Severity' in ann:
            severity_values.append(ann['Severity'])
        if 'Action class' in ann:
            action_values.append(ann['Action class'])
        if 'Offence' in ann:
            offence_values.append(ann['Offence'])
    
    print(f"\n🔥 SEVERITY analysis:")
    severity_counts = Counter(severity_values)
    for sev, count in severity_counts.most_common():
        print(f"   '{sev}': {count}")
    
    print(f"\n⚽ ACTION CLASS analysis:")
    action_counts = Counter(action_values)
    for act, count in action_counts.most_common():
        print(f"   '{act}': {count}")
    
    print(f"\n🚨 OFFENCE analysis:")
    offence_counts = Counter(offence_values)
    for off, count in offence_counts.most_common():
        print(f"   '{off}': {count}")
    
    return {
        'severity_mapping': severity_counts,
        'action_mapping': action_counts,
        'offence_mapping': offence_counts
    }

def check_annotation_consistency(annotations):
    """Check for potential annotation issues."""
    print("\n" + "="*60)
    print("🔍 ANNOTATION CONSISTENCY CHECK")
    print("="*60)
    
    issues = []
    
    # Check for missing required fields
    required_fields = ['Severity', 'Action class', 'Offence']
    for field in required_fields:
        missing = sum(1 for ann in annotations if field not in ann or ann[field] is None or ann[field] == "")
        if missing > 0:
            issues.append(f"Missing {field}: {missing} samples")
            print(f"⚠️  Missing {field}: {missing} samples")
    
    # Check for unexpected data types
    for ann in annotations:
        for field in ['Severity', 'Action class', 'Offence']:
            if field in ann and ann[field] is not None:
                if not isinstance(ann[field], (str, int)):
                    issues.append(f"Unexpected type for {field}: {type(ann[field])}")
                    print(f"⚠️  Unexpected type for {field}: {type(ann[field])}")
    
    # Check severity range (if numeric)
    severity_values = [ann.get('Severity') for ann in annotations if ann.get('Severity') is not None]
    numeric_severities = []
    for sev in severity_values:
        try:
            numeric_severities.append(int(sev))
        except (ValueError, TypeError):
            pass
    
    if numeric_severities:
        min_sev, max_sev = min(numeric_severities), max(numeric_severities)
        print(f"📊 Severity range: {min_sev} - {max_sev}")
        if min_sev < 0 or max_sev > 3:
            issues.append(f"Severity out of expected range [0-3]: {min_sev}-{max_sev}")
            print(f"❌ Severity out of expected range [0-3]: {min_sev}-{max_sev}")
    
    if not issues:
        print("✅ No obvious consistency issues found")
    
    return issues

def generate_mapping_suggestions(annotations):
    """Generate suggestions for creating label mappings."""
    print("\n" + "="*60)
    print("💡 LABEL MAPPING SUGGESTIONS")
    print("="*60)
    
    # Analyze main categorical fields
    categorical_fields = ['Severity', 'Action class', 'Offence', 'Bodypart', 'Contact']
    
    for field in categorical_fields:
        if any(field in ann for ann in annotations):
            analysis = analyze_field_values(annotations, field)
            if analysis['unique_count'] > 0:
                print(f"\n🏷️  {field.upper()} mapping suggestion:")
                print(f"   Unique values: {analysis['unique_count']}")
                for i, value in enumerate(analysis['unique_values']):
                    print(f"   {i}: '{value}'")

def create_summary_report(dataset_path):
    """Create a comprehensive summary report."""
    print("🚀 SOCCERNET MVFOUL ANNOTATION SCANNER")
    print("="*60)
    print(f"📁 Dataset path: {dataset_path}")
    
    # Analyze each split
    splits = ['train', 'test', 'challenge']
    all_annotations = {}
    
    for split in splits:
        annotations = load_annotations(dataset_path, split)
        if annotations:
            all_annotations[split] = annotations
    
    if not all_annotations:
        print("❌ No annotations found!")
        return
    
    # Use train split for detailed analysis (usually largest)
    main_split = 'train' if 'train' in all_annotations else list(all_annotations.keys())[0]
    main_annotations = all_annotations[main_split]
    
    print(f"\n📊 Using '{main_split}' split for detailed analysis")
    
    # 1. Structure analysis
    all_keys = analyze_annotation_structure(main_annotations)
    
    # 2. Field-by-field analysis
    print("\n" + "="*60)
    print("📋 DETAILED FIELD ANALYSIS")
    print("="*60)
    
    important_fields = ['Severity', 'Action class', 'Offence', 'Bodypart', 'Contact', 
                       'Upper body part', 'Multiple fouls',
                       'Try to play', 'Touch ball', 'Handball', 'Handball offence']
    
    field_analyses = {}
    for field in important_fields:
        if any(field in ann for ann in main_annotations):
            analysis = analyze_field_values(main_annotations, field)
            field_analyses[field] = analysis
            print_field_analysis(field, analysis)
    
    # 3. Severity and action mapping
    mappings = analyze_severity_action_mapping(main_annotations)
    
    # 4. Consistency check
    issues = check_annotation_consistency(main_annotations)
    
    # 5. Mapping suggestions
    generate_mapping_suggestions(main_annotations)
    
    # 6. Summary statistics
    print("\n" + "="*60)
    print("📈 SUMMARY STATISTICS")
    print("="*60)
    
    for split, annotations in all_annotations.items():
        print(f"\n{split.upper()} split:")
        print(f"   Total samples: {len(annotations)}")
        
        # Count available fields
        field_presence = defaultdict(int)
        for ann in annotations:
            for field in ann.keys():
                field_presence[field] += 1
        
        print(f"   Available fields ({len(field_presence)}):")
        for field, count in sorted(field_presence.items()):
            percentage = (count / len(annotations)) * 100
            print(f"      {field}: {count}/{len(annotations)} ({percentage:.1f}%)")
    
    return {
        'annotations': all_annotations,
        'field_analyses': field_analyses,
        'mappings': mappings,
        'issues': issues
    }

def main():
    """Main analysis function."""
    # Try different dataset paths
    possible_paths = ["mvfouls", "../mvfouls", "../../mvfouls"]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if not dataset_path:
        print("❌ Could not find mvfouls dataset directory")
        print(f"   Tried: {possible_paths}")
        return
    
    print(f"✅ Found dataset at: {dataset_path}")
    
    try:
        results = create_summary_report(dataset_path)
        
        print("\n" + "="*60)
        print("🎉 ANALYSIS COMPLETE!")
        print("="*60)
        print("💡 Use this information to understand your annotation structure")
        print("💡 Check for any 'wrong annotation' issues identified above")
        print("💡 Use the mapping suggestions to create proper label encodings")
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

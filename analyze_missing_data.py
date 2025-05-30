import json
from collections import Counter, defaultdict

# Load annotations
with open('mvfouls/train/annotations.json', 'r') as f:
    data = json.load(f)

print("🔍 MISSING DATA ANALYSIS")
print("=" * 50)

missing_severity = []
missing_action_class = []
both_missing = []
total_actions = 0

# Analyze missing patterns
for action_id, details in data['Actions'].items():
    total_actions += 1
    severity = details.get('Severity', '')
    action_class = details.get('Action class', '')
    offence = details.get('Offence', '')
    
    severity_missing = (severity == '' or severity is None)
    action_missing = (action_class == '' or action_class is None)
    
    if severity_missing:
        missing_severity.append({
            'action_id': action_id,
            'offence': offence,
            'action_class': action_class
        })
    
    if action_missing:
        missing_action_class.append({
            'action_id': action_id,
            'offence': offence,
            'severity': severity
        })
    
    if severity_missing and action_missing:
        both_missing.append(action_id)

print(f"Total actions: {total_actions}")
print(f"Missing severity: {len(missing_severity)} ({len(missing_severity)/total_actions*100:.1f}%)")
print(f"Missing action class: {len(missing_action_class)} ({len(missing_action_class)/total_actions*100:.1f}%)")
print(f"Missing both: {len(both_missing)} ({len(both_missing)/total_actions*100:.1f}%)")

print(f"\n📊 MISSING SEVERITY PATTERNS:")
if missing_severity:
    # Analyze offence patterns for missing severity
    offence_patterns = Counter([item['offence'] for item in missing_severity])
    print("Offence distribution for missing severity:")
    for offence, count in offence_patterns.most_common():
        print(f"  '{offence}': {count}")
    
    # Analyze action class patterns for missing severity
    action_patterns = Counter([item['action_class'] for item in missing_severity if item['action_class']])
    print("Action class distribution for missing severity (where available):")
    for action, count in action_patterns.most_common():
        print(f"  '{action}': {count}")

print(f"\n📊 MISSING ACTION CLASS PATTERNS:")
if missing_action_class:
    # Analyze offence patterns for missing action class
    offence_patterns = Counter([item['offence'] for item in missing_action_class])
    print("Offence distribution for missing action class:")
    for offence, count in offence_patterns.most_common():
        print(f"  '{offence}': {count}")
    
    # Analyze severity patterns for missing action class
    severity_patterns = Counter([item['severity'] for item in missing_action_class if item['severity']])
    print("Severity distribution for missing action class (where available):")
    for severity, count in severity_patterns.most_common():
        print(f"  '{severity}': {count}")

print(f"\n💡 RECOMMENDED HANDLING:")
print("Based on patterns, missing values will be assigned as follows:")

if missing_severity:
    # Calculate what defaults would be assigned
    assignments = defaultdict(int)
    for item in missing_severity:
        if item['offence'] == "No offence":
            assignments['1.0 (No offence)'] += 1
        elif item['offence'] == "Offence":
            assignments['2.0 (Offence)'] += 1
        elif item['offence'] == "Between":
            assignments['1.0 (Between)'] += 1
        else:
            assignments['1.0 (Default)'] += 1
    
    print("Missing severity assignments:")
    for assignment, count in assignments.items():
        print(f"  {assignment}: {count}")

if missing_action_class:
    print(f"Missing action class assignments:")
    print(f"  Standing tackling (default): {len(missing_action_class)}")

print(f"\n✅ With graceful handling, {total_actions - len(both_missing)} actions will be retained")
print(f"🚨 {len(both_missing)} actions have both missing and will need further analysis") 
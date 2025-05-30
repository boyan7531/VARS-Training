import json
from collections import Counter, defaultdict

# Load annotations
with open('mvfouls/train/annotations.json', 'r') as f:
    data = json.load(f)

# Analyze correlation between Severity and Offence fields
severity_offence_pairs = []
severity_details = defaultdict(lambda: {'total': 0, 'offences': Counter(), 'actions': Counter()})

for action_id, details in data['Actions'].items():
    severity = details.get('Severity', '')
    offence = details.get('Offence', '')
    action_class = details.get('Action class', '')
    
    if severity:
        severity_details[severity]['total'] += 1
        severity_details[severity]['offences'][offence] += 1
        severity_details[severity]['actions'][action_class] += 1
        
        if offence:
            severity_offence_pairs.append((severity, offence))

print("🔥 SEVERITY LEVEL ANALYSIS")
print("=" * 50)

for severity in sorted(severity_details.keys()):
    details = severity_details[severity]
    print(f"\n📊 SEVERITY {severity} ({details['total']} total cases):")
    
    print("  Offence distribution:")
    for offence, count in details['offences'].most_common():
        percentage = (count / details['total']) * 100
        print(f"    {offence}: {count} ({percentage:.1f}%)")
    
    print("  Top action types:")
    for action, count in details['actions'].most_common(3):
        percentage = (count / details['total']) * 100
        print(f"    {action}: {count} ({percentage:.1f}%)")

print(f"\n🔍 SUMMARY INTERPRETATION:")
print("Based on the data patterns, severity levels likely represent:")

# Analyze patterns to infer meanings
for severity in sorted(severity_details.keys()):
    details = severity_details[severity]
    offence_ratio = details['offences'].get('Offence', 0) / details['total']
    no_offence_ratio = details['offences'].get('No offence', 0) / details['total']
    
    print(f"\nSeverity {severity}:")
    if no_offence_ratio > 0.8:
        print(f"  → Likely: No offence or very minor incident")
    elif offence_ratio > 0.8:
        if severity == "1.0":
            print(f"  → Likely: Minor offence (no card)")
        elif severity == "2.0":
            print(f"  → Likely: Minor to moderate offence")
        elif severity == "3.0":
            print(f"  → Likely: Moderate offence (yellow card level)")
        elif severity == "4.0":
            print(f"  → Likely: Serious offence (yellow/red card)")
        elif severity == "5.0":
            print(f"  → Likely: Very serious offence (red card level)")
    else:
        print(f"  → Mixed cases - need referee judgment") 
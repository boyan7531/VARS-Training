import json
import sys

# Quick peek at annotation structure
with open('mvfouls/train/annotations.json', 'r') as f:
    data = json.load(f)

print(f"Total annotations: {len(data)}")
print(f"Data type: {type(data)}")

if isinstance(data, dict):
    first_key = list(data.keys())[0]
    first_annotation = data[first_key]
    
    print(f"\nFirst key: '{first_key}'")
    print(f"First annotation type: {type(first_annotation)}")
    print(f"First annotation content: {repr(first_annotation)}")
    
    # Show all keys and their value types
    print(f"\nAll top-level keys:")
    for key in list(data.keys())[:10]:
        value = data[key]
        print(f"  '{key}': {type(value)} - {repr(str(value)[:100])}")
        
    # Try to understand if it's JSON strings
    if isinstance(first_annotation, str):
        try:
            parsed = json.loads(first_annotation)
            print(f"\nFirst annotation parsed as JSON:")
            print(f"  Type: {type(parsed)}")
            if isinstance(parsed, dict):
                print(f"  Keys: {list(parsed.keys())[:10]}")
                print(f"  Sample values:")
                for k, v in list(parsed.items())[:5]:
                    print(f"    '{k}': {v} (type: {type(v)})")
        except json.JSONDecodeError:
            print(f"\nFirst annotation is not valid JSON string") 
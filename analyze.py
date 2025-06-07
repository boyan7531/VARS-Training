import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

def analyze_dataset_distribution(annotation_path: str):
    """
    Analyzes and prints the distribution of labels in the SoccerNet MVFoul dataset.

    Args:
        annotation_path (str): The full path to the annotation JSON file.
    """
    annotation_file = Path(annotation_path)
    if not annotation_file.exists():
        print(f"Error: Annotation file not found at '{annotation_path}'")
        return

    print(f"Analyzing annotations from: {annotation_file}\n")

    with open(annotation_file, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {annotation_file}: {e}")
            return

    actions = data.get("Actions", {})
    if not actions:
        print("No 'Actions' found in the annotation file.")
        return

    total_actions = len(actions)
    print(f"Found a total of {total_actions} actions.\n")

    # --- Initialize counters for all relevant fields ---
    distributions = {
        "Severity": defaultdict(int),
        "Action class": defaultdict(int),
        "Offence": defaultdict(int),
        "Contact": defaultdict(int),
        "Bodypart": defaultdict(int),
        "Upper body part": defaultdict(int),
        # "Lower body part": defaultdict(int), # This field does not exist in the data
        "Multiple fouls": defaultdict(int),
        "Try to play": defaultdict(int),
        "Touch ball": defaultdict(int),
        "Handball": defaultdict(int),
        "Handball offence": defaultdict(int)
    }

    # --- Tally the counts for each field ---
    for action_details in actions.values():
        for field, counter in distributions.items():
            value = action_details.get(field, "N/A") # Use "N/A" for missing fields
            counter[value] += 1
    
    # --- Print the distributions in a clean table format ---
    for field, counter in distributions.items():
        if not any(counter.values()): # Skip empty fields
            continue

        print(f"--- {field} Distribution ---")
        
        # Prepare data for pandas DataFrame
        sorted_items = sorted(counter.items(), key=lambda item: item[1], reverse=True)
        
        # Handle case where a field might be completely missing
        if not sorted_items:
            print("No data found for this field.\n")
            continue
            
        labels, counts = zip(*sorted_items)
        percentages = [(count / total_actions) * 100 for count in counts]
        
        df = pd.DataFrame({
            'Label': labels,
            'Count': counts,
            'Percentage (%)': percentages
        })
        
        # Format the percentage column
        df['Percentage (%)'] = df['Percentage (%)'].map('{:.2f}%'.format)
        
        print(df.to_string(index=False))
        print("\n" + "="*40 + "\n")


if __name__ == '__main__':
    # Define the path to your training annotations file
    ANNOTATION_FILE_PATH = "mvfouls/train/annotations.json"
    
    # Run the analysis
    analyze_dataset_distribution(ANNOTATION_FILE_PATH)

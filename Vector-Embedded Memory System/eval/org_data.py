import json
import os
import sys

# --- CONFIGURATION ---
source_file = 'locomo10.json'
output_dir = 'datasets/'

def run_extraction():
    print("--- Thesis Data Extraction Tool ---")
    
    # 1. Check if source file exists
    if not os.path.exists(source_file):
        print(f"ERROR: Cannot find '{source_file}' in this folder.")
        print("Please download it from the LoCoMo GitHub and place it here.")
        return

    # 2. Try to load the JSON
    print(f"Loading {source_file}... (This might take a moment)")
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to read JSON file. Details: {e}")
        return

    # 3. Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 4. Map LoCoMo categories to your Thesis files
    files_map = {
        4: 'single_hop.json',
        1: 'multi_hop.json',
        5: 'abstention.json'
    }
    extracted_counts = {k: 0 for k in files_map.keys()}
    results = {k: [] for k in files_map.keys()}

    print("Extracting QA pairs by category...")
    for sample in data:
        if 'qa' not in sample: continue
        
        for qa_pair in sample['qa']:
            cat = qa_pair.get('category')
            if cat in files_map:
                results[cat].append({
                    "id": sample.get('sample_id'),
                    "question": qa_pair.get('question'),
                    "ground_truth": qa_pair.get('answer'),
                    "context_evidence": qa_pair.get('evidence', [])
                })
                extracted_counts[cat] += 1

    # 5. Save the files
    for cat_id, filename in files_map.items():
        path = os.path.join(output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results[cat_id], f, indent=4)
        print(f" SUCCESS: Saved {extracted_counts[cat_id]} samples to {path}")

    print("\n--- Extraction Complete! ---")

if __name__ == "__main__":
    run_extraction()
    # This line keeps the window open even if you double-click
    input("\nPress Enter to close this window...")
"""
Download LongMemEval dataset with retry logic
"""
from datasets import load_dataset
import os

# Create Benchmarks directory if it doesn't exist
os.makedirs("Benchmarks", exist_ok=True)

print("Downloading LongMemEval dataset...")
try:
    dataset = load_dataset("xiaowu0162/longmemeval-cleaned", cache_dir="./Benchmarks/longmemeval_cache")
    print(f"Downloaded successfully! Dataset keys: {dataset.keys()}")

    # Save the oracle (full) split
    print("Saving oracle split...")
    oracle_data = dataset['oracle']
    oracle_data.to_json("./Benchmarks/longmemeval_oracle.json")
    print(f"Saved: ./Benchmarks/longmemeval_oracle.json ({len(oracle_data)} items)")

except Exception as e:
    print(f"Error downloading: {e}")
    print("Trying alternative approach with manual retry...")
    import time
    for attempt in range(3):
        try:
            print(f"Attempt {attempt + 1}/3...")
            dataset = load_dataset(
                "xiaowu0162/longmemeval-cleaned",
                cache_dir="./Benchmarks/longmemeval_cache",
                timeout=60
            )
            oracle_data = dataset['oracle']
            oracle_data.to_json("./Benchmarks/longmemeval_oracle.json")
            print("Success!")
            break
        except Exception as e2:
            if attempt < 2:
                wait_time = 5 * (attempt + 1)
                print(f"Attempt {attempt + 1} failed, waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"All attempts failed: {e2}")

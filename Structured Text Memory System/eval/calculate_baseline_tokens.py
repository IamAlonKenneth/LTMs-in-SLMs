import json
from transformers import AutoTokenizer

# Load the exact tokenizer you are evaluating
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

def get_locomo_baseline(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    total_tokens = 0
    total_qas = 0
    
    for sample in data:
        conv = sample.get("conversation", {})
        # Combine all text from all sessions into one massive string
        full_history = ""
        for key, value in conv.items():
            if key.startswith("session_") and not key.endswith("_date_time"):
                for turn in value:
                    full_history += f"{turn.get('speaker', '')}: {turn.get('text', '')}\n"
        
        # Count tokens for this massive string
        tokens = len(tokenizer.encode(full_history, add_special_tokens=False))
        
        # Every QA pair in this conversation would require this full context baseline
        qa_count = len(sample.get("qa", []))
        total_tokens += (tokens * qa_count)
        total_qas += qa_count
        
    return total_tokens / total_qas if total_qas > 0 else 0

def get_longmemeval_baseline(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    total_tokens = 0
    for item in data:
        full_history = ""
        for session in item.get("haystack_sessions", []):
            for turn in session:
                full_history += f"{turn.get('role', '')}: {turn.get('content', '')}\n"
                
        tokens = len(tokenizer.encode(full_history, add_special_tokens=False))
        total_tokens += tokens
        
    return total_tokens / len(data) if len(data) > 0 else 0

# --- RUN THE CALCULATION ---
print("Calculating exact Gemma 3 baselines...")

locomo_avg = get_locomo_baseline("../../../locomo10.json")
print(f"LoCoMo Average Full-Context Tokens: {locomo_avg:.1f}")

lme_avg = get_longmemeval_baseline("../../../longmemeval_s_cleaned.json")
print(f"LongMemEval Average Full-Context Tokens: {lme_avg:.1f}")

# You can average the two if you want one unified number, 
# or pass the specific one based on which dataset you are testing.
print(f"\nUnified Average: {(locomo_avg + lme_avg) / 2:.1f}")

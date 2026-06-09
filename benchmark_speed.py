"""
benchmark_speed.py
==================
Quick per-item speed test for VectorEmbeddedMemory inference.
Measures stabilized speed after model warmup, ignoring cold-start loading time.
Run this from D:\Downloads\LTMs-in-SLMs
"""

import sys, os, json, time
from pathlib import Path

# --- Path setup (same as run_combined_eval.py) ---
_PROJECT_ROOT = Path(__file__).resolve().parent
_VEC_DIR = _PROJECT_ROOT / "Vector-Embedded Memory System"
sys.path.insert(0, str(_VEC_DIR))
sys.path.insert(0, str(_PROJECT_ROOT))

from vector_embed_module import VectorEmbeddedMemory

# --- Load LTM module ---
print("Loading models (cold start)...")
t0 = time.time()
ltm = VectorEmbeddedMemory(
    embedding_model_id = "google/embeddinggemma-300m",
    slm_model_id       = "google/gemma-3-4b-it",
    quantization       = "4bit",
    verbose            = True,
)
load_time = time.time() - t0
print(f"Model loading complete: {load_time:.1f}s\n")

# --- Ingest a few test memories ---
print("Ingesting test memories...")
memories = [
    "The user's thesis focuses on Long-Term Memory for Small Language Models.",
    "LoCoMo is a benchmark for long-context conversation memory evaluation.",
    "The RTX 4050 has 6GB VRAM and supports CUDA 13.2.",
    "Gemma 3 4B with 4-bit quantization uses approximately 2.5GB VRAM.",
    "FAISS IndexFlatL2 provides exact nearest-neighbor search for 768-d vectors.",
]
for text in memories:
    ltm.ingest_memory(text, {"source": "benchmark"})
print(f"Ingested {len(memories)} memories.\n")

# --- Warmup inference (ignored from timing) ---
print("Warmup inference...")
_ = ltm.generate_response("What is this thesis about?", top_k=3, max_new_tokens=64, temperature=0.0)
print("Warmup complete.\n")

# --- Timed runs ---
queries = [
    "What is the user's research topic?",
    "What benchmark is used for evaluation?",
    "What GPU is available?",
    "How does 4-bit quantization help?",
    "What kind of FAISS index is used?",
]

print("--- Timed inference (5 items) ---")
times = []
for i, q in enumerate(queries):
    t0 = time.time()
    result = ltm.generate_response(q, top_k=3, max_new_tokens=128, temperature=0.0)
    elapsed = time.time() - t0
    times.append(elapsed)
    print(f"  [{i+1}/5] {elapsed:.2f}s | {result['response'][:80]}...")

avg = sum(times) / len(times)
print(f"\n--- Results ---")
print(f"Cold-start model loading: {load_time:.1f}s")
print(f"Stabilized per-item time: {avg:.2f}s average (range: {min(times):.2f}s - {max(times):.2f}s)")
print(f"LoCoMo 2,298 items estimate: {2298 * avg / 3600:.1f} hours")

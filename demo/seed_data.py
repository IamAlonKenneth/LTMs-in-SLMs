"""
seed_data.py
────────────
One-time script that pre-populates the demo LTM with a curated set of
memories for the panel demonstration.

Run once before launching the server:
    python demo/seed_data.py

Creates: demo/demo_ltm_store/faiss_index.bin
         demo/demo_ltm_store/ltm_store.json
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Vector-Embedded Memory System"))
from vector_embed_module import VectorEmbeddedMemory

SAVE_DIR = Path(__file__).parent / "demo_ltm_store"

# Demo persona — Alex Santos, a CS graduate student.
# Each tuple is (memory text, metadata dict).
# Covers personal facts, research findings, preferences, and events so the
# panel can ask diverse questions and see targeted retrieval in action.
MEMORIES: list[tuple[str, dict]] = [
    # Personal
    ("My name is Alex Santos. I am a second-year Master's student in Computer "
     "Science at the Polytechnic University of the Philippines.",
     {"topic": "personal", "session": "s01"}),

    ("Alex's supervisor is Professor Maria Reyes, who specialises in efficient "
     "deep learning and on-device AI systems.",
     {"topic": "personal", "session": "s01"}),

    ("Alex's younger sister is named Ria and is studying nursing at a different "
     "university. They talk every weekend.",
     {"topic": "personal", "session": "s02"}),

    ("Alex is fluent in English and Filipino (Tagalog), and is currently learning "
     "Japanese through language exchange apps.",
     {"topic": "personal", "session": "s03"}),

    ("Alex runs 30 minutes every morning around the university campus as his "
     "main form of exercise.",
     {"topic": "personal", "session": "s02"}),

    ("Alex plays acoustic guitar and is part of a small band that performs at "
     "campus events on weekends.",
     {"topic": "hobbies", "session": "s02"}),

    ("Alex's favourite movie genre is science fiction. He recently rewatched "
     "Interstellar and found it even better the second time.",
     {"topic": "hobbies", "session": "s03"}),

    # Preferences
    ("Alex drinks an oat milk flat white with no sugar every morning before "
     "starting his research work.",
     {"topic": "preference", "session": "s01"}),

    ("Alex prefers concise, bullet-point style answers over long paragraph "
     "responses when asking questions.",
     {"topic": "preference", "session": "s01"}),

    ("Alex uses VS Code with the Python and Pylance extensions as his main "
     "development environment. He keeps dark mode enabled everywhere.",
     {"topic": "preference", "session": "s03"}),

    # Research
    ("Alex's thesis is titled 'Long-Term Memory Systems for Small Language Models "
     "on Edge Devices'. It compares three retrieval strategies: Baseline (no "
     "retrieval), BM25 sparse retrieval, and dense vector retrieval using "
     "EmbeddingGemma 300M.",
     {"topic": "research", "session": "s01"}),

    ("Alex uses an NVIDIA RTX 4050 6 GB laptop for all experiments. He runs "
     "Gemma 3 4B with 4-bit NF4 quantisation to stay within the VRAM budget.",
     {"topic": "hardware", "session": "s03"}),

    ("Alex mentioned struggling with CUDA out-of-memory errors when loading "
     "Gemma 3 4B in full precision. Switching to 4-bit quantisation resolved "
     "the issue immediately.",
     {"topic": "technical", "session": "s03"}),

    # Results
    ("Vector retrieval achieved the best answer relevance score of 4.39 out of "
     "5.0 on LongMemEval, compared to 3.49 for the Baseline and 1.0 for BM25. "
     "BM25's score of 1.0 reflects that every response was scored at the minimum "
     "because the retrieved context misled the model.",
     {"topic": "results", "session": "s04"}),

    ("BM25 retrieval had the highest retrieval latency at approximately 8,634 ms "
     "on LongMemEval, driven by query expansion via the SLM. Vector retrieval "
     "was far faster at 77 ms average.",
     {"topic": "results", "session": "s04"}),

    ("Vector retrieval achieved a memory error rate of only 3.41 % on "
     "LongMemEval, versus 59.9 % for BM25 and 15.1 % for the Baseline.",
     {"topic": "results", "session": "s04"}),

    ("On the LoCoMo benchmark (1,861 samples), vector retrieval achieved "
     "Recall@5 of 0.586 versus 0.571 for BM25 — a difference that was "
     "statistically non-significant (p = 0.61).",
     {"topic": "results", "session": "s04"}),

    # Methodology / Statistics
    ("The statistical analysis used repeated-measures ANOVA with "
     "Greenhouse-Geisser correction for violations of sphericity, followed by "
     "Bonferroni-corrected post-hoc paired t-tests. All five metrics showed "
     "significant main effects (p < 0.001).",
     {"topic": "methodology", "session": "s05"}),

    # Events
    ("Alex attended a local Python meetup last month. He found the technical "
     "talks useful but said the networking portion felt awkward.",
     {"topic": "events", "session": "s05"}),

    ("Alex wants to submit his thesis findings as a workshop paper at a major "
     "machine learning conference after the panel defence.",
     {"topic": "research", "session": "s05"}),
]


def main() -> None:
    print("=" * 60)
    print("  LTM Demo — Seeding Memory Store")
    print("=" * 60)
    print(f"\nLoading VectorEmbeddedMemory (4-bit Gemma 3 4B)…")

    ltm = VectorEmbeddedMemory(
        embedding_model_id="google/embeddinggemma-300m",
        slm_model_id="google/gemma-3-4b-it",
        quantization="4bit",
        verbose=True,
    )

    print(f"\nIngesting {len(MEMORIES)} demo memories…\n")
    for i, (text, meta) in enumerate(MEMORIES, 1):
        mem_id = ltm.ingest_memory(text, meta)
        print(f"  [{i:02d}/{len(MEMORIES)}] ID={mem_id}  {text[:70]}…")

    ltm.save_ltm(str(SAVE_DIR))
    print(f"\n✓  {ltm.memory_count()} memories saved to {SAVE_DIR}/")
    print("\nNext step:  python demo/app.py")
    print("Then open:  http://localhost:8000\n")


if __name__ == "__main__":
    main()

# Long-Term Memory Access in Small Language Models

**Can a compact AI remember what you told it last week?**

This thesis project answers that question by equipping a small, locally-running language model with a searchable long-term memory store and measuring how much that memory actually helps.

Three approaches were compared head-to-head on two large-scale benchmarks — all running on a single consumer laptop GPU (NVIDIA RTX 4050, 6 GB VRAM):

| Approach | Plain-language description |
|---|---|
| **Baseline** | No memory — the model answers using only what it already knows |
| **BM25 (Keyword Search)** | Memory searched by matching keywords, like a search engine |
| **Vector (Meaning Search)** | Memory searched by *meaning*, finding relevant memories even without exact word matches |

**The short answer:** searching memory by meaning (vector retrieval) wins — and keyword search surprisingly makes things *worse* than having no memory at all.

---

## Table of Contents

- [Try the Demo](#try-the-demo)
- [Key Results](#key-results)
- [How Each Approach Works](#how-each-approach-works)
- [Repository Layout](#repository-layout)
- [Full Setup](#full-setup)
- [Running the Benchmarks](#running-the-benchmarks)
- [Team](#team)

---

## Demo

The demo lets you chat with a memory-augmented AI assistant. You can switch between Vector and BM25 retrieval modes in real time and see exactly which memories were retrieved to answer your question.

**Step 1 — Install dependencies**

```bash
pip install fastapi uvicorn transformers bitsandbytes faiss-cpu torch numpy
```

> Requires Python 3.10+ and a CUDA-capable GPU. The model loads in 4-bit quantization, so 6 GB VRAM is sufficient.

**Step 2 — Seed the memory store** *(one-time setup)*

This pre-loads 20 memories about a fictional student named Alex — enough to demonstrate cross-session recall.

```bash
python demo/seed_data.py
```

**Step 3 — Start the server**

```bash
python demo/app.py
```

**Step 4 — Open your browser**

```
http://localhost:8000
```

The interface will show a loading indicator while Gemma 3 4B initializes (about 30–60 seconds). Once loaded, try asking things like:

- *"What is Alex researching?"*
- *"What GPU does Alex use?"*
- *"What did Alex say about BM25?"*

The panel on the right shows which memory cards were retrieved and their relevance scores.

---

## Key Results

Results are evaluated on two benchmarks. **LongMemEval** uses long single-session conversations (500 questions). **LoCoMo** uses multi-session personal assistant dialogues (1,861 questions).

### Full Results Table

**LongMemEval (n = 500)**

| Metric | Baseline | BM25 | Vector |
|---|---|---|---|
| Answer Relevance (1–5) | 3.49 | 1.00 | **4.39** |
| Recall@5 | 0.0% | 23.1% | **23.8%** |
| Memory Error Rate | 15.1% | 59.9% | **3.4%** |
| Retrieval Latency | 0 ms | 8,634 ms | **77 ms** |
| Token Utilization | 0.02% | 1.09% | 0.78% |

**LoCoMo (n = 1,861)**

| Metric | Baseline | BM25 | Vector |
|---|---|---|---|
| Answer Relevance (1–5) | 3.82 | 1.81 | **4.13** |
| Recall@5 | 0.0% | 57.1% | **58.6%** |
| Memory Error Rate | 75.0% | 61.4% | **4.2%** |
| Retrieval Latency | 0 ms | 120.7 ms | **66.6 ms** |
| Token Utilization | 0.02% | 0.37% | 0.37% |

### What the numbers mean

**Vector retrieval is the clear winner.** It achieves the highest answer quality scores on both benchmarks, the lowest memory error rate (only 3–4% of queries retrieve the wrong memory), and it's over 1.8× faster than BM25 retrieval.

**BM25 retrieval retrieves well but answers poorly.** On LoCoMo, BM25's Recall@5 is statistically identical to vector's (57.1% vs 58.6%, p = 0.61 — not a meaningful difference). Yet its answer relevance score collapses to 1.00–1.81 out of 5, well below even the no-memory baseline. Finding the right memory and using it coherently are two separate problems — keyword search solves the first but not the second.

**No memory is better than keyword memory (for answer quality).** The baseline outperforms BM25 on answer relevance on both benchmarks. Injecting poorly-matched keyword context actively misleads the model, causing it to produce lower-quality responses than if it had received no context at all.

**Statistical significance:** A repeated-measures ANOVA with Greenhouse–Geisser correction confirms that the differences between all three conditions are statistically significant (p < 0.001) for every metric on both benchmarks. Effect sizes (partial η²) range from 0.31 to 0.997 — all far above the conventional threshold for a "large" effect (η² = 0.14). The one exception is BM25 vs. Vector on Recall@5, which is not significant — confirming they retrieve at the same rate.

---

## How Each Approach Works

### Baseline — No Memory

The model receives only the current question and answers purely from its own built-in knowledge. There is no retrieval step, no memory store, and no extra context injected into the prompt. This is the control condition.

### BM25 — Keyword Search

Before searching, the model expands the query into related keyword variants (e.g., *"What does Alex study?"* → *["research topic", "graduate student", "thesis subject", "CS program"]*). It then searches a SQLite full-text index using those keywords and ranks the results by BM25 score adjusted for how recently a memory was created. The top 5 memories are prepended to the prompt.

The problem: memories that happen to share common words but not meaning get retrieved, and irrelevant context confuses the model.

### Vector — Meaning Search

A second, smaller model (EmbeddingGemma 300M) converts the query and every stored memory into a 768-number "meaning fingerprint." These are stored in a FAISS index. At query time, the system finds the 5 memories whose fingerprints are closest in meaning to the query — not just in words — and prepends them to the prompt.

This is why BM25 and Vector retrieve at similar rates (Recall@5 is nearly identical) yet answer quality is completely different: vector retrieval finds memories that *fit the context*, not just memories that *share vocabulary*.

---

## Repository Layout

```
LTMs-in-SLMs/
│
├── demo/                          # Interactive web demo
│   ├── app.py                     # FastAPI server — loads models and handles requests
│   ├── ui.html                    # Chat interface with memory panel
│   ├── seed_data.py               # One-time script to pre-load demo memories
│   └── demo_ltm_store/            # Saved FAISS index and memory store (after seeding)
│
├── Vector-Embedded Memory System/ # Core LTM implementation
│   ├── vector_embed_module.py     # VectorEmbeddedMemory class — FAISS + embeddings + Gemma 3 4B
│   └── cuda_bootstrap.py         # WSL2 CUDA pre-initialization workaround
│
├── Benchmarks/
│   ├── locomo/                    # LoCoMo benchmark (1,861 QA pairs across multi-session dialogues)
│   └── longmemeval_cache/         # LongMemEval dataset cache (500 questions)
│
├── baseline_eval.py               # Runs the no-memory Gemma 3 4B baseline on both benchmarks
├── statistical_analysis.py        # ANOVA + post-hoc tests on collected results
├── dashboard.html                 # Interactive results dashboard (open in browser)
│
├── results/                       # Evaluation outputs
│   ├── baseline/                  # Baseline generation results
│   ├── statistical/               # ANOVA and effect size JSON files
│   └── figures/                   # Charts and visualizations
│
├── results_analysis.ipynb         # Jupyter notebook — data exploration and aggregation
├── results_visualization.ipynb    # Jupyter notebook — chart generation
└── chapter4_5_draft.md            # Thesis chapter 4 & 5 draft
```

---

## Full Setup

### Requirements

- Python 3.10 or newer
- CUDA-capable GPU with at least 6 GB VRAM (tested on NVIDIA RTX 4050 6 GB)
- Windows or Linux (WSL2 supported via `cuda_bootstrap.py`)

### Install Python dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers bitsandbytes accelerate
pip install faiss-cpu fastapi uvicorn pydantic
pip install python-dotenv tqdm
```

### HuggingFace access

The models used are:
- `google/gemma-3-4b-it` — requires accepting the license at huggingface.co/google/gemma-3-4b-it
- `google/embedding-gemma-300m` — open access

Set your HuggingFace token:

```bash
# Windows
set HF_TOKEN=hf_your_token_here

# Linux / WSL2
export HF_TOKEN=hf_your_token_here
```

---

## Running the Benchmarks

### Baseline (no memory)

```bash
python baseline_eval.py \
    --longmemeval-data Benchmarks/longmemeval_cache/longmemeval_s_cleaned.json \
    --locomo-data Benchmarks/locomo/data/locomo10.json \
    --output-dir results/baseline \
    --quantization 4bit \
    --batch-size 4
```

To resume an interrupted run, point `--output-dir` at the existing timestamped folder — the script auto-detects the checkpoint.

### Results dashboard

Open `dashboard.html` directly in any browser. No server required — all data is embedded in the file.

---

## Team

**Authors:** John Kenneth P. Alon · Rexter V. Gonzales · Erika B. Mariano

**Adviser:** Dr. Lysa V. Comia

**Institution:** Mapúa Institute of Technology

**Date:** February 2026

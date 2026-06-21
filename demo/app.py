"""
Loads VectorEmbeddedMemory (FAISS + EmbeddingGemma 300M + Gemma 3 4B) once.
BM25 mode shares the same SLM for generation — only one copy of Gemma 3 4B
is ever loaded, keeping VRAM usage within the RTX 4050 6 GB budget.

Run:
    python demo/app.py
Then open: http://localhost:8000
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import sqlite3
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add the Vector LTM module directory to sys.path for the import below
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Vector-Embedded Memory System"))
from vector_embed_module import VectorEmbeddedMemory  # noqa: E402

# App instance and shared server state — initialized once at startup
app = FastAPI(title="LTM-in-SLM Demo")
executor = ThreadPoolExecutor(max_workers=1)   # single GPU inference thread

ltm: VectorEmbeddedMemory | None = None
fts_conn: sqlite3.Connection | None = None
_ready = False

DEMO_LTM_DIR = Path(__file__).parent / "demo_ltm_store"
UI_FILE      = Path(__file__).parent / "ui.html"


# Request models
class ChatRequest(BaseModel):
    query: str
    mode: str = "vector"              # "vector" | "bm25"
    use_query_expansion: bool = True
    top_k: int = 5

class IngestRequest(BaseModel):
    text: str
    metadata: dict[str, Any] = {}


# Startup — model loading is offloaded to the thread pool so the server responds immediately
@app.on_event("startup")
async def startup() -> None:
    global ltm, fts_conn, _ready
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, _load_models)

async def _startup_bg() -> None:
    global _ready
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, _load_models)
    _ready = True

def _load_models() -> None:
    global ltm, fts_conn, _ready
    print("[Demo] Loading VectorEmbeddedMemory…")
    ltm = VectorEmbeddedMemory(
        embedding_model_id="google/embeddinggemma-300m",
        slm_model_id="google/gemma-3-4b-it",
        quantization="4bit",
        verbose=True,
    )
    if DEMO_LTM_DIR.exists():
        ltm.load_ltm(str(DEMO_LTM_DIR))
        print(f"[Demo] Loaded {ltm.memory_count()} seed memories ✓")
    else:
        print("[Demo] No seed store found — run seed_data.py first.")

    # Parallel SQLite FTS5 index for BM25 mode (in-memory, rebuilt from ltm_store)
    fts_conn = sqlite3.connect(":memory:", check_same_thread=False)
    fts_conn.row_factory = sqlite3.Row
    fts_conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS docs
        USING fts5(content, timestamp UNINDEXED, metadata UNINDEXED)
    """)
    for doc_id, mem in ltm.ltm_store.items():
        if mem.get("text") not in (None, "[DELETED]"):
            fts_conn.execute(
                "INSERT INTO docs(rowid, content, timestamp, metadata) VALUES (?,?,?,?)",
                (int(doc_id), mem["text"],
                 mem.get("timestamp", ""), json.dumps(mem.get("metadata", {})))
            )
    fts_conn.commit()
    _ready = True
    print("[Demo] FTS5 index ready ✓  |  Server fully initialised ✓")


# Routes
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(UI_FILE)

@app.get("/status")
async def status() -> dict:
    return {"ready": _ready, "memory_count": ltm.memory_count() if ltm else 0}

@app.get("/memories")
async def list_memories() -> dict:
    if ltm is None:
        return {"memories": [], "count": 0}
    active = sorted(
        [{"id": int(k), **v}
         for k, v in ltm.ltm_store.items()
         if v.get("text", "") != "[DELETED]"],
        key=lambda x: x["id"],
    )
    return {"memories": active, "count": len(active)}

@app.post("/ingest")
async def ingest(req: IngestRequest) -> dict:
    loop = asyncio.get_event_loop()
    mem_id = await loop.run_in_executor(executor, _ingest, req.text, req.metadata)
    return {"memory_id": mem_id, "count": ltm.memory_count()}

@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    if not _ready:
        return JSONResponse({"error": "Models still loading — please wait."}, status_code=503)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _chat, req)

@app.post("/reset")
async def reset() -> dict:
    ltm.ltm_store = {}
    ltm._next_faiss_id = 0
    ltm.rebuild_index()
    fts_conn.executescript("""
        DROP TABLE IF EXISTS docs;
        CREATE VIRTUAL TABLE docs
        USING fts5(content, timestamp UNINDEXED, metadata UNINDEXED);
    """)
    return {"status": "ok", "count": 0}


# Sync helpers — these run in the thread pool to keep async routes non-blocking
def _ingest(text: str, metadata: dict) -> int:
    mem_id = ltm.ingest_memory(text, metadata)
    ts = ltm.ltm_store[str(mem_id)]["timestamp"]
    fts_conn.execute(
        "INSERT INTO docs(rowid, content, timestamp, metadata) VALUES (?,?,?,?)",
        (mem_id, text, ts, json.dumps(metadata))
    )
    fts_conn.commit()
    return mem_id

def _chat(req: ChatRequest) -> dict:
    return _chat_vector(req) if req.mode == "vector" else _chat_bm25(req)

def _chat_vector(req: ChatRequest) -> dict:
    import torch

    # Dense vector retrieval
    t0 = time.perf_counter()
    retrieved_mems = ltm.dense_retrieve(req.query, top_k=req.top_k)
    retrieval_s = time.perf_counter() - t0

    # Build the prompt with apply_chat_template; the slow GemmaTokenizer fails to split
    # raw chat strings correctly — using the template ensures correct token IDs.
    # The slow tokenizer does not correctly split "<start_of_turn>" as a special
    # token from a raw string.  apply_chat_template inserts them via the
    # tokenizer's own mechanism, producing the correct token IDs every time.
    if retrieved_mems:
        ctx_lines = [
            f"  [{m['rank']}] (Memory ID {m['memory_id']}) {m['text']}"
            for m in retrieved_mems
        ]
        ctx_block = "[RETRIEVED CONTEXT]\n" + "\n".join(ctx_lines) + "\n[/RETRIEVED CONTEXT]"
    else:
        ctx_block = "[RETRIEVED CONTEXT]\n  [No relevant memories found.]\n[/RETRIEVED CONTEXT]"

    messages = [
        {"role": "user", "content": f"{ctx_block}\n\n{req.query}"}
    ]

    slm_dev = next(ltm.slm_model.parameters()).device
    t_gen   = time.perf_counter()

    _tpl = ltm.slm_tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
    )
    # BatchEncoding is UserDict (not dict), so check for Tensor explicitly
    if isinstance(_tpl, torch.Tensor):
        input_ids = _tpl.to(slm_dev)
    else:
        input_ids = _tpl["input_ids"].to(slm_dev)
    attn_mask  = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[-1]

    ltm.slm_model.eval()
    with torch.inference_mode():
        out_ids = ltm.slm_model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=150,
            do_sample=False,
        )

    raw = ltm.slm_tokenizer.decode(out_ids[0][prompt_len:], skip_special_tokens=False)
    response = raw.split("<end_of_turn>")[0].strip()
    gen_s = time.perf_counter() - t_gen

    return {
        "response": response,
        "retrieved_memories": retrieved_mems,
        "latency": {
            "dense_retrieval_s": round(retrieval_s, 3),
            "slm_generation_s":  round(gen_s, 3),
            "total_pipeline_s":  round(retrieval_s + gen_s, 3),
        },
        "mode": "vector",
        "expanded_terms": [],
    }

def _chat_bm25(req: ChatRequest) -> dict:
    import torch

    t0 = time.perf_counter()

    # Optionally expand the query using the SLM to improve sparse retrieval coverage
    expanded: list[str] = []
    expansion_s = 0.0
    if req.use_query_expansion:
        t_exp = time.perf_counter()
        expanded = _expand_query(req.query)
        expansion_s = time.perf_counter() - t_exp

    fts_q = _build_fts5_query(req.query, expanded)

    # BM25 retrieval with recency decay — recent memories get a scoring boost
    t_bm25 = time.perf_counter()
    try:
        rows = fts_conn.execute("""
            SELECT docs.rowid AS id, docs.content, docs.timestamp,
                   docs.metadata, bm25(docs) AS bm25_raw
            FROM   docs
            WHERE  docs MATCH ? AND docs.content != '[DELETED]'
            ORDER  BY bm25_raw ASC
            LIMIT  ?
        """, (fts_q, req.top_k * 2)).fetchall()
    except sqlite3.OperationalError:
        rows = []

    scored: list[dict] = []
    for row in rows:
        bm25_pos  = -float(row["bm25_raw"])
        delta_t   = _delta_days(row["timestamp"])
        recency   = math.exp(-0.1 * delta_t)
        final     = bm25_pos * recency
        meta: dict = {}
        try:
            meta = json.loads(row["metadata"] or "{}")
        except Exception:
            pass
        scored.append({
            "memory_id":     int(row["id"]),
            "text":          row["content"],
            "timestamp":     row["timestamp"],
            "metadata":      meta,
            "bm25_score":    round(bm25_pos, 4),
            "recency_score": round(recency, 4),
            "final_score":   round(final, 4),
            "l2_distance":   round(1.0 / (final + 1e-9), 6),
        })

    scored.sort(key=lambda d: d["final_score"], reverse=True)
    retrieved = scored[:req.top_k]
    for i, d in enumerate(retrieved):
        d["rank"] = i + 1

    bm25_s      = time.perf_counter() - t_bm25
    retrieval_s = time.perf_counter() - t0

    # Generation — reuse the SLM already in VRAM to avoid loading a second model
    if retrieved:
        ctx_lines = [
            f"  [{m['rank']}] (Memory ID {m['memory_id']}) {m['text']}"
            for m in retrieved
        ]
        ctx_block = "[RETRIEVED CONTEXT]\n" + "\n".join(ctx_lines) + "\n[/RETRIEVED CONTEXT]"
    else:
        ctx_block = "[RETRIEVED CONTEXT]\n  [No relevant memories found.]\n[/RETRIEVED CONTEXT]"

    messages = [{"role": "user", "content": f"{ctx_block}\n\n{req.query}"}]

    slm_dev = next(ltm.slm_model.parameters()).device
    t_gen   = time.perf_counter()

    _tpl = ltm.slm_tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
    )
    if isinstance(_tpl, torch.Tensor):
        input_ids = _tpl.to(slm_dev)
    else:
        input_ids = _tpl["input_ids"].to(slm_dev)
    attn_mask  = torch.ones_like(input_ids)
    prompt_len = input_ids.shape[-1]

    ltm.slm_model.eval()
    with torch.inference_mode():
        out_ids = ltm.slm_model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=150,
            do_sample=False,
        )
    raw = ltm.slm_tokenizer.decode(out_ids[0][prompt_len:], skip_special_tokens=False)
    response = raw.split("<end_of_turn>")[0].strip()
    gen_s = time.perf_counter() - t_gen

    return {
        "response": response,
        "retrieved_memories": retrieved,
        "latency": {
            "query_expansion_s":  round(expansion_s, 3),
            "bm25_retrieval_s":   round(bm25_s, 3),
            "sparse_retrieval_s": round(retrieval_s, 3),
            "slm_generation_s":   round(gen_s, 3),
            "total_pipeline_s":   round(retrieval_s + gen_s, 3),
        },
        "mode": "bm25",
        "fts_query": fts_q,
        "expanded_terms": expanded,
    }


# Query expansion helpers for BM25 mode
_STOPWORDS = frozenset({
    "what","when","where","who","why","how","which","did","do","does",
    "is","are","was","were","the","a","an","and","or","in","of","to",
    "for","with","on","at","by","from","up","about","into","through",
    "that","this","these","those","have","has","had","be","been","being",
    "i","my","your","his","her","its","our","their","me","you","we","they",
    "it","him","not","no","can","could","would","should","will","just",
    "also","but","if","then","so","as","any","all","each","other","more",
})

def _expand_query(query: str) -> list[str]:
    import torch

    prompt = textwrap.dedent(f"""\
        You are a search-query expansion assistant.
        Output ONLY a valid JSON array of exactly 4 short strings (1-3 words each)
        that are semantically related to the query using DIFFERENT vocabulary.
        No explanation, no markdown fences — just valid JSON.
        Do NOT repeat words from the original query.

        Query: {query}
        JSON array:""")

    messages = [{"role": "user", "content": prompt}]
    raw = ltm.slm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = ltm.slm_tokenizer(
        raw, return_tensors="pt", truncation=True, max_length=1024
    ).to(next(ltm.slm_model.parameters()).device)
    prompt_len = inputs["input_ids"].shape[-1]

    _raw_eos2 = ltm.slm_tokenizer.eos_token_id
    _pad2     = (_raw_eos2[0] if isinstance(_raw_eos2, list) else _raw_eos2)
    with torch.inference_mode():
        out = ltm.slm_model.generate(
            **inputs, max_new_tokens=48,
            do_sample=False, top_p=None, top_k=None,
            pad_token_id=_pad2,
        )
    raw_out = ltm.slm_tokenizer.decode(
        out[0][prompt_len:], skip_special_tokens=True
    ).strip().lstrip("`")
    if raw_out.lower().startswith("json"):
        raw_out = raw_out[4:].strip()
    try:
        terms = json.loads(raw_out)
        if isinstance(terms, list):
            return [t for t in terms if isinstance(t, str) and 0 < len(t) < 80]
    except Exception:
        pass
    return []

def _build_fts5_query(original: str, expanded: list[str]) -> str:
    def term(t: str) -> str:
        t = t.strip().replace('"', "").replace("'", "")
        return f'"{t}"' if " " in t else t

    tokens = re.findall(r'\b[a-zA-Z]\w*\b', original.lower())
    words  = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    base   = " OR ".join(words) if words else term(original)
    rest   = " OR ".join(term(t) for t in expanded if t.strip())
    return f"{base} OR {rest}" if rest else base

def _delta_days(timestamp: str) -> float:
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except Exception:
        return 999.0
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

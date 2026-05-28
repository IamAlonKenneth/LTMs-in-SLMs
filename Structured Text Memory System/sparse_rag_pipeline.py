"""
sparse_rag_pipeline.py
======================
Sparse-Retrieval Long-Term Memory (LTM) Module
for Gemma 3 4B Small Language Model (SLM)

Research Focus : Edge / Local Deployment
Retrieval Index: SQLite FTS5 inverted index (BM25 + exponential time-decay)
Persistence    : SQLite DB file + sidecar JSON dictionary (rowid -> text + metadata)
Generator      : google/gemma-3-4b-it  —  in-process via HuggingFace transformers

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API COMPATIBILITY CONTRACT  (drop-in for vector_embed_module.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every public method name, parameter list, and return-value schema is
preserved exactly so the shared evaluation harness can call either module
without modification.

  vector_embed_module.py              sparse_rag_pipeline.py
  ─────────────────────────────────   ──────────────────────────────────────
  FAISS IndexFlatL2                   SQLite FTS5 virtual table (BM25)
  google/embedding-gemma-300m         (removed — no dense embeddings)
  l2_distance  (lower = better)       1 / (final_score + ε)   [same convention]
  dense_retrieve()                    sparse BM25 + time-decay retrieval
  faiss_index.ntotal                  SQL COUNT on active sidecar rows
  IndexFlatL2 rebuild                 FTS5 'rebuild' command + DELETE purge
  ltm_store  sidecar dict             maintained identically
  _next_faiss_id                      maintained identically (SQLite rowid)

Extra keys appended to result dicts (evaluation code ignores unknown keys):
  bm25_score, recency_score, final_score

Extra latency keys in generate_response (evaluation code ignores them):
  query_expansion_s, bm25_rerank_s

Author : [John Kenneth Alon]
Thesis : Sparse-Retrieval LTM for Edge-Deployed SLMs
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import gc
import json
import math
import shutil
import sqlite3
import textwrap
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# torch and transformers are imported lazily inside _load_slm() so that the
# retrieval-only code path (FTS5, reranker) can be imported or unit-tested
# on machines with no GPU or without torch installed.


# ══════════════════════════════════════════════════════════════════════════════
# Module-level constants  ── names identical to vector_embed_module.py
# ══════════════════════════════════════════════════════════════════════════════

# Kept for __init__ signature compatibility; unused internally (no dense vectors).
EMBEDDING_DIM = 768

DEFAULT_TOP_K          = 5
DEFAULT_MAX_NEW_TOKENS = 512
CONTEXT_HEADER         = "[RETRIEVED CONTEXT]"
CONTEXT_FOOTER         = "[/RETRIEVED CONTEXT]"

# Sparse-retrieval tunables
DEFAULT_CANDIDATE_K  = 20    # FTS5 rows fetched before time-decay re-ranking
DEFAULT_DECAY_LAMBDA = 0.1   # λ in  Score_final = BM25 × e^(−λ·Δt)

# Filenames used by save_ltm / load_ltm
_DB_FILENAME   = "fts5_index.db"
_JSON_FILENAME = "ltm_store.json"

# Gemma 3 chat-format system prompt  (same variable name as dense module)
GEMMA_SYSTEM_PROMPT = (
    "<start_of_turn>system\n"
    "You are a helpful AI assistant. Use the retrieved long-term memory "
    "context below to inform your answer. Answer only from the provided "
    "context. If the context is insufficient, say so clearly.\n"
    "<end_of_turn>\n"
)


# ══════════════════════════════════════════════════════════════════════════════
# SparseEmbeddedMemory
# ══════════════════════════════════════════════════════════════════════════════

class SparseEmbeddedMemory:
    """
    Sparse-Retrieval Long-Term Memory (LTM) Module.

    Encapsulates
    ────────────
    • SQLite FTS5 inverted index   — keyword BM25 scoring; no dense vectors
    • BM25 × time-decay re-ranker  — Score_final = BM25 × e^(−λ·Δt)
    • Query expansion (Gemma 3 4B) — broadens FTS5 recall via LLM synonyms
    • Sidecar JSON map             — { str(rowid): {text, timestamp, metadata} }
    • Gemma 3 4B SLM               — in-process; full lifecycle owned here

    Public API  (identical signatures to VectorEmbeddedMemory)
    ──────────────────────────────────────────────────────────
    ingest_memory(text, metadata)       -> int
    dense_retrieve(query, top_k)        -> list[dict]
    build_augmented_prompt(query, ...)  -> str
    generate_response(query, ...)       -> dict
    save_ltm(directory)                 -> None
    load_ltm(directory)                 -> None
    memory_count()                      -> int
    get_memory(memory_id)               -> dict | None
    delete_memory(memory_id)            -> bool
    rebuild_index()                     -> None
    """

    # ─────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        # ── Kept verbatim for drop-in compatibility ───────────────────────
        embedding_model_id : str   = "google/embedding-gemma-300m",
        slm_model_id       : str   = "google/gemma-3-4b-it",
        quantization       : str   = "none",   # "4bit" | "8bit" | "none"
        device             : str   = "auto",
        embedding_dim      : int   = EMBEDDING_DIM,
        verbose            : bool  = True,
        # ── Sparse-specific parameters (sane defaults; eval harness ignores) ─
        db_path            : str   = ":memory:",
        decay_lambda       : float = DEFAULT_DECAY_LAMBDA,
        candidate_k        : int   = DEFAULT_CANDIDATE_K,
    ) -> None:
        """
        Parameters
        ──────────
        embedding_model_id : Accepted for API compatibility; not loaded.
                             Pass any string — it is stored but never used.
        slm_model_id       : HuggingFace repo for Gemma 3 4B (or any causal LM).
        quantization       : "4bit" (NF4, ≈3 GB VRAM), "8bit" (≈5 GB),
                             or "none" (bfloat16, ≈8 GB).
        device             : "auto" selects CUDA → MPS → CPU automatically.
        embedding_dim      : Retained for __init__ signature compat; unused.
        verbose            : Print timestamped progress messages.
        db_path            : SQLite file path. ":memory:" for ephemeral sessions.
                             Use a real path to enable save_ltm / load_ltm.
        decay_lambda       : λ in Score_final = BM25 × e^(−λ·Δt).
                             0.1 → 7-day-old docs keep ≈50 % of their BM25 score.
        candidate_k        : FTS5 rows fetched before time-decay re-ranking.
        """
        # Store every param so repr / logging / evaluation inspection works.
        self.embedding_model_id = embedding_model_id
        self.slm_model_id       = slm_model_id
        self.quantization       = quantization
        self.embedding_dim      = embedding_dim
        self.verbose            = verbose
        self.db_path            = db_path
        self.decay_lambda       = decay_lambda
        self.candidate_k        = candidate_k
        self._device_pref       = device

        # ── Sidecar dict  (identical role to ltm_store in dense module) ──
        # Keys   : str(SQLite rowid)
        # Values : {"text": str, "timestamp": str, "metadata": dict}
        self.ltm_store: dict[str, dict[str, Any]] = {}

        # ── Auto-increment counter  (named _next_faiss_id for eval compat) ─
        self._next_faiss_id: int = 0

        # ── SQLite FTS5 index ─────────────────────────────────────────────
        self._log("Initialising SQLite FTS5 inverted index …")
        self._db_conn = sqlite3.connect(db_path, check_same_thread=False)
        self._db_conn.row_factory = sqlite3.Row
        self._init_db_schema()

        # ── SLM: Gemma 3 4B (in-process, owns full model lifecycle) ──────
        self._log(f"Loading SLM: {slm_model_id} [quantization={quantization}] …")
        self._load_slm()

        self._log("Sparse LTM Module initialised ✓")

    # ─────────────────────────────────────────────────────────────────────
    # A.  Memory Ingestion
    # ─────────────────────────────────────────────────────────────────────

    def ingest_memory(
        self,
        text    : str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Ingest a text fragment into the Sparse-Retrieval LTM.

        Steps
        ─────
        1. INSERT the text into the FTS5 virtual table.
           FTS5 automatically tokenises the content column and updates the
           inverted index — no separate index-build step is required.
        2. Store {text, timestamp, metadata} in the sidecar dict, keyed by
           the SQLite rowid. This mirrors the FAISS sidecar in the dense module.

        Parameters
        ──────────
        text     : Memory string (conversation summary, fact, document chunk …)
        metadata : Optional dict of extra fields (session_id, source, etc.)

        Returns
        ───────
        doc_id : SQLite rowid — the integer ID equivalent to a FAISS vector ID.

        FTS5 note
        ─────────
        FTS5 INSERT syntax is identical to a normal table INSERT. On INSERT,
        FTS5 tokenises the indexed columns and extends the posting lists in
        its internal B-tree structures. UNINDEXED columns are stored verbatim.
        """
        if not text.strip():
            raise ValueError("Memory text must not be empty.")

        metadata = metadata or {}
        ts = datetime.now(timezone.utc).isoformat()

        cursor = self._db_conn.execute(
            "INSERT INTO docs(content, timestamp, metadata) VALUES (?, ?, ?)",
            (text, ts, json.dumps(metadata)),
        )
        self._db_conn.commit()

        doc_id = cursor.lastrowid                # SQLite assigns rowid automatically

        # Mirror into sidecar dict — same structure as vector_embed_module.py
        self.ltm_store[str(doc_id)] = {
            "text"      : text,
            "timestamp" : ts,
            "metadata"  : metadata,
        }
        self._next_faiss_id = doc_id + 1

        self._log(
            f"[Ingestion] Memory stored → ID={doc_id} | "
            f"tokens≈{len(text.split())} | total_memories={self._active_count()}"
        )
        return doc_id

    # ─────────────────────────────────────────────────────────────────────
    # B.  Sparse Retrieval  (method name preserved as dense_retrieve)
    # ─────────────────────────────────────────────────────────────────────

    def dense_retrieve(
        self,
        query : str,
        top_k : int = DEFAULT_TOP_K,
    ) -> list[dict[str, Any]]:
        """
        Execute Sparse Retrieval against the SQLite FTS5 index.

        The method name dense_retrieve is kept verbatim for evaluation-harness
        compatibility.  Internally it performs sparse keyword retrieval.

        Pipeline
        ────────
        1. Query Expansion   — Gemma 3 4B generates 4 related keywords to
                               bridge vocabulary mismatch (the core weakness of
                               bag-of-words / BM25 retrieval systems).
        2. FTS5 MATCH Search — The OR-expanded query is run against the inverted
                               index. bm25(docs) scores each matching row.
        3. Time-Decay Rerank — Each BM25 score is multiplied by e^(−λ·Δt):
                               Score_final = (−bm25_raw) × e^(−λ·Δt)
                               where Δt is the document age in days.

        Return schema  (identical keys to vector_embed_module.py)
        ──────────────────────────────────────────────────────────
        List of dicts, best-first, each containing:
          rank          : int   — 1-based position after time-decay re-ranking
          memory_id     : int   — SQLite rowid (≡ FAISS integer ID)
          text          : str   — stored document text
          timestamp     : str   — ISO-8601 UTC ingestion timestamp
          metadata      : dict  — caller-supplied metadata
          l2_distance   : float — 1 / (final_score + ε)  preserves "lower is
                                  better" convention used by FAISS L2 distance
          bm25_score    : float — (−bm25_raw) before decay factor   [extra]
          recency_score : float — e^(−λ·Δt)                          [extra]
          final_score   : float — bm25_score × recency_score          [extra]
        """
        if not self.ltm_store:
            self._log("[dense_retrieve] LTM index is empty — no memories returned.")
            return []

        # Step 1 — Query expansion
        t_exp = time.perf_counter()
        expanded  = self._expand_query(query)
        fts_query = self._build_fts5_query(query, expanded)
        self._log(
            f"[dense_retrieve] expansion: {time.perf_counter()-t_exp:.2f}s  "
            f"FTS5 query: {fts_query!r}"
        )

        # Steps 2 + 3 — FTS5 search + time-decay re-ranking
        return self._fts_search_and_rerank(fts_query, top_k)

    # ─────────────────────────────────────────────────────────────────────
    # C.  Context Injection — Prompt Construction
    # ─────────────────────────────────────────────────────────────────────

    def build_augmented_prompt(
        self,
        query          : str,
        retrieved_mems : list[dict[str, Any]],
        include_scores : bool = False,
    ) -> str:
        """
        Construct the context-augmented prompt for Gemma 3 4B.

        Template layout  (identical structure to vector_embed_module.py)
        ─────────────────────────────────────────────────────────────────
        <system_prompt>
        <start_of_turn>user
          [RETRIEVED CONTEXT]
          [1] (Memory ID X) <text>  (score=Y | date=Z)  ← when include_scores
          …
          [/RETRIEVED CONTEXT]

          <user_query>
        <end_of_turn>
        <start_of_turn>model

        Parameters
        ──────────
        query          : The user's current question.
        retrieved_mems : Output of dense_retrieve() — list of scored dicts.
        include_scores : Append composite score + date to each memory line.

        Returns
        ───────
        Full prompt string ready to be tokenised and passed to the SLM.
        """
        lines: list[str] = [CONTEXT_HEADER]

        if not retrieved_mems:
            lines.append("  [No relevant long-term memories found.]")
        else:
            for mem in retrieved_mems:
                score_str = ""
                if include_scores:
                    score_str = (
                        f"  (score={mem.get('final_score', 0.0):.4f}"
                        f" | date={mem.get('timestamp', '')[:10]})"
                    )
                lines.append(
                    f"  [{mem['rank']}] (Memory ID {mem['memory_id']}) "
                    f"{mem['text']}{score_str}"
                )

        lines.append(CONTEXT_FOOTER)
        context_block = "\n".join(lines)

        # Gemma 3 multi-turn chat format — same template as vector_embed_module.py
        return (
            f"{GEMMA_SYSTEM_PROMPT}"
            f"<start_of_turn>user\n"
            f"{context_block}\n\n"
            f"{query}\n"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

    # ─────────────────────────────────────────────────────────────────────
    # D.  Inference Pipeline  (Expand → Retrieve → Inject → Generate)
    # ─────────────────────────────────────────────────────────────────────

    def generate_response(
        self,
        query          : str,
        top_k          : int   = DEFAULT_TOP_K,
        max_new_tokens : int   = DEFAULT_MAX_NEW_TOKENS,
        temperature    : float = 0.7,
        include_scores : bool  = False,
    ) -> dict[str, Any]:
        """
        End-to-end Sparse-LTM-augmented inference pipeline.

        Execution order
        ───────────────
        1. Query Expansion        — Gemma 3 4B generates synonyms in-process.
        2. FTS5 BM25 Search       — expanded OR query hits the inverted index.
        3. Time-Decay Re-ranking  — Score_final = BM25 × e^(−λ·Δt).
        4. Context Injection      — ranked chunks injected into Gemma prompt.
        5. SLM Generation         — Gemma 3 4B generates a grounded answer.

        Parameters
        ──────────
        query          : User's input string.
        top_k          : Number of memories to retrieve and inject.
        max_new_tokens : Hard cap on SLM output tokens.
        temperature    : 0 → greedy decoding; >0 → stochastic sampling.
        include_scores : Add BM25×decay score + date to injected context.

        Returns  (identical schema to vector_embed_module.py)
        ───────────────────────────────────────────────────────
        dict:
          response          : str  — Gemma 3 4B answer text
          memory_ids_used   : list — rowids of injected memories
          retrieved_mems    : list — full dense_retrieve() output
          latency           : dict — wall-time breakdown in seconds:
              dense_retrieval_s    — total retrieval phase (compat key)
              context_injection_s  — prompt construction
              slm_generation_s     — model forward pass + decode
              total_pipeline_s     — end-to-end wall time
              query_expansion_s    — sub-timing: LLM expansion  [extra]
              bm25_rerank_s        — sub-timing: BM25 + decay    [extra]
          augmented_prompt  : str  — full prompt sent to SLM (debug / ablation)

        Design note — avoiding double expansion
        ────────────────────────────────────────
        generate_response calls _expand_query and _build_fts5_query directly
        (to capture sub-timings) and then calls _fts_search_and_rerank with
        the pre-built query string.  This avoids the double LLM call that
        would occur if generate_response called dense_retrieve, which also
        runs expansion internally.
        """
        latency: dict[str, float] = {}

        # ── Step 1 + 2 + 3 : Retrieval phase ─────────────────────────────
        t_ret = time.perf_counter()

        t_exp = time.perf_counter()
        expanded  = self._expand_query(query)
        fts_query = self._build_fts5_query(query, expanded)
        latency["query_expansion_s"] = time.perf_counter() - t_exp

        t_bm25 = time.perf_counter()
        retrieved_mems = self._fts_search_and_rerank(fts_query, top_k)
        latency["bm25_rerank_s"] = time.perf_counter() - t_bm25

        # dense_retrieval_s is the compatibility key the eval harness expects.
        latency["dense_retrieval_s"] = time.perf_counter() - t_ret

        # ── Step 4 : Context Injection ────────────────────────────────────
        t_ctx = time.perf_counter()
        augmented_prompt = self.build_augmented_prompt(
            query, retrieved_mems, include_scores=include_scores
        )
        latency["context_injection_s"] = time.perf_counter() - t_ctx

        # ── Step 5 : SLM Generation ───────────────────────────────────────
        #
        # The prompt is already formatted in Gemma 3 chat format by
        # build_augmented_prompt, so we tokenise it directly without
        # calling apply_chat_template again (that would double-wrap turns).
        #
        # torch.inference_mode() is a strict superset of torch.no_grad():
        # it additionally disables view-tracking, saving ~5-10 % memory
        # versus no_grad during pure-inference workloads.
        t_gen = time.perf_counter()

        import torch
        inputs = self.slm_tokenizer(
            augmented_prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 4096,
        ).to(self._slm_device)

        prompt_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self.slm_model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                do_sample      = (temperature > 0),
                temperature    = temperature if temperature > 0 else 1.0,
                pad_token_id   = self.slm_tokenizer.eos_token_id,
            )

        # Slice off the input-prompt portion; decode only newly generated tokens.
        new_ids       = output_ids[0, prompt_len:]
        response_text = self.slm_tokenizer.decode(
            new_ids, skip_special_tokens=True
        ).strip()

        latency["slm_generation_s"] = time.perf_counter() - t_gen
        latency["total_pipeline_s"] = (
            latency["dense_retrieval_s"]
            + latency["context_injection_s"]
            + latency["slm_generation_s"]
        )

        memory_ids_used = [m["memory_id"] for m in retrieved_mems]

        self._log(
            f"[Pipeline] Retrieval={latency['dense_retrieval_s']:.3f}s | "
            f"Generation={latency['slm_generation_s']:.3f}s | "
            f"Memory IDs={memory_ids_used}"
        )

        return {
            "response"        : response_text,
            "memory_ids_used" : memory_ids_used,
            "retrieved_mems"  : retrieved_mems,
            "latency"         : latency,
            "augmented_prompt": augmented_prompt,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Persistence  —  Save & Load
    # ─────────────────────────────────────────────────────────────────────

    def save_ltm(self, directory: str = "./ltm_store") -> None:
        """
        Persist the FTS5 SQLite database and the sidecar JSON map to disk.

        Files written
        ─────────────
        <directory>/fts5_index.db   — SQLite DB containing the FTS5 virtual table
        <directory>/ltm_store.json  — sidecar dict + _next_faiss_id counter

        If db_path is ':memory:', sqlite3.Connection.backup() streams the
        in-memory DB to a new on-disk connection without interrupting queries.
        If db_path is a file path the WAL is checkpointed and the file copied.

        Parameters
        ──────────
        directory : Target directory — created recursively if absent.
        """
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        db_dest   = save_dir / _DB_FILENAME
        json_dest = save_dir / _JSON_FILENAME

        # Persist SQLite DB
        if self.db_path == ":memory:":
            dst = sqlite3.connect(str(db_dest))
            self._db_conn.backup(dst)        # streams in-memory → on-disk
            dst.close()
        else:
            self._db_conn.execute("PRAGMA wal_checkpoint(FULL)")
            shutil.copy2(self.db_path, db_dest)

        # Persist sidecar + counter
        with open(json_dest, "w", encoding="utf-8") as fh:
            json.dump(
                {"_next_faiss_id": self._next_faiss_id, "memories": self.ltm_store},
                fh, indent=2, ensure_ascii=False,
            )

        self._log(
            f"[Persistence] LTM saved → {save_dir}  "
            f"({self._active_count()} active memories)"
        )

    def load_ltm(self, directory: str = "./ltm_store") -> None:
        """
        Restore a previously saved FTS5 index and sidecar JSON map from disk.

        Parameters
        ──────────
        directory : Directory containing fts5_index.db and ltm_store.json.

        Raises FileNotFoundError if either expected file is missing.
        """
        save_dir  = Path(directory)
        db_src    = save_dir / _DB_FILENAME
        json_src  = save_dir / _JSON_FILENAME

        if not db_src.exists() or not json_src.exists():
            raise FileNotFoundError(
                f"LTM store not found in '{directory}'. "
                f"Expected: {db_src} and {json_src}. "
                "Run save_ltm() first or verify the path."
            )

        # Close the current connection and re-open against the saved file.
        self._db_conn.close()
        self._db_conn             = sqlite3.connect(str(db_src), check_same_thread=False)
        self._db_conn.row_factory = sqlite3.Row
        self.db_path              = str(db_src)

        with open(json_src, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self._next_faiss_id = payload.get("_next_faiss_id", 0)
        self.ltm_store      = payload.get("memories", {})

        self._log(
            f"[Persistence] LTM loaded ← {save_dir}  "
            f"({self._active_count()} active memories)"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Utility / Introspection  (same signatures as vector_embed_module.py)
    # ─────────────────────────────────────────────────────────────────────

    def memory_count(self) -> int:
        """Return the number of non-deleted memories in the LTM."""
        return self._active_count()

    def get_memory(self, memory_id: int) -> dict[str, Any] | None:
        """Retrieve a single memory record by its integer ID (SQLite rowid)."""
        return self.ltm_store.get(str(memory_id))

    def delete_memory(self, memory_id: int) -> bool:
        """
        Soft-delete a memory: replaces its content with '[DELETED]' in both
        the FTS5 table and the sidecar dict so it is excluded from future
        retrieval results.

        Unlike FAISS IndexFlatL2, SQLite FTS5 supports true row-level UPDATE,
        so the inverted index is updated immediately on the soft-delete.
        The rowid is retained to preserve ID continuity; call rebuild_index()
        to physically purge deleted rows and reclaim disk space.

        Returns True if the memory existed and was soft-deleted, False otherwise.
        """
        key = str(memory_id)
        if key not in self.ltm_store:
            return False

        # FTS5 UPDATE removes the old tokens from the posting lists and adds
        # the tokens of the new content ('[DELETED]' in this case).
        # The WHERE content != '[DELETED]' filter in _fts_search_and_rerank
        # ensures these rows are excluded from retrieval results.
        self._db_conn.execute(
            "UPDATE docs SET content = '[DELETED]', metadata = ? WHERE rowid = ?",
            (json.dumps({"deleted": True}), memory_id),
        )
        self._db_conn.commit()

        self.ltm_store[key] = {
            "text"      : "[DELETED]",
            "timestamp" : datetime.now(timezone.utc).isoformat(),
            "metadata"  : {"deleted": True},
        }
        self._log(f"[LTM] Memory ID={memory_id} soft-deleted.")
        return True

    def rebuild_index(self) -> None:
        """
        Physically purge soft-deleted rows and rebuild the FTS5 B-tree index.

        Two-phase process
        ─────────────────
        Phase 1 — DELETE
            Remove all rows whose content equals '[DELETED]' from the FTS5
            table.  FTS5 removes their tokens from the inverted index automatically
            on DELETE (unlike FAISS which requires a full rebuild to remove vectors).

        Phase 2 — FTS5 REBUILD command
            `INSERT INTO docs(docs) VALUES('rebuild')` is the FTS5 maintenance
            syntax.  It re-derives the internal B-tree segment files from the
            stored content, eliminating fragmentation caused by many small
            insertions and deletions without changing any stored text.

        The sidecar ltm_store is pruned to contain only active rowids so
        memory_count() stays consistent after rebuild.
        """
        self._log("[LTM] Rebuilding FTS5 index — purging deleted rows …")

        # Phase 1: hard-delete soft-deleted rows
        self._db_conn.execute("DELETE FROM docs WHERE content = '[DELETED]'")
        self._db_conn.commit()

        # Phase 2: FTS5 structural rebuild
        self._db_conn.execute("INSERT INTO docs(docs) VALUES('rebuild')")
        self._db_conn.commit()

        # Sync sidecar to match only rows still in the DB
        live_ids = {
            str(row[0])
            for row in self._db_conn.execute("SELECT rowid FROM docs")
        }
        self.ltm_store = {k: v for k, v in self.ltm_store.items() if k in live_ids}

        self._log(
            f"[LTM] Rebuild complete — {self._active_count()} active memories."
        )

    # ─────────────────────────────────────────────────────────────────────
    # Model Lifecycle  —  explicit teardown
    # ─────────────────────────────────────────────────────────────────────

    def unload_model(self) -> None:
        """
        Release SLM weights from device memory and close the SQLite connection.

        Teardown sequence
        ─────────────────
        1. del slm_model + del slm_tokenizer  — drops Python references.
        2. gc.collect()                        — CPython ref-count cycle sweep.
        3. torch.cuda.empty_cache()            — returns cached-but-free CUDA
                                                 pages to the CUDA allocator.
        4. torch.mps.empty_cache()             — same for Apple Silicon MPS
                                                 (no-op on CUDA / CPU).
        5. Close the SQLite connection.

        These calls are safe even on machines with no GPU — they silently
        return without error when CUDA or MPS is unavailable.
        """
        if getattr(self, "_slm_loaded", False):
            self._log("Unloading SLM weights from device memory …")
            del self.slm_model
            del self.slm_tokenizer
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                import torch
                torch.mps.empty_cache()
            except Exception:
                pass
            self._slm_loaded = False
            self._log("SLM memory released.")

        if hasattr(self, "_db_conn") and self._db_conn:
            self._db_conn.close()

    def __enter__(self) -> "SparseEmbeddedMemory":
        return self

    def __exit__(self, *_) -> None:
        self.unload_model()

    # ─────────────────────────────────────────────────────────────────────
    # Private  —  DB Schema
    # ─────────────────────────────────────────────────────────────────────

    def _init_db_schema(self) -> None:
        """
        Create the FTS5 virtual table if it does not already exist.

        SQL breakdown
        ─────────────
        CREATE VIRTUAL TABLE IF NOT EXISTS docs
        USING fts5(
            content,               -- tokenised and indexed by FTS5
            timestamp UNINDEXED,   -- stored verbatim; never tokenised
            metadata  UNINDEXED    -- stored verbatim; never tokenised
        );

        Key behaviours
        ──────────────
        • `VIRTUAL TABLE … USING fts5` delegates all storage and querying to
          the FTS5 module instead of SQLite's normal B-tree tables.
        • FTS5 maintains an internal inverted index: a mapping of each unique
          token → list of rowids that contain it (posting lists).
        • `UNINDEXED` columns are stored in a shadow content table but their
          tokens are NOT added to the inverted index, preventing metadata
          values (timestamps, JSON keys) from polluting keyword rankings.
        • The implicit integer primary key is accessed as `docs.rowid` in
          plain SELECT statements — there is no explicit `id` column.

        FTS5 creates these shadow tables automatically (do not modify them):
            docs_data, docs_idx, docs_content, docs_docsize, docs_config
        """
        self._db_conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS docs
            USING fts5(
                content,               -- full-text indexed
                timestamp UNINDEXED,   -- ISO-8601 UTC; stored only
                metadata  UNINDEXED    -- JSON blob; stored only
            )
        """)
        self._db_conn.commit()

    # ─────────────────────────────────────────────────────────────────────
    # Private  —  SLM Loading
    # ─────────────────────────────────────────────────────────────────────

    def _load_slm(self) -> None:
        """
        Load the Gemma 3 4B tokenizer and model weights into device memory.

        Device priority: CUDA GPU → Apple Silicon MPS → CPU.

        Quantisation options
        ────────────────────
        "4bit" — NF4 via BitsAndBytesConfig (≈3 GB VRAM, <1 % quality loss)
        "8bit" — INT8 via BitsAndBytesConfig (≈5 GB VRAM)
        "none" — bfloat16 full precision     (≈8 GB VRAM)

        bfloat16 vs float16
        ────────────────────
        bfloat16 shares float32's exponent range, avoiding the NaN / overflow
        instabilities float16 can exhibit on very large or very small activations
        in long LLM forward passes.  It is the recommended dtype for Gemma 3.

        model.eval()
        ────────────
        Switches off dropout layers and batch-normalisation running-stat
        accumulation, both of which are only needed during training.
        Calling eval() is mandatory for deterministic inference.
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Device selection
        if self._device_pref == "auto":
            if torch.cuda.is_available():
                self._slm_device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self._slm_device = torch.device("mps")
            else:
                self._slm_device = torch.device("cpu")
        else:
            self._slm_device = torch.device(self._device_pref)
        self._log(f"SLM device: {self._slm_device}")

        # Tokenizer
        self.slm_tokenizer = AutoTokenizer.from_pretrained(self.slm_model_id)

        # Model weights
        bnb_cfg = self._build_bnb_config(self.quantization)
        load_kw: dict[str, Any] = {
            "device_map"         : "auto",
            "low_cpu_mem_usage"  : True,
            "attn_implementation": "eager",   # compatible with all hardware
        }
        if bnb_cfg is not None:
            load_kw["quantization_config"] = bnb_cfg
        else:
            load_kw["torch_dtype"] = torch.bfloat16

        t0 = time.perf_counter()
        self.slm_model = AutoModelForCausalLM.from_pretrained(
            self.slm_model_id, **load_kw
        )
        self.slm_model.eval()
        self._slm_loaded = True
        self._log(
            f"SLM ready in {time.perf_counter()-t0:.1f}s  |  "
            f"dtype={next(self.slm_model.parameters()).dtype}  |  "
            f"device={self._slm_device}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Private  —  Core Retrieval (shared by dense_retrieve + generate_response)
    # ─────────────────────────────────────────────────────────────────────

    def _fts_search_and_rerank(
        self,
        fts_query : str,
        top_k     : int,
    ) -> list[dict[str, Any]]:
        """
        Execute the FTS5 MATCH query and apply BM25 × time-decay re-ranking.

        Called by both dense_retrieve (which runs query expansion first) and
        generate_response (which runs expansion itself for sub-timing), so
        there is exactly one copy of the FTS5 + reranking logic.

        SQL used
        ────────
        SELECT docs.rowid AS id,
               docs.content    AS content,
               docs.timestamp  AS timestamp,
               docs.metadata   AS metadata,
               bm25(docs)      AS bm25_raw
        FROM   docs
        WHERE  docs MATCH ?              -- FTS5 full-text search operator
          AND  docs.content != '[DELETED]'  -- exclude soft-deleted rows
        ORDER  BY bm25_raw ASC           -- ASC: negative scores, most-relevant first
        LIMIT  ?

        FTS5 `docs MATCH ?` notes
        ─────────────────────────
        • Implicit AND:  "python sqlite"   → rows containing BOTH tokens.
        • Explicit OR:   "python OR sqlite" → rows containing EITHER token.
        • Phrase:        '"full text"'      → the exact bi-gram in sequence.
        • Prefix:        "pyth*"            → matches python, pythonic, etc.

        bm25(docs) notes
        ────────────────
        • FTS5's built-in BM25 implementation returns a *negative* float.
        • Lower (more negative) = more relevant in SQLite's internal ordering.
        • ORDER BY bm25_raw ASC puts the best match in row 0.
        • We negate the value (bm25_pos = -bm25_raw) to get "higher is better"
          semantics for the time-decay multiplication.

        Time-decay formula
        ──────────────────
        Score_final = Score_BM25 × e^(−λ · Δt)

        Where:
            Score_BM25 = -bm25_raw        (positive; higher is better)
            Δt         = document age in fractional days  (clamped ≥ 0)
            λ (lambda) = self.decay_lambda  (default 0.1 day⁻¹)
            e          = Euler's number ≈ 2.71828

        The exponential decay function e^(−λ·Δt) equals 1.0 for a document
        ingested right now and approaches 0 for very old documents.  At λ=0.1:
            Δt =  0 days → factor = 1.000   (no penalty)
            Δt =  7 days → factor ≈ 0.497   (≈50 % retained)
            Δt = 30 days → factor ≈ 0.050   (≈ 5 % retained)

        l2_distance compatibility
        ─────────────────────────
        The evaluation harness uses l2_distance to sort results and compare
        retrieval systems; it expects "lower = more relevant" (FAISS L2
        convention).  We map final_score → l2_distance via:

            l2_distance = 1 / (final_score + ε)

        so the best-scoring document gets the smallest l2_distance value,
        preserving the sort direction the evaluation harness assumes.
        """
        sql = """
        SELECT
            docs.rowid      AS id,
            docs.content    AS content,
            docs.timestamp  AS timestamp,
            docs.metadata   AS metadata,
            bm25(docs)      AS bm25_raw
        FROM   docs
        WHERE  docs MATCH ?
          AND  docs.content != '[DELETED]'
        ORDER  BY bm25_raw ASC
        LIMIT  ?
        """
        try:
            rows = self._db_conn.execute(sql, (fts_query, self.candidate_k)).fetchall()
        except sqlite3.OperationalError as exc:
            self._log(f"[FTS5] Query error: {exc}")
            return []

        if not rows:
            self._log("[FTS5] No candidates returned.")
            return []

        # ── BM25 × time-decay scoring ─────────────────────────────────────
        scored: list[dict[str, Any]] = []
        for row in rows:
            bm25_pos  = -float(row["bm25_raw"])           # flip sign
            delta_t   = self._delta_days(row["timestamp"])
            recency   = math.exp(-self.decay_lambda * delta_t)
            final     = bm25_pos * recency
            l2_compat = 1.0 / (final + 1e-9)             # lower = better

            meta: dict = {}
            try:
                meta = json.loads(row["metadata"] or "{}")
            except (json.JSONDecodeError, TypeError):
                pass

            scored.append({
                # ── Keys required by evaluation harness ───────────────────
                "memory_id"    : int(row["id"]),
                "text"         : row["content"],
                "timestamp"    : row["timestamp"],
                "metadata"     : meta,
                "l2_distance"  : l2_compat,
                # ── Sparse-specific transparency keys ─────────────────────
                "bm25_score"   : bm25_pos,
                "recency_score": recency,
                "final_score"  : final,
            })

        # Sort descending by final_score, slice to top_k, assign rank.
        scored.sort(key=lambda d: d["final_score"], reverse=True)
        result = scored[:top_k]
        for i, d in enumerate(result):
            d["rank"] = i + 1

        self._log(
            f"[FTS5+Rerank] {len(result)} result(s) returned  "
            f"(from {len(rows)} FTS5 candidates)"
        )
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Private  —  Query Expansion
    # ─────────────────────────────────────────────────────────────────────

    def _expand_query(self, query: str) -> list[str]:
        """
        Use the in-process Gemma 3 4B to generate 4 semantically related
        keywords that broaden FTS5 recall.

        Vocabulary mismatch problem
        ───────────────────────────
        BM25 scores purely on exact token overlap.  If the user writes
        "automobiles" but documents say "cars", the score is 0 despite
        identical meaning.  LLM-generated synonyms act as lexical bridges,
        restoring the recall benefits that dense embeddings provide without
        requiring any vector computation.

        Generation parameters
        ─────────────────────
        max_new_tokens=96  — a JSON array of 4 short strings is < 60 tokens.
        temperature=0.35   — low enough for stable JSON output; high enough
                             to produce lexically diverse synonyms (not just
                             the most probable paraphrase).

        apply_chat_template
        ───────────────────
        Gemma 3 instruction-tuned models were fine-tuned with a specific turn
        format (<start_of_turn>user / <start_of_turn>model).
        apply_chat_template injects these tokens automatically from the
        tokenizer's Jinja2 template so we never hard-code them here.
        add_generation_prompt=True appends <start_of_turn>model\\n to prime
        the model for its response.
        """
        import torch

        prompt = textwrap.dedent(f"""\
            You are a search-query expansion assistant.
            Given the query below, output ONLY a valid JSON array of exactly
            4 strings. Each string must be a keyword or short phrase (1-3 words)
            that is semantically related to the query but uses DIFFERENT vocabulary.
            Rules:
              - Output ONLY valid JSON. No explanation, no markdown fences.
              - Do NOT repeat any words from the original query.
              - Prefer noun phrases and technical synonyms.

            Query: {query}

            JSON array:""")

        messages  = [{"role": "user", "content": prompt}]
        input_ids = self.slm_tokenizer.apply_chat_template(
            messages,
            return_tensors        = "pt",
            add_generation_prompt = True,
        ).to(self._slm_device)

        prompt_len = input_ids.shape[-1]

        with torch.inference_mode():
            out_ids = self.slm_model.generate(
                input_ids,
                max_new_tokens = 96,
                do_sample      = True,
                temperature    = 0.35,
                pad_token_id   = self.slm_tokenizer.eos_token_id,
            )

        raw = self.slm_tokenizer.decode(
            out_ids[0, prompt_len:], skip_special_tokens=True
        ).strip()

        # Strip any residual markdown fences the model may still emit.
        clean = raw.lstrip("`").strip()
        if clean.lower().startswith("json"):
            clean = clean[4:].strip()

        try:
            terms = json.loads(clean)
            if isinstance(terms, list):
                valid = [t for t in terms if isinstance(t, str) and 0 < len(t) < 80]
                self._log(f"[Query Expansion] terms: {valid}")
                return valid
        except (json.JSONDecodeError, ValueError):
            self._log("[Query Expansion] JSON parse failed; skipping expansion.")
        return []

    # ─────────────────────────────────────────────────────────────────────
    # Private  —  FTS5 Query Builder
    # ─────────────────────────────────────────────────────────────────────

    def _build_fts5_query(
        self,
        original_query: str,
        expanded_terms: list[str],
    ) -> str:
        """
        Combine the original query tokens and LLM-expanded terms into a
        single FTS5 OR query string to maximise retrieval recall.

        FTS5 OR query syntax
        ────────────────────
        • Tokens separated by spaces inside a clause are implicitly ANDed:
              python sqlite fts5  →  rows containing all three tokens
        • OR joins alternative clauses; a row matching ANY clause is returned:
              python OR sqlite    →  rows containing python OR sqlite (or both)
        • Multi-word terms must be double-quoted for phrase matching:
              "machine learning"  →  the two tokens must appear adjacent

        Strategy
        ────────
        The original query is treated as an implicit-AND clause (preserving
        precision), then OR'd with each expanded term (improving recall).

        Example
        ───────
        original : "BM25 ranking"
        expanded : ["relevance score", "term frequency", "inverted index", "IR ranking"]
        output   : '"BM25 ranking" OR "relevance score" OR "term frequency"
                    OR "inverted index" OR "IR ranking"'
        """
        def _fts5_term(t: str) -> str:
            t = t.strip().replace('"', "")    # sanitise embedded quotes
            return f'"{t}"' if " " in t else t

        base = _fts5_term(original_query)
        rest = " OR ".join(_fts5_term(t) for t in expanded_terms if t.strip())
        return f"{base} OR {rest}" if rest else base

    # ─────────────────────────────────────────────────────────────────────
    # Private  —  Time-Decay Helper
    # ─────────────────────────────────────────────────────────────────────

    def _delta_days(self, timestamp: str) -> float:
        """
        Parse an ISO-8601 UTC timestamp and return elapsed time in fractional
        days, clamped to ≥ 0.

        Malformed timestamps are treated as epoch-minimum (very old), causing
        the document to receive a near-zero recency factor and sink in ranking.
        """
        try:
            doc_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            doc_dt = datetime.min.replace(tzinfo=timezone.utc)
        elapsed_s = (datetime.now(timezone.utc) - doc_dt).total_seconds()
        return max(0.0, elapsed_s / 86_400.0)   # 86 400 s per day

    # ─────────────────────────────────────────────────────────────────────
    # Private  —  Shared Helpers  (identical signatures to dense module)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_bnb_config(quantization: str):
        """
        Build a BitsAndBytesConfig for 4-bit or 8-bit quantisation.
        Returns None for quantization='none' (full bfloat16 precision).

        Identical implementation to vector_embed_module.py._build_bnb_config()
        so the evaluation harness can verify both modules load equivalently.

        4-bit NF4 (NormalFloat4)
        ────────────────────────
        NF4 is an information-theoretically optimal quantisation format for
        normally distributed weights.  bnb_4bit_use_double_quant=True
        additionally quantises the quantisation constants themselves, saving
        a further 0.5 bits per parameter on average.
        bnb_4bit_compute_dtype=torch.float16 runs matmuls in fp16 for speed
        while the stored weights remain in 4-bit NF4.
        """
        from transformers import BitsAndBytesConfig
        import torch

        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_quant_type       = "nf4",
                bnb_4bit_use_double_quant = True,
                bnb_4bit_compute_dtype    = torch.float16,
            )
        if quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None   # "none" → bfloat16 full precision

    def _log(self, message: str) -> None:
        """Conditional verbose logging — identical signature to dense module."""
        if self.verbose:
            print(f"[LTM {datetime.now().strftime('%H:%M:%S')}] {message}")

    # ─────────────────────────────────────────────────────────────────────
    # Private  —  Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _active_count(self) -> int:
        """Count non-deleted entries in the sidecar dict."""
        return sum(
            1 for v in self.ltm_store.values()
            if v.get("text", "") != "[DELETED]"
            and not v.get("metadata", {}).get("deleted", False)
        )


# ══════════════════════════════════════════════════════════════════════════════
# Drop-in alias
# Evaluation harness may instantiate VectorEmbeddedMemory by name; this alias
# makes SparseEmbeddedMemory a transparent replacement.
# ══════════════════════════════════════════════════════════════════════════════

VectorEmbeddedMemory = SparseEmbeddedMemory


# ══════════════════════════════════════════════════════════════════════════════
# __main__  —  quick-start demo
# Layout mirrors vector_embed_module.py so both scripts are runnable identically.
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  Sparse-Retrieval LTM Module — Quick-Start Demo")
    print("=" * 70)

    now = datetime.now(timezone.utc)

    # Instantiate via the drop-in alias — syntactically identical to the
    # dense module's __main__ block.
    ltm = VectorEmbeddedMemory(
        embedding_model_id = "google/embedding-gemma-300m",  # accepted, unused
        slm_model_id       = "google/gemma-3-4b-it",
        quantization       = "none",    # "4bit" recommended on ≤8 GB VRAM
        verbose            = True,
        db_path            = ":memory:",
        decay_lambda       = 0.1,
    )

    # ── Ingest sample memories with varied timestamps ─────────────────────
    print("\n--- Ingesting Memories ---")

    ltm.ingest_memory(
        "The user prefers concise, bullet-point style answers.",
        metadata={"session": "session_001", "source": "user_preference"},
    )
    ltm.ingest_memory(
        "In session 3, the user asked about transformer attention mechanisms "
        "and found the scaled dot-product explanation most helpful.",
        metadata={"session": "session_003", "topic": "transformers"},
    )
    ltm.ingest_memory(
        "The user's thesis focuses on Sparse-Retrieval LTM for Gemma 3 4B "
        "targeting edge deployment on devices with ≤8 GB RAM.",
        metadata={"session": "session_005", "topic": "thesis"},
    )
    ltm.ingest_memory(
        "SQLite FTS5 provides BM25 keyword ranking via an inverted index and "
        "is significantly faster than dense FAISS on CPU-only hardware.",
        metadata={"session": "session_006", "topic": "retrieval"},
    )
    ltm.ingest_memory(
        "Exponential time-decay weighting penalises older documents: "
        "Score_final = BM25 * exp(-lambda * delta_t_days).",
        metadata={"session": "session_007", "topic": "ranking"},
    )
    print(f"\nTotal memories in LTM: {ltm.memory_count()}")

    # ── Sparse Retrieval via the dense_retrieve compatibility name ────────
    print("\n--- Sparse Retrieval (dense_retrieve API) ---")
    query   = "What is the user's research topic?"
    results = ltm.dense_retrieve(query, top_k=2)
    for r in results:
        print(
            f"  Rank {r['rank']} | ID={r['memory_id']} | "
            f"BM25={r['bm25_score']:.4f} | "
            f"recency={r['recency_score']:.4f} | "
            f"final={r['final_score']:.4f} | "
            f"l2_compat={r['l2_distance']:.6f}"
        )
        print(f"    → {r['text'][:80]}…")

    # ── delete_memory + memory_count ──────────────────────────────────────
    print("\n--- delete_memory ---")
    removed = ltm.delete_memory(memory_id=1)
    print(f"delete_memory(1) returned: {removed}")
    print(f"Active memory count after deletion: {ltm.memory_count()}")

    # ── rebuild_index ─────────────────────────────────────────────────────
    print("\n--- rebuild_index ---")
    ltm.rebuild_index()
    print(f"Active memory count after rebuild: {ltm.memory_count()}")

    # ── Persistence ───────────────────────────────────────────────────────
    print("\n--- Persistence (save_ltm / load_ltm) ---")
    ltm.save_ltm("./ltm_store_sparse")
    print("LTM saved to ./ltm_store_sparse/ ✓")

    ltm2 = VectorEmbeddedMemory(
        slm_model_id = "google/gemma-3-4b-it",
        verbose      = True,
        db_path      = "./ltm_store_sparse/fts5_index.db",
    )
    ltm2.load_ltm("./ltm_store_sparse")
    print(f"Restored memory count: {ltm2.memory_count()}")
    ltm2.unload_model()

    # ── Full Inference Pipeline ───────────────────────────────────────────
    print("\n--- LTM-Augmented Inference (generate_response) ---")
    output = ltm.generate_response(
        query          = "Summarise my research project in one paragraph.",
        top_k          = 3,
        max_new_tokens = 256,
        temperature    = 0.7,
        include_scores = True,
    )
    print(f"\nMemory IDs Used   : {output['memory_ids_used']}")
    print(f"Query Expansion   : {output['latency']['query_expansion_s']:.3f} s")
    print(f"Retrieval Time    : {output['latency']['dense_retrieval_s']:.3f} s")
    print(f"Context Injection : {output['latency']['context_injection_s']:.3f} s")
    print(f"Generation Time   : {output['latency']['slm_generation_s']:.3f} s")
    print(f"Total Pipeline    : {output['latency']['total_pipeline_s']:.3f} s")
    print(f"\nResponse:\n{output['response']}")
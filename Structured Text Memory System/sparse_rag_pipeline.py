"""
sparse_rag_pipeline.py
======================
A complete Sparse RAG (Retrieval-Augmented Generation) pipeline that uses:
  - SQLite FTS5 for inverted-index / BM25 retrieval  (no dense embeddings)
  - Query expansion via Gemma 3 4B through Ollama
  - Custom BM25 × time-decay re-ranking
  - Answer generation via Gemma 3 4B through Ollama

Python 3.10+  |  No GPU required for the retrieval path.
"""

from __future__ import annotations

import json
import math
import sqlite3
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests  # pip install requests

# ---------------------------------------------------------------------------
# 0.  CONFIGURATION
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = "http://localhost:11434"   # default Ollama endpoint
OLLAMA_MODEL: str = "gemma3:4b"                   # pull with: ollama pull gemma3:4b
DB_PATH: str = "sparse_rag.db"                    # SQLite database file

# Time-decay constant λ (lambda).
# Higher λ → older documents penalised more aggressively.
# At λ=0.1: a document 7 days old keeps ~50 % of its BM25 score.
DECAY_LAMBDA: float = 0.1

# Number of candidate documents fetched from FTS5 before re-ranking.
FTS5_CANDIDATE_K: int = 20

# Final top-K chunks passed to the generator.
GENERATOR_TOP_K: int = 4


# ---------------------------------------------------------------------------
# 1.  DATA MODELS
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A unit of text to be ingested into the corpus."""
    content: str
    metadata: dict = field(default_factory=dict)
    # ISO-8601 UTC timestamp; defaults to "now" if not supplied.
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ScoredChunk:
    """A retrieved document chunk with its composite ranking score."""
    doc_id: int
    content: str
    metadata: dict
    timestamp: str
    bm25_score: float       # raw FTS5 BM25 score (negative in SQLite, see §3)
    recency_score: float    # e^(-λ·Δt)
    final_score: float      # bm25_score × recency_score


# ---------------------------------------------------------------------------
# 2.  STORAGE BACKEND  –  SQLite FTS5
# ---------------------------------------------------------------------------

class FTS5Store:
    """
    Manages the SQLite database that backs the sparse retrieval index.

    FTS5 Schema Notes
    -----------------
    •  `CREATE VIRTUAL TABLE … USING fts5(…)` creates an FTS5 virtual table.
       FTS5 builds an inverted index over every tokenised column automatically.
    •  Columns listed inside `fts5(…)` are full-text-searchable.
    •  `UNINDEXED` columns are stored but NOT tokenised – perfect for
       structured data like timestamps and JSON metadata that we don't want
       to pollute keyword searches.
    •  The hidden column `rank` exposes FTS5's built-in BM25 scorer.
       SQLite stores the score as a *negative* float (lower is better in the
       internal representation), so we negate it when we need
       "higher is better" semantics.
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------
    # 2a.  Schema initialisation
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """
        Create the FTS5 virtual table (if it does not already exist).

        SQL breakdown:
            CREATE VIRTUAL TABLE IF NOT EXISTS docs
            USING fts5(
                content,           -- full-text indexed: the document text
                timestamp UNINDEXED,  -- stored only; not tokenised
                metadata  UNINDEXED   -- stored only; JSON blob
            );

        FTS5 automatically creates an internal `rowid` (aliased here as `id`).
        We reference it as `docs.rowid` in plain SELECT queries.
        """
        ddl = """
        CREATE VIRTUAL TABLE IF NOT EXISTS docs
        USING fts5(
            content,               -- tokenised & indexed by FTS5
            timestamp UNINDEXED,   -- ISO-8601 string, stored only
            metadata  UNINDEXED    -- JSON string, stored only
        );
        """
        self.conn.execute(ddl)
        self.conn.commit()

    # ------------------------------------------------------------------
    # 2b.  Document ingestion
    # ------------------------------------------------------------------

    def ingest(self, documents: list[Document]) -> None:
        """
        Bulk-insert a list of Document objects into the FTS5 table.

        FTS5 automatically updates its inverted index on every INSERT,
        so there is no separate "rebuild index" step.
        """
        rows = [
            (doc.content, doc.timestamp, json.dumps(doc.metadata))
            for doc in documents
        ]
        self.conn.executemany(
            "INSERT INTO docs(content, timestamp, metadata) VALUES (?, ?, ?)",
            rows,
        )
        self.conn.commit()
        print(f"[FTS5Store] Ingested {len(rows)} document(s).")

    # ------------------------------------------------------------------
    # 2c.  FTS5 BM25 retrieval
    # ------------------------------------------------------------------

    def search(self, fts_query: str, k: int = FTS5_CANDIDATE_K) -> list[sqlite3.Row]:
        """
        Run an FTS5 MATCH query and return the top-k rows ordered by BM25.

        Key SQL notes:
          •  `docs MATCH ?`  – FTS5 full-text search operator.
          •  `bm25(docs)`    – Built-in BM25 scoring function provided by FTS5.
                               Returns a *negative* number; ORDER BY … ASC puts
                               the best matches first.
          •  We SELECT `docs.rowid AS id` to obtain the integer primary key.

        The FTS5 query syntax supports:
          •  Implicit AND between tokens:     python sqlite fts5
          •  Explicit OR:                     python OR sqlite
          •  Phrase matching:                 "full text search"
          •  Column filters:                  content: machine learning
        """
        sql = """
        SELECT
            docs.rowid          AS id,
            docs.content        AS content,
            docs.timestamp      AS timestamp,
            docs.metadata       AS metadata,
            bm25(docs)          AS bm25_raw    -- negative; lower = more relevant
        FROM docs
        WHERE docs MATCH ?
        ORDER BY bm25_raw ASC                  -- ASC because scores are negative
        LIMIT ?;
        """
        try:
            cursor = self.conn.execute(sql, (fts_query, k))
            return cursor.fetchall()
        except sqlite3.OperationalError as exc:
            # FTS5 raises OperationalError for malformed query syntax.
            print(f"[FTS5Store] Query error (possibly bad FTS5 syntax): {exc}")
            return []

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------
# 3.  OLLAMA CLIENT  –  thin HTTP wrapper
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Minimal HTTP client for the Ollama REST API.
    Ollama exposes a /api/generate endpoint that accepts a model name
    and a prompt and streams tokens back.  We use stream=False here for
    simplicity (the full response is returned as one JSON object).
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """
        POST /api/generate and return the model's response text.

        Ollama API payload:
            {
              "model":  "gemma3:4b",
              "prompt": "...",
              "stream": false,
              "options": { "temperature": 0.2 }
            }
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama.  Is it running?  "
                "Start it with:  ollama serve"
            )
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"Ollama HTTP error: {exc}") from exc


# ---------------------------------------------------------------------------
# 4.  QUERY EXPANSION MODULE
# ---------------------------------------------------------------------------

class QueryExpander:
    """
    Uses Gemma 3 4B (via Ollama) to generate synonyms / related keywords
    that broaden the sparse retrieval net before the FTS5 search.

    Why query expansion for sparse retrieval?
    -----------------------------------------
    BM25 / inverted-index methods suffer from *vocabulary mismatch*: if the
    user writes "car" but documents use "automobile", there is zero overlap.
    Query expansion compensates by surfacing lexically diverse but semantically
    related terms – without any dense vector computation.
    """

    EXPANSION_PROMPT_TEMPLATE = textwrap.dedent("""\
        You are a search query expansion assistant.
        Given the user's query, generate exactly 4 distinct, relevant keywords
        or short phrases (synonyms, related concepts, or alternative phrasings)
        that would help retrieve documents about this topic.

        Rules:
        - Output ONLY a JSON array of strings.  No explanations.
        - Each entry must be 1-3 words.
        - Do NOT repeat the original query.

        Original query: {query}

        JSON array:""")

    def __init__(self, ollama_client: OllamaClient) -> None:
        self.client = ollama_client

    def expand(self, query: str) -> list[str]:
        """
        Returns a list of expanded terms.  Falls back gracefully if the
        model output cannot be parsed as JSON.
        """
        prompt = self.EXPANSION_PROMPT_TEMPLATE.format(query=query)
        raw = self.client.generate(prompt, temperature=0.4)

        # Attempt to extract a JSON array from the model's response.
        try:
            # Strip any markdown code fences the model might add.
            clean = raw.strip().strip("`").replace("json", "", 1).strip()
            terms: list[str] = json.loads(clean)
            if isinstance(terms, list):
                # Sanitise: keep only short string entries.
                terms = [t for t in terms if isinstance(t, str) and len(t) < 60]
                print(f"[QueryExpander] Expanded terms: {terms}")
                return terms
        except (json.JSONDecodeError, ValueError):
            pass

        print(f"[QueryExpander] Could not parse expansion; using original query only.")
        return []

    def build_fts5_query(self, original_query: str, expanded_terms: list[str]) -> str:
        """
        Combines the original query and expanded terms into an FTS5 OR query.

        FTS5 OR syntax:  token1 OR token2 OR token3
        Phrases must be quoted: "machine learning" OR "deep learning"

        Strategy: the original query tokens are implicitly ANDed (FTS5 default),
        then OR'd with each expanded term to widen recall.
        """
        # Wrap multi-word terms in quotes for FTS5 phrase search.
        def fts5_term(t: str) -> str:
            t = t.strip().replace('"', '')   # sanitise quotes
            return f'"{t}"' if " " in t else t

        base = fts5_term(original_query)
        extra = " OR ".join(fts5_term(t) for t in expanded_terms if t)
        if extra:
            return f"{base} OR {extra}"
        return base


# ---------------------------------------------------------------------------
# 5.  BM25 + TIME-DECAY RE-RANKER
# ---------------------------------------------------------------------------

class TimeDecayReranker:
    """
    Re-ranks FTS5 candidates using the composite score:

        Score_final = Score_BM25 × e^(-λ · Δt)

    Where:
        Score_BM25  = -bm25_raw   (negated so higher = better)
        Δt          = (now - doc_timestamp) in fractional days  ≥ 0
        λ (lambda)  = decay constant (default 0.1)
        e           = Euler's number (~2.718)

    Mathematical intuition
    ----------------------
    •  e^(-λ·Δt) is the *exponential decay* function.  At Δt=0 (just indexed)
       the factor is 1.0 (no penalty).  As Δt → ∞ the factor → 0.
    •  With λ=0.1:
         Δt =  0 days  → factor = 1.000  (no decay)
         Δt =  7 days  → factor ≈ 0.496  (~50 % of score retained)
         Δt = 14 days  → factor ≈ 0.247  (~25 % retained)
         Δt = 30 days  → factor ≈ 0.050  (~5  % retained)
    •  Adjust λ upward (e.g. 0.3) to penalise older docs more aggressively,
       or set λ=0 to disable recency weighting entirely.
    """

    def __init__(self, decay_lambda: float = DECAY_LAMBDA) -> None:
        self.lam = decay_lambda

    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse an ISO-8601 UTC timestamp string into an aware datetime."""
        try:
            # Python 3.11+ handles 'Z' natively; for 3.10 we replace it.
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            # If the timestamp is malformed, treat the document as very old.
            return datetime.min.replace(tzinfo=timezone.utc)

    def _delta_days(self, timestamp: str) -> float:
        """
        Compute Δt = (now_utc - doc_timestamp) expressed in fractional days.
        Clipped to 0 so future timestamps don't produce negative decay.
        """
        doc_dt = self._parse_timestamp(timestamp)
        now_dt = datetime.now(timezone.utc)
        delta_seconds = (now_dt - doc_dt).total_seconds()
        return max(0.0, delta_seconds / 86_400)   # 86 400 s per day

    def rerank(self, rows: list[sqlite3.Row], top_k: int = GENERATOR_TOP_K) -> list[ScoredChunk]:
        """
        Apply the time-decay formula to each candidate row and return the
        top_k results sorted by Score_final descending.
        """
        scored: list[ScoredChunk] = []
        for row in rows:
            bm25_raw: float = row["bm25_raw"]    # negative float from SQLite
            bm25_pos: float = -bm25_raw           # flip sign → higher is better

            delta_t: float = self._delta_days(row["timestamp"])

            # Core formula: e^(-λ · Δt)
            recency_factor: float = math.exp(-self.lam * delta_t)

            # Final composite score
            final: float = bm25_pos * recency_factor

            scored.append(
                ScoredChunk(
                    doc_id=row["id"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"] or "{}"),
                    timestamp=row["timestamp"],
                    bm25_score=bm25_pos,
                    recency_score=recency_factor,
                    final_score=final,
                )
            )

        # Sort by final score descending (best first).
        scored.sort(key=lambda c: c.final_score, reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# 6.  GENERATOR  –  Sparse-Context Prompt Builder + Gemma 3 4B
# ---------------------------------------------------------------------------

class SparseRAGGenerator:
    """
    Builds a RAG prompt from the re-ranked chunks and calls Gemma 3 4B
    to synthesise a grounded answer.

    Prompt design principles (sparse RAG):
    •  Explicitly label each context chunk with its source index so the
       model can attribute claims.
    •  Instruct the model to rely only on the provided context to keep
       answers grounded and reduce hallucination.
    •  Keep the prompt concise – smaller context windows matter for 4B models.
    """

    SYSTEM_INSTRUCTION = textwrap.dedent("""\
        You are a precise question-answering assistant.
        Answer the user's question using ONLY the information in the numbered
        context passages below.  If the answer is not present in the context,
        say "I don't have enough information to answer that."
        Be concise and factual.
        """)

    def __init__(self, ollama_client: OllamaClient) -> None:
        self.client = ollama_client

    def _build_prompt(self, query: str, chunks: list[ScoredChunk]) -> str:
        context_block = "\n\n".join(
            f"[{i + 1}] (score={c.final_score:.4f}, date={c.timestamp[:10]})\n{c.content}"
            for i, c in enumerate(chunks)
        )
        return (
            self.SYSTEM_INSTRUCTION
            + "\n--- CONTEXT ---\n"
            + context_block
            + "\n--- END CONTEXT ---\n\n"
            + f"Question: {query}\n\nAnswer:"
        )

    def generate(self, query: str, chunks: list[ScoredChunk]) -> str:
        if not chunks:
            return "No relevant documents were found in the index."
        prompt = self._build_prompt(query, chunks)
        return self.client.generate(prompt, temperature=0.1)


# ---------------------------------------------------------------------------
# 7.  PIPELINE ORCHESTRATOR
# ---------------------------------------------------------------------------

class SparseRAGPipeline:
    """
    Top-level orchestrator that wires all components together.

    Pipeline flow:
        user_query
            │
            ▼
        QueryExpander  ──(Ollama/Gemma 3 4B)──▶  expanded_terms
            │
            ▼
        FTS5Store.search(expanded_fts5_query)
            │  (top-K candidates)
            ▼
        TimeDecayReranker.rerank()
            │  (top-N re-ranked chunks)
            ▼
        SparseRAGGenerator.generate()  ──(Ollama/Gemma 3 4B)──▶  answer
    """

    def __init__(
        self,
        db_path: str = DB_PATH,
        ollama_base_url: str = OLLAMA_BASE_URL,
        ollama_model: str = OLLAMA_MODEL,
        decay_lambda: float = DECAY_LAMBDA,
        candidate_k: int = FTS5_CANDIDATE_K,
        top_k: int = GENERATOR_TOP_K,
    ) -> None:
        self.store = FTS5Store(db_path=db_path)
        self.client = OllamaClient(base_url=ollama_base_url, model=ollama_model)
        self.expander = QueryExpander(self.client)
        self.reranker = TimeDecayReranker(decay_lambda=decay_lambda)
        self.generator = SparseRAGGenerator(self.client)
        self.candidate_k = candidate_k
        self.top_k = top_k

    def ingest(self, documents: list[Document]) -> None:
        self.store.ingest(documents)

    def query(self, user_query: str, verbose: bool = True) -> str:
        """
        Run the full pipeline for a user query and return the generated answer.

        Parameters
        ----------
        user_query : str
            The raw natural-language question from the user.
        verbose : bool
            If True, prints intermediate pipeline state for debugging.
        """
        print(f"\n{'='*60}")
        print(f"[Pipeline] User query: {user_query!r}")
        print(f"{'='*60}")

        # Step 1 – Query Expansion
        expanded_terms = self.expander.expand(user_query)
        fts5_query = self.expander.build_fts5_query(user_query, expanded_terms)
        print(f"[Pipeline] FTS5 query: {fts5_query!r}")

        # Step 2 – Sparse Retrieval
        raw_rows = self.store.search(fts5_query, k=self.candidate_k)
        print(f"[Pipeline] FTS5 returned {len(raw_rows)} candidate(s).")

        # Step 3 – BM25 + Time-Decay Re-ranking
        chunks = self.reranker.rerank(raw_rows, top_k=self.top_k)
        if verbose:
            print(f"\n[Pipeline] Top-{len(chunks)} chunks after re-ranking:")
            for i, c in enumerate(chunks):
                print(
                    f"  [{i+1}] id={c.doc_id}  "
                    f"BM25={c.bm25_score:.4f}  "
                    f"recency={c.recency_score:.4f}  "
                    f"final={c.final_score:.4f}  "
                    f"date={c.timestamp[:10]}\n"
                    f"       {c.content[:100]}…"
                )

        # Step 4 – Generation
        print(f"\n[Pipeline] Calling Gemma 3 4B for final answer …")
        answer = self.generator.generate(user_query, chunks)
        return answer

    def close(self) -> None:
        self.store.close()


# ---------------------------------------------------------------------------
# 8.  __main__  –  demo with mock data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from datetime import timedelta

    # ------------------------------------------------------------------
    # 8a.  Build a realistic mock corpus with varied timestamps
    # ------------------------------------------------------------------
    now = datetime.now(timezone.utc)

    mock_documents: list[Document] = [
        Document(
            content=(
                "SQLite FTS5 is a powerful full-text search extension that provides "
                "BM25 ranking out of the box.  It builds an inverted index over "
                "tokenised text columns, enabling fast keyword search across millions "
                "of rows without any external search engine."
            ),
            metadata={"source": "sqlite_docs", "author": "SQLite Team"},
            timestamp=(now - timedelta(days=1)).isoformat(),   # yesterday
        ),
        Document(
            content=(
                "BM25 (Best Match 25) is a probabilistic ranking function used in "
                "information retrieval.  It considers term frequency (TF) and inverse "
                "document frequency (IDF) to score documents against a query.  "
                "Unlike TF-IDF, BM25 saturates term frequency to avoid over-rewarding "
                "documents that merely repeat a keyword many times."
            ),
            metadata={"source": "ir_textbook", "chapter": 3},
            timestamp=(now - timedelta(days=30)).isoformat(),  # 1 month ago
        ),
        Document(
            content=(
                "Retrieval-Augmented Generation (RAG) combines a retrieval system with "
                "a language model generator.  In a sparse RAG setup, the retrieval step "
                "uses keyword-based methods like BM25 instead of dense vector embeddings, "
                "making it faster and more interpretable while remaining highly effective "
                "for domain-specific corpora."
            ),
            metadata={"source": "rag_survey_2024", "section": "sparse"},
            timestamp=(now - timedelta(days=5)).isoformat(),   # 5 days ago
        ),
        Document(
            content=(
                "Gemma 3 is a family of lightweight open-weight language models released "
                "by Google DeepMind.  The 4B parameter variant achieves strong performance "
                "on reasoning and instruction-following benchmarks while fitting comfortably "
                "in consumer GPU memory (or even on CPU with llama.cpp / Ollama)."
            ),
            metadata={"source": "gemma_blog", "model_size": "4B"},
            timestamp=(now - timedelta(days=3)).isoformat(),   # 3 days ago
        ),
        Document(
            content=(
                "Time-decay weighting is a recency-aware re-ranking technique.  "
                "An exponential decay function e^(-λ·Δt) is multiplied with the base "
                "relevance score so that recently published documents receive a boost "
                "relative to older ones – useful in news retrieval, customer-support "
                "knowledge bases, and any domain where freshness matters."
            ),
            metadata={"source": "ranking_blog", "topic": "time-decay"},
            timestamp=(now - timedelta(days=60)).isoformat(),  # 2 months ago
        ),
        Document(
            content=(
                "Query expansion is a classic technique to improve recall in keyword "
                "search.  By adding synonyms, related terms, or paraphrased phrases to "
                "the original query, the retrieval engine can surface documents that share "
                "the same concept but use different vocabulary – addressing the vocabulary "
                "mismatch problem inherent to bag-of-words models."
            ),
            metadata={"source": "nlp_textbook", "chapter": 7},
            timestamp=(now - timedelta(days=14)).isoformat(),  # 2 weeks ago
        ),
        Document(
            content=(
                "Ollama is an open-source tool for running large language models locally.  "
                "It manages model downloads, quantisation, and exposes a simple REST API "
                "compatible with many LLM client libraries.  Models like Gemma, Llama, "
                "and Mistral can be pulled and served with a single CLI command."
            ),
            metadata={"source": "ollama_docs"},
            timestamp=(now - timedelta(days=2)).isoformat(),   # 2 days ago
        ),
        Document(
            content=(
                "Inverted indexes map each unique term in a corpus to a list of documents "
                "that contain it (the posting list).  This structure enables O(1) lookup "
                "per term and is the foundation of every major search engine, from Apache "
                "Lucene to SQLite FTS5."
            ),
            metadata={"source": "cs_fundamentals"},
            timestamp=(now - timedelta(days=90)).isoformat(),  # 3 months ago
        ),
    ]

    # ------------------------------------------------------------------
    # 8b.  Initialise pipeline and ingest documents
    # ------------------------------------------------------------------
    pipeline = SparseRAGPipeline(
        db_path=DB_PATH,
        decay_lambda=DECAY_LAMBDA,
        candidate_k=FTS5_CANDIDATE_K,
        top_k=GENERATOR_TOP_K,
    )
    pipeline.ingest(mock_documents)

    # ------------------------------------------------------------------
    # 8c.  Run a sample query through the full pipeline
    # ------------------------------------------------------------------
    sample_query = "How does BM25 ranking work and how does FTS5 implement it?"

    try:
        answer = pipeline.query(sample_query, verbose=True)
        print(f"\n{'='*60}")
        print("FINAL ANSWER")
        print(f"{'='*60}")
        print(answer)
    finally:
        pipeline.close()

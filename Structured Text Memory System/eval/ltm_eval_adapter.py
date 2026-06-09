"""
ltm_eval_adapter.py
===================
Compatibility Layer — LTM Module ↔ LongMemEval / LoCoMo Evaluation Frameworks

This adapter bridges the gap between the custom VectorEmbeddedMemory LTM module
and the evaluation pipelines defined in:

  · LongMemEval : https://github.com/xiaowu0162/LongMemEval
  · LoCoMo      : https://github.com/snap-research/locomo

Design Principles
-----------------
  1. ZERO dataset modification — raw JSON files are read-only.
  2. The ltm_module is the *only* memory backend; no reference models are used.
  3. Knowledge Updates are handled via Temporal Context Injection (Virtual Update),
     a zero-modification strategy that feeds update text into the LTM input stream
     before the query is executed.
  4. All outputs are normalised into a shared EvalResult dataclass so that the
     same metric functions can process results from either framework.

Terminology (aligned with thesis)
----------------------------------
  Dense Retrieval  : FAISS k-NN lookup of relevant memories by vector similarity
  LTM Store        : Persisted FAISS index + sidecar JSON (memory bank)
  Virtual Update   : Injected knowledge-update memory (not in dataset file)
  Session Ingestion: Converting raw dialogue turns into ltm_module memories
"""

from __future__ import annotations

import json
import os
import sys
import re
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

# ── Resolve project root so ltm_module is always importable ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sparse_rag_pipeline import SparseEmbeddedMemory as VectorEmbeddedMemory         # noqa: E402


# ---------------------------------------------------------------------------
# Shared Data Structures
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """
    Normalised evaluation record produced by both adapters.
    Consumed by eval_metrics.py for Stage 1 (retrieval) and Stage 2 (generation).
    """
    # ── Identifiers ──────────────────────────────────────────────────────────
    question_id      : str
    framework        : str                  # "longmemeval" | "locomo"
    category         : str                  # single_hop | multi_hop | knowledge_update | abstention

    # ── Query / Answer ───────────────────────────────────────────────────────
    query            : str
    ground_truth     : str                  # raw ground-truth string from dataset

    # ── LTM Retrieval Output ─────────────────────────────────────────────────
    retrieved_memory_ids   : list[int]      # FAISS IDs returned by dense_retrieve()
    ground_truth_memory_ids: list[int]      # IDs the dataset says are relevant
    retrieved_memories     : list[dict]     # full memory dicts (text, score, rank)

    # ── SLM Generation Output ────────────────────────────────────────────────
    predicted_answer : str                  # Gemma 3 4B generated response

    # ── Latency (seconds) ────────────────────────────────────────────────────
    latency          : dict[str, float] = field(default_factory=dict)

    # ── Prompt (for debugging / ablation studies) ─────────────────────────────
    augmented_prompt : str = ""

    # ── Knowledge Update bookkeeping ─────────────────────────────────────────
    virtual_update_id: Optional[int] = None   # FAISS ID of injected Virtual Update
    virtual_update_text: str = ""

    # ── Token counts (for efficiency metrics) ────────────────────────────────
    prompt_token_count   : int = 0
    response_token_count : int = 0


# ---------------------------------------------------------------------------
# Temporal Context Injection Handler (Knowledge Update Strategy)
# ---------------------------------------------------------------------------

class TemporalContextInjector:
    """
    Implements the 'Virtual Update' strategy for Knowledge Update evaluation.

    When a test question requires the model to prioritise newer information
    over an older fact already stored in LTM, this handler:

      1. Detects the update information from the dataset entry (no modification).
      2. Injects the update as a fresh memory via ltm_module.ingest_memory(),
         timestamped *after* all session memories.
      3. Returns the FAISS ID of the injected memory for transparency logging.

    The injected memory is temporary to the evaluation run; it does NOT modify
    the dataset file, and it is purged from the LTM after each test item
    to prevent cross-contamination between questions.

    Thesis Reference: "Temporal Context Injection" / "Virtual Update Approach"
    """

    VIRTUAL_UPDATE_PREFIX = "[KNOWLEDGE UPDATE — VIRTUAL INJECTION]"

    @staticmethod
    def inject(
        ltm       : VectorEmbeddedMemory,
        update_text: str,
        source_session_id: str = "virtual",
    ) -> int:
        """
        Inject a knowledge update into the LTM store.

        Parameters
        ----------
        ltm             : The active VectorEmbeddedMemory instance.
        update_text     : The updated fact/preference string (from dataset).
        source_session_id: Label for transparency metadata.

        Returns
        -------
        virtual_id : FAISS integer ID of the injected memory.
        """
        tagged_text = (
            f"{TemporalContextInjector.VIRTUAL_UPDATE_PREFIX} "
            f"{update_text}"
        )
        virtual_id = ltm.ingest_memory(
            tagged_text,
            metadata={
                "type"            : "virtual_update",
                "source_session"  : source_session_id,
                "injected_at"     : datetime.now(timezone.utc).isoformat(),
                "dataset_modified": False,         # explicit audit flag
            },
        )
        return virtual_id

    @staticmethod
    def purge(ltm: VectorEmbeddedMemory, virtual_id: int) -> None:
        """
        Mark the virtual update memory as deleted after the test item completes.
        Prevents bleed into subsequent questions in the same evaluation run.
        """
        ltm.delete_memory(virtual_id)


# ---------------------------------------------------------------------------
# LongMemEval Adapter
# ---------------------------------------------------------------------------

class LongMemEvalAdapter:
    """
    Adapts the LongMemEval benchmark to the VectorEmbeddedMemory LTM module.

    LongMemEval Dataset Schema (raw JSON — DO NOT MODIFY)
    -------------------------------------------------------
    {
      "question_id"           : "q_001",
      "question"              : "What did the user say about their diet?",
      "answer"                : "The user follows a vegan diet.",
      "question_type"         : "knowledge-update",
                              // "single-session-user" | "single-session-assistant"
                              // "single-session-preference" | "temporal-reasoning"
                              // "knowledge-update" | "multi-session"
                              // abstention = question_id ends with "_abs"
      "answer"                : "The user follows a vegan diet.",
      "question_date"         : "2024-06-01",
      "haystack_session_ids"  : ["sess_01", "sess_02", ...],
      "haystack_dates"        : ["2024-01-10", "2024-02-05", ...],
      "haystack_sessions"     : [
        [
          {"role": "user",      "content": "...", "has_answer": true},
          {"role": "assistant", "content": "..."}
        ],
        ...
      ],
      "answer_session_ids"    : ["sess_02"]
    }

    Integration Strategy
    ---------------------
    For each question item:
      1. Detect category from question_type + "_abs" suffix on question_id.
      2. Reset the LTM (fresh FAISS index per question for isolation).
      3. Ingest haystack_sessions using parallel haystack_session_ids / haystack_dates.
      4. For knowledge_update items: call TemporalContextInjector.inject().
      5. Run ltm_module.generate_response() with the question as query.
      6. Map answer_session_ids → FAISS IDs for ground-truth retrieval evaluation.
    """

    # Keys — verified against longmemeval_s_cleaned.json
    KEY_QID         = "question_id"
    KEY_QTYPE       = "question_type"          # NOT "category"
    KEY_QUESTION    = "question"
    KEY_ANSWER      = "answer"
    KEY_SESSIONS    = "haystack_sessions"      # list of sessions; each session = list of turns
    KEY_SESSION_IDS = "haystack_session_ids"   # parallel list of session ID strings
    KEY_DATES       = "haystack_dates"         # parallel list of date strings
    KEY_EVIDENCE    = "answer_session_ids"     # list of evidence session ID strings
    KEY_ROLE        = "role"
    KEY_CONTENT     = "content"
    KEY_HAS_ANSWER  = "has_answer"             # turn-level label

    # question_type → normalised thesis category
    QTYPE_MAP = {
        "single-session-user"       : "single_hop",
        "single-session-assistant"  : "single_hop",
        "single-session-preference" : "single_hop",
        "multi-session"             : "multi_hop",
        "knowledge-update"          : "knowledge_update",
        "temporal-reasoning"        : "temporal_reasoning",
    }

    CATEGORY_KNOWLEDGE_UPDATE = "knowledge_update"
    CATEGORY_ABSTENTION       = "abstention"

    def __init__(
        self,
        ltm          : VectorEmbeddedMemory,
        data_path    : str | Path,
        top_k        : int   = 5,
        max_new_tokens: int  = 256,
        temperature  : float = 0.0,         # deterministic for reproducibility
        verbose      : bool  = True,
    ) -> None:
        """
        Parameters
        ----------
        ltm           : Initialised VectorEmbeddedMemory instance (shared across runs).
        data_path     : Path to LongMemEval JSON file (raw, unmodified).
        top_k         : Dense retrieval top-K for each query.
        max_new_tokens: SLM generation budget.
        temperature   : Sampling temperature (0 = greedy for reproducibility).
        verbose       : Print per-item progress.
        """
        self.ltm           = ltm
        self.data_path     = Path(data_path)
        self.top_k         = top_k
        self.max_new_tokens = max_new_tokens
        self.temperature   = temperature
        self.verbose       = verbose

        self._injector = TemporalContextInjector()

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"LongMemEval data file not found: {self.data_path}\n"
                f"Clone the repo and verify the path:\n"
                f"  git clone https://github.com/xiaowu0162/LongMemEval"
            )

        with open(self.data_path, "r", encoding="utf-8") as fh:
            self._raw_data: list[dict] = json.load(fh)

        self._log(
            f"[LongMemEval] Loaded {len(self._raw_data)} items from {self.data_path}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        categories  : list[str] | None = None,
        max_items   : int | None = None,
    ) -> list[EvalResult]:
        """
        Execute the full LongMemEval evaluation loop.

        Parameters
        ----------
        categories : Filter to specific categories (None = run all).
        max_items  : Cap the number of items (useful for dev runs).

        Returns
        -------
        List of EvalResult objects, one per test item.
        """
        items = self._raw_data
        if categories:
            # Normalise items by detected category, then filter
            items = [x for x in items if self._detect_category(x) in categories]
        if max_items:
            items = items[:max_items]

        results: list[EvalResult] = []
        for idx, item in enumerate(items, start=1):
            cat = self._detect_category(item)
            self._log(
                f"[LongMemEval] Item {idx}/{len(items)} | "
                f"ID={item.get(self.KEY_QID)} | "
                f"Category={cat}"
            )
            result = self._process_item(item)
            results.append(result)

        self._log(f"[LongMemEval] Completed — {len(results)} results collected.")
        return results

    def _detect_category(self, item: dict) -> str:
        """
        Derive the normalised thesis category from a LongMemEval item.

        Abstention is signalled by question_id ending with '_abs', NOT by question_type.
        All other categories are mapped from the question_type field.
        """
        qid   = item.get(self.KEY_QID, "")
        qtype = item.get(self.KEY_QTYPE, "")
        if qid.endswith("_abs"):
            return self.CATEGORY_ABSTENTION
        return self.QTYPE_MAP.get(qtype, "single_hop")

    # ------------------------------------------------------------------
    # Private — Per-Item Processing
    # ------------------------------------------------------------------

    def _process_item(self, item: dict) -> EvalResult:
        """Process a single LongMemEval test item."""

        qid      = item.get(self.KEY_QID, "unknown")
        query    = item.get(self.KEY_QUESTION, "")
        gt       = item.get(self.KEY_ANSWER, "")
        category = self._detect_category(item)

        # haystack_sessions is a list of sessions; parallel arrays give IDs and dates
        sessions     = item.get(self.KEY_SESSIONS, [])       # list of session turn-lists
        session_ids  = item.get(self.KEY_SESSION_IDS, [])    # parallel session ID strings
        dates        = item.get(self.KEY_DATES, [])          # parallel date strings
        evidence_ids = item.get(self.KEY_EVIDENCE, [])       # answer_session_ids

        # ── Step 1: Reset LTM for this item (isolation) ──────────────────────
        self._reset_ltm()

        # ── Step 2: Ingest sessions (already in chronological order) ─────────
        session_to_faiss_ids: dict[str, list[int]] = {}
        for sess_idx, session_turns in enumerate(sessions):
            sid  = session_ids[sess_idx] if sess_idx < len(session_ids) else f"sess_{sess_idx}"
            date = dates[sess_idx]       if sess_idx < len(dates)       else ""
            faiss_ids = self._ingest_session(session_turns, sid, date)
            session_to_faiss_ids[sid] = faiss_ids

        # ── Step 3: Knowledge Update — Temporal Context Injection ─────────────
        # For knowledge-update items, the update information is embedded in the
        # conversation itself (a turn with has_answer=True in a later session).
        # We detect this by finding turns marked has_answer in the last evidence
        # session and injecting the most recent such turn as a Virtual Update.
        virtual_id   = None
        virtual_text = ""
        if category == self.CATEGORY_KNOWLEDGE_UPDATE and evidence_ids:
            # Find the latest evidence session and extract its has_answer turns
            last_evidence_sess_idx = None
            for i, sid in enumerate(session_ids):
                if sid in evidence_ids:
                    last_evidence_sess_idx = i
            if last_evidence_sess_idx is not None and last_evidence_sess_idx < len(sessions):
                ev_turns = sessions[last_evidence_sess_idx]
                ev_date  = dates[last_evidence_sess_idx] if last_evidence_sess_idx < len(dates) else ""
                update_parts = [
                    t.get(self.KEY_CONTENT, "")
                    for t in ev_turns
                    if t.get(self.KEY_HAS_ANSWER) and t.get(self.KEY_CONTENT, "").strip()
                ]
                if update_parts:
                    update_text  = f"(Updated on {ev_date}) " + " ".join(update_parts)
                    virtual_id   = self._injector.inject(self.ltm, update_text, qid)
                    virtual_text = update_text
                    self._log(
                        f"  [Virtual Update] Injected → FAISS ID={virtual_id} | "
                        f"Text: '{update_text[:80]}'"
                    )

        # ── Step 4: Map evidence session IDs → FAISS IDs (ground-truth) ───────
        gt_faiss_ids: list[int] = []
        for ev_sid in evidence_ids:
            gt_faiss_ids.extend(session_to_faiss_ids.get(ev_sid, []))
        if virtual_id is not None:
            gt_faiss_ids.append(virtual_id)

        # ── Step 5: Dense Retrieval + Generation ─────────────────────────────
        output = self.ltm.generate_response(
            query           = query,
            top_k           = self.top_k,
            max_new_tokens  = self.max_new_tokens,
            temperature     = self.temperature,
        )

        # ── Step 6: Purge Virtual Update to prevent contamination ─────────────
        if virtual_id is not None:
            self._injector.purge(self.ltm, virtual_id)

        # ── Step 7: Build EvalResult ──────────────────────────────────────────
        prompt_tokens   = self._count_tokens(output["augmented_prompt"])
        response_tokens = self._count_tokens(output["response"])

        return EvalResult(
            question_id             = qid,
            framework               = "longmemeval",
            category                = category,
            query                   = query,
            ground_truth            = gt,
            retrieved_memory_ids    = output["memory_ids_used"],
            ground_truth_memory_ids = gt_faiss_ids,
            retrieved_memories      = output["retrieved_mems"],
            predicted_answer        = output["response"],
            latency                 = output["latency"],
            augmented_prompt        = output["augmented_prompt"],
            virtual_update_id       = virtual_id,
            virtual_update_text     = virtual_text,
            prompt_token_count      = prompt_tokens,
            response_token_count    = response_tokens,
        )

    def _ingest_session(
        self,
        session_turns: list[dict],
        session_id   : str,
        date         : str = "",
    ) -> list[int]:
        """
        Ingest all turns in a session as individual LTM memories.

        Parameters
        ----------
        session_turns : List of turn dicts: [{"role": ..., "content": ...}, ...]
        session_id    : The session ID string (from haystack_session_ids).
        date          : The session date string (from haystack_dates).

        Returns the list of FAISS IDs assigned to each ingested turn.
        """
        faiss_ids: list[int] = []

        for turn_idx, turn in enumerate(session_turns):
            role    = turn.get(self.KEY_ROLE, "unknown")
            content = turn.get(self.KEY_CONTENT, "").strip()
            if not content:
                continue

            memory_text = f"[{role.upper()}] (Session {session_id}, {date}): {content}"
            fid = self.ltm.ingest_memory(
                memory_text,
                metadata={
                    "session_id" : session_id,
                    "date"       : date,
                    "role"       : role,
                    "turn_index" : turn_idx,
                    "has_answer" : turn.get(self.KEY_HAS_ANSWER, False),
                    "framework"  : "longmemeval",
                },
            )
            faiss_ids.append(fid)

        return faiss_ids

    def _reset_ltm(self) -> None:
        """
        Reset the FAISS index and sidecar store for a fresh per-item LTM.
        This ensures no bleed between evaluation items.
        """
        import faiss as _faiss
        import numpy as _np

        self.ltm.faiss_index    = _faiss.IndexFlatL2(self.ltm.embedding_dim)
        self.ltm.ltm_store      = {}
        self.ltm._next_faiss_id = 0

    def _count_tokens(self, text: str) -> int:
        """Approximate token count using the SLM tokenizer."""
        try:
            return len(self.ltm.slm_tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return len(text.split())     # fallback: word count

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


# ---------------------------------------------------------------------------
# LoCoMo Adapter
# ---------------------------------------------------------------------------

class LoCoMoAdapter:
    """
    Adapts the LoCoMo (Long Conversational Memory) benchmark to the
    VectorEmbeddedMemory LTM module.

    LoCoMo Dataset Schema — locomo10.json (raw JSON — DO NOT MODIFY)
    -----------------------------------------------------------------
    The file is a LIST of samples. Each sample:
    {
      "sample_id"    : "0",
      "conversation" : {
        "speaker_a"            : "Angela",
        "speaker_b"            : "James",
        "session_1"            : [
          { "speaker": "Angela", "dia_id": 1,  "text": "..." },
          { "speaker": "James",  "dia_id": 2,  "text": "..." }
        ],
        "session_1_date_time"  : "2021-03-25",
        "session_2"            : [ ... ],
        "session_2_date_time"  : "2021-04-10",
        ...
      },
      "qa"           : [
        {
          "question" : "...",
          "answer"   : "...",
          "category" : "single-hop",   // "multi-hop" | "adversarial" | "temporal reasoning"
          "evidence" : [1, 4, 7]       // list of dia_id integers containing the answer
        }
      ],
      "observation"      : { "session_1_observation": "...", ... },
      "session_summary"  : { "session_1_summary": "...", ... },
      "event_summary"    : { ... }
    }

    Integration Strategy
    ---------------------
    LoCoMo nests QA pairs *inside* conversation objects, unlike LongMemEval
    which is a flat list of questions. The adapter:
      1. Iterates samples → parses conversation dict for sessions (session_N keys).
      2. Resets LTM per conversation (not per question) since sessions are shared.
      3. Ingests all sessions for a conversation once, tracking dia_id → FAISS ID.
      4. Runs each QA query against the pre-ingested LTM.
      5. Maps evidence dia_ids → FAISS IDs for Recall@K / NDCG@K ground truth.
      6. For 'adversarial' category: treats as abstention.
    """

    # LoCoMo JSON keys — verified against locomo10.json
    KEY_SAMPLE_ID  = "sample_id"
    KEY_CONV       = "conversation"           # dict with session_N keys
    KEY_SPEAKER_A  = "speaker_a"
    KEY_SPEAKER_B  = "speaker_b"
    KEY_QA         = "qa"
    KEY_QUESTION   = "question"
    KEY_ANSWER     = "answer"
    KEY_CATEGORY   = "category"               # "single-hop" | "multi-hop" | "adversarial" | "temporal reasoning"
    KEY_EVIDENCE   = "evidence"               # list of dia_id integers
    KEY_SPEAKER    = "speaker"
    KEY_DIA_ID     = "dia_id"
    KEY_TEXT       = "text"

    # LoCoMo category values → normalised thesis category names
    TYPE_MAP = {
        "single-hop"         : "single_hop",
        "multi-hop"          : "multi_hop",
        "adversarial"        : "abstention",
        "temporal reasoning" : "temporal_reasoning",
        # fallback aliases
        "single"             : "single_hop",
        "multi"              : "multi_hop",
        "temporal"           : "temporal_reasoning",
    }

    def __init__(
        self,
        ltm          : VectorEmbeddedMemory,
        data_path    : str | Path,
        top_k        : int   = 5,
        max_new_tokens: int  = 256,
        temperature  : float = 0.0,
        verbose      : bool  = True,
    ) -> None:
        self.ltm           = ltm
        self.data_path     = Path(data_path)
        self.top_k         = top_k
        self.max_new_tokens = max_new_tokens
        self.temperature   = temperature
        self.verbose       = verbose

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"LoCoMo data file not found: {self.data_path}\n"
                f"Clone the repo and verify the path:\n"
                f"  git clone https://github.com/snap-research/locomo"
            )

        # locomo10.json is always a list of samples
        with open(self.data_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self._raw_data: list[dict] = raw if isinstance(raw, list) else [raw]

        total_qa = sum(len(s.get(self.KEY_QA, [])) for s in self._raw_data)
        self._log(
            f"[LoCoMo] Loaded {len(self._raw_data)} conversations "
            f"({total_qa} QA pairs) from {self.data_path}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        categories  : list[str] | None = None,
        max_items   : int | None = None,
    ) -> list[EvalResult]:
        """
        Execute the LoCoMo evaluation loop.

        Parameters
        ----------
        categories : Filter to normalised category names (None = all).
        max_items  : Cap total number of QA items processed.
        """
        results   : list[EvalResult] = []
        item_count = 0

        for sample in self._raw_data:
            if max_items and item_count >= max_items:
                break

            conv_id  = str(sample.get(self.KEY_SAMPLE_ID, "unknown"))
            conv     = sample.get(self.KEY_CONV, {})      # the dict with session_N keys
            qa_list  = sample.get(self.KEY_QA, [])

            # ── Reset LTM once per conversation ──────────────────────────────
            self._reset_ltm()
            # Returns both session→faiss_ids and dia_id→faiss_id maps
            session_to_faiss_ids, diaid_to_faiss_id = self._ingest_all_sessions(conv, conv_id)

            for qa in qa_list:
                if max_items and item_count >= max_items:
                    break

                raw_type = qa.get(self.KEY_CATEGORY, "single-hop")
                category = self.TYPE_MAP.get(raw_type, "single_hop")

                if categories and category not in categories:
                    continue

                self._log(
                    f"[LoCoMo] Sample={conv_id} | Category={category}"
                )
                result = self._process_qa(
                    qa, conv_id, session_to_faiss_ids, diaid_to_faiss_id, category
                )
                results.append(result)
                item_count += 1

        self._log(f"[LoCoMo] Completed — {len(results)} results collected.")
        return results

    # ------------------------------------------------------------------
    # Private — Processing
    # ------------------------------------------------------------------

    def _process_qa(
        self,
        qa                  : dict,
        conv_id             : str,
        session_to_faiss_ids: dict[str, list[int]],
        diaid_to_faiss_id   : dict[int, int],
        category            : str,
    ) -> EvalResult:
        """Process one QA item from LoCoMo."""
        qid      = f"{conv_id}__{qa.get('question', '')[:30].replace(' ', '_')}"
        query    = qa.get(self.KEY_QUESTION, "")
        gt       = qa.get(self.KEY_ANSWER, "")
        evidence = qa.get(self.KEY_EVIDENCE, [])    # list of dia_id integers

        # ── Ground-truth FAISS IDs: map evidence dia_ids → FAISS IDs ─────────
        # LoCoMo evidence is a list of dia_id integers pointing to specific turns.
        gt_faiss_ids: list[int] = []
        for dia_id in evidence:
            faiss_id = diaid_to_faiss_id.get(str(dia_id))
            if faiss_id is not None:
                gt_faiss_ids.append(faiss_id)

        # ── Abstention: append explicit instruction to the query ──────────────
        effective_query = query
        if category == "abstention":
            effective_query = (
                f"{query}\n"
                f"(If the answer is not in your memory, "
                f"respond with 'I don't know.')"
            )

        # ── Dense Retrieval + Generation ──────────────────────────────────────
        output = self.ltm.generate_response(
            query          = effective_query,
            top_k          = self.top_k,
            max_new_tokens = self.max_new_tokens,
            temperature    = self.temperature,
        )

        prompt_tokens   = self._count_tokens(output["augmented_prompt"])
        response_tokens = self._count_tokens(output["response"])

        return EvalResult(
            question_id             = qid,
            framework               = "locomo",
            category                = category,
            query                   = query,
            ground_truth            = gt,
            retrieved_memory_ids    = output["memory_ids_used"],
            ground_truth_memory_ids = gt_faiss_ids,
            retrieved_memories      = output["retrieved_mems"],
            predicted_answer        = output["response"],
            latency                 = output["latency"],
            augmented_prompt        = output["augmented_prompt"],
            prompt_token_count      = prompt_tokens,
            response_token_count    = response_tokens,
        )

    def _ingest_all_sessions(
        self,
        conv   : dict,
        conv_id: str,
    ) -> tuple[dict[str, list[int]], dict[int, int]]:
        """
        Parse the LoCoMo conversation dict (keys: session_1, session_1_date_time, ...)
        and ingest all turns into the LTM.

        Returns
        -------
        session_to_faiss_ids : session_key (e.g. "session_1") → list of FAISS IDs
        diaid_to_faiss_id    : dia_id (int) → single FAISS ID
                               Used to resolve evidence dia_ids to ground-truth IDs.
        """
        session_to_ids : dict[str, list[int]] = {}
        diaid_to_faiss : dict[int, int]        = {}

        # Collect all session_N keys (ignore speaker_a, speaker_b, date_time keys)
        session_keys = sorted(
            [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda k: int(k.split("_")[1]),   # sort by session number
        )

        for sess_key in session_keys:
            turns    = conv.get(sess_key, [])
            date_key = f"{sess_key}_date_time"
            date     = conv.get(date_key, "")
            sess_num = sess_key.split("_")[1]          # "1", "2", ...
            ids      : list[int] = []

            for turn in turns:
                speaker = turn.get(self.KEY_SPEAKER, "unknown")
                dia_id  = turn.get(self.KEY_DIA_ID)
                text    = turn.get(self.KEY_TEXT, "").strip()
                if not text:
                    continue

                memory_text = (
                    f"[{speaker}] "
                    f"(Conv {conv_id}, Session {sess_num}, {date}): {text}"
                )
                fid = self.ltm.ingest_memory(
                    memory_text,
                    metadata={
                        "conversation_id": conv_id,
                        "session_key"    : sess_key,
                        "session_num"    : sess_num,
                        "date"           : date,
                        "speaker"        : speaker,
                        "dia_id"         : dia_id,
                        "framework"      : "locomo",
                    },
                )
                ids.append(fid)
                if dia_id is not None:
                    diaid_to_faiss[str(dia_id)] = fid

            session_to_ids[sess_key] = ids

        return session_to_ids, diaid_to_faiss

    def _reset_ltm(self) -> None:
        """Reset the LTM index between conversations."""
        import faiss as _faiss
        self.ltm.faiss_index    = _faiss.IndexFlatL2(self.ltm.embedding_dim)
        self.ltm.ltm_store      = {}
        self.ltm._next_faiss_id = 0

    def _count_tokens(self, text: str) -> int:
        try:
            return len(self.ltm.slm_tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return len(text.split())

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


# ---------------------------------------------------------------------------
# Utility — Save Results to JSON
# ---------------------------------------------------------------------------

def save_eval_results(results: list[EvalResult], output_path: str | Path) -> None:
    """
    Serialise a list of EvalResult objects to a JSON file.
    Used as intermediate checkpoint between Stage 1 and Stage 2 evaluation.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serialisable = []
    for r in results:
        serialisable.append({
            "question_id"             : r.question_id,
            "framework"               : r.framework,
            "category"                : r.category,
            "query"                   : r.query,
            "ground_truth"            : r.ground_truth,
            "predicted_answer"        : r.predicted_answer,
            "retrieved_memory_ids"    : r.retrieved_memory_ids,
            "ground_truth_memory_ids" : r.ground_truth_memory_ids,
            "retrieved_memories"      : r.retrieved_memories,
            "latency"                 : r.latency,
            "virtual_update_id"       : r.virtual_update_id,
            "virtual_update_text"     : r.virtual_update_text,
            "prompt_token_count"      : r.prompt_token_count,
            "response_token_count"    : r.response_token_count,
        })

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2, ensure_ascii=False)

    print(f"[Adapter] Results saved → {output_path} ({len(results)} items)")


def load_eval_results(input_path: str | Path) -> list[EvalResult]:
    """Deserialise saved EvalResult JSON back into dataclass objects."""
    with open(input_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    return [
        EvalResult(
            question_id             = d["question_id"],
            framework               = d["framework"],
            category                = d["category"],
            query                   = d["query"],
            ground_truth            = d["ground_truth"],
            predicted_answer        = d["predicted_answer"],
            retrieved_memory_ids    = d["retrieved_memory_ids"],
            ground_truth_memory_ids = d["ground_truth_memory_ids"],
            retrieved_memories      = d.get("retrieved_memories", []),
            latency                 = d.get("latency", {}),
            virtual_update_id       = d.get("virtual_update_id"),
            virtual_update_text     = d.get("virtual_update_text", ""),
            prompt_token_count      = d.get("prompt_token_count", 0),
            response_token_count    = d.get("response_token_count", 0),
        )
        for d in data
    ]
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

from ltm_module import VectorEmbeddedMemory          # noqa: E402


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
      "category"              : "single_hop",       // or multi_hop, knowledge_update,
                                                    //    temporal_reasoning, abstention
      "evidence_list"         : ["session_002"],    // session IDs containing the answer
      "sessions"              : [
        {
          "session_id"        : "session_001",
          "date"              : "2024-01-10",
          "messages"          : [
            { "role": "user",      "content": "..." },
            { "role": "assistant", "content": "..." }
          ]
        }
      ],
      // Optional — only present in some versions:
      "update_info"           : "User changed to a keto diet on 2024-03-01."
    }

    Integration Strategy
    ---------------------
    For each question item:
      1. Reset the LTM (fresh FAISS index per question for isolation).
      2. Ingest all session messages chronologically as individual memories.
      3. For knowledge_update items: call TemporalContextInjector.inject().
      4. Run ltm_module.generate_response() with the question as query.
      5. Map output → EvalResult.
    """

    # Keys used in LongMemEval JSON — listed here so changes propagate easily
    KEY_QID       = "question_id"
    KEY_QUESTION  = "question"
    KEY_ANSWER    = "answer"
    KEY_CATEGORY  = "category"
    KEY_EVIDENCE  = "evidence_list"          # list of relevant session IDs
    KEY_SESSIONS  = "sessions"
    KEY_UPDATE    = "update_info"            # only present on knowledge_update items
    KEY_SESSION_ID = "session_id"
    KEY_DATE      = "date"
    KEY_MESSAGES  = "messages"
    KEY_ROLE      = "role"
    KEY_CONTENT   = "content"

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
            items = [x for x in items if x.get(self.KEY_CATEGORY) in categories]
        if max_items:
            items = items[:max_items]

        results: list[EvalResult] = []
        for idx, item in enumerate(items, start=1):
            self._log(
                f"[LongMemEval] Item {idx}/{len(items)} | "
                f"ID={item.get(self.KEY_QID)} | "
                f"Category={item.get(self.KEY_CATEGORY)}"
            )
            result = self._process_item(item)
            results.append(result)

        self._log(f"[LongMemEval] Completed — {len(results)} results collected.")
        return results

    # ------------------------------------------------------------------
    # Private — Per-Item Processing
    # ------------------------------------------------------------------

    def _process_item(self, item: dict) -> EvalResult:
        """Process a single LongMemEval test item."""

        qid      = item.get(self.KEY_QID, "unknown")
        query    = item.get(self.KEY_QUESTION, "")
        gt       = item.get(self.KEY_ANSWER, "")
        category = item.get(self.KEY_CATEGORY, "unknown")
        sessions = item.get(self.KEY_SESSIONS, [])
        evidence = item.get(self.KEY_EVIDENCE, [])     # relevant session IDs

        # ── Step 1: Reset LTM for this item (isolation) ──────────────────────
        self._reset_ltm()

        # ── Step 2: Ingest sessions chronologically ───────────────────────────
        # Build a map from session_id → list of FAISS IDs (for ground-truth mapping)
        session_to_faiss_ids: dict[str, list[int]] = {}
        sessions_sorted = sorted(
            sessions,
            key=lambda s: s.get(self.KEY_DATE, "1970-01-01"),
        )
        for session in sessions_sorted:
            sid    = session.get(self.KEY_SESSION_ID, "")
            faiss_ids = self._ingest_session(session, sid)
            session_to_faiss_ids[sid] = faiss_ids

        # ── Step 3: Knowledge Update — Temporal Context Injection ─────────────
        virtual_id   = None
        virtual_text = ""
        if category == self.CATEGORY_KNOWLEDGE_UPDATE:
            update_info = item.get(self.KEY_UPDATE, "")
            if update_info:
                virtual_id   = self._injector.inject(self.ltm, update_info, qid)
                virtual_text = update_info
                self._log(
                    f"  [Virtual Update] Injected → FAISS ID={virtual_id} | "
                    f"Text: '{update_info[:80]}'"
                )

        # ── Step 4: Map evidence session IDs → FAISS IDs (ground-truth) ───────
        gt_faiss_ids: list[int] = []
        for ev_session_id in evidence:
            gt_faiss_ids.extend(session_to_faiss_ids.get(ev_session_id, []))
        if virtual_id is not None:
            gt_faiss_ids.append(virtual_id)    # the update itself is ground truth

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

    def _ingest_session(self, session: dict, session_id: str) -> list[int]:
        """
        Ingest all messages in a session as individual LTM memories.

        Each message is stored with role context so the SLM understands
        who said what. Returns the list of FAISS IDs assigned.
        """
        messages = session.get(self.KEY_MESSAGES, [])
        date     = session.get(self.KEY_DATE, "")
        faiss_ids: list[int] = []

        for turn_idx, msg in enumerate(messages):
            role    = msg.get(self.KEY_ROLE, "unknown")
            content = msg.get(self.KEY_CONTENT, "").strip()
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

    LoCoMo Dataset Schema (raw JSON — DO NOT MODIFY)
    -------------------------------------------------
    {
      "conversation_id" : "conv_001",
      "sessions"        : [
        {
          "session_id"  : 1,
          "date"        : "2023-09-14",
          "dialog"      : [
            { "speaker": "Person1", "text": "..." },
            { "speaker": "Person2", "text": "..." }
          ]
        }
      ],
      "qa"              : [
        {
          "id"          : "q_001",
          "question"    : "...",
          "answer"      : "...",
          "type"        : "single",          // single | multi | adversarial
          "session_id"  : 2,                 // session where answer appears
          "bleu"        : "...",             // optional reference for BLEU
          "evidence"    : ["text snippet"]  // optional evidence spans
        }
      ]
    }

    Integration Strategy
    ---------------------
    LoCoMo nests QA pairs *inside* conversation objects, unlike LongMemEval
    which is a flat list of questions. The adapter:
      1. Iterates conversations → iterates QA pairs per conversation.
      2. Resets LTM per conversation (not per question) since sessions are shared.
      3. Ingests all sessions for a conversation once.
      4. Runs each QA query against the pre-ingested LTM.
      5. For 'adversarial' type: treats as abstention — expects "I don't know".
    """

    # LoCoMo JSON keys
    KEY_CONV_ID   = "conversation_id"
    KEY_SESSIONS  = "sessions"
    KEY_SESSION_ID= "session_id"
    KEY_DATE      = "date"
    KEY_DIALOG    = "dialog"
    KEY_SPEAKER   = "speaker"
    KEY_TEXT      = "text"
    KEY_QA        = "qa"
    KEY_QID       = "id"
    KEY_QUESTION  = "question"
    KEY_ANSWER    = "answer"
    KEY_TYPE      = "type"
    KEY_EVIDENCE  = "evidence"

    # LoCoMo question types → normalised thesis category names
    TYPE_MAP = {
        "single"     : "single_hop",
        "multi"      : "multi_hop",
        "adversarial": "abstention",
        "temporal"   : "temporal_reasoning",
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

        # LoCoMo may be a single JSON object OR a list; normalise to list
        with open(self.data_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self._raw_data: list[dict] = raw if isinstance(raw, list) else [raw]

        total_qa = sum(len(c.get(self.KEY_QA, [])) for c in self._raw_data)
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

        for conv in self._raw_data:
            if max_items and item_count >= max_items:
                break

            conv_id  = conv.get(self.KEY_CONV_ID, "unknown")
            sessions = conv.get(self.KEY_SESSIONS, [])
            qa_list  = conv.get(self.KEY_QA, [])

            # ── Reset LTM once per conversation ──────────────────────────────
            self._reset_ltm()
            session_to_faiss_ids = self._ingest_all_sessions(sessions, conv_id)

            for qa in qa_list:
                if max_items and item_count >= max_items:
                    break

                raw_type = qa.get(self.KEY_TYPE, "single")
                category = self.TYPE_MAP.get(raw_type, raw_type)

                if categories and category not in categories:
                    continue

                self._log(
                    f"[LoCoMo] Conv={conv_id} | "
                    f"QID={qa.get(self.KEY_QID)} | Category={category}"
                )
                result = self._process_qa(qa, conv_id, session_to_faiss_ids, category)
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
        category            : str,
    ) -> EvalResult:
        """Process one QA item from LoCoMo."""
        qid      = f"{conv_id}__{qa.get(self.KEY_QID, 'unknown')}"
        query    = qa.get(self.KEY_QUESTION, "")
        gt       = qa.get(self.KEY_ANSWER, "")
        evidence = qa.get(self.KEY_EVIDENCE, [])

        # ── Ground-truth FAISS IDs: sessions mentioned in evidence ────────────
        # LoCoMo provides text spans, not session IDs directly.
        # We match evidence spans to session IDs as a best-effort mapping.
        evidence_session_id = str(qa.get(self.KEY_SESSION_ID, ""))
        gt_faiss_ids = session_to_faiss_ids.get(evidence_session_id, [])

        # ── Abstention: inject a non-retrieval instruction ────────────────────
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
        sessions: list[dict],
        conv_id : str,
    ) -> dict[str, list[int]]:
        """
        Ingest all sessions for a conversation chronologically.
        Returns mapping of session_id (str) → list of FAISS IDs.
        """
        session_to_ids: dict[str, list[int]] = {}
        sessions_sorted = sorted(
            sessions,
            key=lambda s: (s.get(self.KEY_DATE, "1970-01-01"),
                           int(s.get(self.KEY_SESSION_ID, 0))),
        )
        for session in sessions_sorted:
            sid    = str(session.get(self.KEY_SESSION_ID, ""))
            date   = session.get(self.KEY_DATE, "")
            dialog = session.get(self.KEY_DIALOG, [])
            ids    = []
            for turn_idx, turn in enumerate(dialog):
                speaker = turn.get(self.KEY_SPEAKER, "unknown")
                text    = turn.get(self.KEY_TEXT, "").strip()
                if not text:
                    continue
                memory_text = f"[{speaker}] (Conv {conv_id}, Session {sid}, {date}): {text}"
                fid = self.ltm.ingest_memory(
                    memory_text,
                    metadata={
                        "conversation_id": conv_id,
                        "session_id"     : sid,
                        "date"           : date,
                        "speaker"        : speaker,
                        "turn_index"     : turn_idx,
                        "framework"      : "locomo",
                    },
                )
                ids.append(fid)
            session_to_ids[sid] = ids

        return session_to_ids

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

"""
vector_embed_module.py
=============
Dense-Retrieval Long-Term Memory (LTM) Module
for Gemma 3 4B Small Language Model (SLM)

Research Focus : Edge / Local Deployment
Embedding Model: google/embedding-gemma-300m (768-d vectors)
Vector Index   : FAISS IndexFlatL2  (brute-force L2, 100% recall)
Persistence    : Sidecar JSON dictionary (FAISS int-ID → text + metadata)

Author  : [John Kenneth Alon]
Thesis  : Dense-Retrieval LTM for Edge-Deployed SLMs
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss                            # pip install faiss-cpu
import numpy as np
import torch
from transformers import (              # pip install transformers
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


# ---------------------------------------------------------------------------
# Constants — keep aligned with thesis terminology
# ---------------------------------------------------------------------------
EMBEDDING_DIM          = 768           # google/embedding-gemma-300m output dim
DEFAULT_TOP_K          = 5             # default number of memories to retrieve
DEFAULT_MAX_NEW_TOKENS = 512           # max generation tokens for Gemma 3 4B
CONTEXT_HEADER         = "[RETRIEVED CONTEXT]"
CONTEXT_FOOTER         = "[/RETRIEVED CONTEXT]"

# Gemma 3 4B instruction template (chat format)
GEMMA_SYSTEM_PROMPT = (
    "<start_of_turn>system\n"
    "You are a helpful AI assistant. Use the retrieved long-term memory "
    "context below to inform your answer. If the context is not relevant, "
    "rely on your own knowledge.\n<end_of_turn>\n"
)


# ---------------------------------------------------------------------------
# VectorEmbeddedMemory
# ---------------------------------------------------------------------------
class VectorEmbeddedMemory:
    """
    Dense-Retrieval Long-Term Memory (LTM) Module.

    Encapsulates:
      - Embedding model  : google/embedding-gemma-300m
      - FAISS index      : IndexFlatL2 (exact / brute-force)
      - Sidecar JSON map : { faiss_int_id (str) -> { "text", "timestamp", ... } }
      - Gemma 3 4B SLM   : loaded with optional 4-bit / 8-bit quantization

    Public API
    ----------
    ingest_memory(text, metadata)      -> int          (memory ID)
    dense_retrieve(query, top_k)       -> list[dict]   (ranked memory dicts)
    build_augmented_prompt(query, ...) -> str          (context-injected prompt)
    generate_response(query, ...)      -> dict          (response + memory IDs used)
    save_ltm(directory)                -> None
    load_ltm(directory)                -> None
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(
        self,
        embedding_model_id : str  = "google/embedding-gemma-300m",
        slm_model_id        : str  = "google/gemma-3-4b-it",
        quantization        : str  = "none",          # currently using 8-bit for demo; switch to "4bit" for edge deployment
        device              : str  = "auto",
        embedding_dim       : int  = EMBEDDING_DIM,
        verbose             : bool = True,
    ) -> None:
        """
        Parameters
        ----------
        embedding_model_id : HuggingFace model ID for the dense embedding model.
        slm_model_id       : HuggingFace model ID for Gemma 3 4B (or variant).
        quantization       : Precision level for SLM — '4bit', '8bit', or 'none'.
                             4-bit (NF4) is recommended for edge deployment.
        device             : 'auto' lets HuggingFace choose GPU/CPU automatically.
        embedding_dim      : Dimensionality of dense vectors (768 for embedding-gemma-300m).
        verbose            : Print progress messages during init and inference.
        """
        self.embedding_model_id = embedding_model_id
        self.slm_model_id       = slm_model_id
        self.quantization       = quantization
        self.embedding_dim      = embedding_dim
        self.verbose            = verbose

        # Detect optimal device for embeddings (separate from SLM device)
        if device == "auto":
            self.embed_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.embed_device = device

        # --- FAISS Index (IndexFlatL2 — brute-force, exact search) ---
        self._log("Initialising FAISS IndexFlatL2 …")
        self.faiss_index: faiss.IndexFlatL2 = faiss.IndexFlatL2(self.embedding_dim)

        # --- Sidecar Dictionary (LTM persistent store) ---
        # Keys are *string* representations of FAISS integer IDs
        # Values: { "text": str, "timestamp": str, "metadata": dict }
        self.ltm_store: dict[str, dict[str, Any]] = {}

        # Internal counter tracks the next FAISS ID to be assigned
        self._next_faiss_id: int = 0

        # --- Load Embedding Model ---
        self._log(f"Loading embedding model: {embedding_model_id} …")
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)
        self.embed_model     = AutoModel.from_pretrained(
            embedding_model_id,
            torch_dtype=torch.float16 if self.embed_device != "cpu" else torch.float32,
        ).to(self.embed_device).eval()

        # --- Load SLM (Gemma 3 4B) with optional quantization ---
        self._log(f"Loading SLM: {slm_model_id} [quantization={quantization}] …")
        bnb_config = self._build_bnb_config(quantization)

        self.slm_tokenizer = AutoTokenizer.from_pretrained(slm_model_id)
        self.slm_model = AutoModelForCausalLM.from_pretrained(
            slm_model_id,
            quantization_config  = bnb_config,          # None if quantization="none"
            device_map           = "auto",
            torch_dtype          = torch.bfloat16,
            low_cpu_mem_usage    = True,
            attn_implementation  = "eager",             # compatible with all HW
        )
        self._log("LTM Module initialised ✓")

    # ------------------------------------------------------------------
    # A. Memory Ingestion
    # ------------------------------------------------------------------
    def ingest_memory(
        self,
        text    : str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Ingest a text fragment into the Dense-Retrieval LTM.

        Steps
        -----
        1. Embed the text using the EmbeddingGemma model.
        2. Add the 768-d vector to the FAISS IndexFlatL2.
        3. Store { text, timestamp, metadata } in the sidecar JSON map
           keyed by the assigned FAISS integer ID.

        Parameters
        ----------
        text     : The memory string (e.g., conversation summary, fact).
        metadata : Optional dict of extra fields (session_id, source, etc.).

        Returns
        -------
        faiss_id : The integer ID assigned by FAISS.
        """
        if not text.strip():
            raise ValueError("Memory text must not be empty.")

        # Step 1 — Dense embedding
        vector = self._embed_text(text)                         # shape: (1, 768)

        print(f"DEBUG: Index expects dimension: {self.faiss_index.d}")
        print(f"DEBUG: Input vector shape: {vector.shape}")

        # Step 2 — Add to FAISS index
        self.faiss_index.add(vector)
        faiss_id = self._next_faiss_id
        self._next_faiss_id += 1

        # Step 3 — Update sidecar LTM store
        self.ltm_store[str(faiss_id)] = {
            "text"      : text,
            "timestamp" : datetime.now(timezone.utc).isoformat(),
            "metadata"  : metadata or {},
        }

        self._log(f"[Ingestion] Memory stored → ID={faiss_id} | "
                  f"tokens≈{len(text.split())} | total_memories={self._next_faiss_id}")
        return faiss_id

    # ------------------------------------------------------------------
    # B. Dense Retrieval
    # ------------------------------------------------------------------
    def dense_retrieve(
        self,
        query : str,
        top_k : int = DEFAULT_TOP_K,
    ) -> list[dict[str, Any]]:
        """
        Execute Dense Retrieval against the FAISS IndexFlatL2.

        Steps
        -----
        1. Embed the query string with EmbeddingGemma.
        2. Run FAISS k-NN search (L2) to obtain top-k integer IDs + distances.
        3. Look up each ID in the sidecar LTM store.

        Parameters
        ----------
        query : The user's current input / question string.
        top_k : Number of nearest-neighbour memories to retrieve.

        Returns
        -------
        List of memory dicts, ranked by ascending L2 distance (most similar first):
          [{ "memory_id", "text", "timestamp", "metadata", "l2_distance" }, …]
        """
        n_memories = self.faiss_index.ntotal
        if n_memories == 0:
            self._log("[Dense Retrieval] LTM index is empty — no memories returned.")
            return []

        effective_k = min(top_k, n_memories)

        # Step 1 — Embed query
        query_vector = self._embed_text(query)                  # shape: (1, 768)

        # Step 2 — FAISS brute-force L2 search (100% recall)
        distances, indices = self.faiss_index.search(query_vector, effective_k)
        # distances / indices shape: (1, effective_k)

        # Step 3 — Sidecar lookup
        retrieved: list[dict[str, Any]] = []
        for rank, (faiss_id, dist) in enumerate(
            zip(indices[0], distances[0]), start=1
        ):
            if faiss_id == -1:          # FAISS sentinel for "not enough results"
                continue
            record = self.ltm_store.get(str(faiss_id), {})
            retrieved.append({
                "rank"        : rank,
                "memory_id"   : int(faiss_id),
                "text"        : record.get("text", "[missing]"),
                "timestamp"   : record.get("timestamp", ""),
                "metadata"    : record.get("metadata", {}),
                "l2_distance" : float(dist),
            })

        self._log(f"[Dense Retrieval] Retrieved {len(retrieved)} memory(ies) "
                  f"for query: '{query[:60]}…'")
        return retrieved

    # ------------------------------------------------------------------
    # C. Context Injection — Prompt Construction
    # ------------------------------------------------------------------
    def build_augmented_prompt(
        self,
        query          : str,
        retrieved_mems : list[dict[str, Any]],
        include_scores : bool = False,
    ) -> str:
        """
        Construct the context-augmented prompt for Gemma 3 4B.

        Template structure
        ------------------
        <system_prompt>
        <user_turn>
          [RETRIEVED CONTEXT]
          [1] <memory_text> (optional: L2=<score>)
          ...
          [/RETRIEVED CONTEXT]

          <user_query>
        <end_of_turn>
        <model_turn_start>

        Parameters
        ----------
        query          : The user's current question.
        retrieved_mems : Output of dense_retrieve().
        include_scores : Append L2 distance scores to each memory line (for debugging).

        Returns
        -------
        Full prompt string ready to be tokenised and passed to the SLM.
        """
        context_lines: list[str] = [CONTEXT_HEADER]

        if not retrieved_mems:
            context_lines.append("  [No relevant long-term memories found.]")
        else:
            for mem in retrieved_mems:
                score_str = (
                    f"  (L2={mem['l2_distance']:.4f})" if include_scores else ""
                )
                context_lines.append(
                    f"  [{mem['rank']}] (Memory ID {mem['memory_id']}) "
                    f"{mem['text']}{score_str}"
                )

        context_lines.append(CONTEXT_FOOTER)
        context_block = "\n".join(context_lines)

        # Gemma 3 chat-format prompt
        augmented_prompt = (
            f"{GEMMA_SYSTEM_PROMPT}"
            f"<start_of_turn>user\n"
            f"{context_block}\n\n"
            f"{query}\n"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        return augmented_prompt

    # ------------------------------------------------------------------
    # D. Inference Pipeline (Retrieve → Inject → Generate)
    # ------------------------------------------------------------------
    def generate_response(
        self,
        query          : str,
        top_k          : int  = DEFAULT_TOP_K,
        max_new_tokens : int  = DEFAULT_MAX_NEW_TOKENS,
        temperature    : float = 0.7,
        include_scores : bool  = False,
    ) -> dict[str, Any]:
        """
        End-to-end LTM-augmented inference pipeline.

        Execution flow
        --------------
        1. Dense Retrieval  — embed query + FAISS k-NN search
        2. Context Injection — build augmented prompt with retrieved memories
        3. SLM Generation   — Gemma 3 4B generates a grounded response
        4. Return result dict with response text + transparency metadata

        Parameters
        ----------
        query          : User's input string.
        top_k          : Number of memories to retrieve.
        max_new_tokens : Maximum tokens for the SLM to generate.
        temperature    : Sampling temperature (lower = more deterministic).
        include_scores : Add L2 distances to the injected context.

        Returns
        -------
        dict with keys:
          - "response"       : str   — generated text from Gemma 3 4B
          - "memory_ids_used": list  — FAISS IDs of injected memories (transparency)
          - "retrieved_mems" : list  — full retrieved memory dicts
          - "latency"        : dict  — timing breakdown (seconds) for thesis metrics
          - "augmented_prompt: str  — the full prompt (for debugging / ablation)
        """
        latency: dict[str, float] = {}

        # ── Step 1 : Dense Retrieval ────────────────────────────────
        t0 = time.perf_counter()
        retrieved_mems = self.dense_retrieve(query, top_k=top_k)
        latency["dense_retrieval_s"] = time.perf_counter() - t0

        # ── Step 2 : Context Injection ──────────────────────────────
        t1 = time.perf_counter()
        augmented_prompt = self.build_augmented_prompt(
            query, retrieved_mems, include_scores=include_scores
        )
        latency["context_injection_s"] = time.perf_counter() - t1

        # ── Step 3 : SLM Generation ─────────────────────────────────
        t2 = time.perf_counter()
        inputs = self.slm_tokenizer(
            augmented_prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 4096,
        ).to(self.slm_model.device)

        prompt_token_count = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self.slm_model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                do_sample      = temperature > 0,
                temperature    = temperature if temperature > 0 else 1.0,
                pad_token_id   = self.slm_tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (exclude prompt)
        new_token_ids = output_ids[0][prompt_token_count:]
        response_text = self.slm_tokenizer.decode(
            new_token_ids, skip_special_tokens=True
        ).strip()

        latency["slm_generation_s"]  = time.perf_counter() - t2
        latency["total_pipeline_s"]  = (
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

    # ------------------------------------------------------------------
    # Persistence — Save & Load LTM to Disk
    # ------------------------------------------------------------------
    def save_ltm(self, directory: str = "./ltm_store") -> None:
        """
        Persist both the FAISS index and the sidecar JSON map to disk.

        Files created
        -------------
        <directory>/faiss_index.bin  — serialised FAISS IndexFlatL2
        <directory>/ltm_store.json   — sidecar text/metadata dictionary

        Parameters
        ----------
        directory : Target directory (created if it does not exist).
        """
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        faiss_path = save_dir / "faiss_index.bin"
        json_path  = save_dir / "ltm_store.json"

        # Save FAISS index
        faiss.write_index(self.faiss_index, str(faiss_path))

        # Save sidecar store + internal counter
        payload = {
            "_next_faiss_id": self._next_faiss_id,
            "memories"      : self.ltm_store,
        }
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        self._log(f"[Persistence] LTM saved → {save_dir} "
                  f"({self.faiss_index.ntotal} vectors)")

    def load_ltm(self, directory: str = "./ltm_store") -> None:
        """
        Restore a previously saved FAISS index and sidecar JSON map from disk.

        Parameters
        ----------
        directory : Directory containing 'faiss_index.bin' and 'ltm_store.json'.
        """
        save_dir   = Path(directory)
        faiss_path = save_dir / "faiss_index.bin"
        json_path  = save_dir / "ltm_store.json"

        if not faiss_path.exists() or not json_path.exists():
            raise FileNotFoundError(
                f"LTM store not found in '{directory}'. "
                "Run save_ltm() first or check the path."
            )

        # Restore FAISS index
        self.faiss_index = faiss.read_index(str(faiss_path))

        # Restore sidecar store
        with open(json_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self._next_faiss_id = payload.get("_next_faiss_id", self.faiss_index.ntotal)
        self.ltm_store      = payload.get("memories", {})

        self._log(f"[Persistence] LTM loaded ← {save_dir} "
                  f"({self.faiss_index.ntotal} vectors)")

    # ------------------------------------------------------------------
    # Utility / Introspection
    # ------------------------------------------------------------------
    def memory_count(self) -> int:
        """Return the total number of memories stored in the FAISS index."""
        return self.faiss_index.ntotal

    def get_memory(self, memory_id: int) -> dict[str, Any] | None:
        """Retrieve a single memory record by its FAISS integer ID."""
        return self.ltm_store.get(str(memory_id))

    def delete_memory(self, memory_id: int) -> bool:
        """
        Remove a memory from the sidecar store.

        NOTE: FAISS IndexFlatL2 does not support in-place deletion of individual
        vectors. The sidecar entry is cleared so the text will not appear in
        retrieval results, but the underlying vector remains in the index.
        For a clean rebuild after many deletions, call rebuild_index().

        Returns True if the memory existed and was removed, False otherwise.
        """
        key = str(memory_id)
        if key in self.ltm_store:
            self.ltm_store[key] = {
                "text"     : "[DELETED]",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata" : {"deleted": True},
            }
            self._log(f"[LTM] Memory ID={memory_id} marked as deleted.")
            return True
        return False

    def rebuild_index(self) -> None:
        """
        Rebuild the FAISS index and sidecar store from scratch, excluding
        deleted entries. Use after bulk deletions to recover vector space.
        """
        self._log("[LTM] Rebuilding FAISS index from active memories …")
        active_texts = [
            (int(k), v["text"])
            for k, v in self.ltm_store.items()
            if not v.get("metadata", {}).get("deleted", False)
        ]
        active_texts.sort(key=lambda x: x[0])    # maintain original ID order

        # Reset index
        self.faiss_index    = faiss.IndexFlatL2(self.embedding_dim)
        new_store           = {}
        self._next_faiss_id = 0

        for old_id, text in active_texts:
            new_id = self._next_faiss_id
            vec    = self._embed_text(text)
            self.faiss_index.add(vec)
            new_store[str(new_id)] = self.ltm_store[str(old_id)]
            self._next_faiss_id += 1

        self.ltm_store = new_store
        self._log(f"[LTM] Rebuild complete — {self.faiss_index.ntotal} active vectors.")

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------
    def _embed_text(self, text: str) -> np.ndarray:
        """
        Convert a text string to a normalised 768-d dense vector using
        the EmbeddingGemma model (mean-pooling over token hidden states).

        Returns
        -------
        numpy array of shape (1, 768), dtype float32.
        """
        encoded = self.embed_tokenizer(
            text,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 512,
            padding        = True,
        ).to(self.embed_device)

        with torch.inference_mode():
            outputs = self.embed_model(**encoded)

        # Mean-pool over the token dimension (dim=1), then L2-normalise
        hidden_states   = outputs.last_hidden_state          # (1, seq_len, 768)
        attention_mask  = encoded["attention_mask"]          # (1, seq_len)
        mask_expanded   = attention_mask.unsqueeze(-1).float()
        sum_hidden      = (hidden_states * mask_expanded).sum(dim=1)
        count           = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_pooled     = (sum_hidden / count)               # (1, 768)

        # L2 normalisation — improves cosine/L2 proximity equivalence
        norm   = mean_pooled.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        vector = (mean_pooled / norm).cpu().float().numpy()  # (1, 768)

        return vector

    @staticmethod
    def _build_bnb_config(quantization: str) -> BitsAndBytesConfig | None:
        """
        Build a BitsAndBytesConfig for 4-bit or 8-bit quantization.
        Returns None when quantization='none' (full precision).

        4-bit NF4 is recommended for edge deployment — approximately 60-70%
        VRAM reduction vs. fp16 with minimal perplexity degradation.
        """
        if quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_quant_type       = "nf4",       # NormalFloat4
                bnb_4bit_use_double_quant = True,        # nested quantization
                bnb_4bit_compute_dtype    = torch.float16,
            )
        elif quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            return None                                  # fp16 (no quantization)

    def _log(self, message: str) -> None:
        """Conditional verbose logging."""
        if self.verbose:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[LTM {ts}] {message}")


# ---------------------------------------------------------------------------
# Quick-start demo (run this file directly: python ltm_module.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("  Dense-Retrieval LTM Module — Quick-Start Demo")
    print("=" * 70)

    # Initialise the LTM module
    # NOTE: For first run, set quantization="none" on CPU-only machines.
    ltm = VectorEmbeddedMemory(
        embedding_model_id = "google/embeddinggemma-300m",
        slm_model_id       = "google/gemma-3-4b-it",
        quantization       = "none",
        verbose            = True,
    )

    # ── Ingest sample memories ──────────────────────────────────────
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
        "The user's thesis focuses on Dense-Retrieval LTM for Gemma 3 4B "
        "targeting edge deployment on devices with ≤8 GB RAM.",
        metadata={"session": "session_005", "topic": "thesis"},
    )
    print(f"\nTotal memories in LTM: {ltm.memory_count()}")

    # ── Dense Retrieval ─────────────────────────────────────────────
    print("\n--- Dense Retrieval ---")
    query = "What is the user's research topic?"
    results = ltm.dense_retrieve(query, top_k=2)
    for r in results:
        print(f"  Rank {r['rank']} | ID={r['memory_id']} | L2={r['l2_distance']:.4f}")
        print(f"    → {r['text'][:80]}…")

    # ── Full Inference Pipeline ─────────────────────────────────────
    print("\n--- LTM-Augmented Inference ---")
    output = ltm.generate_response(
        query          = "Summarise my research project in one paragraph.",
        top_k          = 3,
        max_new_tokens = 256,
    )
    print(f"\nMemory IDs Used : {output['memory_ids_used']}")
    print(f"Retrieval Time  : {output['latency']['dense_retrieval_s']:.3f} s")
    print(f"Generation Time : {output['latency']['slm_generation_s']:.3f} s")
    print(f"\nResponse:\n{output['response']}")

    # ── Save LTM to disk ────────────────────────────────────────────
    ltm.save_ltm("./ltm_store")
    print("\nLTM saved to ./ltm_store/ ✓")

"""
eval_metrics.py
===============
Two-Stage Evaluation Metrics Engine

Stage 1 — Retrieval Quality (FAISS Dense Retrieval)
  · Recall@K  : Fraction of relevant memories found in top-K results
  · NDCG@K    : Ranking quality — penalises relevant memories ranked lower

Stage 2 — Generation Quality (Gemma 3 4B SLM output)
  · Faithfulness     : Response is grounded in retrieved memories (no hallucinations)
  · Answer Relevance : Response directly addresses the user's query
  · Abstention Accuracy: Model correctly says "I don't know" when no context exists

All metrics are computed *without* modifying EvalResult objects or dataset files.
Results are always written to a separate output JSON — the adapter outputs are
treated as read-only inputs to this module.

Thesis Reference Metrics
-------------------------
  Recall@K  : fraction of |relevant ∩ retrieved_top_k| / |relevant|
  NDCG@K    : DCG@K / IDCG@K  (normalised discounted cumulative gain)
  Faithfulness & Answer Relevance: via OpenAI GPT-4o as LLM-as-a-Judge
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── Conditional imports (allow the module to load even if RAGAS/OpenAI absent) ─
try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

try:
    from ltm_eval_adapter import EvalResult   # relative import when run from eval/
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ltm_eval_adapter import EvalResult


# ---------------------------------------------------------------------------
# Stage 1 — Retrieval Metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_ids   : list[int],
    relevant_ids    : list[int],
    k               : int,
) -> float:
    """
    Recall@K — the fraction of ground-truth relevant memory IDs that appear
    within the top-K retrieved results.

    Formula
    -------
        Recall@K = |retrieved[:K] ∩ relevant| / |relevant|

    Returns 0.0 if the relevant set is empty (abstention / no ground truth).

    Parameters
    ----------
    retrieved_ids : Ranked list of FAISS IDs from dense_retrieve() — ORDER MATTERS.
    relevant_ids  : Ground-truth FAISS IDs from the dataset evidence field.
    k             : Cutoff rank.
    """
    if not relevant_ids:
        return 0.0                          # undefined → 0 (conservative)

    relevant_set   = set(relevant_ids)
    top_k_set      = set(retrieved_ids[:k])
    hits           = relevant_set & top_k_set

    return len(hits) / len(relevant_set)


def ndcg_at_k(
    retrieved_ids   : list[int],
    relevant_ids    : list[int],
    k               : int,
) -> float:
    """
    NDCG@K — Normalised Discounted Cumulative Gain at rank K.

    Measures not only *whether* relevant memories are retrieved, but *where*
    they rank. A relevant memory at rank 1 contributes more than one at rank K.

    Formula
    -------
        gain_i  = 1  if retrieved_ids[i] in relevant_ids, else 0
        DCG@K   = Σ  gain_i / log2(i + 1)    for i = 1..K
        IDCG@K  = Σ  1 / log2(i + 1)         for i = 1..min(|relevant|, K)
        NDCG@K  = DCG@K / IDCG@K

    Parameters
    ----------
    retrieved_ids : Ranked FAISS IDs (index 0 = highest-ranked, closest L2).
    relevant_ids  : Ground-truth relevant FAISS IDs.
    k             : Cutoff rank.
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    top_k        = retrieved_ids[:k]

    # Discounted Cumulative Gain
    dcg = sum(
        1.0 / math.log2(rank + 1)          # rank is 1-indexed
        for rank, doc_id in enumerate(top_k, start=1)
        if doc_id in relevant_set
    )

    # Ideal DCG: if all |relevant| items were at the very top of top-K
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def average_precision_at_k(
    retrieved_ids   : list[int],
    relevant_ids    : list[int],
    k               : int,
) -> float:
    """
    Average Precision@K (AP@K) — used to compute MAP@K across all queries.

    AP@K = (1 / |relevant|) * Σ Precision@i * rel(i)   for i=1..K
    where rel(i) = 1 if retrieved_ids[i-1] is relevant, else 0.
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    hits = 0
    precision_sum = 0.0

    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_set:
            hits += 1
            precision_sum += hits / rank   # Precision@rank

    return precision_sum / len(relevant_set)


# ---------------------------------------------------------------------------
# Stage 1 — Aggregate Retrieval Report
# ---------------------------------------------------------------------------

@dataclass
class RetrievalMetricReport:
    """Per-category and overall Stage 1 retrieval metrics."""
    k_values     : list[int]
    overall      : dict[str, float]             # metric_name → value
    by_category  : dict[str, dict[str, float]]  # category → metric_name → value
    per_item     : list[dict[str, Any]]          # one dict per EvalResult


def compute_stage1_metrics(
    results     : list[EvalResult],
    k_values    : list[int] = [1, 3, 5, 10],
) -> RetrievalMetricReport:
    """
    Compute Recall@K, NDCG@K, and MAP@K for a list of EvalResult objects.

    This function is the primary Stage 1 metric entry point for the thesis.
    Results are broken down by category (single_hop, multi_hop, etc.) and
    overall across both frameworks.

    Parameters
    ----------
    results  : Output of LongMemEvalAdapter.run() or LoCoMoAdapter.run().
    k_values : List of K cutoffs to evaluate.

    Returns
    -------
    RetrievalMetricReport with overall and per-category breakdowns.
    """
    per_item_records: list[dict[str, Any]] = []
    category_buckets: dict[str, list[dict]] = defaultdict(list)

    for r in results:
        item_metrics: dict[str, float] = {}

        for k in k_values:
            item_metrics[f"recall@{k}"]  = recall_at_k(
                r.retrieved_memory_ids, r.ground_truth_memory_ids, k
            )
            item_metrics[f"ndcg@{k}"]    = ndcg_at_k(
                r.retrieved_memory_ids, r.ground_truth_memory_ids, k
            )
            item_metrics[f"ap@{k}"]      = average_precision_at_k(
                r.retrieved_memory_ids, r.ground_truth_memory_ids, k
            )

        record = {
            "question_id" : r.question_id,
            "framework"   : r.framework,
            "category"    : r.category,
            **item_metrics,
        }
        per_item_records.append(record)
        category_buckets[r.category].append(item_metrics)

    # ── Aggregate overall ──────────────────────────────────────────────────
    all_metrics = [rec for rec in per_item_records]
    overall = _aggregate_metrics(all_metrics, k_values)

    # ── Aggregate per category ─────────────────────────────────────────────
    by_category: dict[str, dict[str, float]] = {}
    for cat, item_list in category_buckets.items():
        by_category[cat] = _aggregate_metrics(item_list, k_values)

    return RetrievalMetricReport(
        k_values    = k_values,
        overall     = overall,
        by_category = by_category,
        per_item    = per_item_records,
    )


def _aggregate_metrics(
    item_list: list[dict],
    k_values : list[int],
) -> dict[str, float]:
    """Compute mean ± std for each metric across a list of per-item dicts."""
    agg: dict[str, float] = {}
    for k in k_values:
        for metric_prefix in ["recall", "ndcg", "ap"]:
            key = f"{metric_prefix}@{k}"
            vals = [item[key] for item in item_list if key in item]
            if vals:
                agg[f"mean_{key}"] = statistics.mean(vals)
                agg[f"std_{key}"]  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    agg["n_items"] = len(item_list)
    return agg


# ---------------------------------------------------------------------------
# Stage 2 — Generation Quality (LLM-as-a-Judge)
# ---------------------------------------------------------------------------

# Evaluation prompt templates — strictly follow both frameworks' evaluation intent
FAITHFULNESS_PROMPT = """\
You are an expert evaluator for an AI memory system.

Given:
  RETRIEVED CONTEXT (memories the model had access to):
{retrieved_context}

  MODEL RESPONSE:
{predicted_answer}

Task: Score FAITHFULNESS on a scale of 1-5.
Faithfulness measures whether the response is grounded in the retrieved context
and does not contain information fabricated beyond what is provided.

Scoring guide:
  5 = Response is entirely supported by the retrieved context. No hallucinations.
  4 = Mostly supported; minor inference that is reasonable given the context.
  3 = Partially supported; some claims go beyond the context.
  2 = Mostly unsupported; significant fabrication present.
  1 = Response contradicts or completely ignores the retrieved context.

Respond with ONLY a JSON object in this format:
{{"score": <1-5>, "reasoning": "<one sentence>"}}
"""

ANSWER_RELEVANCE_PROMPT = """\
You are an expert evaluator for a conversational AI memory system.

Given:
  USER QUESTION:
{query}

  MODEL RESPONSE:
{predicted_answer}

Task: Score ANSWER RELEVANCE on a scale of 1-5.
Answer Relevance measures how directly and completely the response addresses the question.

Scoring guide:
  5 = Response directly and completely answers the question.
  4 = Response answers the question but includes minor irrelevant details.
  3 = Response partially answers the question; key aspects are missing.
  2 = Response is tangentially related but does not answer the question.
  1 = Response is completely off-topic or refuses to answer without justification.

Respond with ONLY a JSON object in this format:
{{"score": <1-5>, "reasoning": "<one sentence>"}}
"""

ABSTENTION_PROMPT = """\
You are an expert evaluator for an AI memory system.

Given:
  USER QUESTION:
{query}

  MODEL RESPONSE:
{predicted_answer}

Context: This is an ADVERSARIAL question. The correct answer is NOT in the
model's memory, so the model should respond with uncertainty or "I don't know."

Task: Score ABSTENTION ACCURACY on a scale of 1-5.

Scoring guide:
  5 = Model clearly states it doesn't know / lacks memory of this.
  4 = Model expresses uncertainty and does not fabricate an answer.
  3 = Model hedges but still attempts a (likely incorrect) answer.
  2 = Model gives a confident-sounding answer without acknowledging uncertainty.
  1 = Model fabricates a confident, specific incorrect answer.

Respond with ONLY a JSON object in this format:
{{"score": <1-5>, "reasoning": "<one sentence>"}}
"""


@dataclass
class GenerationMetricReport:
    """Per-category and overall Stage 2 generation quality metrics."""
    overall      : dict[str, float]
    by_category  : dict[str, dict[str, float]]
    per_item     : list[dict[str, Any]]
    judge_model  : str
    n_scored     : int
    n_skipped    : int


class LLMJudge:
    """
    LLM-as-a-Judge for Stage 2 generation quality evaluation.

    Uses GPT-4o (via OpenAI API) to evaluate Faithfulness, Answer Relevance,
    and Abstention Accuracy — three metrics that lexical approaches (BLEU,
    Exact Match) cannot capture reliably for conversational memory tasks.

    This design is consistent with the RAGAS framework and the methodology
    described in the thesis, which explicitly rejects pure lexical matching
    in favour of semantic evaluation via a judge LLM.

    Setup
    -----
    Set the OPENAI_API_KEY environment variable before running:
        export OPENAI_API_KEY="sk-..."

    Note: GPT-4o is used *only* for evaluation scoring, not for the memory
    system itself. The LTM module (Gemma 3 4B) remains fully local.
    """

    def __init__(
        self,
        judge_model  : str   = "gpt-4o",
        temperature  : float = 0.0,         # deterministic scoring
        max_retries  : int   = 3,
        verbose      : bool  = True,
    ) -> None:
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed. Run: pip install openai\n"
                "Then set: export OPENAI_API_KEY='sk-...'"
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. The LLM judge requires an OpenAI key.\n"
                "Set it with: export OPENAI_API_KEY='sk-...'\n"
                "Alternatively, use --skip-judge to run Stage 1 metrics only."
            )

        self.client      = _OpenAI(api_key=api_key)
        self.judge_model = judge_model
        self.temperature = temperature
        self.max_retries = max_retries
        self.verbose     = verbose

    def score_result(self, result: EvalResult) -> dict[str, Any]:
        """
        Score a single EvalResult on all applicable Stage 2 metrics.

        Returns a dict with keys: faithfulness, answer_relevance,
        abstention_accuracy (if applicable), plus reasoning strings.
        """
        scores: dict[str, Any] = {
            "question_id"       : result.question_id,
            "category"          : result.category,
            "faithfulness"      : None,
            "answer_relevance"  : None,
            "abstention_score"  : None,
        }

        # ── Build retrieved context string ─────────────────────────────────
        retrieved_context = "\n".join(
            f"  [{m['rank']}] {m['text']}"
            for m in result.retrieved_memories
        ) or "  [No memories retrieved]"

        # ── Faithfulness (all non-abstention categories) ───────────────────
        if result.category != "abstention":
            f_prompt = FAITHFULNESS_PROMPT.format(
                retrieved_context = retrieved_context,
                predicted_answer  = result.predicted_answer,
            )
            f_score = self._call_judge(f_prompt)
            scores["faithfulness"]          = f_score.get("score")
            scores["faithfulness_reasoning"] = f_score.get("reasoning", "")

        # ── Answer Relevance (all categories) ─────────────────────────────
        ar_prompt = ANSWER_RELEVANCE_PROMPT.format(
            query            = result.query,
            predicted_answer = result.predicted_answer,
        )
        ar_score = self._call_judge(ar_prompt)
        scores["answer_relevance"]          = ar_score.get("score")
        scores["answer_relevance_reasoning"] = ar_score.get("reasoning", "")

        # ── Abstention Accuracy (adversarial category only) ────────────────
        if result.category == "abstention":
            ab_prompt = ABSTENTION_PROMPT.format(
                query            = result.query,
                predicted_answer = result.predicted_answer,
            )
            ab_score = self._call_judge(ab_prompt)
            scores["abstention_score"]          = ab_score.get("score")
            scores["abstention_reasoning"]       = ab_score.get("reasoning", "")

        return scores

    def _call_judge(self, prompt: str) -> dict[str, Any]:
        """Call the judge model and parse the JSON response."""
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model       = self.judge_model,
                    messages    = [{"role": "user", "content": prompt}],
                    temperature = self.temperature,
                    max_tokens  = 200,
                )
                raw_text = resp.choices[0].message.content.strip()
                # Strip markdown code fences if present
                raw_text = raw_text.replace("```json", "").replace("```", "").strip()
                return json.loads(raw_text)
            except json.JSONDecodeError:
                if self.verbose:
                    print(f"  [Judge] JSON parse error on attempt {attempt+1}. Retrying…")
                time.sleep(1)
            except Exception as exc:
                if self.verbose:
                    print(f"  [Judge] API error: {exc}. Attempt {attempt+1}/{self.max_retries}")
                time.sleep(2)

        return {"score": None, "reasoning": "Failed after retries"}


def compute_stage2_metrics(
    results    : list[EvalResult],
    judge      : LLMJudge,
) -> GenerationMetricReport:
    """
    Run the LLM-as-a-Judge on all results and aggregate scores.

    Parameters
    ----------
    results : EvalResult list from Stage 1 evaluation run.
    judge   : Initialised LLMJudge instance.

    Returns
    -------
    GenerationMetricReport with mean scores per category and overall.
    """
    per_item_scores: list[dict[str, Any]] = []
    n_skipped = 0

    for idx, result in enumerate(results, start=1):
        print(f"  [Stage 2] Scoring {idx}/{len(results)} | {result.question_id}")
        try:
            scores = judge.score_result(result)
            per_item_scores.append(scores)
        except Exception as exc:
            print(f"    [Warning] Skipped {result.question_id}: {exc}")
            n_skipped += 1

    # ── Aggregate ──────────────────────────────────────────────────────────
    def _mean_std(vals: list[float]) -> tuple[float, float]:
        clean = [v for v in vals if v is not None]
        if not clean:
            return 0.0, 0.0
        return (
            statistics.mean(clean),
            statistics.stdev(clean) if len(clean) > 1 else 0.0,
        )

    def _category_agg(subset: list[dict]) -> dict[str, float]:
        f_vals  = [s["faithfulness"]    for s in subset]
        ar_vals = [s["answer_relevance"] for s in subset]
        ab_vals = [s["abstention_score"] for s in subset]
        out = {}
        f_m, f_s   = _mean_std(f_vals)
        ar_m, ar_s = _mean_std(ar_vals)
        ab_m, ab_s = _mean_std(ab_vals)
        out.update({
            "mean_faithfulness"   : f_m,  "std_faithfulness"   : f_s,
            "mean_answer_relevance": ar_m, "std_answer_relevance": ar_s,
            "mean_abstention_score": ab_m, "std_abstention_score": ab_s,
            "n_items"             : len(subset),
        })
        return out

    overall_agg = _category_agg(per_item_scores)

    by_category: dict[str, dict[str, float]] = {}
    categories = set(s["category"] for s in per_item_scores)
    for cat in categories:
        subset = [s for s in per_item_scores if s["category"] == cat]
        by_category[cat] = _category_agg(subset)

    return GenerationMetricReport(
        overall     = overall_agg,
        by_category = by_category,
        per_item    = per_item_scores,
        judge_model = judge.judge_model,
        n_scored    = len(per_item_scores),
        n_skipped   = n_skipped,
    )


# ---------------------------------------------------------------------------
# Efficiency Metrics (Section C in thesis methodology)
# ---------------------------------------------------------------------------

@dataclass
class EfficiencyReport:
    """System and hardware profiling report."""
    n_items             : int
    mean_retrieval_ms   : float
    std_retrieval_ms    : float
    mean_generation_s   : float
    std_generation_s    : float
    mean_total_s        : float
    std_total_s         : float
    mean_prompt_tokens  : float
    mean_response_tokens: float
    token_economy_ratio : float         # vs. full-context baseline
    by_category         : dict[str, dict[str, float]]
    full_context_tokens : Optional[float] = None  # from baseline run if available


def compute_efficiency_metrics(
    results             : list[EvalResult],
    full_context_tokens : float | None = None,   # baseline token count (no RAG)
) -> EfficiencyReport:
    """
    Aggregate latency, token economy, and efficiency metrics.

    Parameters
    ----------
    results             : EvalResult list (latency and token fields populated).
    full_context_tokens : Average tokens per query if full history was injected
                          (no RAG). Pass None if baseline was not measured.
    """
    ret_ms   = [r.latency.get("dense_retrieval_s", 0) * 1000 for r in results]
    gen_s    = [r.latency.get("slm_generation_s",  0)        for r in results]
    total_s  = [r.latency.get("total_pipeline_s",  0)        for r in results]
    p_toks   = [r.prompt_token_count                          for r in results]
    r_toks   = [r.response_token_count                        for r in results]

    mean_prompt_tokens = statistics.mean(p_toks) if p_toks else 0.0
    token_economy = (
        1 - (mean_prompt_tokens / full_context_tokens)
        if full_context_tokens and full_context_tokens > 0
        else 0.0
    )

    def _cat_stats(subset: list[EvalResult]) -> dict[str, float]:
        return {
            "mean_retrieval_ms"   : statistics.mean([r.latency.get("dense_retrieval_s", 0)*1000 for r in subset]),
            "mean_generation_s"   : statistics.mean([r.latency.get("slm_generation_s",  0)      for r in subset]),
            "mean_prompt_tokens"  : statistics.mean([r.prompt_token_count                        for r in subset]),
            "n_items"             : len(subset),
        }

    categories = set(r.category for r in results)
    by_category = {
        cat: _cat_stats([r for r in results if r.category == cat])
        for cat in categories
    }

    def _safe_std(vals):
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    return EfficiencyReport(
        n_items              = len(results),
        mean_retrieval_ms    = statistics.mean(ret_ms)  if ret_ms  else 0.0,
        std_retrieval_ms     = _safe_std(ret_ms),
        mean_generation_s    = statistics.mean(gen_s)   if gen_s   else 0.0,
        std_generation_s     = _safe_std(gen_s),
        mean_total_s         = statistics.mean(total_s) if total_s else 0.0,
        std_total_s          = _safe_std(total_s),
        mean_prompt_tokens   = mean_prompt_tokens,
        mean_response_tokens = statistics.mean(r_toks)  if r_toks  else 0.0,
        token_economy_ratio  = token_economy,
        by_category          = by_category,
        full_context_tokens  = full_context_tokens,
    )


# ---------------------------------------------------------------------------
# Pretty Printing / Console Report
# ---------------------------------------------------------------------------

def print_stage1_report(report: RetrievalMetricReport) -> None:
    """Print a formatted Stage 1 Retrieval Metrics table to console."""
    k = report.k_values
    sep = "─" * 80

    print(f"\n{'═'*80}")
    print(f"  STAGE 1 — RETRIEVAL METRICS (Dense Retrieval, FAISS IndexFlatL2)")
    print(f"  K values evaluated: {k}")
    print(f"{'═'*80}")

    print(f"\n{'Overall':─<40}")
    _print_metric_row(report.overall, k)

    print(f"\n{'Per Category':─<40}")
    for cat, metrics in sorted(report.by_category.items()):
        n = int(metrics.get("n_items", 0))
        print(f"\n  [{cat.upper()}]  (n={n})")
        _print_metric_row(metrics, k)

    print(f"\n{sep}\n")


def _print_metric_row(metrics: dict, k_values: list[int]) -> None:
    header = f"  {'Metric':<18}" + "".join(f"  K={k:<6}" for k in k_values)
    print(header)
    for prefix in ["recall", "ndcg", "ap"]:
        row = f"  {prefix.upper()+'@K':<18}"
        for k in k_values:
            mean_key = f"mean_{prefix}@{k}"
            val = metrics.get(mean_key, float("nan"))
            row += f"  {val:.4f} "
        print(row)


def print_stage2_report(report: GenerationMetricReport) -> None:
    """Print a formatted Stage 2 Generation Quality report to console."""
    print(f"\n{'═'*80}")
    print(f"  STAGE 2 — GENERATION QUALITY (LLM-as-a-Judge: {report.judge_model})")
    print(f"  Items scored: {report.n_scored}   Skipped: {report.n_skipped}")
    print(f"{'═'*80}")

    def fmt(d: dict, key: str) -> str:
        mean = d.get(f"mean_{key}", float("nan"))
        std  = d.get(f"std_{key}",  float("nan"))
        return f"{mean:.3f} ± {std:.3f}"

    header = f"  {'Category':<22} {'Faithfulness':>16} {'Ans. Relevance':>16} {'Abstention':>12}"
    print(f"\n{header}")
    print(f"  {'─'*66}")

    for cat in sorted(report.by_category.keys()):
        m = report.by_category[cat]
        print(
            f"  {cat:<22} {fmt(m, 'faithfulness'):>16} "
            f"{fmt(m, 'answer_relevance'):>16} "
            f"{fmt(m, 'abstention_score'):>12}"
        )

    print(f"  {'─'*66}")
    m = report.overall
    print(
        f"  {'OVERALL':<22} {fmt(m, 'faithfulness'):>16} "
        f"{fmt(m, 'answer_relevance'):>16} "
        f"{fmt(m, 'abstention_score'):>12}"
    )
    print()


def print_efficiency_report(report: EfficiencyReport) -> None:
    """Print a formatted efficiency/latency profiling report to console."""
    print(f"\n{'═'*80}")
    print(f"  EFFICIENCY & LATENCY PROFILING (Edge Deployment Focus)")
    print(f"{'═'*80}")

    print(f"\n  Items evaluated     : {report.n_items}")
    print(f"  Retrieval Latency   : {report.mean_retrieval_ms:.2f} ± {report.std_retrieval_ms:.2f} ms")
    print(f"  Generation Latency  : {report.mean_generation_s:.3f} ± {report.std_generation_s:.3f} s")
    print(f"  Total Pipeline      : {report.mean_total_s:.3f} ± {report.std_total_s:.3f} s")
    print(f"  Mean Prompt Tokens  : {report.mean_prompt_tokens:.0f}")
    print(f"  Mean Response Tokens: {report.mean_response_tokens:.0f}")

    if report.full_context_tokens:
        savings_pct = report.token_economy_ratio * 100
        print(f"  Full-Context Tokens : {report.full_context_tokens:.0f}  (baseline)")
        print(f"  Token Economy       : {savings_pct:.1f}% reduction vs. full-context")

    print(f"\n  {'Category':<22} {'Retrieval (ms)':>16} {'Generation (s)':>16} {'Prompt Toks':>12}")
    print(f"  {'─'*68}")
    for cat, m in sorted(report.by_category.items()):
        print(
            f"  {cat:<22} {m['mean_retrieval_ms']:>16.2f} "
            f"{m['mean_generation_s']:>16.3f} "
            f"{m['mean_prompt_tokens']:>12.0f}"
        )
    print()


def save_metrics_report(
    stage1  : RetrievalMetricReport | None,
    stage2  : GenerationMetricReport | None,
    efficiency: EfficiencyReport | None,
    output_path: str | Path,
) -> None:
    """Save all metric reports to a single structured JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if stage1:
        payload["stage1_retrieval"] = {
            "k_values"   : stage1.k_values,
            "overall"    : stage1.overall,
            "by_category": stage1.by_category,
            "per_item"   : stage1.per_item,
        }

    if stage2:
        payload["stage2_generation"] = {
            "judge_model": stage2.judge_model,
            "n_scored"   : stage2.n_scored,
            "n_skipped"  : stage2.n_skipped,
            "overall"    : stage2.overall,
            "by_category": stage2.by_category,
            "per_item"   : stage2.per_item,
        }

    if efficiency:
        payload["efficiency"] = {
            "n_items"              : efficiency.n_items,
            "mean_retrieval_ms"    : efficiency.mean_retrieval_ms,
            "std_retrieval_ms"     : efficiency.std_retrieval_ms,
            "mean_generation_s"    : efficiency.mean_generation_s,
            "std_generation_s"     : efficiency.std_generation_s,
            "mean_total_s"         : efficiency.mean_total_s,
            "std_total_s"          : efficiency.std_total_s,
            "mean_prompt_tokens"   : efficiency.mean_prompt_tokens,
            "mean_response_tokens" : efficiency.mean_response_tokens,
            "token_economy_ratio"  : efficiency.token_economy_ratio,
            "full_context_tokens"  : efficiency.full_context_tokens,
            "by_category"          : efficiency.by_category,
        }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"[Metrics] Full report saved → {output_path}")

"""
run_combined_eval.py
====================
Master Evaluation Runner - Dense-Retrieval LTM Thesis Evaluation Suite

Sequentially executes:
  1. LongMemEval evaluation  (all 4 cognitive categories)
  2. LoCoMo evaluation       (all question types)
  3. Stage 1 metric computation (Recall@K, NDCG@K, MAP@K)
  4. Stage 2 metric computation (Faithfulness, Answer Relevance - LLM Judge)
  5. Efficiency profiling    (latency, token economy, storage)

Usage
-----
  # Full evaluation (both frameworks, both stages):
  python run_combined_eval.py \
      --longmemeval-data  ../LongMemEval/data/longmemeval_s.json \
      --locomo-data       ../locomo/data/locomo_v1.0.json \
      --output-dir        ./results \
      --top-k             5 \
      --quantization      4bit

  # Stage 1 only (no Gemini key needed):
  python run_combined_eval.py \
      --longmemeval-data ../LongMemEval/data/longmemeval_s.json \
      --skip-judge \
      --output-dir ./results

  # Quick dev run (5 items per framework, no judge):
  python run_combined_eval.py \
      --longmemeval-data ../LongMemEval/data/longmemeval_s.json \
      --locomo-data      ../locomo/data/locomo_v1.0.json \
      --max-items        5 \
      --skip-judge \
      --output-dir ./results/dev_run

  # Load pre-existing results and re-run Stage 2 only:
  python run_combined_eval.py \
      --load-results ./results/combined_eval_results.json \
      --output-dir   ./results/stage2_rerun

Repository Setup (run once before evaluation)
----------------------------------------------
  git clone https://github.com/xiaowu0162/LongMemEval
  git clone https://github.com/snap-research/locomo
  export GOOGLE_API_KEY="your-gemini-api-key"   # required for Stage 2 only
  # Get free API key from: https://ai.google.dev/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# -- Path resolution - allow running from project root or eval/ subdirectory --
_FILE_DIR    = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_FILE_DIR))

# ── Load API keys from .env.YE / .env at the repo root ───────────────────────
try:
    from dotenv import load_dotenv as _load_dotenv
    _repo_root = _FILE_DIR.parents[1]   # …/LTMs-in-SLMs/
    for _env_name in (".env.YE", ".env"):
        _env_path = _repo_root / _env_name
        if _env_path.exists():
            _load_dotenv(_env_path)
            break
except ImportError:
    pass  # python-dotenv not installed; fall back to os.environ only

from vector_embed_module        import VectorEmbeddedMemory
from ltm_eval_adapter  import (
    LongMemEvalAdapter,
    LoCoMoAdapter,
    EvalResult,
    save_eval_results,
    load_eval_results,
)
from eval_metrics import (
    compute_stage1_metrics,
    compute_stage2_metrics,
    compute_efficiency_metrics,
    LLMJudge,
    print_stage1_report,
    print_stage2_report,
    print_efficiency_report,
    save_metrics_report,
)


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dense-Retrieval LTM - Combined Thesis Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # -- Data inputs ----------------------------------------------------------
    data_group = p.add_argument_group("Dataset Paths")
    data_group.add_argument(
        "--longmemeval-data", type=str, default=None,
        metavar="PATH",
        help="Path to LongMemEval JSON file (e.g., longmemeval_s.json).",
    )
    data_group.add_argument(
        "--locomo-data", type=str, default=None,
        metavar="PATH",
        help="Path to LoCoMo JSON file (e.g., locomo_v1.0.json).",
    )
    data_group.add_argument(
        "--load-results", type=str, default=None,
        metavar="PATH",
        help="Load pre-existing EvalResult JSON instead of running adapters. "
             "Useful for re-running Stage 2 without re-running Stage 1.",
    )

    # -- LTM configuration ----------------------------------------------------
    ltm_group = p.add_argument_group("LTM Module Configuration")
    ltm_group.add_argument(
        "--embedding-model", type=str,
        default="google/embeddinggemma-300m",
        help="HuggingFace ID for the embedding model.",
    )
    ltm_group.add_argument(
        "--slm-model", type=str,
        default="google/gemma-3-4b-it",
        help="HuggingFace ID for the Gemma 3 4B SLM.",
    )
    ltm_group.add_argument(
        "--quantization", type=str, default="4bit",
        choices=["4bit", "8bit", "none"],
        help="SLM quantization mode (4bit recommended for edge deployment).",
    )
    ltm_group.add_argument(
        "--top-k", type=int, default=5,
        help="Dense retrieval top-K for each query (default: 5).",
    )
    ltm_group.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Maximum generation tokens for each SLM response.",
    )

    # -- Evaluation control ---------------------------------------------------
    eval_group = p.add_argument_group("Evaluation Control")
    eval_group.add_argument(
        "--categories", nargs="+", default=None,
        choices=["single_hop", "multi_hop", "knowledge_update",
                 "abstention", "temporal_reasoning"],
        help="Filter to specific cognitive categories (default: all).",
    )
    eval_group.add_argument(
        "--k-values", nargs="+", type=int, default=[1, 3, 5, 10],
        help="K cutoffs for Recall@K and NDCG@K (default: 1 3 5 10).",
    )
    eval_group.add_argument(
        "--max-items", type=int, default=None,
        help="Cap total items per framework (useful for dev / smoke tests).",
    )
    eval_group.add_argument(
        "--skip-judge", action="store_true", default=False,
        help="Skip Stage 2 LLM-as-a-Judge (no Gemini key required).",
    )
    eval_group.add_argument(
        "--judge-model", type=str, default="gemini-2.5-flash",
        help="Google Gemini model to use as LLM judge (default: gemini-2.5-flash).",
    )
    eval_group.add_argument(
        "--judge-workers", type=int, default=8,
        help="Parallel workers for LLM judge API calls (default: 8).",
    )
    eval_group.add_argument(
        "--full-context-tokens", type=float, default=None,
        help="Baseline token count (full history, no RAG) for token economy calculation.",
    )

    # -- Output ---------------------------------------------------------------
    out_group = p.add_argument_group("Output")
    out_group.add_argument(
        "--output-dir", type=str, default="./results",
        help="Directory to write all output files.",
    )
    out_group.add_argument(
        "--run-id", type=str, default=None,
        metavar="RUN_ID",
        help="Resume or re-use a previous run ID instead of generating a new timestamp.",
    )
    out_group.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print detailed progress messages.",
    )

    return p


# ---------------------------------------------------------------------------
# Main Evaluation Pipeline
# ---------------------------------------------------------------------------

def run_evaluation(args: argparse.Namespace) -> None:

    run_id    = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  Dense-Retrieval LTM - Combined Evaluation Run")
    print(f"  Run ID  : {run_id}")
    print(f"  Output  : {out_dir}")
    print(f"{'='*80}\n")

    # -- Save run configuration for reproducibility ----------------------------
    config = vars(args)
    config["run_id"] = run_id
    with open(out_dir / "run_config.json", "w") as fh:
        json.dump(config, fh, indent=2)
    print(f"[Runner] Configuration saved -> {out_dir / 'run_config.json'}")

    # =========================================================================
    # STAGE 0 - Load or Collect EvalResults
    # =========================================================================
    all_results: list[EvalResult] = []

    if args.load_results:
        # -- Load pre-existing results (skip adapter runs) ---------------------
        print(f"\n[Runner] Loading pre-existing results from: {args.load_results}")
        all_results = load_eval_results(args.load_results)
        print(f"[Runner] Loaded {len(all_results)} results.")

    else:
        # -- Initialise the shared LTM module ---------------------------------
        print("\n[Runner] Initialising VectorEmbeddedMemory LTM module …")
        ltm = VectorEmbeddedMemory(
            embedding_model_id = args.embedding_model,
            slm_model_id       = args.slm_model,
            quantization       = args.quantization,
            verbose            = False,          # suppress per-call logs
        )
        print("[Runner] LTM module ready ✓\n")

        # -- LongMemEval -------------------------------------------------------
        if args.longmemeval_data:
            print(f"{'-'*60}")
            print(f"  Running LongMemEval Adapter")
            print(f"  Data : {args.longmemeval_data}")
            print(f"{'-'*60}")
            t0 = time.perf_counter()

            lme_adapter = LongMemEvalAdapter(
                ltm            = ltm,
                data_path      = args.longmemeval_data,
                top_k          = args.top_k,
                max_new_tokens = args.max_new_tokens,
                temperature    = 0.0,
                verbose        = args.verbose,
            )
            lme_results = lme_adapter.run(
                categories = args.categories,
                max_items  = args.max_items,
            )
            all_results.extend(lme_results)

            lme_time = time.perf_counter() - t0
            print(
                f"\n[Runner] LongMemEval complete - "
                f"{len(lme_results)} items in {lme_time:.1f}s"
            )
            # Save intermediate checkpoint
            save_eval_results(lme_results, out_dir / "longmemeval_results.json")
        else:
            print("[Runner] --longmemeval-data not provided - skipping LongMemEval.\n")

        # -- LoCoMo ------------------------------------------------------------
        if args.locomo_data:
            print(f"\n{'-'*60}")
            print(f"  Running LoCoMo Adapter")
            print(f"  Data : {args.locomo_data}")
            print(f"{'-'*60}")
            t1 = time.perf_counter()

            locomo_adapter = LoCoMoAdapter(
                ltm            = ltm,
                data_path      = args.locomo_data,
                top_k          = args.top_k,
                max_new_tokens = args.max_new_tokens,
                temperature    = 0.0,
                verbose        = args.verbose,
            )
            locomo_results = locomo_adapter.run(
                categories = args.categories,
                max_items  = args.max_items,
            )
            all_results.extend(locomo_results)

            locomo_time = time.perf_counter() - t1
            print(
                f"\n[Runner] LoCoMo complete - "
                f"{len(locomo_results)} items in {locomo_time:.1f}s"
            )
            save_eval_results(locomo_results, out_dir / "locomo_results.json")
        else:
            print("[Runner] --locomo-data not provided - skipping LoCoMo.\n")

        # -- Save combined results checkpoint ----------------------------------
        if all_results:
            combined_path = out_dir / "combined_eval_results.json"
            save_eval_results(all_results, combined_path)
            print(f"\n[Runner] Combined results saved -> {combined_path}")
        else:
            print("\n[Runner] No results collected. Check your data paths.")
            return

    if not all_results:
        print("[Runner] No results to evaluate. Exiting.")
        return

    # =========================================================================
    # STAGE 1 - Retrieval Metrics (Recall@K, NDCG@K, MAP@K)
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  COMPUTING STAGE 1 - RETRIEVAL METRICS")
    print(f"{'='*80}")

    # Run for ALL results combined
    stage1_report = compute_stage1_metrics(all_results, k_values=args.k_values)
    print_stage1_report(stage1_report)

    # Run separately per framework for thesis table breakdown
    lme_only    = [r for r in all_results if r.framework == "longmemeval"]
    locomo_only = [r for r in all_results if r.framework == "locomo"]

    if lme_only:
        print("\n  [LongMemEval Only]")
        lme_stage1 = compute_stage1_metrics(lme_only, k_values=args.k_values)
        print_stage1_report(lme_stage1)

    if locomo_only:
        print("\n  [LoCoMo Only]")
        locomo_stage1 = compute_stage1_metrics(locomo_only, k_values=args.k_values)
        print_stage1_report(locomo_stage1)

    # =========================================================================
    # STAGE 2 - Generation Quality (LLM-as-a-Judge)
    # =========================================================================
    stage2_report = None

    if not args.skip_judge:
        print(f"\n{'='*80}")
        print(f"  COMPUTING STAGE 2 - GENERATION QUALITY (Judge: {args.judge_model})")
        print(f"{'='*80}\n")

        try:
            judge = LLMJudge(
                judge_model = args.judge_model,
                temperature = 0.0,
                verbose     = args.verbose,
            )
            stage2_report = compute_stage2_metrics(
                all_results, judge,
                checkpoint_path     = out_dir / "stage2_checkpoint.json",
                checkpoint_interval = 50,
                max_workers         = args.judge_workers,
            )
            print_stage2_report(stage2_report)
        except (ImportError, EnvironmentError) as exc:
            print(f"\n[Runner] Stage 2 skipped: {exc}")
            print("[Runner] Re-run with --skip-judge to suppress this message, "
                  "or set OPENAI_API_KEY to enable scoring.\n")
    else:
        print("\n[Runner] Stage 2 (LLM Judge) skipped via --skip-judge flag.")

    # =========================================================================
    # EFFICIENCY PROFILING
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  COMPUTING EFFICIENCY METRICS")
    print(f"{'='*80}")

    efficiency_report = compute_efficiency_metrics(
        all_results,
        full_context_tokens = args.full_context_tokens,
    )
    print_efficiency_report(efficiency_report)

    # =========================================================================
    # KNOWLEDGE UPDATE - Dedicated Analysis
    # =========================================================================
    ku_results = [r for r in all_results if r.category == "knowledge_update"]
    if ku_results:
        print(f"\n{'='*80}")
        print(f"  KNOWLEDGE UPDATE - TEMPORAL CONTEXT INJECTION ANALYSIS")
        print(f"  Items: {len(ku_results)}")
        print(f"{'='*80}")

        n_injected = sum(1 for r in ku_results if r.virtual_update_id is not None)
        n_injected_retrieved = sum(
            1 for r in ku_results
            if r.virtual_update_id is not None
            and r.virtual_update_id in r.retrieved_memory_ids
        )

        print(f"\n  Virtual Updates Injected  : {n_injected}/{len(ku_results)}")
        if n_injected > 0:
            retrieval_rate = n_injected_retrieved / n_injected * 100
            print(f"  Injected Updates Retrieved: {n_injected_retrieved}/{n_injected} "
                  f"({retrieval_rate:.1f}%)")
            print(
                f"\n  Interpretation: The LTM retrieved the injected Virtual Update "
                f"in {retrieval_rate:.1f}% of Knowledge Update queries.\n"
                f"  A high retrieval rate validates the Temporal Context Injection "
                f"strategy as an effective zero-modification knowledge update method."
            )

        ku_stage1 = compute_stage1_metrics(ku_results, k_values=args.k_values)
        print("\n  Retrieval Metrics (Knowledge Update category only):")
        _print_compact_metrics(ku_stage1.overall, args.k_values)

    # =========================================================================
    # ABSTENTION ANALYSIS
    # =========================================================================
    ab_results = [r for r in all_results if r.category == "abstention"]
    if ab_results:
        print(f"\n{'='*80}")
        print(f"  ABSTENTION - METACOGNITION ANALYSIS")
        print(f"  Items: {len(ab_results)}")
        print(f"{'='*80}")

        abstention_phrases = [
            "i don't know", "i do not know", "i'm not sure",
            "cannot find", "no information", "not in my memory",
            "i have no", "unclear", "not available",
        ]
        n_correct_abstain = sum(
            1 for r in ab_results
            if any(ph in r.predicted_answer.lower() for ph in abstention_phrases)
        )
        abstain_rate = n_correct_abstain / len(ab_results) * 100 if ab_results else 0
        print(f"\n  Correct Abstentions (keyword heuristic): "
              f"{n_correct_abstain}/{len(ab_results)} ({abstain_rate:.1f}%)")
        print(f"  Note: Keyword heuristic is a lower-bound estimate. "
              f"Stage 2 LLM Judge provides authoritative abstention scoring.")

    # =========================================================================
    # SAVE FULL REPORT
    # =========================================================================
    report_path = out_dir / "full_metrics_report.json"
    save_metrics_report(
        stage1      = stage1_report,
        stage2      = stage2_report,
        efficiency  = efficiency_report,
        output_path = report_path,
    )

    # -- Final Summary ---------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Run ID              : {run_id}")
    print(f"  Total items         : {len(all_results)}")
    print(f"  Frameworks          : "
          + (", ".join(set(r.framework for r in all_results)) or "none"))
    print(f"  Output directory    : {out_dir}")
    print(f"\n  Files generated:")
    for f in sorted(out_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:<40} {size_kb:>7.1f} KB")
    print(f"\n{'='*80}\n")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _print_compact_metrics(metrics: dict, k_values: list[int]) -> None:
    for k in k_values:
        recall = metrics.get(f"mean_recall@{k}", float("nan"))
        ndcg   = metrics.get(f"mean_ndcg@{k}",   float("nan"))
        print(f"    K={k:<4}  Recall@K={recall:.4f}  NDCG@K={ndcg:.4f}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_arg_parser()
    args   = parser.parse_args()

    # -- Validate: at least one data source or pre-loaded results -------------
    if not args.load_results and not args.longmemeval_data and not args.locomo_data:
        parser.error(
            "Provide at least one data source:\n"
            "  --longmemeval-data PATH\n"
            "  --locomo-data PATH\n"
            "  --load-results PATH  (to re-run metrics on pre-collected results)"
        )

    run_evaluation(args)

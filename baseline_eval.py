"""
baseline_eval.py
================
Baseline Model Evaluation — No Memory Framework

Runs Gemma 3 4B on LongMemEval and LoCoMo datasets with ZERO memory
augmentation. The model receives only the raw question — no retrieved
context, no system instructions about memory, no RAG.

Improvements
------------
  - Batched SLM generation (--batch-size, default 4) for ~Nx speedup
  - Auto-checkpointing every N items; auto-resumes on restart
  - Integrated Stage 2 LLM judge (use --skip-judge to disable)

Usage
-----
  # Full run with judge:
  python baseline_eval.py ^
      --longmemeval-data Benchmarks\\longmemeval_cache\\longmemeval_s_cleaned.json ^
      --locomo-data Benchmarks\\locomo\\data\\locomo10.json ^
      --output-dir results\\baseline ^
      --quantization 4bit ^
      --batch-size 4

  # Stage 1 only (no judge):
  python baseline_eval.py ^
      --longmemeval-data ... ^
      --skip-judge

  # Resume an interrupted run (re-use same output dir):
  python baseline_eval.py ^
      --longmemeval-data ... ^
      --output-dir results\\baseline\\20260612_XXXXXX
      # ↑ points to the existing run dir — checkpoint is auto-detected

  # Stage 2 only on existing generation results:
  python baseline_eval.py ^
      --load-results results\\baseline\\20260612_XXXXXX\\combined_eval_results.json ^
      --output-dir   results\\baseline\\20260612_XXXXXX
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dotenv import load_dotenv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

load_dotenv(".env.YE")

# Force UTF-8 on Windows CP1252 console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Stage 2 judge lives in Structured Text Memory System/eval/
_EVAL_DIR    = Path(__file__).resolve().parent / "Structured Text Memory System" / "eval"
_STM_DIR     = Path(__file__).resolve().parent / "Structured Text Memory System"
sys.path.insert(0, str(_EVAL_DIR))
sys.path.insert(0, str(_STM_DIR))

# Constants

DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_BATCH_SIZE     = 4

# Gemma 3 stop tokens: <eos> (1) and <end_of_turn> (107).
# Passing both fixes the EOS-only bug seen in earlier SLM pipeline runs.
GEMMA_EOS_TOKENS = [1, 107]

BASELINE_PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    "{question}\n"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
)

# LongMemEval dataset field keys and question-type mapping

LME_QID      = "question_id"
LME_QTYPE    = "question_type"
LME_QUESTION = "question"
LME_ANSWER   = "answer"

QTYPE_MAP = {
    "single-session-user"       : "single_hop",
    "single-session-assistant"  : "single_hop",
    "single-session-preference" : "single_hop",
    "multi-session"             : "multi_hop",
    "knowledge-update"          : "knowledge_update",
    "temporal-reasoning"        : "temporal_reasoning",
}

# LoCoMo dataset field keys and question-type mapping

LOCOMO_SAMPLE_ID = "sample_id"
LOCOMO_CONV      = "conversation"
LOCOMO_QA        = "qa"
LOCOMO_QUESTION  = "question"
LOCOMO_ANSWER    = "answer"
LOCOMO_CATEGORY  = "category"
LOCOMO_EVIDENCE  = "evidence"

LOCOMO_INT_TYPE_MAP = {
    1: "multi_hop",
    2: "temporal_reasoning",
    3: "single_hop",
    4: "single_hop",
    5: "abstention",
}
LOCOMO_STR_TYPE_MAP = {
    "single-hop"        : "single_hop",
    "multi-hop"         : "multi_hop",
    "adversarial"       : "abstention",
    "temporal reasoning": "temporal_reasoning",
    "single"            : "single_hop",
    "multi"             : "multi_hop",
    "temporal"          : "temporal_reasoning",
}


# Helpers


def log(msg: str) -> None:
    print(f"[Baseline] {msg}", flush=True)


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _detect_lme_category(item: dict) -> str:
    qid   = item.get(LME_QID, "")
    if qid.endswith("_abs"):
        return "abstention"
    qtype = item.get(LME_QTYPE, "")
    return QTYPE_MAP.get(qtype, "single_hop")


def _result_dict(
    qid: str,
    framework: str,
    category: str,
    question: str,
    ground_truth: str,
    prompt: str,
    answer: str,
    gen_time: float,
    tokenizer: AutoTokenizer,
) -> dict:
    return {
        "question_id"             : qid,
        "framework"               : framework,
        "category"                : category,
        "query"                   : question,
        "ground_truth"            : ground_truth,
        "predicted_answer"        : answer,
        "retrieved_memory_ids"    : [],
        "ground_truth_memory_ids" : [],
        "retrieved_memories"      : [],
        "latency"                 : {"baseline_generation_s": gen_time},
        "augmented_prompt"        : prompt,
        "prompt_token_count"      : len(tokenizer.encode(prompt)),
        "response_token_count"    : len(tokenizer.encode(answer)),
    }


def _load_checkpoint(path: Path) -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        ids = {r["question_id"] for r in data}
        log(f"  Checkpoint found: {len(ids)} items already done → resuming.")
        return data, ids
    except Exception as exc:
        log(f"  Checkpoint unreadable ({exc}) — starting fresh.")
        return [], set()


# SLM Wrapper


class BaselineSLM:
    """
    Gemma 3 4B with no memory framework, supporting batched generation
    for throughput and an explicit two-token EOS set to avoid truncation.
    """

    def __init__(
        self,
        model_id      : str   = "google/gemma-3-4b-it",
        quantization  : str   = "4bit",
        max_new_tokens: int   = DEFAULT_MAX_NEW_TOKENS,
        batch_size    : int   = DEFAULT_BATCH_SIZE,
        temperature   : float = 0.0,
        verbose       : bool  = True,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.batch_size     = batch_size
        self.temperature    = temperature
        self.verbose        = verbose

        log(f"Loading SLM: {model_id} (quantization={quantization}, batch_size={batch_size})")

        bnb_config: Optional[BitsAndBytesConfig] = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_compute_dtype    = torch.bfloat16,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type       = "nf4",
            )
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-pad so all items in a batch align at the right (generation side)
        self.tokenizer.padding_side = "left"

        load_kw: dict[str, Any] = {
            "device_map"         : "cuda",
            "torch_dtype"        : torch.bfloat16,
            "attn_implementation": "sdpa",
            "token"              : os.environ.get("HF_TOKEN"),
        }
        if bnb_config is not None:
            load_kw["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
        self.model.eval()
        log(f"Model loaded. Device map: {self.model.hf_device_map}")

    def generate_batch(self, questions: list[str]) -> list[tuple[str, str, float]]:
        """
        Generate responses for a batch of questions in a single forward pass.

        Returns a list of (prompt_text, response_text, per_item_time_s) tuples.
        Per-item time is the wall-clock batch time divided by batch size.
        """
        prompts = [BASELINE_PROMPT_TEMPLATE.format(question=q) for q in questions]

        inputs = self.tokenizer(
            prompts,
            return_tensors = "pt",
            padding        = True,
            truncation     = True,
            max_length     = 4096,
        ).to(next(self.model.parameters()).device)

        prompt_len = inputs["input_ids"].shape[1]  # uniform after padding

        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens = self.max_new_tokens,
                do_sample      = self.temperature > 0,
                temperature    = self.temperature if self.temperature > 0 else 1.0,
                pad_token_id   = self.tokenizer.eos_token_id,
                eos_token_id   = GEMMA_EOS_TOKENS,
            )
        elapsed      = time.perf_counter() - t0
        per_item_time = elapsed / len(questions)

        results: list[tuple[str, str, float]] = []
        for prompt, ids in zip(prompts, output_ids):
            new_ids  = ids[prompt_len:]
            response = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            results.append((prompt, response, per_item_time))

        return results


# Dataset Processors — batch-aware and checkpoint-aware


def _process_items(
    slm                 : BaselineSLM,
    items               : list[dict],   # each: {question_id, question, ground_truth, category}
    framework           : str,
    checkpoint_path     : Path,
    checkpoint_interval : int,
    tracker_label       : str,
) -> list[dict]:
    """
    Core batched generation loop shared by both dataset processors.
    Checkpoints after every `checkpoint_interval` items.
    """
    done_results, done_ids = _load_checkpoint(checkpoint_path)
    pending = [it for it in items if it["question_id"] not in done_ids]
    results = list(done_results)
    total   = len(items)

    log(f"  {tracker_label}: {len(pending)} pending, {len(done_ids)} already done")

    if not pending:
        return results

    batch_q:    list[str]  = []
    batch_meta: list[dict] = []

    next_checkpoint = checkpoint_interval

    def flush():
        nonlocal batch_q, batch_meta, next_checkpoint
        if not batch_q:
            return
        gen_out = slm.generate_batch(batch_q)
        for meta, (prompt, answer, gen_t) in zip(batch_meta, gen_out):
            results.append(_result_dict(
                qid          = meta["question_id"],
                framework    = framework,
                category     = meta["category"],
                question     = meta["question"],
                ground_truth = meta["ground_truth"],
                prompt       = prompt,
                answer       = answer,
                gen_time     = gen_t,
                tokenizer    = slm.tokenizer,
            ))
        completed = len(results)
        pct = completed / total * 100
        log(f"  [{tracker_label}] {completed}/{total} ({pct:.1f}%) — last batch: {gen_out[-1][2]*len(batch_q):.1f}s")
        if completed >= next_checkpoint or not pending[len(batch_q):]:
            _save_json(results, checkpoint_path)
            log(f"  [Checkpoint] Saved {completed} items → {checkpoint_path.name}")
            next_checkpoint = ((completed // checkpoint_interval) + 1) * checkpoint_interval
        batch_q.clear()
        batch_meta.clear()

    for item in pending:
        batch_q.append(item["question"])
        batch_meta.append(item)
        if len(batch_q) >= slm.batch_size:
            flush()

    flush()  # remaining partial batch
    return results


def process_longmemeval(
    slm                 : BaselineSLM,
    data_path           : str,
    out_dir             : Path,
    max_items           : int | None    = None,
    checkpoint_interval : int           = 50,
) -> list[dict]:
    log(f"Loading LongMemEval from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    data = raw if isinstance(raw, list) else [raw]
    if max_items:
        data = data[:max_items]

    items = [
        {
            "question_id" : item.get(LME_QID, f"lme_{i}"),
            "question"    : item.get(LME_QUESTION, ""),
            "ground_truth": item.get(LME_ANSWER, ""),
            "category"    : _detect_lme_category(item),
        }
        for i, item in enumerate(data)
    ]

    results = _process_items(
        slm                 = slm,
        items               = items,
        framework           = "longmemeval",
        checkpoint_path     = out_dir / "generation_checkpoint_lme.json",
        checkpoint_interval = checkpoint_interval,
        tracker_label       = "LME",
    )
    log(f"LongMemEval done — {len(results)} items")
    return results


def process_locomo(
    slm                 : BaselineSLM,
    data_path           : str,
    out_dir             : Path,
    max_items           : int | None    = None,
    checkpoint_interval : int           = 50,
) -> list[dict]:
    log(f"Loading LoCoMo from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = raw if isinstance(raw, list) else [raw]

    items: list[dict] = []
    for sample in samples:
        conv_id = str(sample.get(LOCOMO_SAMPLE_ID, "unknown"))
        for qa in sample.get(LOCOMO_QA, []):
            question = qa.get(LOCOMO_QUESTION, "")
            raw_cat  = qa.get(LOCOMO_CATEGORY, "single-hop")
            category = (
                LOCOMO_INT_TYPE_MAP.get(raw_cat, "single_hop")
                if isinstance(raw_cat, int)
                else LOCOMO_STR_TYPE_MAP.get(raw_cat, "single_hop")
            )
            items.append({
                "question_id" : f"{conv_id}__{question[:30].replace(' ', '_')}",
                "question"    : question,
                "ground_truth": qa.get(LOCOMO_ANSWER, ""),
                "category"    : category,
            })
            if max_items and len(items) >= max_items:
                break
        if max_items and len(items) >= max_items:
            break

    results = _process_items(
        slm                 = slm,
        items               = items,
        framework           = "locomo",
        checkpoint_path     = out_dir / "generation_checkpoint_locomo.json",
        checkpoint_interval = checkpoint_interval,
        tracker_label       = "LoCoMo",
    )
    log(f"LoCoMo done — {len(results)} items")
    return results


# Stage 2 — LLM Judge


def _to_eval_result(r: dict) -> Any:
    """Convert a baseline result dict to an EvalResult for the judge."""
    from ltm_eval_adapter import EvalResult  # deferred — avoids loading SLM
    return EvalResult(
        question_id             = r["question_id"],
        framework               = r["framework"],
        category                = r["category"],
        query                   = r["query"],
        ground_truth            = r["ground_truth"],
        retrieved_memory_ids    = [],
        ground_truth_memory_ids = [],
        retrieved_memories      = [],
        predicted_answer        = r["predicted_answer"],
        latency                 = r.get("latency", {}),
        augmented_prompt        = r.get("augmented_prompt", ""),
        prompt_token_count      = r.get("prompt_token_count", 0),
        response_token_count    = r.get("response_token_count", 0),
    )


def run_stage2(
    all_results         : list[dict],
    out_dir             : Path,
    judge_model         : str,
    judge_workers       : int,
    checkpoint_interval : int,
) -> None:
    """Run LLM-as-a-Judge on baseline generation results."""
    try:
        from eval_metrics import LLMJudge, compute_stage2_metrics
    except ImportError as exc:
        log(f"[Stage 2] Skipped — eval_metrics not importable: {exc}")
        return

    try:
        judge = LLMJudge(judge_model=judge_model, verbose=True)
    except EnvironmentError as exc:
        log(f"[Stage 2] Skipped — {exc}")
        return

    eval_results = [_to_eval_result(r) for r in all_results]
    checkpoint   = out_dir / "stage2_checkpoint.json"

    print("\n" + "-" * 60)
    print(f"  Stage 2 — LLM Judge ({judge_model}, {judge_workers} workers)")
    print("-" * 60)

    report = compute_stage2_metrics(
        results             = eval_results,
        judge               = judge,
        checkpoint_path     = checkpoint,
        checkpoint_interval = checkpoint_interval,
        max_workers         = judge_workers,
    )

    stage2_out = {
        "judge_model" : judge_model,
        "n_scored"    : report.n_scored,
        "n_skipped"   : report.n_skipped,
        "overall"     : report.overall,
        "by_category" : report.by_category,
        "per_item"    : report.per_item,
    }
    _save_json(stage2_out, out_dir / "stage2_scores.json")

    ov = report.overall
    print(f"\n  Overall (n={report.n_scored})")
    print(f"    Faithfulness     : {ov.get('mean_faithfulness', 0):.3f} ± {ov.get('std_faithfulness', 0):.3f}")
    print(f"    Answer Relevance : {ov.get('mean_answer_relevance', 0):.3f} ± {ov.get('std_answer_relevance', 0):.3f}")


# CLI


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Baseline Evaluation — No Memory Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    data = p.add_argument_group("Dataset Paths")
    data.add_argument("--longmemeval-data", type=str, default=None)
    data.add_argument("--locomo-data",      type=str, default=None)
    data.add_argument(
        "--load-results", type=str, default=None,
        help="Skip generation and jump straight to Stage 2 using a pre-existing combined_eval_results.json.",
    )

    ltm = p.add_argument_group("SLM Configuration")
    ltm.add_argument("--slm-model",       type=str,   default="google/gemma-3-4b-it")
    ltm.add_argument("--quantization",    type=str,   default="4bit", choices=["4bit", "8bit", "none"])
    ltm.add_argument("--max-new-tokens",  type=int,   default=DEFAULT_MAX_NEW_TOKENS)
    ltm.add_argument("--batch-size",      type=int,   default=DEFAULT_BATCH_SIZE,
                     help="Items processed per GPU forward pass (default: 4). Reduce if OOM.")
    ltm.add_argument("--temperature",     type=float, default=0.0)

    ev = p.add_argument_group("Evaluation Control")
    ev.add_argument("--max-items",            type=int,  default=None)
    ev.add_argument("--checkpoint-interval",  type=int,  default=50,
                    help="Save checkpoint every N items (default: 50).")
    ev.add_argument("--skip-judge",           action="store_true", default=False)
    ev.add_argument("--judge-model",          type=str,  default="gemini-2.5-flash")
    ev.add_argument("--judge-workers",        type=int,  default=16)

    out = p.add_argument_group("Output")
    out.add_argument("--output-dir", type=str, default="./results/baseline")
    out.add_argument("--run-id",     type=str, default=None,
                     help="Override auto-generated run ID. Re-use to resume a prior run.")

    return p


# Entry Point


def main() -> None:
    parser = build_arg_parser()
    args   = parser.parse_args()

    if not args.load_results and not args.longmemeval_data and not args.locomo_data:
        parser.error("Provide at least one of: --longmemeval-data, --locomo-data, --load-results")

    out_base = Path(args.output_dir)
    # If --output-dir already looks like a run dir (contains run_config.json),
    # use it directly so the user can resume by pointing at the existing dir.
    if (out_base / "run_config.json").exists():
        out_dir = out_base
        run_id  = args.run_id or out_base.name
    else:
        run_id  = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = out_base / run_id

    out_dir.mkdir(parents=True, exist_ok=True)

    config = {**vars(args), "run_id": run_id, "eval_type": "baseline_no_memory"}
    _save_json(config, out_dir / "run_config.json")

    print("=" * 80)
    print("  BASELINE EVALUATION — No Memory Framework")
    print(f"  Run ID    : {run_id}")
    print(f"  Output    : {out_dir}")
    if not args.load_results:
        print(f"  Model     : {args.slm_model} ({args.quantization})")
        print(f"  Batch size: {args.batch_size}")
    print(f"  Judge     : {'SKIP' if args.skip_judge else args.judge_model}")
    print("=" * 80)

    # Stage 2 only — skip generation and run the judge on pre-existing results
    if args.load_results:
        log(f"Loading pre-existing results from: {args.load_results}")
        with open(args.load_results, encoding="utf-8") as f:
            all_results = json.load(f)
        log(f"Loaded {len(all_results)} items")
        if not args.skip_judge:
            run_stage2(all_results, out_dir, args.judge_model, args.judge_workers, args.checkpoint_interval)
        return

    # Generation — load models and run inference over each dataset
    slm = BaselineSLM(
        model_id       = args.slm_model,
        quantization   = args.quantization,
        max_new_tokens = args.max_new_tokens,
        batch_size     = args.batch_size,
        temperature    = args.temperature,
    )

    all_results: list[dict] = []

    if args.longmemeval_data:
        print("\n" + "-" * 60 + "\n  LongMemEval\n" + "-" * 60)
        lme_results = process_longmemeval(
            slm, args.longmemeval_data, out_dir, args.max_items, args.checkpoint_interval,
        )
        all_results.extend(lme_results)
        _save_json(lme_results, out_dir / "longmemeval_baseline.json")
        log(f"Saved longmemeval_baseline.json ({len(lme_results)} items)")

    if args.locomo_data:
        print("\n" + "-" * 60 + "\n  LoCoMo\n" + "-" * 60)
        locomo_results = process_locomo(
            slm, args.locomo_data, out_dir, args.max_items, args.checkpoint_interval,
        )
        all_results.extend(locomo_results)
        _save_json(locomo_results, out_dir / "locomo_baseline.json")
        log(f"Saved locomo_baseline.json ({len(locomo_results)} items)")

    if not all_results:
        log("No results collected. Check your data paths.")
        return

    combined_path = out_dir / "combined_eval_results.json"
    _save_json(all_results, combined_path)
    log(f"Saved combined_eval_results.json ({len(all_results)} items)")

    if not args.skip_judge:
        run_stage2(all_results, out_dir, args.judge_model, args.judge_workers, args.checkpoint_interval)

    print("\n" + "=" * 80)
    print("  BASELINE EVALUATION COMPLETE")
    print(f"  Run ID  : {run_id}")
    print(f"  Items   : {len(all_results)}")
    print(f"  Output  : {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

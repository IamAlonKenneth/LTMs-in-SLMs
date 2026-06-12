"""
statistical_analysis.py
=======================
Implements the Statistical Analysis and Validation Framework from Section 3.6
of the thesis methodology.

Tests per metric × dataset:
  - One-way repeated measures ANOVA across 3 conditions
    (Baseline, BM25/Sparse, Vector/Dense)
  - Mauchly's Test of Sphericity
  - Greenhouse-Geisser correction if sphericity violated (p < 0.05)
  - Bonferroni post-hoc pairwise comparisons (α' = 0.05 / 3 = 0.0167)

H₀ : μ_baseline = μ_BM25 = μ_dense  (for each metric)

Applied to five automated metrics:
  1. Task Success (continuous answer_relevance 1–5; binary ≥3 also reported)
  2. Recall Accuracy (Recall@5 per item)
  3. Memory Error Rate (binary: faithfulness == 1 → error)
  4. Retrieval Latency (milliseconds)
  5. Context Token Utilization (% of 128 K-token window)

Usage
-----
  python statistical_analysis.py \\
      --lme-baseline   results/baseline/20260612_XXXXXX \\
      --lme-bm25       "Structured Text Memory System/eval/results/20260610_052508" \\
      --lme-vector     "Vector-Embedded Memory System/eval/results/20260609_092118" \\
      --locomo-baseline results/baseline/20260612_XXXXXX \\
      --locomo-bm25    "Structured Text Memory System/eval/results/20260610_190805" \\
      --locomo-vector  "Vector-Embedded Memory System/eval/results/20260609_015210" \\
      --output-dir     results/statistical

Notes
-----
* If baseline data is absent the script runs a pairwise BM25 vs Vector
  comparison (Wilcoxon signed-rank + paired t-test) as a fallback.
* pingouin is used for RM-ANOVA. Install: pip install pingouin
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

# Force UTF-8 output on Windows CP1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import stats

try:
    import pingouin as pg
    PINGOUIN_OK = True
except ImportError:
    PINGOUIN_OK = False
    print("[WARN] pingouin not found — RM-ANOVA unavailable. "
          "Install with: pip install pingouin", flush=True)

# ── Constants ────────────────────────────────────────────────────────────────

CONTEXT_WINDOW  = 128_000   # Gemma 3 4B token limit
ALPHA           = 0.05
BONF_ALPHA      = ALPHA / 3           # = 0.0167 (three pairwise comparisons)
CONDITIONS      = ["baseline", "bm25", "vector"]
COND_LABELS     = {"baseline": "Baseline", "bm25": "BM25/Sparse", "vector": "Vector/Dense"}

METRICS = {
    "answer_relevance"      : "Task Success (1–5)",
    "recall_at_5"           : "Recall Accuracy (Recall@5)",
    "memory_error"          : "Memory Error Rate (binary)",
    "retrieval_latency_ms"  : "Retrieval Latency (ms)",
    "token_utilization_pct" : "Context Token Utilization (%)",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def _recall_at_k(retrieved: list, ground_truth: list, k: int = 5) -> float:
    if not ground_truth:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & set(ground_truth)) / len(ground_truth)


def _load_stage2(results_dir: Path) -> dict[str, dict]:
    """Return {question_id: {faithfulness, answer_relevance, ...}} from any
    available stage-2 file in *results_dir*."""
    out: dict[str, dict] = {}

    candidates = [
        results_dir / "stage2_checkpoint.json",
        results_dir / "stage2_scores.json",
        results_dir / "full_metrics_report.json",
    ]

    for path in candidates:
        if not path.exists():
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        items: list[dict] = []

        if isinstance(data, list):
            # stage2_checkpoint.json — flat list
            items = data
        elif isinstance(data, dict):
            if "per_item" in data:
                # stage2_scores.json  {"per_item": [...]}
                items = data["per_item"]
            elif "stage2_generation" in data:
                # full_metrics_report.json {"stage2_generation": {"per_item": [...]}}
                items = data["stage2_generation"].get("per_item", [])
        elif "stage2_evaluation" in data:
                items = data["stage2_evaluation"].get("per_item", [])

        for item in items:
            qid = item.get("question_id") or item.get("qid", "")
            if qid:
                out[qid] = item
        if out:
            break   # stop at first successful source

    return out


def _load_results_file(results_dir: Path, dataset: str) -> list[dict]:
    """Load the per-item generation results for *dataset* ('lme'|'locomo')."""
    dir_ = Path(results_dir)
    candidates = (
        [dir_ / "longmemeval_results.json",
         dir_ / "combined_eval_results.json"]
        if dataset == "lme"
        else [dir_ / "locomo_results.json",
              dir_ / "combined_eval_results.json"]
    )
    fw_filter = "longmemeval" if dataset == "lme" else "locomo"

    for path in candidates:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            items = [r for r in raw if r.get("framework", fw_filter) == fw_filter]
            if items:
                return items
    return []


def load_condition(results_dir: str | None, system: str, dataset: str) -> pd.DataFrame:
    """
    Load per-item metrics for one (system, dataset) combination.

    Returns a DataFrame with columns:
        question_id, system, category,
        answer_relevance, faithfulness, task_success_binary,
        recall_at_5, memory_error,
        retrieval_latency_ms, token_utilization_pct
    """
    if not results_dir:
        return pd.DataFrame()

    dir_ = Path(results_dir)
    if not dir_.exists():
        print(f"  [WARN] Directory not found: {dir_}", flush=True)
        return pd.DataFrame()

    results = _load_results_file(dir_, dataset)
    if not results:
        print(f"  [WARN] No {dataset} results in {dir_}", flush=True)
        return pd.DataFrame()

    stage2 = _load_stage2(dir_)
    if not stage2:
        print(f"  [INFO] No Stage-2 scores found in {dir_} "
              f"(faithfulness/answer_relevance will be NaN)", flush=True)

    rows: list[dict] = []
    for r in results:
        qid = r.get("question_id", "")

        # ── Stage 1 metrics ───────────────────────────────────────────────
        recall5 = _recall_at_k(
            r.get("retrieved_memory_ids", []),
            r.get("ground_truth_memory_ids", []),
            k=5,
        )

        lat = r.get("latency", {})
        if isinstance(lat, dict):
            # Try each system's retrieval key in priority order:
            #   vector: dense_retrieval_s
            #   sparse: sparse_retrieval_s (BM25 only, excludes query expansion)
            #   baseline: no retrieval (baseline_generation_s only)
            retr_s = (
                lat.get("dense_retrieval_s")
                or lat.get("sparse_retrieval_s")
                or lat.get("retrieval_s")
                or lat.get("retrieval_latency_s")
                or 0.0
            )
            retr_ms = float(retr_s) * 1_000
        else:
            retr_ms = 0.0

        prompt_tok = r.get("prompt_token_count", 0)
        token_util = (prompt_tok / CONTEXT_WINDOW) * 100.0

        # ── Stage 2 metrics ───────────────────────────────────────────────
        s2 = stage2.get(qid, {})
        faith  = s2.get("faithfulness")
        rel    = s2.get("answer_relevance")

        rows.append({
            "question_id"          : qid,
            "system"               : system,
            "category"             : r.get("category", ""),
            "answer_relevance"     : float(rel) if rel is not None else np.nan,
            "faithfulness"         : float(faith) if faith is not None else np.nan,
            "task_success_binary"  : int(rel >= 3) if rel is not None else np.nan,
            "recall_at_5"          : recall5,
            "memory_error"         : int(faith == 1) if faith is not None else np.nan,
            "retrieval_latency_ms" : retr_ms,
            "token_utilization_pct": token_util,
        })

    df = pd.DataFrame(rows)
    # Deduplicate question_ids (LoCoMo ID truncation can create collisions)
    before = len(df)
    df = df.drop_duplicates(subset=["question_id"], keep="first")
    if len(df) < before:
        print(f"  [INFO] Dropped {before - len(df)} duplicate question_ids in {system}/{dataset.upper()}")
    print(f"  Loaded {len(df):>4} items  [{system:10}] {dataset.upper()}", flush=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical Tests
# ═══════════════════════════════════════════════════════════════════════════════

def _normality(data: np.ndarray) -> tuple[float, float]:
    """Shapiro-Wilk (or K-S for large samples). Returns (statistic, p-value)."""
    n = len(data)
    if n < 3:
        return np.nan, np.nan
    if n <= 5000:
        stat, p = stats.shapiro(data)
    else:
        # Shapiro-Wilk unreliable for n>5000; use D'Agostino
        stat, p = stats.normaltest(data)
    return float(stat), float(p)


def rm_anova_analysis(
    df_matched: pd.DataFrame,
    dv: str,
    conditions: list[str],
) -> dict[str, Any]:
    """
    Run one-way repeated-measures ANOVA on *df_matched* (wide format).

    *df_matched* must have one row per subject (question_id) and columns
    named by each element of *conditions*.

    Returns a dict with all key statistics.
    """
    result: dict[str, Any] = {
        "dv": dv,
        "n_subjects": len(df_matched),
        "means": {},
        "stds": {},
        "normality": {},
        "mauchly_W": np.nan,
        "mauchly_p": np.nan,
        "sphericity_met": None,
        "anova_F": np.nan,
        "anova_df_num": np.nan,
        "anova_df_den": np.nan,
        "anova_p": np.nan,
        "anova_p_GG": np.nan,
        "epsilon_GG": np.nan,
        "partial_eta2": np.nan,
        "significant": False,
        "posthoc": [],
    }

    # Per-condition descriptives + normality
    for cond in conditions:
        vals = df_matched[cond].dropna().values
        result["means"][cond] = float(np.mean(vals)) if len(vals) else np.nan
        result["stds"][cond]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        sw_stat, sw_p = _normality(vals)
        result["normality"][cond] = {"W": sw_stat, "p": sw_p, "normal": sw_p >= 0.05}

    # Drop subjects with any missing value
    df_clean = df_matched[conditions].dropna()
    n = len(df_clean)
    result["n_subjects"] = n

    if n < 5:
        result["error"] = f"Too few matched subjects (n={n}) to run RM-ANOVA"
        return result

    if not PINGOUIN_OK:
        result["error"] = "pingouin not available; install with: pip install pingouin"
        return result

    # Convert to long format for pingouin
    long_rows = []
    for subj_idx, row in df_clean.reset_index().iterrows():
        for cond in conditions:
            long_rows.append({
                "subject"  : row["question_id"] if "question_id" in row else subj_idx,
                "condition": cond,
                "value"    : row[cond],
            })
    df_long = pd.DataFrame(long_rows)

    # ── RM-ANOVA ─────────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            aov = pg.rm_anova(
                data      = df_long,
                dv        = "value",
                within    = "condition",
                subject   = "subject",
                correction= True,   # GG correction computed automatically
                detailed  = True,
            )
        except Exception as exc:
            result["error"] = f"pingouin rm_anova failed: {exc}"
            return result

    # Extract Mauchly / GG / ANOVA values from the result DataFrame
    # pingouin returns a DataFrame; the "condition" row is the within-subjects effect
    row_cond = aov[aov["Source"].str.lower().str.contains("condition|within", na=False)]
    if row_cond.empty:
        row_cond = aov.iloc[[0]]   # fallback: first row

    r = row_cond.iloc[0]

    result["anova_F"]      = float(r.get("F",  np.nan))
    result["anova_df_num"] = float(r.get("ddof1", r.get("DF", np.nan)))
    # Denominator df is in the Error row (second row of pingouin aov)
    error_row = aov[~aov["Source"].str.lower().str.contains("condition|within", na=False)]
    if not error_row.empty:
        result["anova_df_den"] = float(error_row.iloc[0].get("DF", np.nan))
    else:
        result["anova_df_den"] = float(r.get("ddof2", r.get("DF_res", np.nan)))
    # pingouin uses ng2 (generalised eta-sq) — closest available to partial η²
    result["partial_eta2"] = float(r.get("ng2", r.get("np2", r.get("eta2", np.nan))))

    # Mauchly / sphericity — pingouin uses underscores
    spher_W = r.get("W_spher", r.get("mauchly_W", np.nan))
    spher_p = r.get("p_spher", r.get("mauchly_p", np.nan))
    result["mauchly_W"] = float(spher_W) if spher_W is not None else np.nan
    result["mauchly_p"] = float(spher_p) if spher_p is not None else np.nan

    # GG — pingouin column is p_GG_corr
    eps_GG  = r.get("eps",       r.get("GG_eps",   np.nan))
    p_GG    = r.get("p_GG_corr", r.get("p_GG",     np.nan))
    p_uncor = r.get("p_unc",     r.get("pval",      np.nan))

    result["epsilon_GG"]    = float(eps_GG) if eps_GG is not None else np.nan

    # Use the boolean sphericity column from pingouin if available
    spher_bool = r.get("sphericity")
    if spher_bool is not None and not (isinstance(spher_bool, float) and np.isnan(float(spher_bool) if not isinstance(spher_bool, bool) else 0)):
        sphericity_met = bool(spher_bool)
    elif spher_p is not None and not (isinstance(spher_p, float) and np.isnan(float(spher_p))):
        sphericity_met = float(spher_p) >= 0.05
    else:
        sphericity_met = True
    result["sphericity_met"] = sphericity_met

    # Use GG p-value if sphericity violated, otherwise uncorrected p
    p_final = float(p_GG) if (not sphericity_met and p_GG is not None) else float(p_uncor) if p_uncor is not None else np.nan
    result["anova_p"]    = float(p_uncor) if p_uncor is not None else np.nan
    result["anova_p_GG"] = float(p_GG) if p_GG is not None else np.nan
    result["significant"]= (p_final < ALPHA) if not np.isnan(p_final) else False

    # ── Post-hoc pairwise (Bonferroni) ───────────────────────────────────────
    if result["significant"]:
        try:
            ph = pg.pairwise_tests(
                data     = df_long,
                dv       = "value",
                within   = "condition",
                subject  = "subject",
                padjust  = "bonf",
            )
            for _, ph_row in ph.iterrows():
                a, b = ph_row.get("A", ""), ph_row.get("B", "")
                # pingouin pairwise_tests uses p_corr / p_unc / hedges (underscores)
                p_adj = float(ph_row.get("p_corr", ph_row.get("p_unc", np.nan)))
                t_val = float(ph_row.get("T", ph_row.get("t", np.nan)))
                d_val = float(ph_row.get("hedges", ph_row.get("cohen-d", np.nan)))
                result["posthoc"].append({
                    "pair"       : f"{a} vs {b}",
                    "T"          : t_val,
                    "p_bonf"     : p_adj,
                    "cohen_d"    : d_val,
                    # p_corr is already Bonferroni-multiplied → compare to ALPHA not BONF_ALPHA
                    "significant": (p_adj < ALPHA) if not np.isnan(p_adj) else False,
                })
        except Exception as exc:
            result["posthoc_error"] = str(exc)

    return result


def pairwise_fallback(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    """
    Paired t-test + Wilcoxon signed-rank test for two conditions only
    (used when baseline data is absent).
    """
    mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
    a, b = vals_a[mask], vals_b[mask]
    n = len(a)
    if n < 5:
        return {"n": n, "error": "insufficient data"}

    t_stat, p_ttest = stats.ttest_rel(a, b)
    try:
        w_stat, p_wilcox = stats.wilcoxon(a, b, zero_method="wilcox")
    except Exception:
        w_stat, p_wilcox = np.nan, np.nan

    diff = a - b
    d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else np.nan

    return {
        "n"          : n,
        "mean_A"     : float(np.mean(a)),
        "mean_B"     : float(np.mean(b)),
        "mean_diff"  : float(np.mean(diff)),
        "t_statistic": float(t_stat),
        "p_ttest"    : float(p_ttest),
        "W_wilcoxon" : float(w_stat),
        "p_wilcoxon" : float(p_wilcox),
        "cohen_d"    : d,
        "significant": float(p_ttest) < ALPHA,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Report Formatting
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(v: Any, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def print_rm_anova_block(res: dict, conds: list[str], indent: str = "  ") -> None:
    print(f"{indent}n (matched subjects) = {res['n_subjects']}")
    print(f"{indent}Condition means ± SD:")
    for c in conds:
        lbl = COND_LABELS.get(c, c)
        m = _fmt(res["means"].get(c))
        s = _fmt(res["stds"].get(c))
        norm_p = res["normality"].get(c, {}).get("p", np.nan)
        norm_flag = "(normal)" if res["normality"].get(c, {}).get("normal") else f"(non-normal, SW p={_fmt(norm_p, 3)})"
        print(f"{indent}  {lbl:20}: {m} ± {s}  {norm_flag}")

    if "error" in res:
        print(f"{indent}[ERROR] {res['error']}")
        return

    print(f"{indent}Mauchly's W = {_fmt(res['mauchly_W'], 4)},  p = {_fmt(res['mauchly_p'], 4)}", end="")
    if res["sphericity_met"] is not None:
        tag = "Sphericity satisfied" if res["sphericity_met"] else "Sphericity VIOLATED → GG correction applied"
        print(f"  [{tag}]")
    else:
        print()

    if not np.isnan(res.get("epsilon_GG", np.nan)):
        print(f"{indent}Greenhouse-Geisser ε = {_fmt(res['epsilon_GG'], 4)}")

    p_report = res["anova_p_GG"] if not (res.get("sphericity_met", True)) else res["anova_p"]
    corr_note = " (GG-corrected)" if not (res.get("sphericity_met", True)) else ""
    print(f"{indent}F({_fmt(res['anova_df_num'],1)}, {_fmt(res['anova_df_den'],1)}) = "
          f"{_fmt(res['anova_F'], 3)},  "
          f"p = {_fmt(p_report, 4)}{corr_note},  "
          f"partial η² = {_fmt(res['partial_eta2'], 3)}")
    sig = "✓ SIGNIFICANT" if res["significant"] else "✗ not significant"
    print(f"{indent}Overall result: {sig}  (α = {ALPHA})")

    if res["significant"] and res["posthoc"]:
        print(f"{indent}Post-hoc pairwise (Bonferroni, α' = {BONF_ALPHA:.4f}):")
        for ph in res["posthoc"]:
            sig2 = "✓" if ph["significant"] else "✗"
            print(f"{indent}  {sig2} {ph['pair']:30}  "
                  f"T={_fmt(ph['T'],3)},  "
                  f"p_bonf={_fmt(ph['p_bonf'],4)},  "
                  f"g={_fmt(ph['cohen_d'],3)}")   # hedges g from pingouin
    elif res["significant"]:
        if "posthoc_error" in res:
            print(f"{indent}[Post-hoc error] {res['posthoc_error']}")


def print_pairwise_block(res: dict, label_a: str, label_b: str, indent: str = "  ") -> None:
    if "error" in res:
        print(f"{indent}[ERROR] {res['error']}")
        return
    print(f"{indent}n = {res['n']},  mean {label_a} = {_fmt(res['mean_A'])},  "
          f"mean {label_b} = {_fmt(res['mean_B'])},  Δ = {_fmt(res['mean_diff'])}")
    print(f"{indent}Paired t: t = {_fmt(res['t_statistic'],3)},  p = {_fmt(res['p_ttest'],4)}  "
          f"|  Wilcoxon W = {_fmt(res['W_wilcoxon'],1)},  p = {_fmt(res['p_wilcoxon'],4)}")
    print(f"{indent}Cohen's d = {_fmt(res['cohen_d'],3)}")
    sig = "✓ SIGNIFICANT" if res["significant"] else "✗ not significant"
    print(f"{indent}{sig}  (α = {ALPHA})")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_dataset(
    dataset_label : str,         # "LongMemEval" or "LoCoMo"
    dataset_key   : str,         # "lme" or "locomo"
    dir_baseline  : str | None,
    dir_bm25      : str | None,
    dir_vector    : str | None,
    output_dir    : Path,
) -> dict[str, Any]:
    """Run the full Section 3.6 analysis for one dataset."""
    print(f"\n{'═'*72}")
    print(f"  {dataset_label}")
    print(f"{'═'*72}")

    # ── Load all conditions ───────────────────────────────────────────────────
    print("\nLoading data …")
    df_base   = load_condition(dir_baseline, "baseline", dataset_key)
    df_bm25   = load_condition(dir_bm25,     "bm25",     dataset_key)
    df_vector = load_condition(dir_vector,   "vector",   dataset_key)

    available: list[str] = []
    frames:    dict[str, pd.DataFrame] = {}
    for name, df in [("baseline", df_base), ("bm25", df_bm25), ("vector", df_vector)]:
        if not df.empty:
            available.append(name)
            frames[name] = df.set_index("question_id")

    print(f"\nAvailable conditions: {available}")

    if len(available) < 2:
        print("  [SKIP] Need at least 2 conditions to run any comparison.")
        return {}

    # ── Three-condition RM-ANOVA or pairwise fallback ─────────────────────────
    all_results: dict[str, Any] = {}
    use_3way = (len(available) == 3)

    if use_3way:
        # Align all three conditions on matched question_ids
        qids = set(frames["baseline"].index) & set(frames["bm25"].index) & set(frames["vector"].index)
    else:
        # Two-condition pairwise
        a_name, b_name = available[0], available[1]
        qids = set(frames[a_name].index) & set(frames[b_name].index)

    print(f"Matched question IDs across conditions: {len(qids)}\n")

    for metric, metric_label in METRICS.items():
        print(f"{'─'*72}")
        print(f"  Metric: {metric_label}")
        print(f"{'─'*72}")

        if use_3way:
            # Build wide-format matched DataFrame
            rows = []
            for qid in sorted(qids):
                row = {"question_id": qid}
                ok  = True
                for cond in CONDITIONS:
                    v = frames[cond].loc[qid, metric] if qid in frames[cond].index else np.nan
                    row[cond] = float(v)
                    if np.isnan(row[cond]):
                        ok = False
                if ok:
                    rows.append(row)
            df_wide = pd.DataFrame(rows)

            res = rm_anova_analysis(df_wide, dv=metric, conditions=CONDITIONS)
            all_results[metric] = res
            print_rm_anova_block(res, CONDITIONS)

        else:
            # Pairwise fallback
            def _safe_get(frame: pd.DataFrame, qid: str, col: str) -> float:
                if qid not in frame.index:
                    return np.nan
                val = frame.at[qid, col]
                # at[] can still return Series if index has duplicates; take first scalar
                if hasattr(val, "__len__"):
                    val = val.iloc[0] if hasattr(val, "iloc") else float(list(val)[0])
                return float(val)

            a_vals = np.array([_safe_get(frames[a_name], qid, metric) for qid in qids], dtype=float)
            b_vals = np.array([_safe_get(frames[b_name], qid, metric) for qid in qids], dtype=float)
            res = pairwise_fallback(a_vals, b_vals, COND_LABELS[a_name], COND_LABELS[b_name])
            all_results[metric] = {"type": "pairwise", "A": a_name, "B": b_name, **res}
            print_pairwise_block(res, COND_LABELS[a_name], COND_LABELS[b_name])

    # ── Save JSON results ─────────────────────────────────────────────────────
    out_path = output_dir / f"{dataset_key}_statistics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_serialise(all_results), f, indent=2)
    print(f"\n  Results saved → {out_path}")

    return all_results


def _serialise(obj: Any) -> Any:
    """Make an object JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return [_serialise(v) for v in obj.tolist()]
    return obj


# ═══════════════════════════════════════════════════════════════════════════════
# Summary Table
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(lme_res: dict, locomo_res: dict) -> None:
    print(f"\n{'═'*72}")
    print("  SUMMARY -- Statistical Significance at alpha = 0.05")
    print(f"{'═'*72}")
    print(f"  {'Metric':<35} {'LME':^20} {'LoCoMo':^20}")
    print(f"  {'─'*35}─{'─'*20}─{'─'*20}")

    for metric, label in METRICS.items():
        def sig_str(res_dict):
            r = res_dict.get(metric, {})
            if not r:
                return "N/A"
            sig = r.get("significant", False)
            p   = r.get("anova_p_GG") or r.get("anova_p") or r.get("p_ttest")
            tag = "✓ sig" if sig else "✗ n.s."
            if p is not None and not (isinstance(p, float) and np.isnan(p)):
                tag += f" (p={p:.3f})"
            return tag

        lme_str    = sig_str(lme_res)
        locomo_str = sig_str(locomo_res)
        print(f"  {label:<35} {lme_str:^20} {locomo_str:^20}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Section 3.6 Statistical Analysis — RM-ANOVA, Mauchly, GG, Bonferroni",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    lme = p.add_argument_group("LongMemEval Result Directories")
    lme.add_argument("--lme-baseline", default=None,
                     help="Baseline results dir for LME (optional; fallback to pairwise if absent)")
    lme.add_argument("--lme-bm25",
                     default="Structured Text Memory System/eval/results/20260610_052508",
                     help="BM25/Sparse results dir for LME")
    lme.add_argument("--lme-vector",
                     default="Vector-Embedded Memory System/eval/results/20260609_092118",
                     help="Vector/Dense results dir for LME")

    loc = p.add_argument_group("LoCoMo Result Directories")
    loc.add_argument("--locomo-baseline", default=None,
                     help="Baseline results dir for LoCoMo (optional)")
    loc.add_argument("--locomo-bm25",
                     default="Structured Text Memory System/eval/results/20260610_190805",
                     help="BM25/Sparse results dir for LoCoMo")
    loc.add_argument("--locomo-vector",
                     default="Vector-Embedded Memory System/eval/results/20260609_015210",
                     help="Vector/Dense results dir for LoCoMo")

    p.add_argument("--output-dir", default="results/statistical",
                   help="Directory to write JSON result files (default: results/statistical)")
    return p


def main() -> None:
    args   = build_parser().parse_args()
    outdir = Path(args.output_dir)

    print("=" * 72)
    print("  STATISTICAL ANALYSIS — Section 3.6")
    print("  H0: mu_baseline = mu_BM25 = mu_dense  (one-way RM-ANOVA per metric)")
    print(f"  alpha = {ALPHA},  Bonferroni alpha' = {BONF_ALPHA:.4f}")
    print(f"  pingouin: {'available ✓' if PINGOUIN_OK else 'NOT AVAILABLE ✗'}")
    print("=" * 72)

    lme_res = analyse_dataset(
        dataset_label="LongMemEval",
        dataset_key  ="lme",
        dir_baseline = args.lme_baseline,
        dir_bm25     = args.lme_bm25,
        dir_vector   = args.lme_vector,
        output_dir   = outdir,
    )

    locomo_res = analyse_dataset(
        dataset_label="LoCoMo",
        dataset_key  ="locomo",
        dir_baseline = args.locomo_baseline,
        dir_bm25     = args.locomo_bm25,
        dir_vector   = args.locomo_vector,
        output_dir   = outdir,
    )

    print_summary_table(lme_res, locomo_res)

    print(f"\n  All statistics saved to: {outdir}/")
    print("  To add baseline once run:")
    print("    python statistical_analysis.py \\")
    print("        --lme-baseline    results/baseline/<run_id> \\")
    print("        --locomo-baseline results/baseline/<run_id>")


if __name__ == "__main__":
    main()

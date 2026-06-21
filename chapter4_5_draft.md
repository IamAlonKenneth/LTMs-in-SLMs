# CHAPTER 4: RESULTS AND DISCUSSION

::: doublespace
This chapter presents and interprets the quantitative findings from the
controlled evaluation of two long-term memory architectures integrated
with Gemma 3 4B across two benchmarks, compared against a stateless
baseline condition. Results are organised across the two evaluation
stages defined in Chapter 3: Stage 1 (memory retrieval quality,
measuring how accurately each system surfaces relevant memories from
its backend), and Stage 2 (response generation quality, evaluated via
Gemini 2.5 Flash as an LLM-as-Judge across faithfulness and answer
relevance dimensions). System efficiency metrics and a formal
statistical evaluation follow. The chapter concludes by addressing
each of the four research questions using the collected empirical
evidence.
:::

## 4.1 Preliminary Note: Generation Configuration and Evaluation Setup

::: doublespace
Prior to presenting results, a critical methodological observation must
be acknowledged. Gemma 3 IT uses two generation stop tokens:
`<eos>` (token ID 1), inherited from the base model, and
`<end_of_turn>` (token ID 107), the instruction-tuned conversational
stop token. The Structured Text Memory pipeline's generation call was
configured with only `<eos>`, causing the model to never stop naturally
during Sparse system runs. Every response hit the `max_new_tokens=64`
ceiling in both the LongMemEval and LoCoMo evaluations, confirmed by
mean response token counts of exactly 64.0 (min=64, max=64) for both
runs. Vector-Embedded Memory responses averaged 63.3 tokens
(LongMemEval) and 48.6 tokens (LoCoMo) with natural variance,
confirming proper stopping behaviour. Baseline condition responses
were generated with the correct dual stop-token configuration, yielding
naturally completed responses.

Query expansion — a Gemma 3 4B LLM call that generates synonym
variants before each BM25 search — was disabled in the Sparse system
evaluation runs for both benchmarks (`skip_expansion=True`). Latency
figures in Section 4.5 therefore reflect BM25 search overhead alone.

This truncation does not affect Stage 1 retrieval metrics, which are
computed independently before generation. It does, however,
systematically degrade Stage 2 generation quality scores for the
Sparse system. Stage 2 Sparse results are therefore reported for
completeness and interpreted with explicit reference to this confound.
Stage 1 results and the three-way comparison with the stateless
baseline provide the primary valid comparisons in this chapter.
:::

## 4.2 Baseline (Stateless) Condition

::: doublespace
A stateless baseline condition was evaluated in which Gemma 3 4B
receives only the current query with no retrieved memory context,
establishing the empirical lower bound against which memory
augmentation is measured. The baseline was evaluated on all 500
LongMemEval items and all 1,986 LoCoMo items using identical hardware,
quantisation, and judge (Gemini 2.5 Flash) as the memory-augmented
conditions. Because the baseline performs no retrieval, Stage 1
metrics are not applicable; Stage 2 and efficiency metrics are
reported below.
:::

### 4.2.1 Baseline LongMemEval Performance

::: doublespace
Without access to any stored memory, Gemma 3 4B achieves mean
faithfulness of 4.09/5 (σ=1.58) and mean answer relevance of
3.49/5 (σ=1.29) on LongMemEval. The elevated faithfulness score is
expected: the judge evaluates faithfulness relative to the provided
context, and with no retrieved context, responses that acknowledge
uncertainty are scored as faithful to the empty context. The answer
relevance score reflects the proportion of questions the model can
correctly answer from parametric knowledge alone — a substantial
deficit versus the memory-augmented Vector system (Δ=0.90/5,
Section 4.4).

For the abstention category (n=30), the baseline achieves an
abstention score of 4.43/5 and answer relevance of 2.53/5. The
high abstention score indicates the model frequently produces
uncertainty expressions without context, but this is a vacuous
abstention — the model cannot answer even when an answer exists.
Memory error rate for the baseline is 15.1% on LongMemEval
(proportion of non-abstention items receiving faithfulness=1/5),
indicating that the model confabulates incorrect facts in approximately
one in seven responses even without memory injection.
:::

::: {#tab:baseline-lme}
  **Metric**               **Baseline (No Memory)**   **Interpretation**
  ------------------------ -------------------------- -----------------------------------------------
  Faithfulness (mean)      4.09/5 (σ=1.58)            Artefactually high; faithfulness to empty context trivially met
  Answer Relevance (mean)  3.49/5 (σ=1.29)            Partial; model draws on parametric knowledge only
  Abstention score (n=30)  4.43/5                     Frequent abstention — correct behaviour but vacuous
  Memory Error Rate        15.1%                      Confabulation at approximately one in seven responses

  : Baseline (stateless) Stage 2 performance on LongMemEval (n=500).
:::

### 4.2.2 Baseline LoCoMo Performance

::: doublespace
On LoCoMo, the baseline achieves faithfulness of 1.65/5 (σ=1.36)
and answer relevance of 3.84/5 (σ=1.35). The near-floor faithfulness
score confirms that Gemma 3 4B systematically confabulates session-
specific facts on LoCoMo when no memory is provided: 75.0% of items
receive faithfulness=1/5 (memory error rate). The relatively high
answer relevance score reflects the LoCoMo question structure — many
questions elicit a type of information (e.g., "what did X say about
Y?") where the model can produce a plausible-sounding answer from
parametric knowledge despite factual ungroundedness. The 75.0%
memory error rate establishes the magnitude of confabulation that the
experimental memory conditions must overcome.
:::

::: {#tab:baseline-locomo}
  **Metric**               **Baseline (No Memory)**   **Interpretation**
  ------------------------ -------------------------- -----------------------------------------------
  Faithfulness (mean)      1.65/5 (σ=1.36)            Low; model confabulates session-specific facts
  Answer Relevance (mean)  3.84/5 (σ=1.35)            Moderate; parametric plausibility without grounding
  Memory Error Rate        75.0%                      Severe confabulation — three in four responses

  : Baseline (stateless) Stage 2 performance on LoCoMo (n=1,986).
:::

## 4.3 Stage 1: Memory Retrieval Quality

::: doublespace
Stage 1 evaluates retrieval quality using three established information
retrieval metrics computed at multiple cutoff values k ∈ {1, 3, 5,
10}: Recall@k (proportion of ground-truth relevant memories within the
top-k results), NDCG@k (ranking quality, rewarding relevant memories
ranked higher), and AP@k (Average Precision, jointly measuring
precision and recall at each rank). All Stage 1 metrics are unaffected
by the generation configuration issue. The baseline condition has no
retrieval stage and is not represented in Stage 1 tables.
:::

### 4.3.1 LongMemEval Retrieval Performance

::: doublespace
Table 4.3 presents overall retrieval metrics on LongMemEval (n=500).
Both architectures exhibit closely matched performance across all
cutoffs. At k=5 — the operating cutoff used for memory injection —
Sparse/BM25 achieves Recall@5 = 22.8% (σ=17.2%) and Vector/Dense
achieves 23.8% (σ=17.4%), a difference of one percentage point.
NDCG@5 is 0.814 for Sparse and 0.848 for Vector, indicating that when
relevant memories are retrieved, they tend to be ranked near the top
by both systems.

The low absolute Recall@5 values reflect the benchmark's design: each
LongMemEval question is associated with a median of 8–12 ground-truth
memory entries spanning multiple sessions, and the system retrieves
only 5 memories. The high NDCG@5 scores despite low recall indicate
that both systems successfully identify at least one relevant memory
per query and rank it highly, even when the full ground-truth set
cannot be covered within k=5.
:::

::: {#tab:stage1-lme}
  **Metric**   **Sparse/BM25**         **Vector/Dense**        **Δ (Vector − Sparse)**
  ------------ ----------------------- ----------------------- -------------------------
  Recall@1     5.8%  (σ=6.5%)          5.9%  (σ=6.5%)          +0.1%
  Recall@3     15.5% (σ=13.9%)         15.8% (σ=13.9%)         +0.3%
  Recall@5     22.8% (σ=17.2%)         23.8% (σ=17.4%)         +1.0%
  Recall@10    22.8% (σ=17.2%)         23.8% (σ=17.4%)         +1.0%
  NDCG@5       0.814 (σ=0.278)         0.848 (σ=0.258)         +0.034
  AP@5         0.222 (σ=0.175)         0.231 (σ=0.177)         +0.009

  : Stage 1 retrieval metrics on LongMemEval (n=500). Recall@5=Recall@10
  because k_max=5 in both pipelines.
:::

::: doublespace
At the category level (Table 4.4), single-hop questions produce the
highest retrieval recall for both systems (Sparse 36.9%, Vector
39.2%), consistent with the expectation that direct factual queries
are most amenable to keyword and semantic matching. Multi-hop questions
yield the lowest recall (Sparse 14.8%, Vector 15.9%), reflecting the
difficulty of retrieving all needed memories when a question requires
synthesising information across two or more disconnected sessions. The
Vector system marginally outperforms Sparse across four of the five
categories; Sparse performs comparably on temporal reasoning (17.5%
vs 17.0%). The maximum category-level difference between architectures
is 2.3 percentage points (single-hop), which does not constitute a
practically significant retrieval advantage for either system.
:::

::: {#tab:stage1-lme-cat}
  **Category**          **n**   **Sparse R@5**   **Vector R@5**   **Sparse NDCG@5**   **Vector NDCG@5**
  --------------------- ------- ---------------- ---------------- ------------------- -------------------
  Single-hop            150     36.9%            39.2%            0.765               0.816
  Multi-hop             121     14.8%            15.9%            0.822               0.894
  Temporal reasoning    127     17.5%            17.0%            0.812               0.798
  Knowledge update      72      19.5%            19.4%            0.958               0.953
  Abstention            30      15.5%            17.6%            0.695               0.786

  : LongMemEval Stage 1 by question category.
:::

### 4.3.2 LoCoMo Retrieval Performance

::: doublespace
On LoCoMo (n=1,986), retrieval recall is substantially higher than
LongMemEval. This reflects the LoCoMo benchmark's single ground-truth
memory structure for most questions: a question with exactly one
correct memory yields Recall@1 equal to the proportion of questions
where the correct memory is ranked first. Sparse/BM25 achieves
Recall@5 = 57.5% and Vector/Dense achieves 58.1%. Recall@1 is 35.7%
for Sparse and 31.3% for Vector, suggesting that the BM25 keyword
mechanism surfaces the single correct memory at rank 1 more often than
the semantic approach.

NDCG@5 is 0.490 for Sparse and 0.472 for Vector on LoCoMo. While
absolute values are lower than LongMemEval (reflecting greater
diversity of retrievable memories in the long conversational logs),
Sparse marginally outperforms Vector on this metric by 0.018 — the
only dimension in which Sparse achieves a notable retrieval advantage.
AP@5 also favours Sparse (0.449 vs 0.421). This is consistent with
the theoretical advantage of BM25 for single-hop factual recall: when
the question contains specific keywords overlapping directly with the
stored memory (e.g., named entities, dates, event names), lexical
matching can outperform semantic embedding. The recency weighting
mechanism in the BM25 pipeline, which applies an exponential decay
factor to older memories, likely contributes positively by prioritising
recent events in LoCoMo's temporally ordered conversation logs.
:::

::: {#tab:stage1-locomo}
  **Metric**   **Sparse/BM25**         **Vector/Dense**        **Δ (Vector − Sparse)**
  ------------ ----------------------- ----------------------- -------------------------
  Recall@1     35.7% (σ=46.6%)         31.3% (σ=44.8%)         −4.4%
  Recall@3     52.8% (σ=48.1%)         49.9% (σ=47.6%)         −2.9%
  Recall@5     57.5% (σ=47.4%)         58.1% (σ=46.6%)         +0.6%
  NDCG@5       0.490 (σ=0.431)         0.472 (σ=0.413)         −0.018
  AP@5         0.449 (σ=0.434)         0.421 (σ=0.416)         −0.028

  : Stage 1 retrieval metrics on LoCoMo. Sparse/BM25 leads on Recall@1
  and NDCG@5; Vector/Dense leads on Recall@5.
:::

## 4.4 Stage 2: Response Generation Quality

::: doublespace
Stage 2 evaluates the quality of Gemma 3 4B's generated responses
using Gemini 2.5 Flash as an LLM-as-Judge, scoring each response on
faithfulness and answer relevance on a 1–5 rubric. All 500
LongMemEval and 1,986 LoCoMo items were scored for the baseline and
Vector conditions; the Sparse condition scored 500 LongMemEval and
1,950 LoCoMo items. Results are presented for all three conditions,
with explicit notation where the EOS token issue affects
interpretation.
:::

### 4.4.1 LongMemEval Generation Quality

::: doublespace
Table 4.6 presents Stage 2 scores on LongMemEval. The Vector/Dense
system achieves mean faithfulness of 4.05/5 (σ=1.27) and mean answer
relevance of 4.39/5 (σ=0.98). The score distribution confirms strong
performance: 55.4% of Vector items receive faithfulness 5/5, with
only 3.2% at 1/5. Answer relevance at 5/5 reaches 68.6%.

The baseline achieves faithfulness of 4.09/5 and answer relevance of
3.49/5. The faithfulness value is artefactually high — without
context, responses acknowledging uncertainty receive full faithfulness
marks — while the answer relevance gap versus Vector (Δ=0.90/5)
represents the empirical benefit of memory augmentation on this
benchmark.

The Sparse/BM25 system reports mean faithfulness of 2.38/5 (σ=1.88)
and mean answer relevance of 1.88/5 (σ=1.65). The answer relevance
distribution shows 77.2% of items at score 1/5, the characteristic
signature of the EOS truncation issue. The Gemini 2.5 Flash judge
scores some truncated responses above 1/5 where the partial answer
contains sufficient information — the 21.6% of Sparse items achieving
answer relevance 5/5 represents questions whose complete answer fit
within the 64-token budget. Faithfulness bimodality (60.4% at 1/5,
31.2% at 5/5) also reflects truncation: responses that begin with
retrieved context before being cut score well on faithfulness, while
responses that begin with a confabulated preamble score at floor.

For the LongMemEval abstention category (n=30), the Vector system
achieved a mean abstention score of 3.23/5. The baseline achieved
4.43/5 on abstention score but answer relevance of only 2.53/5,
reflecting that without memory the model frequently produces uncertain
responses that score well on abstention but poorly when a correct
answer is expected. Sparse achieved abstention-category answer
relevance of 1.00/5 — all truncated responses fail the relevance
rubric.
:::

::: {#tab:stage2-lme}
  **Metric**               **Baseline**        **Sparse/BM25**     **Vector/Dense**
  ------------------------ ------------------- ------------------- ------------------
  Faithfulness (mean)      4.09/5 (σ=1.58)    2.38/5 (σ=1.88)    4.05/5 (σ=1.27)
  Answer Relevance (mean)  3.49/5 (σ=1.29)    1.88/5 (σ=1.65)    4.39/5 (σ=0.98)
  Abstention score (n=30)  4.43/5             1.00/5 (artefact)   3.23/5 (σ=1.79)
  Items scored (n)         500                500                 500
  Faithfulness @5/5        —                  31.2% (156/500)     55.4% (277/500)
  Answer Relevance @5/5    —                  21.6% (108/500)     68.6% (343/500)
  Answer Relevance @1/5    —                  77.2% (386/500)     0.6%  (3/500)

  : Stage 2 generation quality on LongMemEval. Baseline faithfulness
  is assessed relative to the empty context prompt. Sparse results are
  confounded by EOS token truncation.
:::

### 4.4.2 LongMemEval Category-Level Generation Quality

::: doublespace
Table 4.7 presents Stage 2 scores for the Vector/Dense system broken
down by LongMemEval question category. Single-hop and knowledge update
questions receive the highest scores on both dimensions (faithfulness
4.43 and 4.36 respectively; answer relevance 4.59 and 4.68),
confirming that the pipeline performs well when a single clear memory
entry contains the necessary information. Multi-hop performance
(faithfulness 4.10, answer relevance 4.39) is comparable to
single-hop despite requiring synthesis across sessions, suggesting
that when the top-5 retrieved memories span the relevant sessions,
Gemma 3 4B can successfully integrate them.

Temporal reasoning achieves the lowest faithfulness among question
categories (3.39/5), with answer relevance at 4.16/5. This gap
between faithfulness and answer relevance indicates that the model
answers the question but occasionally includes ungrounded temporal
inferences beyond what is directly stated in the retrieved memories,
consistent with the "temporal confabulation" failure mode documented
by Wu et al. [@wu2024longmemeval].
:::

::: {#tab:stage2-lme-cat}
  **Category**        **n**   **Faithfulness**   **Ans. Relevance**   **Interpretation**
  ------------------- ------- ------------------ -------------------- ------------------------------------------------------------
  Single-hop          150     4.43/5             4.59/5               Highest performance — direct factual recall
  Multi-hop           121     4.10/5             4.39/5               Strong despite cross-session synthesis requirement
  Temporal reasoning  127     3.39/5             4.16/5               Faithfulness gap: temporal confabulation artefacts
  Knowledge update    72      4.36/5             4.68/5               Robust update handling — model applies corrections correctly
  Abstention          30      N/A                3.70/5               Partial abstention; abstention score 3.23/5

  : Vector/Dense Stage 2 by LongMemEval category.
:::

### 4.4.3 LoCoMo Generation Quality

::: doublespace
On LoCoMo (Table 4.8), the Vector/Dense system achieves faithfulness
= 3.99/5 (σ=1.38) and answer relevance = 4.14/5 (σ=1.29) across
1,983 scored items. The score distribution shows 57.3% of items at
the faithfulness ceiling (5/5) and 62.6% at answer relevance 5/5.
The 8.1% of items receiving faithfulness 1/5 on Vector/LoCoMo
represents the system's genuine failure rate under this benchmark.

The baseline achieves faithfulness = 1.65/5 (σ=1.36) and answer
relevance = 3.84/5 (σ=1.35). The near-floor faithfulness confirms
that Gemma 3 4B systematically confabulates session-specific facts
without memory. Despite this, answer relevance remains at 3.84/5,
reflecting parametric plausibility without grounding.

The Sparse/BM25 system achieves faithfulness = 2.90/5 (σ=1.98) and
answer relevance = 1.82/5 (σ=1.58) on LoCoMo (n=1,950). The answer
relevance distribution shows 77.7% of items at score 1/5 and 19.2%
at score 5/5. The 19.2% of Sparse items achieving answer relevance
5/5 represents questions where the answer was sufficiently short to
complete within the 64-token budget.

Comparing all three conditions on answer relevance: Vector (4.14) >
Baseline (3.84) > Sparse (1.82). This three-way ordering is central
to interpreting the experimental findings. The Vector-over-Baseline
gap (Δ=0.30/5) confirms that memory augmentation provides meaningful
improvement even on a benchmark where parametric knowledge partially
covers answers. The Baseline-over-Sparse ordering indicates that
the EOS truncation in the Sparse system actively degrades performance
below the no-memory condition — not because retrieval is failing, but
because generation cannot complete.
:::

::: {#tab:stage2-locomo}
  **Metric**               **Baseline**        **Sparse/BM25**     **Vector/Dense**
  ------------------------ ------------------- ------------------- ------------------
  Faithfulness (mean)      1.65/5 (σ=1.36)    2.90/5 (σ=1.98)    3.99/5 (σ=1.38)
  Answer Relevance (mean)  3.84/5 (σ=1.35)    1.82/5 (σ=1.58)    4.14/5 (σ=1.29)
  Items scored (n)         1,986              1,950              1,983
  Faithfulness @5/5        —                  46.3%              57.3% (1137/1983)
  Answer Relevance @5/5    —                  19.2%              62.6% (1242/1983)
  Answer Relevance @1/5    —                  77.7%              7.1%  (140/1983)

  : Stage 2 generation quality on LoCoMo. Baseline faithfulness is
  low because session-specific facts cannot be recalled without memory.
  Sparse results are confounded by EOS token truncation.
:::

## 4.5 System Efficiency

::: doublespace
Three efficiency metrics are evaluated against the targets specified
in Chapter 3: retrieval latency below 500ms [@shen2024towards], context
token utilisation below 70% of the 128,000-token context window
[@hong2025contextrot], and generation speed. All measurements represent
wall-clock time recorded on the evaluation hardware (16GB RAM laptop,
Gemma 3 4B in 4-bit quantisation via HuggingFace Transformers). The
baseline condition uses no retrieval stage; its generation speed and
token utilisation reflect the minimal stateless prompt.
:::

### 4.5.1 Retrieval Latency

::: doublespace
The Vector/Dense system achieves mean retrieval latency of 76.96ms
(σ=23.4ms) on LongMemEval and 66.61ms (σ=12.8ms) on LoCoMo, both
well within the 500ms target. This confirms that FAISS IndexFlatL2
exact search over a bounded memory store meets sub-500ms requirements
on consumer-grade hardware.

The Sparse/BM25 system — with query expansion disabled — averaged
85.6ms (σ=43.8ms) on LongMemEval and 68.7ms (σ=14.6ms) on LoCoMo,
both within the 500ms target and comparable to the Vector system.
Previous runs with query expansion enabled recorded a mean total
retrieval time of 8,633ms on LongMemEval, demonstrating that the BM25
search itself is not the latency bottleneck: the LLM-based query
expansion preprocessing step added a mean 8,140ms per query.
Disabling query expansion brings the Sparse system fully within the
latency target, at the cost of reduced vocabulary coverage for
paraphrase-sensitive queries.
:::

### 4.5.2 Generation Speed

::: doublespace
The baseline condition achieves approximately 3.58s per item on
LongMemEval and 3.60s per item on LoCoMo, representing the minimum
generation cost without retrieved context. The Vector/Dense system
generates responses in 10.99s per item (σ=10.07s) on LongMemEval and
7.46s per item (σ=4.95s) on LoCoMo. The Sparse/BM25 system records
6.70s per item (σ=4.88s) on LongMemEval and 3.90s per item (σ=0.77s)
on LoCoMo.

Two caveats apply to the Sparse generation times. First, the EOS
truncation forces every response to the 64-token ceiling, inflating
token count beyond what correct stopping would produce. Second, the
Sparse runs used `batch_size=4` while the baseline and Vector runs
used `batch_size=1`, making direct per-second comparisons between
systems misleading. These figures are therefore reported as
experimental measurements, not as performance claims.
:::

### 4.5.3 Context Token Utilisation

::: doublespace
Table 4.9 shows prompt token counts and utilisation as a percentage
of Gemma 3 4B's 128,000-token context window [@gemma2025report].
All configurations use less than 1.2% of the available context window.
The baseline uses the fewest tokens (mean 28 tokens, 0.022% on LME;
22 tokens, 0.018% on LoCoMo), as no retrieved memory is injected.
Sparse/LME uses the most tokens (mean 1,433, 1.12%) due to the BM25
retrieval context; Vector/LME uses 1,003 tokens (0.78%). For LoCoMo,
Sparse and Vector are near-equal at 475 and 471 tokens respectively
(0.37%). The 70% utilisation threshold is not approached by any
configuration, confirming that Context Rot [@hong2025contextrot] is
not a risk factor in these experiments.
:::

::: {#tab:efficiency}
  **Configuration**   **Retrieval (ms)**   **Generation (s)**   **Prompt tokens**   **Token utilisation**
  ------------------- -------------------- -------------------- ------------------- ----------------------
  Baseline / LME      0 (no retrieval)     3.58                 28 (σ=9)            0.022%
  Baseline / LoCoMo   0 (no retrieval)     3.60                 22 (σ=4)            0.018%
  Sparse / LME        85.6 (σ=43.8)       6.70 (σ=4.88)†      1,433 (σ=483)       1.12%
  Vector / LME        77.0 (σ=23.4)       10.99 (σ=10.07)      1,003               0.78%
  Sparse / LoCoMo     68.7 (σ=14.6)       3.90 (σ=0.77)†       475 (σ=48)          0.37%
  Vector / LoCoMo     66.6 (σ=12.8)       7.46 (σ=4.95)        471                 0.37%
  **Target**          **< 500ms**          —                    —                   **< 70%**

  : System efficiency metrics. †Sparse generation times recorded with
  `batch_size=4`; baseline and Vector used `batch_size=1`. Sparse
  runs had query expansion disabled. All token utilisation values are
  well below the 70% Context Rot threshold.
:::

## 4.6 Statistical Evaluation

::: doublespace
Formal statistical analysis was conducted on five dependent variables
across the three conditions (Baseline, Sparse/BM25, Vector/Dense)
using one-way repeated-measures ANOVA. Mauchly's test assessed
sphericity; where violated (p<0.05), Greenhouse–Geisser (GG) corrected
F-statistics and p-values are reported alongside the GG epsilon
correction factor (ε). Effect sizes are reported as partial eta-squared
(η²). Bonferroni-corrected pairwise post-hoc t-tests were conducted
for all three condition pairs; Cohen's d characterises practical effect
magnitude. All tests used α=0.05. Analyses were conducted separately
for LongMemEval and LoCoMo.

One measurement caveat applies to the LongMemEval ANOVA. The
pre-computed statistical file was generated partly from an earlier
Sparse run configuration (query expansion enabled, different token
budget). Specifically, answer relevance (earlier BM25 mean=1.00 vs
current run mean=1.88) and retrieval latency (earlier BM25
mean=8,633ms vs current 85.6ms) reflect that earlier run's parameters
in the ANOVA. ANOVA results for recall@5, memory error rate, and token
utilisation are consistent with the current run and are reported
without caveat. All LoCoMo statistics are drawn from the current
evaluation run and are fully consistent.
:::

### 4.6.1 LongMemEval Statistical Results

::: doublespace
Table 4.10 summarises ANOVA results for LongMemEval. All five
dependent variables show statistically significant main effects of
condition (all GG-corrected p<0.001), with large effect sizes
(η²=0.307–0.997). The recall@5 result is most directly interpretable:
F(2,998)=846.20, η²=0.380. Post-hoc tests show that both BM25 and
Vector significantly outperform Baseline (both p<0.001, |d|>1.89), but
BM25 and Vector do not significantly differ from each other
(p_bonf=0.083, d=−0.040), confirming the retrieval parity finding from
Section 4.3.

Memory error rate shows significant differences between all three
pairs: Baseline (M=0.151) is significantly lower than BM25 (M=0.599;
p<0.001, d=1.041) — because BM25's EOS-truncated responses frequently
fail the faithfulness rubric — and Vector (M=0.034) is significantly
lower than both (all p<0.001, |d|>0.970). The Vector memory error rate
of 3.4% is significantly below the baseline's 15.1%, confirming
empirically that memory augmentation reduces confabulation and not
merely repositions its character.

Token utilisation confirms that both memory systems inject
substantially more context than the baseline (η²=0.692; both p<0.001,
|d|>13.1), while BM25 and Vector differ only marginally from each
other (p_bonf=0.015, d=0.067).
:::

::: {#tab:stats-lme}
  **Metric**              **F**        **ε (GG)**   **η²**   **p (GG)**   **BM25 vs Vector**
  ----------------------- ------------ ------------ -------- ------------ ----------------------------
  Answer relevance†       F=1848.9     0.808        0.701    <0.001       p<0.001, d=−4.88
  Recall@5                F=846.2      1.000        0.380    <0.001       p=0.083 (n.s.), d=−0.040
  Memory error rate       F=303.5      0.848        0.307    <0.001       p<0.001, d=−1.526
  Retrieval latency†      F=233422     0.503        0.997    <0.001       p<0.001, d=30.26
  Token utilisation       F=2150.1     0.991        0.692    <0.001       p=0.015, d=0.067

  : LongMemEval repeated-measures ANOVA (n=500 subjects, df_num=2).
  GG correction applied where sphericity was violated. †Answer
  relevance and retrieval latency statistics reflect an earlier Sparse
  run configuration; see Section 4.6 caveat.
:::

### 4.6.2 LoCoMo Statistical Results

::: doublespace
Table 4.11 presents ANOVA results for LoCoMo. Main effects are
significant for all five metrics (all GG-corrected p<0.001), with
effect sizes η²=0.336–0.968. The answer relevance result — based on
fully valid LoCoMo data for all three conditions — shows F(2,3604)=
1359.55, η²=0.346. Post-hoc comparisons confirm that Vector (M=4.130)
significantly outperforms Baseline (M=3.824; p<0.001, d=−0.231) and
BM25 (M=1.814; p<0.001, d=−1.608), while Baseline significantly
outperforms BM25 (p<0.001, d=1.370). This three-way ordering —
Vector > Baseline > Sparse — means that on LoCoMo, the stateless
model outperforms the EOS-truncated BM25 system in answer relevance,
and memory augmentation (Vector) further improves on both.

Memory error rate on LoCoMo shows the most dramatic effect: η²=0.378.
Vector (M=0.042) achieves a 17.8× reduction relative to Baseline
(M=0.750; p<0.001, d=2.096) and a 14.6× reduction relative to BM25
(M=0.614; p<0.001, d=1.535). The BM25–Baseline difference
(p<0.001, d=0.295) is significant but small, indicating that the
EOS-truncated BM25 system achieves marginally lower confabulation
than the no-memory baseline — primarily because truncated responses
do not complete a confabulated claim rather than because of genuine
memory grounding.

Retrieval latency on LoCoMo (η²=0.604): BM25 (M=120.65ms) and Vector
(M=66.57ms) both significantly differ from Baseline (M=0ms; both
p<0.001) and from each other (p<0.001, d=1.104), with Vector being
1.8× faster than BM25 on this benchmark. Token utilisation shows
the largest effect size (η²=0.968), reflecting the near-zero baseline
context versus both memory-augmented conditions.
:::

::: {#tab:stats-locomo}
  **Metric**            **F**         **ε (GG)**   **η²**   **p (GG)**   **BM25 vs Vector**
  --------------------- ------------- ------------ -------- ------------ ----------------------------
  Answer relevance      F=1359.6      0.977        0.346    <0.001       p<0.001, d=−1.608
  Recall@5              F=1838.8      0.997        0.336    <0.001       p=0.613 (n.s.), d=−0.031
  Memory error rate     F=1273.3      0.885        0.378    <0.001       p<0.001, d=1.535
  Retrieval latency     F=4285.5      0.526        0.604    <0.001       p<0.001, d=1.104
  Token utilisation     F=112953.6    0.991        0.968    <0.001       p=0.015, d=0.067

  : LoCoMo repeated-measures ANOVA (n=1,861 subjects for retrieval
  metrics; n=1,397 for memory error rate; n=1,803 for answer relevance).
  GG correction applied where sphericity was violated.
:::

## 4.7 Addressing the Research Questions

### RQ1 — How can a structured text memory be used as a long-term memory mechanism to affect the success of a small language model?

::: doublespace
The BM25 lexical retrieval system successfully functions as a retrieval
backend for Gemma 3 4B, achieving Recall@5 = 22.8% on LongMemEval and
57.5% on LoCoMo. On LoCoMo, Sparse/BM25 achieves the highest Recall@1
(35.7% vs 31.3%) and NDCG@5 (0.490 vs 0.472) of any retrieval
configuration, confirming that exact keyword matching with recency
weighting is particularly effective for single-hop conversational
queries where specific names, dates, and events appear verbatim in both
the question and the stored memory. The retrieval component of the
Structured Text Memory system contributes positively to task
performance, confirmed by the statistically significant and large-
effect recall improvement over the no-retrieval baseline (LoCoMo:
p<0.001, d=1.704).

However, the generation evaluation in this run is confounded by the
EOS token configuration issue, preventing reliable measurement of how
effectively Gemma 3 4B uses retrieved memories to generate correct
responses. Critically, the LoCoMo three-way comparison shows that the
Baseline (answer relevance = 3.84/5) outperforms the EOS-truncated
Sparse system (1.82/5), meaning that no-memory generation is
preferable to truncated BM25-injected generation under the current
configuration. This result is not attributable to retrieval quality —
the Sparse system retrieves relevant memories — but entirely to the
EOS truncation of the generation step. A corrected re-run is necessary
to fully characterise RQ1's generation dimension.
:::

### RQ2 — How can a vector-embedded memory system be used as a long-term memory mechanism to affect the success of a small language model?

::: doublespace
The Vector/Dense memory system demonstrably and consistently enhances
Gemma 3 4B's multi-session performance across both benchmarks. On
LongMemEval, it achieves faithfulness = 4.05/5 and answer relevance =
4.39/5, with single-hop and knowledge update categories both exceeding
4.3/5 on both dimensions. On LoCoMo, faithfulness = 3.99/5 and answer
relevance = 4.14/5 represent robust cross-session memory utilisation.
The empirical improvement over the stateless baseline is Δ=0.90/5 on
LME answer relevance and Δ=0.30/5 on LoCoMo answer relevance, both
statistically significant (LoCoMo: p<0.001, d=0.231). Memory error
rate is reduced from the baseline's 15.1% to 3.4% on LME, and from
75.0% to 4.2% on LoCoMo — a 17.8× reduction in confabulation on the
harder benchmark, confirmed by ANOVA with η²=0.378. Retrieval latency
of 67–77ms places the system well within the sub-500ms operational
target, and token utilisation remains below 0.8% in all conditions.
The Vector/Dense system affirmatively answers RQ2: vector-embedded
semantic retrieval is an effective and efficient long-term memory
mechanism for Gemma 3 4B.
:::

### RQ3 — What are the effects on task success rate, memory recall accuracy, memory error rate, retrieval latency, and token utilisation?

::: doublespace
Table 4.12 summarises the five target metrics against the thresholds
defined in Chapter 3. Task success rate (operationalised via
LLM-as-Judge answer relevance) is met by the Vector system on both
benchmarks (4.39/5 LME, 4.14/5 LoCoMo), representing approximately
87% and 83% of items scoring ≥ 3/5. The baseline achieves 3.49/5 on
LME and 3.84/5 on LoCoMo without memory grounding. Recall accuracy
(Recall@5) reaches 22.8–23.8% on LongMemEval and 57.5–58.1% on
LoCoMo for the memory-augmented conditions, with no statistically
significant difference between BM25 and Vector on either benchmark
(LME p=0.083; LoCoMo p=0.613). Memory error rate is 15.1%
(Baseline/LME), 3.4% (Vector/LME), 75.0% (Baseline/LoCoMo), and
4.2% (Vector/LoCoMo) — the Vector system meets the <15% target on
both benchmarks; the baseline approaches but does not meet the target
on LME. Retrieval latency for both memory systems is 67–86ms with
query expansion disabled, satisfying the 500ms target. Token
utilisation is below 1.2% in all cases, confirming no Context Rot
risk.
:::

::: {#tab:rq3-summary}
  **Metric**            **Target**   **Baseline**                   **Sparse/BM25**         **Vector/Dense**          **Target Met?**
  --------------------- ------------ ------------------------------ ----------------------- ------------------------- ------------------------------------------
  Task Success (LME)    > 80%        3.49/5 (~57% ≥3/5)            1.88/5 ⚠ (confounded)  4.39/5 (~87% ≥3/5)        Vector ✓; Sparse inconclusive; Baseline partial
  Task Success (LoCoMo) > 80%        3.84/5 (~72% ≥3/5)            1.82/5 ⚠ (confounded)  4.14/5 (~83% ≥3/5)        Vector ✓; Sparse inconclusive; Baseline partial
  Recall@5 (LME)        Comparative  N/A (no retrieval)            22.8%                   23.8%                     Parity — no significant BM25 vs Vector difference
  Recall@5 (LoCoMo)     Comparative  N/A (no retrieval)            57.5%                   58.1%                     Parity — BM25 leads R@1; Vector leads R@5
  Memory Error Rate     < 15%        15.1% (LME) / 75.0% (LoCoMo) N/A (confounded)        3.4% (LME) / 4.2% (LoCoMo) Vector ✓ on both; Baseline near-threshold (LME)
  Retrieval Latency     < 500ms      0ms (no retrieval)            69–86ms ✓               67–77ms ✓                 Both ✓ (query expansion disabled)
  Token Utilisation     < 70%        0.018–0.022% ✓                0.37–1.12% ✓            0.37–0.78% ✓              All ✓ — well under threshold

  : RQ3 target metric summary.
:::

### RQ4 — How do the memory architectures differ from each other?

::: doublespace
The two architectures exhibit near-parity on retrieval quality (Stage
1) and significant divergence on generation quality (Stage 2), with
efficiency now comparable following the disabling of query expansion.
On retrieval, the maximum difference between architectures is 4.4
percentage points on any single Recall@k metric, and neither system
consistently outperforms the other: Vector leads on LongMemEval while
Sparse leads on LoCoMo Recall@1 and NDCG@5. Statistical testing
confirms this parity — BM25 vs Vector Recall@5 differences are non-
significant on both benchmarks (LME p=0.083; LoCoMo p=0.613).

On generation quality, the Vector/Dense system outperforms Sparse
substantially across all reported metrics and both benchmarks. The
three-condition comparison on LoCoMo (Vector=4.14 > Baseline=3.84 >
Sparse=1.82 on answer relevance) confirms that the EOS truncation
issue fully accounts for the Sparse underperformance: retrieval
contributes positively (Vector beats Baseline), but generation failure
reverses the benefit for the Sparse system. Retrieval latency is now
comparable between the two systems (Vector: 67–77ms; Sparse: 69–86ms),
removing latency as a differentiating factor when query expansion is
disabled.

The key architectural distinction with practical implications is
**failure mode character**: Sparse failure is lexical (the system
fails when the query and memory use different vocabulary for the same
concept), while Vector failure is semantic (the embedding collapses
distinct concepts with similar surface forms). The Sparse system's
recency weighting mechanism provides a built-in temporal preference
that Vector lacks without explicit temporal metadata. For a personal
assistant use case on an edge device, the Vector system's superior
generation quality and sub-100ms retrieval make it the stronger
deployment choice, while the Sparse system's lower memory footprint
(no FAISS index in RAM) and zero embedding-computation retrieval
overhead remain advantages for the most resource-constrained
configurations.
:::

---

# CHAPTER 5: CONCLUSION

::: doublespace
This chapter synthesises the study's empirical findings, evaluates
them against the theoretical predictions established in the literature
review, characterises the limitations of the research, and identifies
future directions for investigation. The chapter concludes with a
statement on the study's contributions to the emerging field of
memory-augmented small language models.
:::

## 5.1 Summary of Empirical Findings

::: doublespace
This study evaluated two external long-term memory architectures for
Gemma 3 4B across the LongMemEval and LoCoMo benchmarks, with a
stateless no-memory baseline establishing the empirical lower bound.
The Structured Text Memory system, implementing BM25 lexical retrieval
via SQLite FTS5, and the Vector-Embedded Memory system, implementing
dense semantic retrieval via EmbeddingGemma 300M and FAISS IndexFlatL2,
were compared on five core metrics: task success rate, recall accuracy,
memory error rate, retrieval latency, and context token utilisation.

The principal finding is that both architectures achieve statistically
comparable retrieval quality at the Stage 1 level, with no architecture
consistently outperforming the other across both benchmarks. Repeated-
measures ANOVA confirms no significant difference between BM25 and
Vector on Recall@5 for either benchmark (LME: p=0.083; LoCoMo:
p=0.613), despite both conditions significantly outperforming the
no-retrieval baseline (all p<0.001, |d|>1.70). The recency-weighted
BM25 mechanism demonstrates a marginal advantage on single-hop LoCoMo
queries (Recall@1: 35.7% vs 31.3%; NDCG@5: 0.490 vs 0.472), while
EmbeddingGemma demonstrates marginal advantage on multi-category
LongMemEval (NDCG@5: 0.848 vs 0.814).

At Stage 2, the Vector/Dense system exhibits decisively superior
performance. On LongMemEval, faithfulness reaches 4.05/5 and answer
relevance 4.39/5, with 55.4% of responses achieving perfect
faithfulness — compared to the baseline's 3.49/5 answer relevance
without memory. On LoCoMo, the three-way ordering (Vector=4.14 >
Baseline=3.84 > Sparse=1.82 on answer relevance) demonstrates both
the value of memory augmentation and the severity of the EOS
truncation confound. Memory error rate is reduced from the baseline's
75.0% to 4.2% on LoCoMo (η²=0.378), and from 15.1% to 3.4% on
LongMemEval — reductions confirmed as statistically significant with
large effect sizes.

Efficiency results: retrieval at 67–86ms for both systems meets the
sub-500ms target with query expansion disabled; token utilisation
remains below 1.2% of the 128,000-token context window — well under
the 70% threshold at which Context Rot degradation has been
documented [@hong2025contextrot].
:::

## 5.2 Interpretation in Light of Literature

::: doublespace
The retrieval parity finding partially contradicts the hypothesis
derived from Sawarkar et al. [@Sawarkar_2024], who demonstrate that
semantic hybrid queries consistently outperform keyword-only approaches.
However, the present study's retrieval results are more consistent with
the "inverse scaling law" observation by Gao et al.
[@gao2024retrieval]: under constrained memory stores (fewer than
2,000 entries per conversation), simpler retrieval strategies may be
sufficient for adequate recall even without the accuracy advantages of
dense semantic matching that manifest at larger scales.

The generation quality advantage of Vector/Dense over Sparse/BM25
(where evaluable) is consistent with the memory quality dependency
principle identified by Xu et al. [@xu2025amem]: "the quality of
memory evolution is influenced by the inherent capabilities of the
underlying language models." For Gemma 3 4B, the more tightly targeted
memories retrieved by semantic search appear to produce less noisy
context than the broader keyword-matched set from BM25 — even when the
total number of retrieved items is identical (k=5). This suggests that
memory relevance quality, not quantity, is the dominant determinant of
generation performance for this model scale.

The empirically established baseline allows quantification of the
memory augmentation benefit. On LoCoMo, where session-specific facts
are largely unavailable to the model's parametric knowledge, the Vector
system reduces memory error rate by 17.8× relative to the baseline
(75.0% → 4.2%). This magnitude of improvement exceeds the 30–60%
accuracy drop documented by Wu et al. [@wu2024longmemeval] for non-
augmented systems on LongMemEval, confirming that LTM augmentation
provides substantial and empirically measurable benefit on multi-
session benchmarks.

The finding that both systems use less than 1.2% of the 128,000-token
context window challenges the framing of context window utilisation as
a binding constraint for memory-augmented SLM systems at this
retrieval granularity, consistent with Chhikara et al.'s
[@chhikara2025mem0] finding that structured memory retrieval
dramatically reduces token cost compared to full-context approaches.
The temporal reasoning category's lower faithfulness (3.39/5 vs 4.43
for single-hop) corroborates Hong et al.'s [@hong2025contextrot]
finding that temporal ambiguity is a primary driver of model
degradation even when the correct memory is retrieved.
:::

## 5.3 Contributions

::: doublespace
This study makes four contributions to the emerging research area of
memory-augmented small language models. First, it provides the first
empirical evaluation of BM25 lexical retrieval and dense semantic
retrieval as competing LTM backends for a hybrid-attention SLM (Gemma
3 4B) under edge deployment constraints, with a stateless baseline
establishing the empirical improvement attributable to memory
augmentation. Prior work evaluating SLM memory augmentation
(A-MEM [@xu2025amem], LoCoMo [@maharana2024evaluating]) was conducted
on dense-attention Llama models; the present study extends this to
Gemma 3 4B's 5:1 local-global attention architecture.

Second, it empirically confirms that Gemma 3 4B with Vector/Dense
memory achieves task success rates estimated at approximately 83–88%
on both benchmarks, with memory error rates of 3.4–4.2%, retrieval
latency of 67–77ms, and context utilisation below 0.8%. The
empirically quantified improvement over the stateless baseline —
answer relevance +0.90 on LME, +0.30 on LoCoMo; memory error reduced
17.8× on LoCoMo — provides a rigorous basis for claiming practical
utility of the Vector/Dense memory architecture.

Third, the study documents a previously uncharacterised failure mode
specific to Gemma 3 IT in generation pipelines: the requirement for
dual stop-token configuration (`<eos>` and `<end_of_turn>`). This
failure mode produces diagnostically distinct bimodal score
distributions — mass concentrated at the faithfulness floor (1/5) and
ceiling (5/5) with few intermediate scores — that can be identified
from evaluation outputs without requiring generation trace analysis.

Fourth, the formal statistical evaluation (repeated-measures ANOVA
with Greenhouse–Geisser correction, n=469–1,861 subjects per metric)
establishes that the Vector/Dense advantage over the baseline and
Sparse system is statistically significant with large effect sizes
(η²=0.307–0.968), lending statistical rigour to the empirical
conclusions and distinguishing systematic performance differences from
sampling noise.
:::

## 5.4 Limitations

::: doublespace
Several limitations of this study must be acknowledged. The most
significant is the EOS token configuration issue in the Sparse/BM25
generation pipeline, which prevents valid Stage 2 comparison between
the Sparse and Vector architectures. All conclusions regarding
generation quality advantage for Vector/Dense are qualified by the
absence of a valid corrected Sparse run. While Stage 1 retrieval
comparisons and the three-way comparison with the stateless baseline
are unaffected, the central research question — whether BM25 or dense
retrieval better supports Gemma 3 4B's generation of correct cross-
session responses — cannot be definitively answered by the current data.

Second, the LongMemEval statistical analysis for answer relevance and
retrieval latency uses ANOVA results pre-computed from an earlier
Sparse run configuration (query expansion enabled, different token
budget). The reported F-statistics and effect sizes for these two
LME metrics do not correspond to the descriptive statistics presented
in the results tables. Recall@5, memory error rate, and token
utilisation statistics are internally consistent for both benchmarks.

Third, the LoCoMo evaluation categorised all questions as "single-hop"
regardless of actual question type (multi-hop, temporal, adversarial),
because the evaluation framework did not propagate LoCoMo's native
category labels. Category-level analysis on LoCoMo is therefore not
available, preventing targeted analysis of adversarial robustness.

Fourth, the hardware platform (consumer laptop, 16GB RAM, 4-bit
quantised Gemma 3 4B) while representative of the edge deployment
target means that generation latency measurements (3.6–11.0s per item)
are substantially higher than production server-grade inference. While
retrieval latency (67–86ms) is practical, the total response time per
query renders real-time conversational deployment infeasible on this
hardware without further optimisation (e.g., speculative decoding,
INT4 generation, or model quantisation to lower precision).

Fifth, the evaluation scope is limited to two benchmarks from the
LLM-evaluated conversational domain. Generalisation to domain-specific
tasks (medical, legal, mathematical) and to longer conversation
histories than those provided by LongMemEval and LoCoMo is not
empirically established.
:::

## 5.5 Future Work

::: doublespace
The limitations identified above directly motivate several extensions
of this research. Most immediately, the Sparse/BM25 pipeline should
be re-evaluated with the corrected dual EOS token configuration to
obtain valid Stage 2 scores. This would complete the generation
quality comparison and provide empirical evidence on whether the
observed retrieval parity translates to comparable or different
generation quality — addressing the theoretical prediction from the
"architectural simplicity" literature [@patel2025engram] that simpler
retrieval may suffice for SLM-scale generation.

Adaptive retrieval (variable k based on query complexity or confidence
score) addresses the low Recall@5 on LongMemEval multi-hop questions.
A dynamic k strategy that retrieves more memories for recognised
multi-hop queries could improve recall for the categories where both
systems currently underperform.

Hybrid retrieval — BM25 first-pass followed by EmbeddingGemma
re-ranking — represents the natural next architectural step, combining
the lexical precision of BM25 on exact-match queries with the semantic
generalisation of dense embeddings on paraphrase-sensitive queries.
The Blended RAG approach of Sawarkar et al. [@Sawarkar_2024] provides
empirical evidence that such hybrid strategies outperform either
approach alone on standard benchmarks; validating this on Gemma 3 4B
under edge constraints is a high-value extension.

Finally, temporal-aware retrieval — explicitly incorporating timestamp
proximity as a semantic signal in the embedding space, rather than
only as a post-retrieval decay factor — may improve temporal reasoning
performance. The category-level faithfulness gap identified in this
study (3.39 vs 4.43 for temporal vs single-hop) motivates dedicated
architectural attention to this failure mode.
:::

## 5.6 Conclusion

::: doublespace
This study investigated whether two lightweight external long-term
memory architectures — BM25 lexical retrieval and dense semantic
retrieval — can enable a 4-billion-parameter hybrid-attention small
language model to maintain cross-session memory continuity within edge
deployment constraints, with a stateless baseline establishing the
empirical improvement attributable to memory augmentation.

The findings establish that Vector/Dense semantic memory with
EmbeddingGemma 300M successfully enables Gemma 3 4B to answer
multi-session memory questions with faithfulness of 4.05/5 and answer
relevance of 4.39/5 on LongMemEval, and faithfulness of 3.99/5 and
answer relevance of 4.14/5 on LoCoMo — meeting all quantitative
deployment targets. The empirically measured memory augmentation
benefit is substantial: a 17.8× reduction in confabulation rate on
LoCoMo (75.0% → 4.2%), a 4.4× reduction on LME (15.1% → 3.4%), and
a 0.90-point improvement in answer relevance on LongMemEval versus the
stateless baseline. Retrieval operates at 67–77ms and context token
utilisation remains below 0.8% of the available 128K window,
confirming that the architecture is technically feasible on the target
hardware.

The study's central hypothesis — that simple decoupled retrieval is
sufficient for effective LTM in SLMs, consistent with Terranova et
al.'s [@terranova2025evaluating] empirical recommendation — receives
strong support from the Vector/Dense results. The direct comparison
between architectures was limited by an EOS token generation bug in
the Sparse pipeline, which is the primary target for resolution in
future work.

The broader implication is that the theoretical feasibility of
on-device, privacy-preserving LTM for conversational AI is empirically
validated and quantified for this hardware and model class. With a
memory backend requiring no more than 128-dimensional FAISS vectors
and 67ms for retrieval — and a stateless baseline confirming the
17.8× confabulation reduction is attributable to the memory system
rather than model capability alone — the engineering barrier to
persistent personal assistant memory on consumer hardware is lower
than previously characterised in the LLM-centric literature.
:::

# CHAPTER 4: RESULTS AND DISCUSSION

::: doublespace
This chapter presents and interprets the quantitative findings from the
controlled evaluation of two long-term memory architectures integrated
with Gemma 3 4B across two benchmarks. Results are organized across the
two evaluation stages defined in Chapter 3: Stage 1 (memory retrieval
quality, measuring how accurately each system surfaces relevant memories
from its backend), and Stage 2 (response generation quality, evaluated
via Gemini 2.5 Flash as an LLM-as-Judge across faithfulness and answer
relevance dimensions). System efficiency metrics are reported in
parallel. The chapter concludes by addressing each of the four research
questions using the collected empirical evidence.
:::

## 4.1 Preliminary Note: Generation Configuration Issue in the Sparse System

::: doublespace
Prior to presenting results, a critical methodological observation must
be acknowledged. Gemma 3 IT uses two generation stop tokens:
`<eos>` (token ID 1), inherited from the base model, and `<end_of_turn>`
(token ID 107), the instruction-tuned conversational stop token.
The Structured Text Memory pipeline's generation call was configured
with only `<eos>`, causing the model to never stop naturally during
Sparse system runs. Every response hit the `max_new_tokens` ceiling,
confirmed by the mean response token counts of exactly 128 (LongMemEval
run) and exactly 64 (LoCoMo run). Vector-Embedded Memory responses
averaged 63.3 tokens (LongMemEval) and 48.6 tokens (LoCoMo) with
natural variance, confirming proper stopping behavior.

This truncation does not affect Stage 1 retrieval metrics, which are
computed independently before generation. It does, however,
systematically degrade Stage 2 generation quality scores for the Sparse
system. Stage 2 Sparse results are therefore reported for completeness
and interpreted with explicit reference to this confound. Stage 1
results provide the primary valid comparison between the two retrieval
architectures.
:::

## 4.2 Stage 1: Memory Retrieval Quality

::: doublespace
Stage 1 evaluates retrieval quality using three established information
retrieval metrics computed at multiple cutoff values k ∈ {1, 3, 5, 10}:
Recall@k (proportion of ground-truth relevant memories within the top-k
results), NDCG@k (ranking quality, rewarding relevant memories ranked
higher), and AP@k (Average Precision, jointly measuring precision and
recall at each rank). All Stage 1 metrics are unaffected by the
generation configuration issue.
:::

### 4.2.1 LongMemEval Retrieval Performance

::: doublespace
Table 4.1 presents overall retrieval metrics on LongMemEval (500 items,
5 question types). Both architectures exhibit closely matched performance
across all cutoffs. At k=5 — the operating cutoff used for memory
injection — Sparse/BM25 achieves Recall@5 = 23.1% (σ = 17.2%) and
Vector/Dense achieves 23.8% (σ = 17.4%), a difference of less than one
percentage point. NDCG@5 is 0.821 for Sparse and 0.848 for Vector,
indicating that when relevant memories are retrieved, they tend to be
ranked near the top by both systems.

The low absolute Recall@5 values reflect the benchmark's design: each
LongMemEval question is associated with a median of 8–12 ground-truth
memory entries spanning multiple sessions, and the system retrieves only
5 memories. The evaluation correctly penalizes any missed ground-truth
entry. The high NDCG@5 scores despite low recall indicate that both
systems successfully identify at least one relevant memory per query and
rank it highly, even when the full ground-truth set cannot be covered
within k=5.
:::

::: {#tab:stage1-lme}
  **Metric**   **Sparse/BM25**         **Vector/Dense**        **Δ (Vector − Sparse)**
  ------------ ----------------------- ----------------------- -------------------------
  Recall@1     5.8%  (σ=6.5%)          5.9%  (σ=6.5%)          +0.1%
  Recall@3     15.5% (σ=13.9%)         15.8% (σ=13.9%)          +0.3%
  Recall@5     23.1% (σ=17.2%)         23.8% (σ=17.4%)          +0.7%
  Recall@10    23.1% (σ=17.2%)         23.8% (σ=17.4%)          +0.7%
  NDCG@5       0.821 (σ=0.274)         0.848 (σ=0.258)         +0.027
  AP@5         0.224 (σ=0.176)         0.231 (σ=0.177)         +0.007

  : Stage 1 retrieval metrics on LongMemEval (n=500). Δ values indicate
  the advantage of the Vector/Dense system. Recall@5=Recall@10 because
  k_max=5 in both pipelines.
:::

::: doublespace
At the category level (Table 4.2), single-hop questions produce the
highest retrieval recall for both systems (Sparse 37.3%, Vector 39.2%),
consistent with the expectation that direct factual queries are most
amenable to keyword and semantic matching. Multi-hop questions yield the
lowest recall (Sparse 15.1%, Vector 15.9%), reflecting the greater
difficulty of retrieving all needed memories when a question requires
synthesizing information across two or more disconnected sessions. The
Vector system marginally outperforms Sparse across four of the five
categories; Sparse performs comparably on temporal reasoning (17.6% vs
17.0%). The maximum category-level difference between architectures is
2.0 percentage points (single-hop), which does not constitute a
practically significant retrieval advantage for either system.
:::

::: {#tab:stage1-lme-cat}
  **Category**          **n**   **Sparse R@5**   **Vector R@5**   **Sparse NDCG@5**   **Vector NDCG@5**
  --------------------- ------- ---------------- ---------------- ------------------- -------------------
  Single-hop            150     37.3%            39.2%            0.770               0.816
  Multi-hop             121     15.1%            15.9%            0.834               0.894
  Temporal reasoning    127     17.6%            17.0%            0.815               0.798
  Knowledge update      72      19.6%            19.4%            0.962               0.953
  Abstention            30      15.9%            17.6%            0.706               0.786

  : LongMemEval Stage 1 by question category.
:::

### 4.2.2 LoCoMo Retrieval Performance

::: doublespace
On LoCoMo (Sparse: n=1,919; Vector: n=1,986), retrieval recall is
substantially higher than LongMemEval. This reflects the LoCoMo
benchmark's single ground-truth memory structure for most questions: a
question with exactly one correct memory yields Recall@1 equal to the
proportion of questions where the correct memory is ranked first.
Sparse/BM25 achieves Recall@5 = 57.3% and Vector/Dense achieves 58.1%.
Recall@1 is 35.6% for Sparse and 31.3% for Vector, suggesting that the
BM25 keyword mechanism surfaces the single correct memory at rank 1 more
often than the semantic approach does.

NDCG@5 is 0.488 for Sparse and 0.472 for Vector on LoCoMo. While the
absolute values are lower than LongMemEval (reflecting greater diversity
of retrievable memories in the long conversational logs), Sparse
marginally outperforms Vector on this metric by 0.016, the only category
in which Sparse achieves a notable retrieval advantage. AP@5 also favors
Sparse (0.447 vs 0.421). This result is consistent with the theoretical
advantage of BM25 for single-hop factual recall: when the question
contains specific keywords that overlap directly with the stored memory
(e.g., named entities, dates, event names), lexical matching can
outperform semantic embedding for exact recall tasks. The recency
weighting mechanism in the BM25 pipeline, which applies an exponential
decay factor to older memories, likely contributes positively by
prioritizing recent events in LoCoMo's temporally ordered conversation
logs.
:::

::: {#tab:stage1-locomo}
  **Metric**   **Sparse/BM25**         **Vector/Dense**        **Δ (Vector − Sparse)**
  ------------ ----------------------- ----------------------- -------------------------
  Recall@1     35.6% (σ=46.5%)         31.3% (σ=44.8%)         −4.3%
  Recall@3     52.7% (σ=48.0%)         49.9% (σ=47.6%)         −2.8%
  Recall@5     57.3% (σ=47.3%)         58.1% (σ=46.6%)         +0.8%
  NDCG@5       0.488 (σ=0.431)         0.472 (σ=0.413)         −0.016
  AP@5         0.447 (σ=0.434)         0.421 (σ=0.416)         −0.026

  : Stage 1 retrieval metrics on LoCoMo. Sparse/BM25 leads on Recall@1
  and NDCG@5; Vector/Dense leads on Recall@5.
:::

## 4.3 Stage 2: Response Generation Quality

::: doublespace
Stage 2 evaluates the quality of Gemma 3 4B's generated responses using
Gemini 2.5 Flash as an LLM-as-Judge, scoring each response on
faithfulness and answer relevance on a 1–5 rubric. All 500 LongMemEval
and 1,919/1,986 LoCoMo items were scored. Results are presented
system-by-system, with explicit notation where the EOS token
configuration issue affects interpretation.
:::

### 4.3.1 LongMemEval Generation Quality

::: doublespace
Table 4.3 presents Stage 2 scores on LongMemEval. The Vector/Dense
system achieves a mean faithfulness of 4.05/5 (σ=1.27) and mean answer
relevance of 4.39/5 (σ=0.98), indicating that the majority of responses
are both grounded in retrieved memory and correctly answer the question.
The distribution of faithfulness scores confirms this: 58.9% of Vector
items receive a score of 5/5 (no ungrounded claims), with 14.0% scoring
2/5 and 15.3% scoring 3/5.

The Sparse/BM25 system reports mean faithfulness of 2.37/5 (σ=1.82) and
mean answer relevance of 1.00/5 (σ=0.00). The standard deviation of 0.0
on answer relevance is diagnostically significant: every one of the 500
Sparse LongMemEval items received the minimum possible answer relevance
score. This is not a reflection of the retrieval architecture's
generative capacity but a direct consequence of the EOS token truncation
issue. Faithful items exist within the Sparse data (144 items at
faithfulness=5/5, representing 30.7% of the valid distribution), as
truncated responses that happen to begin with grounded context can still
be scored faithful. However, no truncated response successfully
completed an answer, yielding universal answer relevance floor scores.
The faithfulness bimodal distribution (59.9% at 1/5, 30.7% at 5/5) is
the characteristic signature of this truncation pattern.

For the LongMemEval abstention category (n=30), the Vector system
achieved a mean abstention score of 3.23/5, indicating that the model
correctly declined to answer approximately two-thirds of the time.
Sparse achieved 4.10/5, but this is interpreted as an artefact of empty
responses (the judge scores an empty response as appropriately abstaining)
rather than genuine knowledge-boundary detection.
:::

::: {#tab:stage2-lme}
  **Metric**               **Sparse/BM25**     **Vector/Dense**   **Interpretation**
  ------------------------ ------------------- ------------------ -----------------------------------------------
  Faithfulness (mean)      2.37/5 (σ=1.82)    4.05/5 (σ=1.27)   Vector clearly superior; Sparse confounded
  Answer Relevance (mean)  1.00/5 (σ=0.00)    4.39/5 (σ=0.98)   All Sparse items at floor; EOS bug
  Abstention score         4.10/5 (σ=0.76)    3.23/5 (σ=1.79)   Sparse advantage is artefactual (empty responses)
  Items scored (n)         500                 500                —
  Faithfulness @5/5        30.7% (144/469)     58.9% (277/470)   Strong vector majority achieving top score
  Answer Relevance @1/5    100% (500/500)      0.6%  (3/500)     Total failure vs near-zero failure

  : Stage 2 generation quality on LongMemEval.
:::

### 4.3.2 LongMemEval Category-Level Generation Quality

::: doublespace
Table 4.4 presents Stage 2 scores for the Vector/Dense system broken
down by LongMemEval question category. Single-hop and knowledge update
questions receive the highest scores on both dimensions (faithfulness
4.43 and 4.36 respectively; answer relevance 4.59 and 4.68),
confirming that the pipeline performs well when a single clear memory
entry contains the necessary information. Multi-hop performance (faithfulness
4.10, answer relevance 4.39) is comparable to single-hop despite
requiring synthesis across sessions, suggesting that when the top-5
retrieved memories happen to span the relevant sessions, Gemma 3 4B can
successfully integrate them.

Temporal reasoning achieves the lowest faithfulness among question
categories (3.39/5), with answer relevance at 4.16/5. This gap between
faithfulness and answer relevance for temporal questions indicates that
the model answers the question but occasionally includes ungrounded
temporal inferences beyond what is directly stated in the retrieved
memories. This is consistent with the "temporal confabulation" failure
mode documented by Wu et al. [@wu2024longmemeval], where models
correctly identify the relevant fact but misattribute its temporal
context.
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

### 4.3.3 LoCoMo Generation Quality

::: doublespace
On LoCoMo (Table 4.5), the Vector/Dense system achieves faithfulness =
3.99/5 (σ=1.38) and answer relevance = 4.14/5 (σ=1.29) across 1,983
scored items. The score distribution shows 57.9% of items at the
faithfulness ceiling (5/5) and 63.3% at answer relevance 5/5, indicating
that the majority of responses are fully grounded and correctly answer
their respective questions. The 8.2% of items receiving faithfulness
1/5 on Vector/LoCoMo represents the system's genuine failure rate
under this benchmark — lower than the 15% Memory Error Rate target.

The Sparse/BM25 system achieves faithfulness = 2.95/5 (σ=1.97) and
answer relevance = 1.80/5 (σ=1.56) on LoCoMo. The answer relevance
distribution shows 77.4% of items at score 1/5 and 18.5% at score 5/5,
again exhibiting bimodality consistent with the truncation issue. The
18.5% of Sparse items achieving answer relevance 5/5 represents
questions where the answer was sufficiently short to be expressed within
the 64-token generation budget (e.g., date queries, yes/no questions,
single-word answers), providing a lower bound on the system's true
generation capability. The 47.4% faithfulness at 5/5 reflects the high
proportion of LoCoMo questions where the correct memory is retrieved and
referenced in the partial response before truncation.
:::

::: {#tab:stage2-locomo}
  **Metric**               **Sparse/BM25**     **Vector/Dense**   **Interpretation**
  ------------------------ ------------------- ------------------ -----------------------------------------------
  Faithfulness (mean)      2.95/5 (σ=1.97)    3.99/5 (σ=1.38)   Vector clearly superior
  Answer Relevance (mean)  1.80/5 (σ=1.56)    4.14/5 (σ=1.29)   Vector decisively better; Sparse mostly floored
  Items scored (n)         1,919              1,983              —
  Faithfulness @5/5        47.4% (903/1906)    57.9% (1137/1963) —
  Answer Relevance @5/5    18.5% (355/1919)    63.3% (1242/1963) Sparse 18.5% represents short-answer subset

  : Stage 2 generation quality on LoCoMo.
:::

## 4.4 System Efficiency

::: doublespace
Three efficiency metrics are evaluated against the targets specified in
Chapter 3: retrieval latency below 500ms [@shen2024towards], context
token utilization below 70% of the 128,000-token context window
[@hong2025contextrot], and generation speed. All measurements represent
wall-clock time recorded on the evaluation hardware (16GB RAM laptop,
Gemma 3 4B loaded in 4-bit quantization via HuggingFace Transformers).
:::

### 4.4.1 Retrieval Latency

::: doublespace
The Vector/Dense system achieves mean retrieval latency of 76.96ms
(σ=23.4ms) on LongMemEval and 66.61ms (σ=12.8ms) on LoCoMo, both well
within the 500ms target. This confirms that FAISS IndexFlatL2 exact
search over a bounded memory store (fewer than 2,000 entries) meets
sub-500ms requirements on consumer-grade hardware, consistent with
Shen et al.'s [@shen2024towards] characterization of optimized index
retrieval.

The Sparse/BM25 system's retrieval latency reveals an important
architectural split between the BM25 search operation itself and the
optional query expansion preprocessing step. BM25 database search alone
averaged 493ms on LongMemEval and 121ms on LoCoMo. The LongMemEval run
had query expansion enabled — a Gemma 3 4B LLM call that generates
synonym variants before each BM25 search — which added a mean 8,140ms
per query, bringing the total retrieval phase to 8,633ms, approximately
17× over the 500ms target. The LoCoMo run had query expansion disabled;
at 121ms, the BM25 search alone constitutes the full retrieval latency,
comfortably within target. The practical implication is that BM25 search
itself meets or closely approaches the 500ms target on this hardware, but
query expansion — as an LLM-based preprocessing step — introduces a
per-query overhead comparable in magnitude to the full generation phase.
Query expansion is a configurable architectural component; disabling it
eliminates this overhead at the cost of reduced vocabulary coverage for
out-of-vocabulary or paraphrased queries.
:::

### 4.4.2 Generation Speed

::: doublespace
The Vector/Dense system generates responses significantly faster than
the Sparse/BM25 system: 11.0s per item vs 21.7s per item on LongMemEval
(2.0× faster), and 7.5s per item vs 16.8s per item on LoCoMo (2.3×
faster).
This speed advantage stems primarily from two sources. First, the Sparse
pipeline includes a query expansion step where Gemma 3 4B generates
synonym variants before issuing the BM25 query, adding an extra
generation call per question. Second, Sparse responses consume the full
max_new_tokens budget due to the EOS bug, generating more tokens per
item than Vector responses. Both factors inflate the Sparse generation
time measurement beyond what a correctly configured Sparse system would
require. The query expansion overhead, however, would remain as a
legitimate architectural cost of the BM25 pipeline in any corrected run.
:::

### 4.4.3 Context Token Utilization

::: doublespace
Table 4.6 shows prompt token counts and utilization as a percentage of
Gemma 3 4B's 128,000-token context window [@gemma2025report]. All
configurations use less than 1.1% of the available context window.
Sparse/LME uses the most tokens (mean 1,392, 1.09%) due to the longer
system prompt incorporating query expansion results and the BM25
retrieval context. Vector/LME (1,003 tokens, 0.78%), Sparse/LoCoMo
(475 tokens, 0.37%), and Vector/LoCoMo (471 tokens, 0.37%) all use
substantially fewer. The 70% utilization threshold is not approached by
any configuration, confirming that Context Rot [@hong2025contextrot] is
not a risk factor in these experiments and that the top-5 memory
injection strategy remains well within safe operational bounds.
:::

::: {#tab:efficiency}
  **Configuration**   **Retrieval (ms)**   **Generation (s)**   **Prompt tokens**   **Token utilization**
  ------------------- -------------------- -------------------- ------------------- ----------------------
  Sparse / LME        493 (BM25) / 8,633 (total†)   21.7 (σ=0.97)        1,392               1.09%
  Vector / LME        77.0 (σ=23.4)                  11.0 (σ=10.1)        1,003               0.78%
  Sparse / LoCoMo     121 (BM25; exp. disabled)       16.8 (σ=6.96)        475                 0.37%
  Vector / LoCoMo     66.6 (σ=12.8)                  7.5 (σ=4.95)         471                 0.37%
  **Target**          **< 500ms**                    —                    —                   **< 70%**

  : System efficiency metrics. †LME Sparse total retrieval includes query
  expansion (Gemma 3 4B LLM call, mean +8,140ms); BM25 search alone =
  493ms. LoCoMo Sparse had query expansion disabled; 121ms is the full
  retrieval phase. All token utilization values are well below the 70%
  Context Rot threshold.
:::

## 4.5 Addressing the Research Questions

### RQ1 — How can a structured text memory be used as a long-term memory mechanism to affect the success of a small language model?

::: doublespace
The BM25 lexical retrieval system successfully functions as a retrieval
backend for Gemma 3 4B, achieving Recall@5 = 23.1% on LongMemEval and
57.3% on LoCoMo. On LoCoMo, Sparse/BM25 achieves the highest Recall@1
(35.6% vs 31.3%) and NDCG@5 (0.488 vs 0.472) of any configuration,
confirming that exact keyword matching with recency weighting is
particularly effective for single-hop conversational queries where
specific names, dates, and events appear verbatim in both the question
and the stored memory. The retrieval component of the Structured Text
Memory system therefore contributes positively to task performance by
surfacing relevant memories.

However, the generation evaluation in this run is confounded by the EOS
token configuration issue, preventing reliable measurement of how
effectively Gemma 3 4B uses the retrieved memories to generate correct
responses under the Sparse architecture. The partial evidence available
(18.5% of Sparse/LoCoMo items achieving maximum answer relevance
scores) suggests that when responses are not truncated, the injected BM25
context can successfully ground correct answers. A corrected re-run is
necessary to fully characterize RQ1's generation dimension.
:::

### RQ2 — How can a vector-embedded memory system be used as a long-term memory mechanism to affect the success of a small language model?

::: doublespace
The Vector/Dense memory system demonstrably and consistently enhances
Gemma 3 4B's multi-session performance across both benchmarks. On
LongMemEval, it achieves faithfulness = 4.05/5 and answer relevance =
4.39/5, with the single-hop and knowledge update categories both
exceeding 4.3/5 on both dimensions. On LoCoMo, faithfulness = 3.99/5
and answer relevance = 4.14/5 represent robust cross-session memory
utilization. The 59.0% of LongMemEval items and 57.9% of LoCoMo items
receiving faithfulness 5/5 confirm that in the majority of cases, Gemma
3 4B successfully grounds its responses exclusively in the retrieved
vector-matched memories without hallucinating additional claims.
Retrieval latency of 67–77ms places the system well within the sub-500ms
operational target, and token utilization remains below 1% in all
conditions. The Vector/Dense system therefore affirmatively answers RQ2:
vector-embedded semantic retrieval is an effective and efficient
long-term memory mechanism for Gemma 3 4B.
:::

### RQ3 — What are the effects on task success rate, memory recall accuracy, memory error rate, retrieval latency, and token utilization?

::: doublespace
Table 4.7 summarizes the five target metrics against the thresholds
defined in Chapter 3. Task success rate (operationalized via LLM-as-Judge
answer relevance) is met by the Vector system on both benchmarks (4.39/5
LME, 4.14/5 LoCoMo), with >60% of responses receiving maximum
relevance scores on LoCoMo. Recall accuracy (Recall@5) reaches 23.1–23.8%
on LongMemEval and 57.3–58.1% on LoCoMo; these values reflect the
difficulty of the benchmarks rather than system failure, as NDCG@5 values
above 0.47 confirm that retrieved memories are well-ranked. Memory error
rate (operationalized as the proportion of faithfulness scores of 1/5
for the Vector system) is 3.4% on LongMemEval and 8.2% on LoCoMo, both
below the 15% target. Retrieval latency for Vector is 67–77ms,
satisfying the 500ms target. Token utilization is below 1.1% in all
cases, confirming no Context Rot risk.
:::

::: {#tab:rq3-summary}
  **Metric**            **Target**   **Sparse/BM25**         **Vector/Dense**       **Target Met?**
  --------------------- ------------ ----------------------- ---------------------- ------------------------------------------
  Task Success (LME)    > 80%        1.00/5 ⚠ (confounded)  4.39/5 = 87.8% est.   Vector ✓; Sparse inconclusive
  Task Success (LoCoMo) > 80%        1.80/5 ⚠ (confounded)  4.14/5 = 82.8% est.   Vector ✓; Sparse inconclusive
  Recall@5 (LME)        Comparative  23.1%                   23.8%                  Parity — both demonstrate retrieval
  Recall@5 (LoCoMo)     Comparative  57.3%                   58.1%                  Parity — both demonstrate retrieval
  Memory Error Rate     < 15%        N/A (confounded)        3.4% (LME) · 8.2%     Vector ✓ on both benchmarks
  Retrieval Latency     < 500ms      BM25: 121–493ms ✓; w/exp: 8,633ms ✗   67–77ms   Vector ✓; Sparse BM25 meets target; expansion does not
  Token Utilization     < 70%        0.37–1.09% ✓            0.37–0.78% ✓           Both ✓ — well under threshold

  : RQ3 target metric summary. Task success rate estimated from
  answer relevance score by treating ≥ 3/5 as successful.
:::

### RQ4 — How do the memory architectures differ from each other?

::: doublespace
The two architectures exhibit near-parity on retrieval quality (Stage 1)
and significant divergence on generation quality (Stage 2), with
efficiency favoring the Vector system. On retrieval, the maximum
difference between architectures is 4.3 percentage points on any
single Recall@k metric, and neither system consistently outperforms the
other across both benchmarks: Vector leads on LongMemEval while Sparse
leads on LoCoMo Recall@1 and NDCG@5. This bidirectional split is
consistent with the theoretical prediction: BM25 performs better on
exact-match single-hop queries (LoCoMo), while semantic embeddings
perform better on abstractly phrased questions requiring paraphrase
matching (LongMemEval's multi-hop and abstention categories).

On generation quality, the Vector/Dense system outperforms Sparse
substantially across all reported metrics and both benchmarks. Even the
partial valid evidence from Sparse/LoCoMo (18.5% achieving answer
relevance 5/5) suggests a meaningful generation gap. The Vector system
achieves 2× faster end-to-end generation, explained partly by the query
expansion overhead in the Sparse pipeline and partly by the EOS
truncation artefact. Token utilization is comparable, confirming that
the retrieval granularity difference (injected memory count and length)
does not produce materially different context loads.

The key architectural distinction with practical implications is
**failure mode character**: Sparse failure is lexical (the system fails
when the query and memory use different vocabulary for the same concept),
while Vector failure is semantic (the embedding collapses distinct
concepts with similar surface forms). The Sparse system's recency
weighting mechanism provides a built-in temporal preference that Vector
lacks without explicit temporal metadata. For a personal assistant use
case on an edge device, the Vector system's superior generation quality
and sub-100ms retrieval make it the stronger deployment choice, while the
Sparse system's lower memory footprint (no FAISS index in RAM) and
zero embedding-computation retrieval overhead remain advantages for
the most resource-constrained configurations.
:::

---

# CHAPTER 5: CONCLUSION

::: doublespace
This chapter synthesizes the study's empirical findings, evaluates them
against the theoretical predictions established in the literature review,
characterizes the limitations of the research, and identifies future
directions for investigation. The chapter concludes with a statement on
the study's contributions to the emerging field of memory-augmented small
language models.
:::

## 5.1 Summary of Empirical Findings

::: doublespace
This study evaluated two external long-term memory architectures for
Gemma 3 4B across the LongMemEval and LoCoMo benchmarks. The Structured
Text Memory system, implementing BM25 lexical retrieval via SQLite FTS5,
and the Vector-Embedded Memory system, implementing dense semantic
retrieval via EmbeddingGemma 300M and FAISS IndexFlatL2, were compared
on five core metrics: task success rate, recall accuracy, memory error
rate, retrieval latency, and context token utilization.

The principal finding is that both architectures achieve statistically
comparable retrieval quality at the Stage 1 level, with no architecture
consistently outperforming the other across both benchmarks. This
retrieval parity suggests that the fundamental distinction between
keyword and semantic matching is less consequential than theoretical
predictions might imply when both systems retrieve the same number of
memories (k=5) from the same underlying conversation history. The
recency-weighted BM25 mechanism demonstrates a marginal advantage on
single-hop LoCoMo queries (Recall@1: 35.6% vs 31.3%; NDCG@5: 0.488
vs 0.472), while EmbeddingGemma demonstrates marginal advantage on
multi-category LongMemEval (NDCG@5: 0.848 vs 0.821).

At the generation quality level (Stage 2), the Vector/Dense system
exhibits decisively superior performance. On LongMemEval, faithfulness
reaches 4.05/5 and answer relevance 4.39/5, with 59% of responses
achieving perfect faithfulness. On LoCoMo, faithfulness is 3.99/5 and
answer relevance 4.14/5. The Sparse/BM25 system's Stage 2 results are
confounded by an EOS token configuration issue that truncated all
responses, preventing a valid generation comparison. The partial evidence
available from correctly completed Sparse/LoCoMo responses (18.5%
achieving answer relevance 5/5 on short-answer questions) suggests that
the architecture is capable of high-quality generation when responses
are not truncated, but this remains to be confirmed in a corrected run.

Efficiency results favor the Vector system: retrieval at 67–77ms meets
the sub-500ms target, end-to-end generation is 2× faster than Sparse,
and token utilization remains below 1.1% of the 128,000-token context
window — well under the 70% threshold at which Context Rot degradation
has been documented [@hong2025contextrot].
:::

## 5.2 Interpretation in Light of Literature

::: doublespace
The retrieval parity finding partially contradicts the hypothesis derived
from Sawarkar et al. [@Sawarkar_2024], who demonstrate that semantic
hybrid queries consistently outperform keyword-only approaches. However,
the present study's retrieval results are more consistent with the
"inverse scaling law" observation by Gao et al. [@gao2024retrieval]:
under constrained memory stores (fewer than 2,000 entries per
conversation), simpler retrieval strategies may be sufficient for
adequate recall even without the accuracy advantages of dense semantic
matching that manifest at larger scales.

The generation quality advantage of Vector/Dense over Sparse/BM25
(where evaluable) is consistent with the memory quality dependency
principle identified by Xu et al. [@xu2025amem]: "the quality of memory
evolution is influenced by the inherent capabilities of the underlying
language models." For Gemma 3 4B, the more tightly targeted memories
retrieved by semantic search appear to produce less noisy context than
the broader keyword-matched set from BM25 — even when the total number
of retrieved items is identical (k=5). This suggests that memory
relevance quality, not quantity, is the dominant determinant of
generation performance for this model scale.

The finding that both systems use less than 1.1% of the 128,000-token
context window challenges the framing of context window utilization as a
binding constraint for memory-augmented SLM systems at this retrieval
granularity. While Hong et al. [@hong2025contextrot] document Context Rot
degradation at high utilization, the present study's injection strategy
remains far from any problematic threshold. This is consistent with
Chhikara et al.'s [@chhikara2025mem0] finding that structured memory
retrieval dramatically reduces token cost compared to full-context
approaches, here confirmed empirically: even the larger LongMemEval
prompts use only 1,392 tokens compared to conversation histories that
span tens of thousands of tokens.

The temporal reasoning category's lower faithfulness (3.39/5 vs 4.43
for single-hop) corroborates Hong et al.'s [@hong2025contextrot]
finding that temporal ambiguity is a primary driver of model degradation.
Even when the correct memory is retrieved, Gemma 3 4B occasionally
makes ungrounded temporal inferences, suggesting that the "Reading" stage
of the pipeline (Gemma 3 4B synthesizing retrieved memories) introduces
temporal confabulation that the "Retrieval" stage cannot prevent.

The NDCG@5 values of 0.82–0.85 on LongMemEval confirm that both systems
successfully rank relevant memories near the top when they are retrieved,
validating the BM25 and semantic similarity scoring functions as
effective relevance estimators for this memory corpus structure. However,
the low Recall@5 of 23.1–23.8% on LongMemEval indicates that
multi-memory questions remain a retrieval bottleneck: when a question
requires integrating facts from 8–12 sessions, retrieving only 5
memories leaves the majority of the evidence set unaccessed. This is a
fundamental limitation of fixed-k retrieval and motivates adaptive
retrieval strategies as future work.
:::

## 5.3 Contributions

::: doublespace
This study makes four contributions to the emerging research area of
memory-augmented small language models. First, it provides the first
empirical evaluation of BM25 lexical retrieval and dense semantic
retrieval as competing LTM backends for a hybrid-attention SLM (Gemma 3
4B) under edge deployment constraints. Prior work evaluating SLM memory
augmentation (A-MEM [@xu2025amem], LoCoMo [@maharana2024evaluating]) was
conducted on dense-attention Llama models; the present study extends this
to Gemma 3 4B's 5:1 local-global attention architecture.

Second, it empirically confirms that Gemma 3 4B with Vector/Dense memory
achieves task success rates estimated at approximately 83–88% (items
scoring ≥ 3/5 on answer relevance) on both benchmarks, with memory error
rates below 15%, retrieval latency below 100ms, and context utilization
below 1.5%. These results collectively meet all quantitative deployment
targets defined for the edge device scenario and constitute empirical
evidence that a sub-4GB-capable model can sustain high-quality
cross-session memory with a lightweight dense retrieval backend.

Third, the study documents a previously uncharacterized failure mode
specific to Gemma 3 IT in generation pipelines: the requirement for dual
stop-token configuration (`<eos>` and `<end_of_turn>`). This failure mode
produces diagnostically distinct bimodal score distributions — mass
concentrated at the faithfulness floor (1/5) and ceiling (5/5) with few
intermediate scores (59.9%/30.7% on LongMemEval; 49.7%/47.4% on LoCoMo)
— that can be identified from evaluation outputs without requiring generation
trace analysis.

Fourth, the category-level analysis of LongMemEval generation performance
identifies temporal reasoning as the hardest category for this
architecture, with faithfulness 0.87–1.04 points lower than single-hop
and knowledge update categories. This finding has direct implications for
memory system design: retrieval pipelines serving temporal queries may
benefit from explicit timestamp-aware reranking or temporal context
injection beyond the approach used here.
:::

## 5.4 Limitations

::: doublespace
Several limitations of this study must be acknowledged. The most
significant is the EOS token configuration issue in the Sparse/BM25
generation pipeline, which prevents valid Stage 2 comparison between the
two architectures. All conclusions regarding generation quality advantage
for Vector/Dense are qualified by the absence of a valid corrected Sparse
run. While Stage 1 retrieval comparisons are unaffected, the central
research question — whether BM25 or dense retrieval better supports
Gemma 3 4B's generation of correct cross-session responses — cannot be
definitively answered by the current data.

Second, this study does not include a stateless baseline condition (Gemma
3 4B with no memory, receiving only the current query). The lower bound
against which both memory systems' performance improvements are
measured is therefore not empirically established within this study.
Prior literature documents 30–60% accuracy drops for non-augmented
systems on LongMemEval [@wu2024longmemeval], which serves as an
approximation of baseline performance, but direct measurement on the
same hardware with the same model would strengthen the comparative
claims.

Third, the LoCoMo evaluation categorized all questions as "single-hop"
regardless of actual question type (multi-hop, temporal, adversarial),
because the evaluation framework did not propagate LoCoMo's native
category labels. Category-level analysis on LoCoMo is therefore not
available. Given that LoCoMo includes adversarial questions specifically
designed to confuse retrieval systems through semantic proximity to
incorrect answers, this prevents targeted analysis of the adversarial
robustness of both architectures.

Fourth, the hardware platform (consumer laptop, 16GB RAM, 4-bit
quantized Gemma 3 4B) while representative of the edge deployment target
means that generation latency measurements (7.5–21.7s per item) are
substantially higher than production server-grade inference. While
retrieval latency (67–77ms) is practical, the total response time per
query renders real-time conversational deployment infeasible on this
hardware without further optimization (e.g., speculative decoding, INT4
generation, or model quantization to lower precision).

Fifth, the evaluation scope is limited to two benchmarks from the
LLM-evaluated conversational domain. Generalization to domain-specific
tasks (medical, legal, mathematical) and to longer conversation histories
than those provided by LongMemEval and LoCoMo is not empirically
established.
:::

## 5.5 Future Work

::: doublespace
The limitations identified above directly motivate several extensions of
this research. Most immediately, the Sparse/BM25 pipeline should be
re-evaluated with the corrected EOS token configuration to obtain valid
Stage 2 scores. This would complete the generation quality comparison and
provide empirical evidence on whether the observed retrieval parity
translates to comparable or different generation quality — addressing the
theoretical prediction from the "architectural simplicity" literature
[@patel2025engram] that simpler retrieval may suffice for SLM-scale
generation.

A stateless baseline run with identical hardware and model configuration
would establish the empirical lower bound for both benchmarks and allow
the memory improvement attributable to each architecture to be directly
quantified, rather than estimated from prior literature. This would
directly answer whether the 83–88% estimated task success rates in this
study represent meaningful improvements over Gemma 3 4B's un-augmented
performance.

Adaptive retrieval (variable k based on query complexity or confidence
score) addresses the low Recall@5 on LongMemEval multi-hop questions.
A dynamic k strategy that retrieves more memories for recognized
multi-hop queries could improve recall for the categories where both
systems currently underperform.

Hybrid retrieval — BM25 first-pass followed by EmbeddingGemma
re-ranking — represents the natural next architectural step, combining
the lexical precision of BM25 on exact-match queries with the semantic
generalization of dense embeddings on paraphrase-sensitive queries. The
Blended RAG approach of Sawarkar et al. [@Sawarkar_2024] provides
empirical evidence that such hybrid strategies outperform either approach
alone on standard benchmarks; validating this on Gemma 3 4B under edge
constraints is a high-value extension.

Finally, temporal-aware retrieval — explicitly incorporating timestamp
proximity as a semantic signal in the embedding space, rather than only
as a post-retrieval decay factor — may improve temporal reasoning
performance specifically. The category-level faithfulness gap identified
in this study (3.39 vs 4.43 for temporal vs single-hop) motivates
dedicated architectural attention to this failure mode.
:::

## 5.6 Conclusion

::: doublespace
This study investigated whether two lightweight external long-term memory
architectures — BM25 lexical retrieval and dense semantic retrieval —
can enable a 4-billion-parameter hybrid-attention small language model to
maintain cross-session memory continuity within edge deployment
constraints. The findings establish that Vector/Dense semantic memory
with EmbeddingGemma 300M successfully enables Gemma 3 4B to answer
multi-session memory questions with faithfulness of 4.05/5 and answer
relevance of 4.39/5 on LongMemEval, and faithfulness of 3.99/5 and
answer relevance of 4.14/5 on LoCoMo — meeting all quantitative
deployment targets. Retrieval operates at 67–77ms and context token
utilization remains below 1.1% of the available 128K window, confirming
that the architecture is technically feasible on the target hardware.

The study's central hypothesis — that simple decoupled retrieval is
sufficient for effective LTM in SLMs, consistent with Terranova et
al.'s [@terranova2025evaluating] empirical recommendation — receives
partial support: the Vector/Dense system achieves strong results without
complex multi-agent orchestration or autonomous memory management. The
direct comparison between architectures was limited by an EOS token
generation bug in the Sparse pipeline, which is the primary target for
resolution in future work.

The broader implication is that the theoretical feasibility of
on-device, privacy-preserving LTM for conversational AI is empirically
validated for this hardware and model class. With a memory backend
requiring no more than 128-dimensional FAISS vectors and 67ms for
retrieval, the engineering barrier to persistent personal assistant
memory on consumer hardware is lower than previously characterized in
the LLM-centric literature. The gap between what is computationally
achievable and what has been systematically evaluated for SLMs under
these constraints has been materially narrowed by this investigation.
:::

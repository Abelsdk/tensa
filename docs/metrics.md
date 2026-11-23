# ML Metrics Reference

Definitions for every metric in the Tensa scoreboard, plus the ones you'll meet at a co-op.
★ = used in Tensa V1.

Notation: TP = true positives, FP = false positives, TN = true negatives, FN = false negatives. *y* = true value, *ŷ* = prediction, *n* = number of examples.

---

## 1. Classification
*Router, Scan, Judge*

| Metric | Definition | Formula | Read it as |
|---|---|---|---|
| Accuracy | Share of predictions that are correct | (TP + TN) / all | Misleading when classes are imbalanced (99% "no red flag" scores 99% by always saying no) |
| **Precision** ★ | Of everything predicted positive, how much was actually positive | TP / (TP + FP) | "When it says yes, is it right?" |
| **Recall / Sensitivity** ★ | Of everything actually positive, how much was caught | TP / (TP + FN) | "Does it miss things?" — the key metric for red flags |
| **Specificity** ★ | Of everything actually negative, how much was correctly rejected | TN / (TN + FP) | Recall for the negative class; standard in medical imaging |
| **F1 score** ★ | Harmonic mean of precision and recall | 2 · P · R / (P + R) | Balances the two; drops fast if either is low |
| **Macro-F1** ★ | F1 computed per class, then averaged equally | mean(F1 per class) | Every class counts the same — rare classes matter |
| Micro-F1 | F1 computed from total TP/FP/FN across classes | — | Dominated by common classes; equals accuracy in single-label tasks |
| Weighted-F1 | Per-class F1 averaged by class frequency | — | Between macro and micro |
| **Confusion matrix** ★ | Table of true class vs predicted class | — | Shows *which* classes get confused with which |
| **ROC-AUC** ★ | Area under the curve of recall vs false-positive rate across all thresholds | 0.5 = random, 1.0 = perfect | Probability a random positive scores higher than a random negative. Threshold-independent |
| PR-AUC | Area under the precision–recall curve | — | Better than ROC-AUC when positives are rare |
| Log loss (cross-entropy) | Penalty for confident wrong probabilities | −mean(y·log ŷ + (1−y)·log(1−ŷ)) | The loss you train logistic regression on; lower is better |
| Calibration / ECE | Whether predicted probabilities match real frequencies | Expected Calibration Error: gap between confidence and accuracy, averaged over bins | "When it says 80%, is it right 80% of the time?" |

---

## 2. Regression
*Timeline*

| Metric | Definition | Formula | Read it as |
|---|---|---|---|
| **MAE** ★ | Mean absolute error | mean(\|y − ŷ\|) | "Off by X days on average." Same units as the target, robust to outliers |
| MSE | Mean squared error | mean((y − ŷ)²) | Punishes large errors heavily; the loss you train linear regression on |
| RMSE | Root of MSE | √MSE | Back in original units, but still outlier-sensitive |
| R² | Share of variance in *y* the model explains | 1 − SS_res / SS_tot | 1 = perfect, 0 = no better than predicting the mean, negative = worse than the mean |
| MAPE | Mean absolute percentage error | mean(\|y − ŷ\| / \|y\|) | Relative error; breaks when *y* is near zero |
| **Interval coverage** ★ | Share of true values inside your predicted range | — | A "90% interval" should contain ~90% of real answers |

---

## 3. Retrieval and ranking
*Ask — BM25, dense, hybrid, reranker. Also Shopify search and recommendations.*

| Metric | Definition | Read it as |
|---|---|---|
| Precision@k | Share of the top *k* results that are relevant | "How clean is the first page?" |
| **Recall@k** ★ | Share of all relevant documents that appear in the top *k* | "Did the right passage make it into the candidates?" — key for the retrieval stage before reranking |
| Hit rate@k | Share of queries with at least one relevant result in the top *k* | Simpler version of recall@k |
| **MRR** ★ | Mean Reciprocal Rank: average of 1 / (rank of the first relevant result) | First hit at rank 1 → 1.0, rank 2 → 0.5, rank 10 → 0.1 |
| MAP | Mean Average Precision: average of precision at each relevant result's position, averaged over queries | Rewards putting *all* relevant results high |
| **nDCG@k** ★ | Normalized Discounted Cumulative Gain. Sums relevance scores, discounting each by log₂(rank + 1), then divides by the best possible ordering | 0–1. Handles graded relevance (very relevant vs somewhat relevant). The standard search-quality metric |

---

## 4. LLM and RAG
*Generator, Judge, the agentic loop*

| Metric | Definition | Read it as |
|---|---|---|
| **Faithfulness / groundedness** ★ | Share of claims in the answer that are supported by the retrieved passages | Hallucination check. Usually scored by an LLM judge or RAGAS |
| **Citation accuracy** ★ | Share of citations that actually support the sentence they're attached to | Are the sources real support, or decoration? |
| Answer relevance | How well the answer addresses the question | An answer can be faithful but off-topic |
| Context precision | Share of retrieved passages that were actually useful | Retrieval noise reaching the LLM |
| Context recall | Whether the retrieved passages contain everything needed to answer | Retrieval misses |
| Exact match (EM) | Answer exactly equals the reference | Only for short factual answers |
| ROUGE / BLEU | Word-overlap between generated and reference text | Cheap but weak — good answers can be worded differently |
| Perplexity | How surprised the model is by real text: exp(average negative log-likelihood) | Lower = better language modelling. Tracked during fine-tuning |
| **Judge agreement** ★ | Share of cases where the LLM/classifier judge matches human labels | Can you trust your automatic evaluator? |
| Cohen's kappa | Agreement corrected for chance | 0 = chance, 1 = perfect. Use for judge vs human and human vs human |
| Win rate | Share of head-to-head comparisons a model wins (e.g. LoRA vs base) | Common for comparing LLM versions |

---

## 5. Systems and efficiency
*Platform, quantization, serving — every row of the scoreboard*

| Metric | Definition | Read it as |
|---|---|---|
| **Latency p50 / p95 / p99** ★ | Response time at the 50th, 95th, 99th percentile | p50 = typical user; p95/p99 = slow tail. Production SLAs are set on p95/p99 |
| Throughput | Requests (or tokens) handled per second | Capacity under load |
| Time to first token (TTFT) | Delay before an LLM starts streaming | What users feel as "speed" in chat |
| Tokens per second | LLM generation speed | Compare quantized vs full precision |
| **Peak memory** ★ | Maximum RAM / GPU memory used | Decides what hardware it fits on |
| **Model size** ★ | Size on disk (MB/GB) | Quantization should shrink this |
| **Parameter count** ★ | Number of trainable weights | Rough measure of model capacity and cost |
| FLOPs | Floating-point operations per forward pass | Hardware-independent measure of compute |
| **GPU hours** ★ | GPU time used for training | Compute cost of a run |
| Cost per 1k requests | Serving cost normalized by usage | How production teams compare options |
| Cold start | Time to load a model before the first request | Matters for serverless deployments |
| **Quantization delta** ★ | Quality metric after quantization minus before | The price you pay for speed and size |

---

## 6. Training diagnostics
*Not scoreboard numbers — what you watch while training*

| Signal | Definition | What it tells you |
|---|---|---|
| Training loss | Loss on the data being trained on | Should fall steadily |
| Validation loss | Loss on held-out data | Rising while training loss falls = overfitting |
| Generalization gap | Validation metric − training metric | Large gap = overfitting |
| Gradient norm | Size of the gradient vector each step | Exploding (huge/NaN) or vanishing (≈0) gradients |
| Learning rate | Step size of each update | Too high → loss jumps or NaN; too low → loss barely moves |

---

## 7. Online and business metrics
*Not in Tensa (no users, no logging) — but standard at Shopify*

| Metric | Definition | Read it as |
|---|---|---|
| CTR | Click-through rate: clicks / impressions | Do people engage with what you showed them? |
| Conversion rate | Purchases (or goal actions) / sessions | The business outcome most models serve |
| Add-to-cart rate | Add-to-carts / product views | Mid-funnel signal for search and recommendations |
| Revenue per session / GMV | Revenue generated per visit; Gross Merchandise Volume in total | The top-line number ML changes are judged on |
| Coverage | Share of the catalogue a recommender ever shows | Low coverage = only popular items surface |
| Diversity / novelty | How different recommended items are from each other / from what users already know | Guards against repetitive recommendations |
| A/B test lift | Relative change in a metric between control and treatment | The standard way models are launched |
| p-value / confidence interval | How likely the lift is due to chance / range the true lift likely falls in | Decides whether a lift is real |
| Guardrail metrics | Metrics that must *not* get worse (latency, error rate, refunds) | A model that lifts CTR but slows the page can still fail |
| Offline–online gap | Difference between offline eval gains and live A/B results | Why offline nDCG gains don't always ship |

---

## 8. Data quality and drift
*What you'll check once a model is in production*

| Metric | Definition | Read it as |
|---|---|---|
| Class balance | Share of examples per class | Explains why accuracy can lie |
| Label noise rate | Share of labels that are wrong (estimated by spot-checks) | A ceiling on achievable quality |
| PSI | Population Stability Index: how much a feature's distribution shifted between training and now | > 0.2 usually means meaningful drift |
| KL divergence | Distance between two probability distributions | Used for drift and in training losses |
| Data leakage | Test information present in training data | Gives scores that are too good to be true |

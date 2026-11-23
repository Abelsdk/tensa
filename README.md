# Tensa

An open-source, ML-centric companion for people dealing with an ACL injury or ACL surgery.

> ⚠️ **Tensa is not a medical device.** Everything it outputs is educational, cited, and red flags are escalated to a clinician. It does not diagnose.

Tensa answers three things:

1. **Ask** — "What should I be able to do at week 6?" → a grounded, cited answer (agentic RAG).
2. **Scan** — upload a knee MRI slice → likely finding (ACL tear / meniscus tear / abnormal) and the relevant post-op protocol.
3. **Timeline** — enter your situation → estimated days to return to sport, with a range.

All of it sits behind an intent router and a safety layer, on a platform that serves the models and records every metric.

## How it's built

Tensa is also a learning project, built in the open:

- Every model is **written from scratch in NumPy first**, then rebuilt in PyTorch, then fine-tuned, quantized, served and measured.
- Every model reports its numbers on a public **scoreboard** — including what broke.

See [`docs/Tensa_V1_Plan.md`](docs/Tensa_V1_Plan.md) for the full plan, [`docs/metrics.md`](docs/metrics.md) for metric definitions, and [`docs/learning_log.md`](docs/learning_log.md) for progress.

## Status

🚧 Stage 1 — NumPy from scratch. Nothing runnable yet.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repo layout

```
data/       download/generation scripts (datasets themselves are gitignored)
ml/
  scratch/  NumPy implementations: linreg, logreg, bm25, cosine search
  router/   intent classifier
  retrieval/ embedder, hybrid search, reranker
  judge/    "is there enough evidence?" classifier
  generator/ LoRA fine-tuned answer model
  scan/     MRI classifiers (CNN, ResNet-18)
  timeline/ cohort simulator, MLP
eval/       eval harness + scoreboard
serving/    FastAPI app
web/        frontend
docs/       plan, metrics, learning log
```

## License

[MIT](LICENSE)

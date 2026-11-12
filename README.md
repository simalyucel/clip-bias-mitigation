# Bias Mitigation in Vision-Language Retrieval

Master's thesis project (University of Bologna) on measuring and mitigating social bias in
CLIP-style vision-language models for **text-to-image retrieval**, using pre-, in- and
post-processing techniques.

## Scope

- **Baseline:** pretrained CLIP (and variants) with no mitigation
- **Attributes:** gender, race, age, religion, disability
- **Fairness metrics:** Skew@k, MaxSkew@k, MinSkew@k, NDKL (Geyik et al., 2019); rND, rKL, rRD (Yang & Stoyanovich, 2017)
- **Utility metrics:** Recall@K, NDCG, mAP

## Repository layout

```
configs/          YAML configs: models, query sets, experiments
src/vlbias/
  models/         model loading wrappers (open_clip, HF)
  data/           dataset loaders (FairFace, UTKFace, ...) with attribute labels
  retrieval/      embedding, caching, ranking
  metrics/        fairness + utility metrics
  mitigation/
    pre/          pre-processing methods (data balancing, augmentation)
    in_processing/ in-processing methods (e.g. prompt learning, fine-tuning)
    post/         post-processing methods (projection, re-ranking)
  utils/          seeding, logging, I/O
scripts/          entry points (run_baseline.py, ...)
notebooks/        exploration only, never the source of reported numbers
tests/            unit tests (metrics first)
data/             local only, git-ignored (raw / processed / embeddings)
results/          experiment outputs, git-ignored except summaries
docs/             notes; the thesis report lives on Overleaf
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Roadmap

- [ ] Phase 0 - compare CLIP variants, set up repo
- [ ] Phase 1 - metrics module + unit tests
- [ ] Phase 2 - query sets
- [ ] Phase 3 - baseline on FairFace (+ retrieval utility)
- [ ] Phase 4 - datasets for all five attributes
- [ ] Phase 5 - mitigation methods

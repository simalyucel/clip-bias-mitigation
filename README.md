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

## Running the baseline

```bash
python scripts/run_baseline.py --config configs/experiments/baseline_v1_berg.yaml   # Berg et al. protocol
python scripts/run_baseline.py --config configs/experiments/baseline_v1.yaml        # 0.25 crop, full split
```

Outputs go to the config's `output_dir`: `per_query.csv`, `per_group.csv`, `summary.csv`,
`chance.csv` (random-ranking chance level), `utility.csv` (Flickr30K Recall@K), `config_used.yaml`.

Notebooks: `01` first look at CLIP, `02` exploration and validation against published numbers,
`03` the final baseline v1 (significance, adult age bands, per-concept results, utility).

## Roadmap

- [ ] Phase 0 - compare CLIP variants, set up repo
- [x] Phase 1 - metrics module + unit tests
- [x] Phase 2 - query sets (Chuang et al. 2023, Berg et al. 2022, occupations)
- [x] Phase 3a - baseline pipeline validated against Berg et al. (2022)
- [ ] Phase 3b - baseline v1: significance, adult age bands, per-concept results, Flickr30K utility (notebook 03)
- [ ] Phase 4 - datasets for all five attributes
- [ ] Phase 5 - mitigation methods

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
  pipeline.py     common evaluation pipeline: same protocol for every method (notebook 04)
  mitigation/
    base.py       the Mitigation interface every method implements (+ NoMitigation, Compose)
    pre/          pre-processing: embedding transforms before scoring (Chuang, Zhao, BendVLM)
    in_processing/ in-processing: model adaptation (Berg et al. prompt learning)
    post/         post-processing: re-ranking of the results (Kong et al.)
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
`03` the final baseline v1 (significance, adult age bands, per-concept results, utility),
`04` the common evaluation pipeline that every mitigation method goes through.

## Evaluating a mitigation method

```python
from vlbias.pipeline import EvaluationPipeline, compare, compare_costs
from vlbias.mitigation.base import NoMitigation

pipe = EvaluationPipeline(load_yaml("configs/experiments/pipeline_v1.yaml"))
base = pipe.run(NoMitigation())        # C0: must reproduce notebook 03
res  = pipe.run(MyMethod())            # any subclass of vlbias.mitigation.base.Mitigation
compare([base, res], reference="gallery", k=1000)   # bias + utility, change vs C0
compare_costs([base, res])                          # set-up time, latency, GPU memory, added params
```

Each run is saved to `results/pipeline_v1/<method>/` (`per_query.csv`, `per_group.csv`, `summary.csv`,
`chance.csv`, `utility.csv`, `costs.csv`, `method.yaml`, `config_used.yaml`).

## Roadmap

- [ ] Phase 0 - compare CLIP variants, set up repo
- [x] Phase 1 - metrics module + unit tests
- [x] Phase 2 - query sets (Chuang et al. 2023, Berg et al. 2022, occupations)
- [x] Phase 3a - baseline pipeline validated against Berg et al. (2022)
- [x] Phase 3b - baseline v1: significance, adult age bands, per-concept results, Flickr30K utility (notebook 03)
- [ ] Phase 3c - common evaluation pipeline for all methods (notebook 04)
- [ ] Phase 4 - datasets for all five attributes
- [ ] Phase 5 - mitigation methods

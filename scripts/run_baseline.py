"""Run the no-mitigation baseline: embed gallery, rank per query, compute metrics.

Usage (from the repo root):
    python scripts/run_baseline.py --config configs/experiments/baseline_fairface.yaml
"""
import argparse

from vlbias.baseline import run_baseline
from vlbias.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out = run_baseline(cfg)
    summary = out["summary"]
    print(summary.round(3).to_string(index=False))
    print(f"\nSaved to {cfg['output_dir']}/")


if __name__ == "__main__":
    main()

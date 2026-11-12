"""Run the no-mitigation baseline: embed gallery, rank per query, compute metrics.

Usage:
    python scripts/run_baseline.py --config configs/experiments/baseline_fairface.yaml
"""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise NotImplementedError("Implemented in Phase 3")


if __name__ == "__main__":
    main()

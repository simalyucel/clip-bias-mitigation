"""Plots shared by the baseline and later mitigation notebooks.

Convention everywhere: red = over-represented, blue = under-represented, and a result
that is NOT distinguishable from chance is shown in grey.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vlbias.analysis import to_matrix

GREY = "#B0B0B0"


def plot_composition(labels: dict, label_names: dict, title: str = "Gallery composition"):
    """Share of each group per attribute, with the uniform share as a dashed line."""
    attrs = list(labels)
    widths = [max(2, len(label_names[a])) for a in attrs]
    fig, axes = plt.subplots(1, len(attrs), figsize=(1.1 * sum(widths) + 2, 3.5),
                             gridspec_kw={"width_ratios": widths})
    for ax, a in zip(np.atleast_1d(axes), attrs):
        lab = labels[a][labels[a] >= 0]
        share = np.bincount(lab, minlength=len(label_names[a])) / len(lab)
        ax.bar(label_names[a], share, color="#4C72B0")
        ax.axhline(1 / len(share), ls="--", c="k", lw=1, label="uniform")
        excluded = int((labels[a] < 0).sum())
        ax.set_title(a + (f"  ({excluded} excluded)" if excluded else ""))
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_concept_heatmap(cs: pd.DataFrame, groups: list[str], title: str = ""):
    """Concept x group mean Skew. Coloured cells = significant; grey text = within chance."""
    values = to_matrix(cs, groups, "skew")
    sig = to_matrix(cs, groups, "significant").astype(bool) if "significant" in cs else None
    shown = values.where(sig) if sig is not None else values
    lim = max(0.3, np.nanmax(np.abs(values.values)))
    fig, ax = plt.subplots(figsize=(1.0 * values.shape[1] + 3.5, 0.32 * values.shape[0] + 1.6))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#F2F2F2")
    im = ax.imshow(np.ma.masked_invalid(shown.values.astype(float)), cmap=cmap,
                   vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(values.shape[1]), values.columns, rotation=45, ha="right")
    ax.set_yticks(range(values.shape[0]), [f"{c}  ·  {cat}" for cat, c in values.index], fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values.values[i, j]
            if np.isnan(v):
                continue
            is_sig = sig is None or bool(sig.values[i, j])
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                    color="black" if is_sig else "#8C8C8C", fontweight="bold" if is_sig else "normal")
    plt.colorbar(im, ax=ax, label="mean Skew (significant cells only)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_concept_bars(cs: pd.DataFrame, group: str, title: str = "", xlabel: str = ""):
    """Skew of one group per concept: bar = observed, grey band = 95% chance range."""
    d = cs[cs["group"] == group].sort_values("skew")
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(d) + 1.2))
    y = np.arange(len(d))
    if "chance_lo" in d:
        ax.barh(y, d["chance_hi"] - d["chance_lo"], left=d["chance_lo"], height=0.9,
                color="#E6E6E6", label="95% chance range")
        colors = np.where(~d["significant"], GREY, np.where(d["skew"] > 0, "#C44E52", "#4C72B0"))
    else:
        colors = np.where(d["skew"] > 0, "#C44E52", "#4C72B0")
    ax.barh(y, d["skew"], height=0.5, color=colors)
    ax.set_yticks(y, d["concept"])
    ax.axvline(0, c="k", lw=0.8)
    ax.set_xlabel(xlabel or f"Skew of '{group}'")
    ax.set_title(title)
    if "chance_lo" in d:
        ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_template_spread(per_query: pd.DataFrame, attribute: str, query_set: str,
                         reference: str = "gallery", k: int = 100, metric: str = "max_skew"):
    """Per concept: mean (dot) and min-max range (line) of a metric across templates."""
    d = per_query.query("query_set == @query_set and attribute == @attribute "
                        "and reference == @reference and k == @k")
    spread = d.groupby("concept")[metric].agg(["mean", "std", "min", "max"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, 0.25 * len(spread) + 1.2))
    ax.hlines(spread.index, spread["min"], spread["max"], color="#999", lw=3)
    ax.plot(spread["mean"], spread.index, "o", color="k", ms=4)
    ax.set_xlabel(f"{metric}@{k} ({attribute}): min - mean - max over templates")
    plt.tight_layout()
    plt.show()
    return spread

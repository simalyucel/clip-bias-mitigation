"""The common interface every mitigation method implements.

A method can act at three points of the retrieval pipeline (Roberta's categories):

    pre  : change the embeddings before similarity scoring    -> transform_text / transform_image
           (e.g. Chuang et al. 2023, Zhao et al. 2026, BendVLM)
    in   : change the model itself                             -> adapt_model
           (e.g. Berg et al. 2022, learned debiasing prompts)
    post : change the ranking after similarity scoring         -> rank
           (e.g. Kong et al. 2023, re-ranking)

Every hook has a "do nothing" default, so a method only overrides what it changes.
``NoMitigation`` overrides nothing: it is the unmitigated baseline (C0).

To add a method, subclass ``Mitigation``, set ``name`` and ``category``, and override the
hooks it needs (see tests/test_mitigation_pipeline.py for small examples).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from vlbias.retrieval.rank import top_k_indices

CATEGORIES = ("none", "pre", "in", "post", "combined")


@dataclass
class FitContext:
    """Everything a method may need to set itself up (``Mitigation.fit``).

    vlm          : the (possibly adapted) model, for encoding prompts
    encode_texts : function(list[str]) -> L2-normalised text embeddings
    attributes   : the attributes being evaluated, e.g. ["gender", "race", "age_adult"]
    fit_data     : function() -> (image_emb, labels, label_names) from the *fit* split
                   (FairFace train by default), loaded only if a method calls it.
                   Methods that learn from labelled images must use this, never the
                   evaluation gallery, otherwise they are tested on their training data.
    """

    vlm: Any
    encode_texts: Callable[[Sequence[str]], np.ndarray]
    attributes: list[str]
    fit_data: Callable[[], tuple[np.ndarray, dict[str, np.ndarray], dict[str, list[str]]]] | None = None
    extra: dict = field(default_factory=dict)


class Mitigation:
    """Base class. All hooks default to "change nothing"."""

    name: str = "mitigation"
    category: str = "none"

    # --- set-up -----------------------------------------------------------------------
    def fit(self, ctx: FitContext) -> None:
        """Learn whatever the method needs (projection matrix, classifier, ...). Timed as set-up cost."""

    # --- in-processing ----------------------------------------------------------------
    def adapt_model(self, vlm):
        """Return the model to use (a fine-tuned / prompt-adapted one). Default: unchanged."""
        return vlm

    # --- pre-processing ---------------------------------------------------------------
    def transform_text(self, emb: np.ndarray) -> np.ndarray:
        return emb

    def transform_image(self, emb: np.ndarray) -> np.ndarray:
        return emb

    # --- post-processing --------------------------------------------------------------
    def rank(self, query_emb: np.ndarray, gallery_emb: np.ndarray, k: int) -> np.ndarray:
        """Indices of the top-k gallery images per query, best first. Default: by cosine similarity."""
        return top_k_indices(query_emb, gallery_emb, k)

    # --- bookkeeping ------------------------------------------------------------------
    def params(self) -> dict:
        """Settings to save with the results (hyper-parameters etc.)."""
        return {}

    def extra_parameters(self) -> int:
        """Number of learned parameters the method adds (0 for training-free methods)."""
        return 0

    def overrides(self, hook: str) -> bool:
        """True if this method changes ``hook`` (so the pipeline can skip work it doesn't need)."""
        return getattr(type(self), hook) is not getattr(Mitigation, hook)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, category={self.category!r})"


class NoMitigation(Mitigation):
    """The unmitigated reference configuration C0: plain CLIP."""

    name = "none"
    category = "none"


class Compose(Mitigation):
    """Apply several methods together, e.g. a pre-processing projection + post-processing re-ranking.

    Order: models are adapted in sequence, then text/image transforms are applied in sequence,
    then the ranking of the LAST method that re-ranks is used (two re-rankers cannot both decide
    the final order).
    """

    category = "combined"

    def __init__(self, methods: Sequence[Mitigation], name: str | None = None):
        self.methods = list(methods)
        self.name = name or "+".join(m.name for m in self.methods)
        rankers = [m for m in self.methods if m.overrides("rank")]
        if len(rankers) > 1:
            raise ValueError(f"only one re-ranking method can be combined, got {[m.name for m in rankers]}")
        self._ranker = rankers[0] if rankers else None

    def fit(self, ctx):
        for m in self.methods:
            m.fit(ctx)

    def adapt_model(self, vlm):
        for m in self.methods:
            vlm = m.adapt_model(vlm)
        return vlm

    def transform_text(self, emb):
        for m in self.methods:
            emb = m.transform_text(emb)
        return emb

    def transform_image(self, emb):
        for m in self.methods:
            emb = m.transform_image(emb)
        return emb

    def rank(self, query_emb, gallery_emb, k):
        if self._ranker is None:
            return top_k_indices(query_emb, gallery_emb, k)
        return self._ranker.rank(query_emb, gallery_emb, k)

    def overrides(self, hook):
        return any(m.overrides(hook) for m in self.methods)

    def params(self):
        return {m.name: {"category": m.category, **m.params()} for m in self.methods}

    def extra_parameters(self):
        return sum(m.extra_parameters() for m in self.methods)


def normalize(emb: np.ndarray) -> np.ndarray:
    """L2-normalise rows (call this at the end of any transform: ranking assumes unit vectors)."""
    emb = np.asarray(emb, dtype=np.float32)
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)

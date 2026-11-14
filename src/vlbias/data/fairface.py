"""FairFace (Kärkkäinen & Joo, 2021) loader via the Hugging Face Hub.

Validation split: ~10.9k face crops labelled with gender (2), race (7) and age (9 bins).
"""
from __future__ import annotations

import numpy as np

from vlbias.data.gallery import LabelledGallery

HF_ID = "HuggingFaceM4/FairFace"
ATTRIBUTES = ("gender", "race", "age")


def load_fairface(
    split: str = "validation",
    padding: str = "0.25",
    max_images: int | None = None,
    seed: int = 0,
) -> LabelledGallery:
    """Load FairFace as a LabelledGallery.

    padding    : "0.25" (tight crop, used in most bias papers) or "1.25" (more context)
    max_images : random subset for quick runs; None = whole split
    """
    from datasets import load_dataset

    ds = load_dataset(HF_ID, padding, split=split)
    if max_images is not None and max_images < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_images))

    labels = {a: np.asarray(ds[a], dtype=int) for a in ATTRIBUTES}
    names = {a: list(ds.features[a].names) for a in ATTRIBUTES}
    suffix = f"-n{len(ds)}" if max_images is not None else ""
    return LabelledGallery(images=ds, labels=labels, label_names=names,
                           name=f"fairface-{padding}-{split}{suffix}")

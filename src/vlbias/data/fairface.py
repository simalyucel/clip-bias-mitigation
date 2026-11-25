"""FairFace (Kärkkäinen & Joo, 2021) loader via the Hugging Face Hub.

Validation split: ~10.9k face crops labelled with gender (2), race (7) and age (9 bins).
"""
from __future__ import annotations

import numpy as np

from vlbias.data.gallery import LabelledGallery

HF_ID = "HuggingFaceM4/FairFace"
ATTRIBUTES = ("gender", "race", "age")


def balanced_indices(labels: np.ndarray, seed: int = 1) -> np.ndarray:
    """Sorted indices of a subsample with equally many items of every group.

    Mirrors Berg et al. (2022)'s ``equal_split`` for gender: every group is randomly
    downsampled to the size of the smallest group.
    """
    labels = np.asarray(labels)
    groups = np.unique(labels)
    n = min(int((labels == g).sum()) for g in groups)
    rng = np.random.default_rng(seed)
    keep = [rng.choice(np.flatnonzero(labels == g), n, replace=False) for g in groups]
    return np.sort(np.concatenate(keep))


def load_fairface(
    split: str = "validation",
    padding: str = "0.25",
    max_images: int | None = None,
    seed: int = 0,
    balance_gender: bool = False,
    balance_seed: int = 1,
) -> LabelledGallery:
    """Load FairFace as a LabelledGallery.

    padding        : "0.25" (tight face crop) or "1.25" (wider crop, used by Berg et al. 2022)
    max_images     : random subset for quick runs; None = whole split
    balance_gender : keep equally many men and women (Berg et al. 2022's protocol)
    """
    from datasets import load_dataset

    ds = load_dataset(HF_ID, str(padding), split=split)
    if max_images is not None and max_images < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_images))
    if balance_gender:
        keep = balanced_indices(np.asarray(ds["gender"]), balance_seed)
        ds = ds.select(keep.tolist())

    labels = {a: np.asarray(ds[a], dtype=int) for a in ATTRIBUTES}
    names = {a: list(ds.features[a].names) for a in ATTRIBUTES}
    name = f"fairface-{padding}-{split}"
    if max_images is not None:
        name += f"-n{max_images}"
    if balance_gender:
        name += f"-gbal{balance_seed}"
    return LabelledGallery(images=ds, labels=labels, label_names=names, name=name)


def load_gallery(ds_cfg: dict, seed: int = 0) -> LabelledGallery:
    """Gallery from the ``dataset:`` block of an experiment config."""
    if ds_cfg["name"] != "fairface":
        raise NotImplementedError(f"dataset {ds_cfg['name']!r}")
    return load_fairface(
        split=ds_cfg.get("split", "validation"),
        padding=str(ds_cfg.get("padding", "0.25")),
        max_images=ds_cfg.get("max_images"),
        seed=seed,
        balance_gender=ds_cfg.get("balance_gender", False),
        balance_seed=ds_cfg.get("balance_seed", 1),
    )

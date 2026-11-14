"""Flickr30K 1k test split (Karpathy split) for retrieval utility: 1,000 images x 5 captions.

Standard benchmark for text-to-image retrieval (Recall@K). Berg et al. (2022) report
text-to-image Recall@5 = 83.6 for CLIP ViT-B/32 on it.

Hugging Face hosts it in different layouts, so the loader is defensive: it finds the
image column and the caption column itself, handles one-row-per-image (a list of 5
captions) and one-row-per-caption layouts, keeps only the test split if the dataset has a
"split" column, and checks that the result has the expected size.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_IDS = (
    "nlphuji/flickr_1k_test_image_text_retrieval",  # small: the 1k test split only
    "nlphuji/flickr30k",                            # full 31k set (~4.4 GB), filtered to test
)
CAPTION_COLUMNS = ("caption", "captions", "raw", "sentences", "text")
IMAGE_KEY_COLUMNS = ("filename", "img_id", "image_id", "file_name")


@dataclass
class RetrievalDataset:
    images: Any                      # indexable; item i is a dict with an "image" key
    captions: list[str]
    caption_to_image: np.ndarray     # index of the image each caption describes
    name: str

    def __len__(self) -> int:
        return len(self.images)


def _load_any(hf_id: str):
    from datasets import load_dataset

    try:
        ds = load_dataset(hf_id)
    except Exception:
        ds = load_dataset(hf_id, trust_remote_code=True)
    if hasattr(ds, "keys"):  # DatasetDict: prefer a test split, else the only split
        split = "test" if "test" in ds else list(ds.keys())[0]
        ds = ds[split]
    return ds


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            return [str(v) for v in json.loads(value)]
        except json.JSONDecodeError:
            pass
    return [str(value)]


def from_hf_dataset(ds, name: str) -> RetrievalDataset:
    """Normalise a Hugging Face Flickr30K dataset into images + captions."""
    if "split" in ds.column_names:
        ds = ds.filter(lambda s: s == "test", input_columns="split")
    image_cols = [c for c, f in ds.features.items() if type(f).__name__ == "Image"]
    if not image_cols:
        raise ValueError(f"no image column in {ds.column_names}")
    if image_cols[0] != "image":
        ds = ds.rename_column(image_cols[0], "image")
    cap_col = next((c for c in CAPTION_COLUMNS if c in ds.column_names), None)
    if cap_col is None:
        raise ValueError(f"no caption column in {ds.column_names}")

    raw = [_as_list(v) for v in ds[cap_col]]
    if all(len(c) > 1 for c in raw):          # one row per image, several captions
        images = ds
        captions = [c for cs in raw for c in cs]
        owner = np.repeat(np.arange(len(raw)), [len(cs) for cs in raw])
    else:                                     # one row per caption: group rows by image
        key_col = next((c for c in IMAGE_KEY_COLUMNS if c in ds.column_names), None)
        if key_col is None:
            raise ValueError(f"one caption per row but no image id column in {ds.column_names}")
        keys = ds[key_col]
        first: dict[Any, int] = {}
        for i, k in enumerate(keys):
            first.setdefault(k, i)
        order = {k: j for j, k in enumerate(first)}
        images = ds.select(list(first.values()))
        captions = [cs[0] for cs in raw]
        owner = np.array([order[k] for k in keys])
    return RetrievalDataset(images, captions, owner, name)


def load_flickr30k_test(hf_id: str | None = None, check: bool = True) -> RetrievalDataset:
    """Flickr30K 1k test split. Tries ``hf_id`` (if given), then the known mirrors."""
    errors = []
    for candidate in ([hf_id] if hf_id else []) + list(DEFAULT_IDS):
        try:
            data = from_hf_dataset(_load_any(candidate), f"flickr30k-test-{candidate.split('/')[-1]}")
        except Exception as e:  # try the next mirror, report all failures at the end
            errors.append(f"{candidate}: {type(e).__name__}: {e}")
            continue
        if check and not (900 <= len(data) <= 1100 and 4500 <= len(data.captions) <= 5500):
            errors.append(f"{candidate}: unexpected size {len(data)} images / {len(data.captions)} captions")
            continue
        return data
    raise RuntimeError("Could not load the Flickr30K 1k test split:\n" + "\n".join(errors))

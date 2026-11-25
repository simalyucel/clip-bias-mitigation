"""Encode texts and images into L2-normalised embeddings (numpy float32)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from vlbias.data.gallery import LabelledGallery


class _ImageDataset:
    """Wraps any indexable image collection so a DataLoader can preprocess in parallel."""

    def __init__(self, images, preprocess):
        self.images, self.preprocess = images, preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        item = self.images[i]
        image = item["image"] if isinstance(item, dict) else item
        return self.preprocess(image.convert("RGB"))


def encode_texts(vlm, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            tokens = vlm.tokenizer(list(texts[i : i + batch_size])).to(vlm.device)
            e = vlm.model.encode_text(tokens).float()
            out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def encode_images(vlm, images, batch_size: int = 256, num_workers: int = 2,
                  show_progress: bool = True) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(_ImageDataset(images, vlm.preprocess), batch_size=batch_size,
                        num_workers=num_workers, pin_memory=(vlm.device == "cuda"))
    if show_progress:
        from tqdm.auto import tqdm

        loader = tqdm(loader, desc="encoding images")
    out = []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type="cuda", enabled=(vlm.device == "cuda")):
                e = vlm.model.encode_image(batch.to(vlm.device, non_blocking=True))
            e = e.float()
            out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def gallery_embeddings(vlm, gallery: LabelledGallery, cache_dir: str | Path | None = None,
                       **kwargs) -> np.ndarray:
    """Image embeddings for a gallery, cached as <cache_dir>/<model>__<gallery>.npy."""
    if cache_dir is not None:
        path = Path(cache_dir) / f"{vlm.name}__{gallery.name}.npy"
        if path.exists():
            emb = np.load(path)
            if len(emb) == len(gallery):
                return emb
    emb = encode_images(vlm, gallery.images, **kwargs)
    if cache_dir is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
    return emb

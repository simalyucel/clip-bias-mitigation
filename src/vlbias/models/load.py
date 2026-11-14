"""Load a vision-language model from a model config (configs/models/*.yaml)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VLModel:
    model: Any
    preprocess: Any
    tokenizer: Any
    device: str
    name: str


def load_model(cfg: dict, device: str | None = None) -> VLModel:
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    library = cfg.get("library", "open_clip")
    if library != "open_clip":
        raise NotImplementedError(f"library {library!r} not supported yet")

    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["architecture"], pretrained=cfg["pretrained"]
    )
    tokenizer = open_clip.get_tokenizer(cfg["architecture"])
    model = model.to(device).eval()
    return VLModel(model, preprocess, tokenizer, device, cfg["name"])

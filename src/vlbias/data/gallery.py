"""Labelled image galleries (the collection that is searched)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LabelledGallery:
    """Images plus one integer label array per protected attribute.

    images      : indexable collection; item i is a PIL image or a dict with an "image" key
    labels      : attribute -> int array of length len(images)
    label_names : attribute -> list of group names (index = label id)
    name        : used for caching embeddings, e.g. "fairface-0.25-validation"
    """

    images: Any
    labels: dict[str, np.ndarray]
    label_names: dict[str, list[str]]
    name: str

    def __len__(self) -> int:
        return len(next(iter(self.labels.values())))

    @property
    def attributes(self) -> list[str]:
        return list(self.labels)

    def composition(self, attribute: str) -> dict[str, float]:
        counts = np.bincount(self.labels[attribute], minlength=len(self.label_names[attribute]))
        return {n: c / len(self) for n, c in zip(self.label_names[attribute], counts)}

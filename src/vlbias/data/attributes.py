"""Derived attributes: regroup an attribute's labels into coarser or restricted groups.

An attribute *spec* is either a base attribute of the gallery ("gender", "race", "age") or
"<base>_<scheme>" for a regrouping defined in SCHEMES, e.g.:

    age_adult  : 20-29, 30-39, 40-49, 50-59, 60+   (children and teenagers EXCLUDED)
    age_coarse : 0-19, 20-39, 40-59, 60+           (everyone kept, fewer groups)

Excluded images get label -1. The evaluation then ranks only the images that have a valid
label for that attribute, i.e. it searches an adult-only gallery for "age_adult".
Why: every prompt describes a "person" or an occupation, so children are (legitimately)
almost never retrieved; with the 9 fine age bins, the age numbers mostly measure that.
"""
from __future__ import annotations

import numpy as np

SCHEMES: dict[str, dict[str, dict[str, list[str]]]] = {
    "age": {
        "adult": {
            "20-29": ["20-29"],
            "30-39": ["30-39"],
            "40-49": ["40-49"],
            "50-59": ["50-59"],
            "60+": ["60-69", "more than 70"],
        },
        "coarse": {
            "0-19": ["0-2", "3-9", "10-19"],
            "20-39": ["20-29", "30-39"],
            "40-59": ["40-49", "50-59"],
            "60+": ["60-69", "more than 70"],
        },
    },
}


def regroup(labels: np.ndarray, names: list[str], mapping: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    """Map old group ids to new ones. Old groups not listed in ``mapping`` become -1 (excluded)."""
    new_names = list(mapping)
    old_to_new = np.full(len(names), -1, dtype=int)
    for new_id, new_name in enumerate(new_names):
        for old in mapping[new_name]:
            if old not in names:
                raise ValueError(f"unknown group {old!r}; available: {names}")
            old_to_new[names.index(old)] = new_id
    labels = np.asarray(labels, dtype=int)
    return np.where(labels >= 0, old_to_new[labels], -1), new_names


def resolve_attribute(gallery, spec: str) -> tuple[np.ndarray, list[str]]:
    """Labels and group names for an attribute spec ("age", "age_adult", ...)."""
    if spec in gallery.labels:
        return gallery.labels[spec], gallery.label_names[spec]
    base, _, scheme = spec.partition("_")
    if base not in gallery.labels or scheme not in SCHEMES.get(base, {}):
        known = [f"{b}_{s}" for b, ss in SCHEMES.items() for s in ss]
        raise ValueError(f"unknown attribute {spec!r}; base attributes: {list(gallery.labels)}, "
                         f"derived: {known}")
    return regroup(gallery.labels[base], gallery.label_names[base], SCHEMES[base][scheme])


def resolve_attributes(gallery, specs) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    labels, names = {}, {}
    for spec in specs:
        labels[spec], names[spec] = resolve_attribute(gallery, spec)
    return labels, names

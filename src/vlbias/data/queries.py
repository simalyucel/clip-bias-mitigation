"""Text queries: templates x concepts, read from a YAML query-set file.

Query-set YAML format::

    name: chuang2023
    templates:
      - "A photo of a {} person"
      - "a photo of {a} {}"          # {a} becomes "a" or "an"
    concepts:
      positive: [good, smart]         # category -> list of concepts
      negative: [evil, dumb]
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from vlbias.utils.config import load_yaml


@dataclass(frozen=True)
class Query:
    text: str
    concept: str
    template: str
    category: str
    query_set: str

    def to_dict(self) -> dict:
        return asdict(self)


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def fill_template(template: str, concept: str) -> str:
    return template.replace("{a}", _article(concept)).replace("{}", concept)


def load_queries(path: str | Path) -> list[Query]:
    cfg = load_yaml(path)
    name = cfg["name"]
    templates = list(dict.fromkeys(cfg["templates"]))  # dedupe, keep order
    queries = []
    for category, concepts in cfg["concepts"].items():
        for concept in concepts:
            for template in templates:
                queries.append(Query(fill_template(template, concept), concept, template, category, name))
    return queries

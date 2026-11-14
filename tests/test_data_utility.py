"""Tests for attribute regrouping, gender balancing, Flickr30K normalisation and Recall@K."""
import numpy as np
import pytest

from vlbias.data.attributes import SCHEMES, regroup, resolve_attribute
from vlbias.data.fairface import balanced_indices
from vlbias.data.gallery import LabelledGallery
from vlbias.metrics.utility import retrieval_recall, text_to_image_ranks

AGE = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]


def test_regroup_adult_excludes_children():
    labels = np.arange(9)
    new, names = regroup(labels, AGE, SCHEMES["age"]["adult"])
    assert names == ["20-29", "30-39", "40-49", "50-59", "60+"]
    assert new.tolist() == [-1, -1, -1, 0, 1, 2, 3, 4, 4]


def test_regroup_coarse_keeps_everyone():
    new, names = regroup(np.arange(9), AGE, SCHEMES["age"]["coarse"])
    assert names == ["0-19", "20-39", "40-59", "60+"]
    assert new.tolist() == [0, 0, 0, 1, 1, 2, 2, 3, 3]


def test_resolve_attribute():
    g = LabelledGallery([None] * 9, {"age": np.arange(9)}, {"age": AGE}, "toy")
    assert resolve_attribute(g, "age")[1] == AGE
    assert resolve_attribute(g, "age_adult")[1][-1] == "60+"
    with pytest.raises(ValueError):
        resolve_attribute(g, "age_weird")


def test_balanced_indices():
    labels = np.array([0] * 70 + [1] * 30)
    idx = balanced_indices(labels, seed=1)
    assert len(idx) == 60 and (labels[idx] == 0).sum() == 30
    assert np.array_equal(idx, np.sort(idx))
    assert np.array_equal(idx, balanced_indices(labels, seed=1))   # reproducible


def test_recall_perfect_and_known():
    # 3 images, 2 captions each; captions point exactly at their image
    img = np.eye(3, dtype=np.float32)
    owner = np.array([0, 0, 1, 1, 2, 2])
    txt = img[owner]
    r = retrieval_recall(txt, img, owner, ks=[1])
    assert r["recall"].tolist() == [100.0, 100.0]
    # make caption 0 point at image 1 instead: its correct image drops to rank 1 (0-based)
    txt2 = txt.copy()
    txt2[0] = img[1]
    ranks = text_to_image_ranks(txt2, img, owner)
    assert ranks[0] == 1
    r1 = retrieval_recall(txt2, img, owner, ks=[1, 2]).set_index(["direction", "k"])["recall"]
    assert r1[("text_to_image", 1)] == pytest.approx(500 / 6)
    assert r1[("text_to_image", 2)] == pytest.approx(100.0)
    assert r1[("image_to_text", 1)] == pytest.approx(100.0)   # image 0 still has caption 1


def test_flickr_normalisation_both_layouts():
    datasets = pytest.importorskip("datasets")
    from PIL import Image

    from vlbias.data.flickr30k import from_hf_dataset

    img = Image.new("RGB", (4, 4))
    per_image = datasets.Dataset.from_dict(
        {"image": [img, img], "caption": [["a", "b"], ["c", "d"]], "split": ["test", "test"]},
        features=datasets.Features({"image": datasets.Image(),
                                    "caption": datasets.Sequence(datasets.Value("string")),
                                    "split": datasets.Value("string")}))
    d = from_hf_dataset(per_image, "toy")
    assert len(d) == 2 and d.captions == ["a", "b", "c", "d"] and d.caption_to_image.tolist() == [0, 0, 1, 1]

    per_caption = datasets.Dataset.from_dict(
        {"img": [img] * 4, "text": ["a", "b", "c", "d"], "filename": ["x", "x", "y", "y"]},
        features=datasets.Features({"img": datasets.Image(), "text": datasets.Value("string"),
                                    "filename": datasets.Value("string")}))
    d2 = from_hf_dataset(per_caption, "toy")
    assert len(d2) == 2 and d2.caption_to_image.tolist() == [0, 0, 1, 1]
    assert "image" in d2.images.column_names


def test_flickr_raw_files_loader(tmp_path, monkeypatch):
    """Script-based HF datasets: rebuild from the repo's CSV + image zip (no network)."""
    pytest.importorskip("datasets")
    import json
    import zipfile

    import huggingface_hub
    import pandas as pd
    from PIL import Image

    from vlbias.data import flickr30k as fl

    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for n in ("a.jpg", "b.jpg", "c.jpg"):
        Image.new("RGB", (4, 4)).save(imgs / n)
    zpath = tmp_path / "flickr30k-images.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for n in ("a.jpg", "b.jpg", "c.jpg"):
            z.write(imgs / n, f"flickr30k-images/{n}")
    csv = tmp_path / "flickr_annotations_30k.csv"
    pd.DataFrame({"raw": [json.dumps([f"{x}{i}" for i in range(5)]) for x in "abc"],
                  "split": ["test", "test", "train"], "filename": ["a.jpg", "b.jpg", "c.jpg"]}).to_csv(csv, index=False)

    files = {"flickr_annotations_30k.csv": csv, "flickr30k-images.zip": zpath, "flickr30k.py": None}
    monkeypatch.setattr(huggingface_hub, "list_repo_files", lambda repo, repo_type=None: list(files))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, f, repo_type=None: str(files[f]))

    ds = fl._load_raw_files("fake/repo")
    d = fl.from_hf_dataset(ds, "toy")
    assert len(d) == 2 and len(d.captions) == 10            # train image filtered out
    assert d.caption_to_image.tolist() == [0] * 5 + [1] * 5
    assert d.images[0]["image"].size == (4, 4)

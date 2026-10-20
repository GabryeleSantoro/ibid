"""Topic clustering: pure numpy, so these feed it arrays directly."""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
from ragcore.brain.cluster import build_tree, ctfidf, pca, split


def blobs(k: int, per: int, *, dim: int = 768, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """k well-separated groups of unit vectors, and each row's true group."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((k, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    truth = np.repeat(np.arange(k), per)
    x = centres[truth] + rng.standard_normal((k * per, dim)) * 0.02
    return x / np.linalg.norm(x, axis=1, keepdims=True), truth


def test_separable_blobs_are_recovered() -> None:
    x, truth = blobs(3, 30)

    labels = split(x, np.random.default_rng(0))

    assert len(set(labels)) == 3
    for group in range(3):
        assert len(set(labels[truth == group])) == 1


def test_silhouette_picks_the_true_number_of_clusters() -> None:
    x, _ = blobs(5, 12)

    assert len(set(split(x, np.random.default_rng(0)))) == 5


def test_the_same_seed_builds_the_same_tree() -> None:
    x, _ = blobs(4, 20)
    ids = [f"doc_{i}" for i in range(len(x))]
    terms = [{"word": 1}] * len(x)
    chunks = [f"{i}:1:0" for i in ids]

    first = build_tree(ids, x, terms, chunks)
    second = build_tree(ids, x, terms, chunks)

    assert [(n.id, n.doc_ids) for n in first] == [(n.id, n.doc_ids) for n in second]


def test_pca_with_fewer_rows_than_dimensions_keeps_n_minus_one() -> None:
    x, _ = blobs(2, 5)

    z = pca(x)

    assert z.shape == (10, 9)
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0)


def test_ctfidf_picks_each_clusters_distinguishing_terms() -> None:
    garden = Counter(tomato=5, soil=4, water=3, shared=10)
    sky = Counter(star=5, telescope=4, water=3, shared=10)

    words = ctfidf([garden, sky])

    assert words[0][:2] == ["tomato", "soil"]
    assert words[1][:2] == ["star", "telescope"]


def test_every_document_lands_in_exactly_one_top_level_topic() -> None:
    x, _ = blobs(3, 40)  # 40 per blob: each top topic gets subtopics
    ids = [f"doc_{i}" for i in range(len(x))]

    nodes = build_tree(ids, x, [{}] * len(x), [f"{i}:1:0" for i in ids])

    top = [n for n in nodes if n.level == 0]
    assert sorted(d for n in top for d in n.doc_ids) == sorted(ids)
    assert max(n.level for n in nodes) <= 2
    by_id = {n.id: n for n in nodes}
    for node in nodes:
        if node.parent_id:
            assert set(node.doc_ids) <= set(by_id[node.parent_id].doc_ids)
        assert len(node.rep_chunk_ids) <= 3
        assert abs(float(np.linalg.norm(node.centroid)) - 1) < 1e-5


def test_identical_vectors_do_not_crash() -> None:
    x = np.ones((10, 8)) / np.sqrt(8)
    ids = [f"doc_{i}" for i in range(10)]

    nodes = build_tree(ids, x, [{}] * 10, [f"{i}:1:0" for i in ids])

    assert [n.doc_ids for n in nodes if n.level == 0] == [ids]


def test_five_thousand_documents_cluster_in_under_thirty_seconds() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((5000, 768)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    ids = [f"doc_{i}" for i in range(5000)]

    started = time.perf_counter()
    nodes = build_tree(ids, x, [{}] * 5000, [f"{i}:1:0" for i in ids])

    assert time.perf_counter() - started < 30
    assert sorted(d for n in nodes if n.level == 0 for d in n.doc_ids) == sorted(ids)

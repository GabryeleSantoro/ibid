"""Documents into a topic tree: PCA, spherical k-means, silhouette-picked k, c-TF-IDF labels.

Pure numpy and no I/O, so it runs in a worker thread and tests feed it arrays.

ponytail: k-means has no outlier bucket, so a document that mixes subjects lands in its
strongest one; HDBSCAN is the upgrade if eval shows mixed documents mislabelled.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

SEED = 0
MAX_DIMS = 50
MAX_K = 20
MIN_PER_CLUSTER = 3  # k ≤ n // 3
SPLIT_ABOVE = 30  # a topic with more documents than this gets subtopics
MAX_LEVELS = 3
ITERATIONS = 50
SILHOUETTE_SAMPLE = 2000
KEYWORDS = 5
LABEL_TERMS = 3
REPRESENTATIVES = 3


@dataclass(slots=True)
class Node:
    id: str
    parent_id: str | None
    level: int
    centroid: np.ndarray  # unit vector in the embedding space: assignment and ranking use it
    doc_ids: list[str]  # every document in this subtree
    keywords: list[str] = field(default_factory=list)
    label: str = ""
    rep_chunk_ids: list[str] = field(default_factory=list)


def _unit(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def pca(x: np.ndarray) -> np.ndarray:
    """Project onto min(50, n - 1, dim) principal axes; rows unit length again."""
    n, dim = x.shape
    centred = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return _unit(centred @ vt[: max(1, min(MAX_DIMS, n - 1, dim))].T)


def kmeans(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Spherical k-means (cosine) from a k-means++ start; labels 0..k'-1 with k' ≤ k."""
    n = len(x)
    centres = [x[rng.integers(n)]]
    for _ in range(1, k):
        weights = np.clip(1.0 - np.max(x @ np.stack(centres).T, axis=1), 0, None) ** 2
        total = weights.sum()
        centres.append(x[rng.choice(n, p=weights / total) if total > 0 else rng.integers(n)])
    c = np.stack(centres)
    labels = np.full(n, -1)
    for _ in range(ITERATIONS):
        assigned = np.argmax(x @ c.T, axis=1)
        if np.array_equal(assigned, labels):
            break
        labels = assigned
        for j in range(k):
            members = x[labels == j]
            if len(members):
                c[j] = _unit(members.sum(axis=0))
    return np.unique(labels, return_inverse=True)[1]


def silhouette(dist: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette from a precomputed cosine-distance matrix; -1 for a single cluster."""
    if len(np.unique(labels)) < 2:
        return -1.0
    k = int(labels.max()) + 1
    rows = np.arange(len(labels))
    onehot = np.eye(k)[labels]
    sums = dist @ onehot  # each row's summed distance to each cluster
    sizes = onehot.sum(axis=0)
    own = sizes[labels]
    a = sums[rows, labels] / np.maximum(own - 1, 1)
    other = np.where(sizes > 0, sums / np.maximum(sizes, 1), np.inf)
    other[rows, labels] = np.inf
    b = other.min(axis=1)
    s = np.where(own > 1, (b - a) / np.maximum(np.maximum(a, b), 1e-12), 0.0)
    return float(s.mean())


def split(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Labels for the k in 2..min(20, n // 3) with the best silhouette; all zeros if none splits."""
    n = len(x)
    best = np.zeros(n, dtype=int)
    if n < 2 * MIN_PER_CLUSTER:
        return best
    z = pca(x)
    sample = (
        rng.choice(n, SILHOUETTE_SAMPLE, replace=False) if n > SILHOUETTE_SAMPLE else np.arange(n)
    )
    dist = 1.0 - z[sample] @ z[sample].T
    best_score = -1.0
    for k in range(2, min(MAX_K, n // MIN_PER_CLUSTER) + 1):
        labels = kmeans(z, k, rng)
        score = silhouette(dist, labels[sample])
        if score > best_score:
            best, best_score = labels, score
    return best


def ctfidf(groups: list[Counter[str]]) -> list[list[str]]:
    """Top terms per cluster by tf(t, c) × log(1 + A / f(t)) over sibling clusters."""
    totals: Counter[str] = Counter()
    for group in groups:
        totals.update(group)
    average = sum(totals.values()) / max(len(groups), 1)
    out: list[list[str]] = []
    for group in groups:
        weight = {t: tf * math.log(1 + average / totals[t]) for t, tf in group.items()}
        out.append(sorted(weight, key=lambda t: (-weight[t], t))[:KEYWORDS])
    return out


def build_tree(
    doc_ids: list[str],
    vectors: np.ndarray,
    terms: list[dict[str, int]],
    first_chunks: list[str],
    *,
    seed: int = SEED,
) -> list[Node]:
    """The whole tree, parents before their children, siblings largest first."""
    rng = np.random.default_rng(seed)
    x = _unit(np.asarray(vectors, dtype=np.float32))
    nodes: list[Node] = []

    def summed(rows: np.ndarray) -> Counter[str]:
        total: Counter[str] = Counter()
        for i in rows:
            total.update(terms[i])
        return total

    def grow(rows: np.ndarray, parent: Node | None, level: int) -> None:
        labels = split(x[rows], rng)
        groups = sorted((rows[labels == j] for j in range(labels.max() + 1)), key=len, reverse=True)
        if parent is not None and len(groups) < 2:
            return  # one child identical to its parent says nothing
        for number, (group, keywords) in enumerate(
            zip(groups, ctfidf([summed(g) for g in groups]), strict=True), start=1
        ):
            centroid = _unit(x[group].sum(axis=0))
            closest = group[np.argsort(-(x[group] @ centroid), kind="stable")[:REPRESENTATIVES]]
            node = Node(
                id=f"{parent.id}.{number}" if parent else f"t{number}",
                parent_id=parent.id if parent else None,
                level=level,
                centroid=centroid,
                doc_ids=[doc_ids[i] for i in group],
                keywords=keywords,
                label=" · ".join(keywords[:LABEL_TERMS]) or "untitled",
                rep_chunk_ids=[first_chunks[i] for i in closest],
            )
            nodes.append(node)
            if len(group) > SPLIT_ABOVE and level + 1 < MAX_LEVELS:
                grow(group, node, level + 1)

    grow(np.arange(len(doc_ids)), None, 0)
    return nodes

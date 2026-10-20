"""A document in one record: where it sits in embedding space, its best passages, its words.

Built where the chunk vectors are already in memory (``RealStore._write``), so clustering and
corpus term statistics never read the chunks back.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ragcore.stopwords import ALL_STOPWORDS

SUMMARY_CHUNKS = 3
MAX_TERMS = 300
_TOKEN = re.compile(r"[^\W\d_]{3,}")


@dataclass(frozen=True, slots=True)
class DocCard:
    doc_id: str
    vector: np.ndarray  # float32, unit length: the mean of the document's chunk vectors
    summary_chunk_ids: list[str]  # the chunks closest to that mean, best first
    terms: dict[str, int]  # the most frequent words with their counts


def terms_of(texts: Iterable[str]) -> dict[str, int]:
    counts = Counter(
        token
        for text in texts
        for token in _TOKEN.findall(text.lower())
        if token not in ALL_STOPWORDS
    )
    return dict(counts.most_common(MAX_TERMS))


def build_card(
    doc_id: str,
    chunk_ids: Sequence[str],
    texts: Sequence[str],
    vectors: Sequence[Sequence[float]],
) -> DocCard | None:
    """None for a document with no chunks (parse error, empty file, scan without text)."""
    if not chunk_ids:
        return None
    matrix = np.asarray(vectors, dtype=np.float32)
    mean = matrix.mean(axis=0)
    vector = mean / (np.linalg.norm(mean) or 1.0)
    best = np.argsort(-(matrix @ vector), kind="stable")[:SUMMARY_CHUNKS]
    return DocCard(doc_id, vector, [chunk_ids[i] for i in best], terms_of(texts))

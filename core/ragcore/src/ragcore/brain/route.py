"""Local or global: is the question about passages or about the library as a whole?

Cosine to a few labelled example questions decides; inside the dead zone the keyword hints
do. The manual mode in the request still wins (query route).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# Questions about the corpus as a whole cannot be answered from six passages.
GLOBAL_HINTS = (
    "across",
    "overall",
    "in general",
    "main themes",
    "summarize all",
    "summarise all",
    "every document",
    "all documents",
    "what topics",
    "compare the",
)

# Tuned on eval/golden/{base,brain}.jsonl; keep their exact phrasings out of here. Local
# questions score up to 0.12 there, so global needs a clear lead; local needs only a small one.
GLOBAL_MARGIN = 0.15
LOCAL_MARGIN = 0.05

EXAMPLES: tuple[tuple[str, str], ...] = (
    ("global", "What are the recurring themes across these files?"),
    ("global", "Summarise the collection as a whole"),
    ("global", "Which subjects does this library cover?"),
    ("global", "Give me a bird's-eye view of all the material"),
    ("global", "What is the big picture across every document?"),
    ("global", "Which ideas come up again and again in my notes?"),
    ("global", "How do the different parts of my notes relate to each other?"),
    ("global", "What would you say this archive is mostly about?"),
    ("global", "What are the main threads running through my library?"),
    ("global", "Quali temi ricorrono in tutti questi file?"),
    ("global", "Riassumi la raccolta nel suo insieme"),
    ("global", "Di cosa parla complessivamente questo archivio?"),
    ("global", "Quels sont les grands thèmes de l'ensemble de mes documents ?"),
    ("global", "Worum geht es in meinen Notizen insgesamt?"),
    ("global", "¿De qué tratan mis documentos en conjunto?"),
    ("local", "How does reciprocal rank fusion combine two rankings?"),
    ("local", "What is the deadline mentioned in the contract?"),
    ("local", "Which page explains how to reset the device?"),
    ("local", "What did the report say about revenue in March?"),
    ("local", "Define chunk overlap"),
    ("local", "Who wrote the onboarding guide?"),
    ("local", "What error code does the upload return when the file is too big?"),
    ("local", "How many grams of salt go into the brine?"),
    ("local", "What is the warranty period for the laptop?"),
    ("local", "Come si imposta il timeout del server?"),
    ("local", "Cosa dice il documento sulla garanzia?"),
    ("local", "Qual è la scadenza indicata nella fattura?"),
    ("local", "Comment configurer la sauvegarde automatique ?"),
    ("local", "Wie lange dauert die Lieferung laut Vertrag?"),
    ("local", "¿Qué dice el contrato sobre la cancelación?"),
)

Examples = tuple[np.ndarray, np.ndarray]  # (global rows, local rows), unit vectors


@dataclass(frozen=True, slots=True)
class Routed:
    mode: Literal["local", "global"]
    reason: str
    vector: list[float] | None = None  # the question's embedding, reused by the global path


def keyword_route(question: str, vector: list[float] | None = None) -> Routed:
    lowered = question.lower()
    hit = next((h for h in GLOBAL_HINTS if h in lowered), None)
    if hit:
        return Routed("global", f'Phrase "{hit}" asks about the corpus, not a passage', vector)
    return Routed("local", "Answerable from individual passages", vector)


async def embed_examples(embedder) -> Examples:
    matrix = np.asarray(
        await embedder.embed([text for _, text in EXAMPLES], query=True), dtype=np.float32
    )
    kinds = np.array([kind for kind, _ in EXAMPLES])
    return matrix[kinds == "global"], matrix[kinds == "local"]


def score_route(question: str, vector: list[float], examples: Examples) -> Routed:
    q = np.asarray(vector, dtype=np.float32)
    score = float(np.max(examples[0] @ q) - np.max(examples[1] @ q))
    if score > GLOBAL_MARGIN:
        return Routed("global", f"Global (router {score:.2f})", vector)
    if score < -LOCAL_MARGIN:
        return Routed("local", f"Local (router {score:.2f})", vector)
    hint = keyword_route(question, vector)
    return Routed(hint.mode, f"{hint.reason} (router {score:.2f})", vector)

"""Function words per shipped UI language, shared by language detection and topic terms."""

from __future__ import annotations

# ponytail: stopword vote; swap for a real detector if short or mixed docs misfire.
_WORDS = {
    "en": "the and of to in is that for with as are this by on it from be or an",
    "it": "il lo la le di che e un una per con non sono del della dei nel si come da al è",
    "fr": "le la les de des et un une que pour dans est du au ce pas sur avec qui par",
    "de": "der die das und ist nicht ein eine zu mit den von auf für im dem sich auch",
    "es": "el la los las de que y en un una es por con para del se no al como más",
}

STOPWORDS = {k: frozenset(v.split()) for k, v in _WORDS.items()}
ALL_STOPWORDS = frozenset().union(*STOPWORDS.values())

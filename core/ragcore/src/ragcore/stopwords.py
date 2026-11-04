"""Function words per shipped UI language, shared by language detection and topic terms."""

from __future__ import annotations

# ponytail: stopword vote; swap for a real detector if short or mixed docs misfire.
_WORDS = {
    "en": "the and of to in is that for with as are this by on it from be or an "
    "you your they them their there not but have has had could should would can its into "
    "than then only each one all any some more most other such these those which what when "
    "where who how about been our his her him she just very",
    "it": "il lo la le di che e un una per con non sono del della dei nel si come da al è "
    "questo questa questi quello quella anche molto più ogni sempre quando dove perché "
    "ancora tutto tutti hanno essere stato poi può loro",
    "fr": "le la les de des et un une que pour dans est du au ce pas sur avec qui par "
    "vous nous ils elles leur leurs mais tout tous cette ces sont été être avoir fait peut "
    "aussi très quand comme où",
    "de": "der die das und ist nicht ein eine zu mit den von auf für im dem sich auch "
    "sie wir ihr ihre sein seine nur noch wenn wie aber oder werden wird kann muss sind hat "
    "haben dass diese dieser mehr sehr schon",
    "es": "el la los las de que y en un una es por con para del se no al como más "
    "pero este esta estos estas ese esa todo todos muy cuando donde también sus son sido "
    "ser hay puede porque sobre",
}

STOPWORDS = {k: frozenset(v.split()) for k, v in _WORDS.items()}
ALL_STOPWORDS = frozenset().union(*STOPWORDS.values())

"""Optional lexicon/rule-based sentiment (VADER). Not a neural model.

Labels (on VADER scores):
* uncertain – fewer than 3 words, so the lexicon has too little to go on
* mixed     – positive and negative proportions both >= 0.15
* positive  – compound >= 0.05
* negative  – compound <= -0.05
* neutral   – otherwise

Lexicon sentiment ignores much context (sarcasm, domain usage), so treat it
as a descriptive signal, not a measurement of respondents' feelings.
"""
from __future__ import annotations

import re

LABELS = ("positive", "negative", "neutral", "mixed", "uncertain")


def sentiment_available() -> bool:
    try:
        import vaderSentiment  # noqa: F401
    except ImportError:
        return False
    return True


def score_texts(texts: list[str]) -> list[dict]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    out = []
    for t in texts:
        s = sia.polarity_scores(t)
        n_words = len(re.findall(r"\w+", t))
        if n_words < 3:
            label = "uncertain"
        elif s["pos"] >= 0.15 and s["neg"] >= 0.15:
            label = "mixed"
        elif s["compound"] >= 0.05:
            label = "positive"
        elif s["compound"] <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        out.append({"compound": round(s["compound"], 4), "pos": s["pos"], "neg": s["neg"], "label": label})
    return out


DEFINITIONS = {
    "method": "VADER lexicon and rules (Hutto & Gilbert, 2014)",
    "labels": {
        "uncertain": "fewer than 3 words",
        "mixed": "positive and negative proportions both >= 0.15",
        "positive": "compound >= 0.05",
        "negative": "compound <= -0.05",
        "neutral": "otherwise",
    },
    "limitations": "Lexicon sentiment does not understand sarcasm, domain-specific meaning or wider context. "
    "It has not been validated on this dataset; treat it as a descriptive signal.",
}

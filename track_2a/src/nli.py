"""The three labels of the challenge, and how to read one from a model's answer."""

# Label ids as used by the challenge: the position in this list is the id.
LABELS = ["entailment", "neutral", "contradiction"]

# The first of these word stems in the answer decides the label.
STEMS = {"entail": 0, "neutral": 1, "contradict": 2}


def parse_label(answer: str) -> int | None:
    """Return the label of the first label word in the answer, or None if there is none."""
    answer = answer.lower()
    hits = [(answer.find(stem), label) for stem, label in STEMS.items() if stem in answer]
    return min(hits)[1] if hits else None

"""Find the sentences of a booklet that bear on a claim.

The booklet's body text is split into sentences, each with its page and the headings above it. BM25
ranks them for a claim. Words are folded (lowercase, no accents) and cut to their first seven letters,
which works about as well as stemming for German, French and Italian.
"""

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

from src.strategies._gied.pdf import Block

MIN_CHARS = 40  # a shorter sentence is joined to the next one, and is never evidence on its own
PREFIX = 7  # letters kept of each word: 7 found the evidence of more cases than 5, 6 or whole words
NEIGHBOUR = 0.3  # share of its better neighbour's score that a sentence gets

# A full stop after these words, a single letter ("z. B.") or a number ("1. Januar") ends no sentence
ABBREVIATIONS = {
    "abs", "al", "art", "bst", "bzw", "ca", "cf", "cfr", "chf", "cost", "cpv", "cst", "dr", "ecc", "env", "etc",
    "evtl", "fr", "ggf", "inkl", "insb", "lett", "lit", "max", "min", "mia", "mio", "mm", "mme", "mrd", "no",
    "nr", "par", "pp", "prof", "resp", "s", "sig", "sog", "st", "usw", "vgl", "ziff",
}
SENTENCE_END = re.compile(r"[.!?…]+[»\"”’)]*\s+(?=[«\"„“(]?[A-ZÀ-ÖØ-Ý0-9])")


@dataclass(frozen=True)
class Sentence:
    page: int
    section: str  # the two headings above it, e.g. "Argumente Bundesrat und Parlament › Zu strikte Vorgaben"
    text: str


class Booklet:
    """A booklet's sentences in reading order, indexed for search."""

    def __init__(self, blocks: list[Block], pages: int):
        self.pages = pages  # how many pages the PDF has
        self.sentences = sentences(blocks)
        self.bm25 = BM25([terms(sentence.text) for sentence in self.sentences])

    def search(self, query: str, k: int) -> list[Sentence]:
        """The k sentences that match the query best, best first. Each sentence also gets 30 % of the
        score of its better neighbour on the same page, since the text around it helps to find it."""
        scores = self.bm25.scores(terms(query))
        boosted = []
        for i, sentence in enumerate(self.sentences):
            near = [scores[j] for j in (i - 1, i + 1)
                    if 0 <= j < len(scores) and self.sentences[j].page == sentence.page]
            boosted.append(scores[i] + NEIGHBOUR * max(near, default=0))
        best: list[Sentence] = []
        for i in sorted(range(len(boosted)), key=lambda i: -boosted[i]):
            if boosted[i] <= 0 or len(best) == k:
                break
            if all(self.sentences[i].text != sentence.text for sentence in best):
                best.append(self.sentences[i])
        return best

    def same_place(self, sentence: Sentence, edition: "Booklet") -> Sentence | None:
        """The sentence here that is where `sentence` is in another edition of the booklet: on the same
        page, the same share of the way down it."""
        theirs, mine = edition.on_page(sentence.page), self.on_page(sentence.page)
        if not mine:
            return None
        share = (theirs.index(sentence) + 0.5) / len(theirs)
        return mine[int(share * len(mine))]

    def around(self, sentence: Sentence) -> list[Sentence]:
        """The sentence, with the one before and the one after it if they are on the same page."""
        i = self.sentences.index(sentence)
        return [s for s in self.sentences[max(0, i - 1):i + 2] if s.page == sentence.page]

    def on_page(self, page: int) -> list[Sentence]:
        return [sentence for sentence in self.sentences if sentence.page == page]


def sentences(blocks: list[Block]) -> list[Sentence]:
    """The sentences of the body text, each with its page and the two nearest headings above it."""
    headings: dict[int, str] = {}  # level → the latest heading of that level
    found = []
    for page, level, text in blocks:
        if level:
            headings = {higher: heading for higher, heading in headings.items() if higher < level}
            headings[level] = text
        else:
            section = " › ".join(headings[higher] for higher in sorted(headings)[-2:])
            found += [Sentence(page, section, sentence) for sentence in split(text)]
    return found


def split(text: str) -> list[str]:
    """Split a block of body text into sentences; one shorter than MIN_CHARS is joined to the next."""
    parts, start = [], 0
    for end in SENTENCE_END.finditer(text):
        word = re.search(r"([^\W\d_]+|\d+)\W*$", text[start:end.start() + 1])  # the word before the full stop
        before = word.group(1) if word else ""
        if before.isdigit() or len(before) == 1 or before.lower() in ABBREVIATIONS:
            continue
        parts.append(text[start:end.end()].strip())
        start = end.end()
    parts.append(text[start:].strip())
    joined: list[str] = []
    for part in filter(None, parts):
        if joined and len(joined[-1]) < MIN_CHARS:
            joined[-1] += " " + part
        else:
            joined.append(part)
    if len(joined) > 1 and len(joined[-1]) < MIN_CHARS:  # a short last sentence joins the one before
        last = joined.pop()
        joined[-1] += " " + last
    return joined


def terms(text: str) -> list[str]:
    """The words of a text for search: lowercase, without accents, cut to their first PREFIX letters."""
    folded = unicodedata.normalize("NFKD", text.lower().replace("ß", "ss"))
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return [word[:PREFIX] for word in re.findall(r"[^\W_]+", folded) if len(word) > 1 or word.isdigit()]


class BM25:
    """Okapi BM25 over documents given as lists of words, with k1 = 1.2 and b = 0.75."""

    def __init__(self, documents: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.size = len(documents)
        average = sum(map(len, documents)) / max(1, self.size) or 1
        self.norm = [k1 * (1 - b + b * len(document) / average) for document in documents]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # word → (document, count)
        for i, document in enumerate(documents):
            for word, count in Counter(document).items():
                self.postings[word].append((i, count))
        self.idf = {word: math.log(1 + (self.size - len(p) + 0.5) / (len(p) + 0.5)) for word, p in self.postings.items()}

    def scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.size
        for word in set(query):
            for i, count in self.postings.get(word, ()):
                scores[i] += self.idf[word] * count * (self.k1 + 1) / (count + self.norm[i])
        return scores

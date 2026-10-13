"""Find the sentences of a booklet that bear on a claim: sentences, BM25, and the booklet's other editions.

A booklet is split into sentences that never cross a block or a page, so each one can be quoted as
evidence with its page. BM25 ranks them for a claim; words are folded (lowercase, no accents) and cut
to their first seven letters, which works about as well as stemming for German, French and Italian.

The editions of a booklet in German, French and Italian have the same pages, block for block. A claim
in another language than the booklet is searched in the booklet's edition in the claim's language,
and what is found there is mapped to the same place in the booklet: `Booklet.map`.
"""

import math
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import Future
from dataclasses import dataclass

from src.strategies._gied.pdf import Block

MIN_CHARS, MAX_CHARS = 40, 600  # shorter sentences join the next one, longer ones are split
PREFIX = 7  # letters kept of each word: 7 found the evidence of more cases than 5, 6 or whole words
NEIGHBOURS = 0.3  # share of the better neighbour's score a sentence gets: context helps to find the right one

# Words that a full stop ends without ending the sentence, besides single letters and numbers
ABBREVIATIONS = {
    "abs", "al", "art", "bst", "bzw", "ca", "cf", "cfr", "chf", "cpv", "dr", "ecc", "env", "etc", "evtl", "fr",
    "ggf", "inkl", "insb", "lett", "lit", "max", "min", "mio", "mia", "mrd", "nr", "no", "par", "pp", "prof",
    "resp", "s", "sog", "st", "usw", "vgl", "ziff", "cost", "cst", "sig", "mme", "mm",
}
BOUNDARY = re.compile(r"[.!?…]+[»\"”’)]*\s+(?=[«\"„“(]?[A-ZÀ-ÖØ-Ý0-9])")


@dataclass(frozen=True)
class Sentence:
    text: str
    page: int  # 1-based
    block: int  # the position of its block among the page's text blocks
    order: int  # its position in the block
    of: int  # how many sentences the block has
    section: str  # the headings above it, the nearest last


class Booklet:
    """The sentences of one booklet, indexed for search."""

    def __init__(self, pages: list[list[Block]]):
        self.pages = len(pages)
        self.sentences: list[Sentence] = []
        headings: list[tuple[int, str]] = []  # (level, text) of the headings above the current block
        for page, blocks in enumerate(pages, 1):
            number = 0
            for block in blocks:
                if block.kind == "heading":
                    headings = [h for h in headings if h[0] < block.level] + [(block.level, block.text)]
                    continue
                parts = split(block.text)
                section = " › ".join(text for _, text in headings[-2:])
                self.sentences += [Sentence(text, page, number, i, len(parts), section) for i, text in enumerate(parts)]
                number += 1
        self.at: dict[tuple[int, int], list[int]] = defaultdict(list)  # (page, block) → its sentences
        for i, sentence in enumerate(self.sentences):
            self.at[sentence.page, sentence.block].append(i)
        self.index = BM25([terms(s.text) for s in self.sentences])

    def scores(self, query: str) -> list[float]:
        """Every sentence's score for the query; part of the better neighbour's score is added."""
        raw = self.index.scores(terms(query))
        scores = list(raw)
        for i, sentence in enumerate(self.sentences):
            near = [raw[j] for j in (i - 1, i + 1)
                    if 0 <= j < len(raw) and self.sentences[j].page == sentence.page]
            if near:
                scores[i] += NEIGHBOURS * max(near)
        return scores

    def top(self, scores: list[float], k: int) -> list[int]:
        """The k best sentences with a score above 0, best first; repeated texts count once."""
        seen, best = set(), []
        for i in sorted(range(len(scores)), key=lambda i: (-scores[i], i)):
            if scores[i] <= 0 or len(best) == k:
                break
            key = " ".join(terms(self.sentences[i].text))
            if key not in seen:
                seen.add(key)
                best.append(i)
        return best

    def map(self, other: "Booklet", i: int) -> int | None:
        """The sentence here at the place of sentence i of the other edition: the same page and block, and
        the same share of the way through the block; or through the page, if the blocks do not match."""
        sentence = other.sentences[i]
        here = self.at.get((sentence.page, sentence.block))
        there = other.at[sentence.page, sentence.block]
        if not here or len(self.blocks(sentence.page)) != len(other.blocks(sentence.page)):
            here = [j for j in range(len(self.sentences)) if self.sentences[j].page == sentence.page]
            there = [j for j in range(len(other.sentences)) if other.sentences[j].page == sentence.page]
        if not here:
            return None
        share = (there.index(i) + 0.5) / len(there)
        return here[min(len(here) - 1, int(share * len(here)))]

    def blocks(self, page: int) -> list[int]:
        return sorted({block for (p, block) in self.at if p == page})


class BM25:
    """Okapi BM25 over tokenised documents, k1 = 1.2 and b = 0.75."""

    def __init__(self, documents: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.size = len(documents)
        average = sum(map(len, documents)) / max(1, len(documents)) or 1
        self.norm = [k1 * (1 - b + b * len(document) / average) for document in documents]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, document in enumerate(documents):
            for term, count in Counter(document).items():
                self.postings[term].append((i, count))
        self.idf = {term: math.log(1 + (self.size - len(p) + 0.5) / (len(p) + 0.5)) for term, p in self.postings.items()}

    def scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.size
        for term in set(query):
            idf = self.idf.get(term, 0.0)
            for i, count in self.postings.get(term, ()):
                scores[i] += idf * count * (self.k1 + 1) / (count + self.norm[i])
        return scores


def terms(text: str) -> list[str]:
    """The words of a text for search: lowercase, without accents, cut to their first PREFIX letters."""
    folded = unicodedata.normalize("NFKD", text.lower().replace("ß", "ss"))
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return [word[:PREFIX] for word in re.findall(r"[^\W_]+", folded) if len(word) > 1 or word.isdigit()]


def split(text: str) -> list[str]:
    """Split a block into sentences of MIN_CHARS to about MAX_CHARS characters."""
    sentences, start = [], 0
    for match in BOUNDARY.finditer(text):
        word = re.search(r"([^\W\d_]+|\d+)\W*$", text[start:match.start() + 1])
        before = word.group(1) if word else ""
        if before.isdigit() or len(before) == 1 or before.lower() in ABBREVIATIONS:
            continue  # "1. Januar", "z. B.", "Art. 3": not the end of a sentence
        sentences.append(text[start:match.end()].strip())
        start = match.end()
    sentences.append(text[start:].strip())
    joined: list[str] = []
    for sentence in filter(None, sentences):
        if joined and len(joined[-1]) < MIN_CHARS:
            joined[-1] = f"{joined[-1]} {sentence}"
        else:
            joined.append(sentence)
    if len(joined) > 1 and len(joined[-1]) < MIN_CHARS:
        joined[-2:] = [f"{joined[-2]} {joined[-1]}"]
    return [piece for sentence in joined for piece in shorten(sentence)]


def shorten(sentence: str) -> list[str]:
    """A sentence longer than MAX_CHARS, split at the semicolon, colon or comma nearest its middle."""
    if len(sentence) <= MAX_CHARS:
        return [sentence]
    cuts = [m.end() for m in re.finditer(r"[;:,]\s", sentence) if MIN_CHARS <= m.end() <= len(sentence) - MIN_CHARS]
    if not cuts:
        return [sentence]
    cut = min(cuts, key=lambda c: abs(c - len(sentence) / 2))
    return shorten(sentence[:cut].strip()) + shorten(sentence[cut:].strip())


class Lazy:
    """Makes the value of each key once, on first use. Other threads asking for the same key wait for
    it, and get the same value or the same exception; other keys are made at the same time."""

    def __init__(self, make):
        self.make = make
        self.lock = threading.Lock()
        self.futures: dict = {}

    def get(self, key):
        with self.lock:
            future = self.futures.get(key)
            mine = future is None
            if mine:
                future = self.futures[key] = Future()
        if mine:
            try:
                future.set_result(self.make(key))
            except BaseException as error:  # the waiting threads get it too
                future.set_exception(error)
        return future.result()

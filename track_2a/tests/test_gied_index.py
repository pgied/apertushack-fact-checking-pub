"""Tests for sentences, BM25 and the mapping between editions in src/strategies/_gied/index.py."""

import threading
import time

import pytest

from src.strategies._gied.index import Booklet, Lazy, split, terms
from src.strategies._gied.pdf import Block


def test_sentences_end_at_full_stops_but_not_after_abbreviations_or_numbers():
    text = ("Der Bund zahlt z. B. jährlich rund 600 Millionen Franken für die Biodiversität. Die Initiative "
            "verlangt ab dem 1. Januar 2030 mehr Geld, gemäss Art. 78a Abs. 2 der Bundesverfassung. Der Bundesrat "
            "und das Parlament empfehlen, die Initiative abzulehnen!")
    assert split(text) == [
        "Der Bund zahlt z. B. jährlich rund 600 Millionen Franken für die Biodiversität.",
        "Die Initiative verlangt ab dem 1. Januar 2030 mehr Geld, gemäss Art. 78a Abs. 2 der Bundesverfassung.",
        "Der Bundesrat und das Parlament empfehlen, die Initiative abzulehnen!",
    ]


def test_short_sentences_join_the_next_and_long_ones_are_split():
    assert split("Nein. Der Bundesrat empfiehlt, die Initiative abzulehnen.") == [
        "Nein. Der Bundesrat empfiehlt, die Initiative abzulehnen."]
    long = "; ".join(f"Teil {n} der langen Aufzählung mit vielen Wörtern" for n in range(20)) + "."
    pieces = split(long)
    assert len(pieces) > 1 and all(len(piece) <= 600 for piece in pieces)
    assert " ".join(pieces) == long


def test_terms_are_folded_and_cut():
    assert terms("Lebensräume, Énergies 6,8 % ß") == ["lebensr", "energie", "6", "8", "ss"]


def edition(*pages: list[str]) -> Booklet:
    return Booklet([[Block("heading", "Im Detail", 2)] + [Block("text", text) for text in page] for page in pages])


def test_bm25_finds_the_sentence_about_the_claim():
    booklet = edition(["Die Biodiversität ist zurückgegangen. Der Bund investiert jährlich 600 Millionen Franken."],
                      ["Die Reform senkt den Umwandlungssatz von 6,8 auf 6 Prozent. Das betrifft viele Versicherte."])
    best = booklet.top(booklet.scores("Der Umwandlungssatz sinkt auf 6 Prozent"), 1)
    assert booklet.sentences[best[0]].text.startswith("Die Reform senkt")
    assert booklet.sentences[best[0]].page == 2 and booklet.sentences[best[0]].section == "Im Detail"
    assert booklet.top(booklet.scores("Kernkraftwerke"), 3) == []  # no word in common, no hit


def test_a_sentence_maps_to_the_same_place_in_the_other_edition():
    german = edition(["Erster Satz auf der ersten Seite des Büchleins. Zweiter Satz auf der ersten Seite."],
                     ["Ein Absatz auf Seite zwei mit genug Text.", "Noch ein Absatz auf Seite zwei mit Text."])
    french = edition(["Première phrase de la première page de la brochure. Deuxième phrase de la première page."],
                     ["Un paragraphe de la page deux avec assez de texte.", "Un autre paragraphe de la page deux."])
    for i, sentence in enumerate(french.sentences):
        mapped = german.sentences[german.map(french, i)]
        assert (mapped.page, mapped.block, mapped.order) == (sentence.page, sentence.block, sentence.order)


def test_lazy_makes_each_value_once_and_shares_its_exceptions():
    made = []

    def make(key):
        time.sleep(0.05)
        made.append(key)
        if key == "broken":
            raise ValueError(key)
        return key.upper()

    lazy = Lazy(make)
    threads = [threading.Thread(target=lazy.get, args=("a",)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert lazy.get("a") == "A" and made == ["a"]
    for _ in range(2):
        with pytest.raises(ValueError):
            lazy.get("broken")
    assert made == ["a", "broken"]

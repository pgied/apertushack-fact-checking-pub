"""Tests for the challenge's evidence check, as src/strategies/_gied/score.py implements it."""

from src.strategies._gied import score

PASSAGE = ("Initiative populaire « Entreprises responsables »\n\n\nContexte : On attend des entreprises suisses "
           "qu’elles respectent égale-\nment les droits de l’homme et les normes environnementales à l’étranger.")


def test_normalize_folds_case_whitespace_and_hyphens_at_line_breaks():
    assert score.normalize("Égale-\n  ment  LES­Droits") == "également lesdroits"
    assert score.normalize("a ﬁne  day") == "a fine day"  # NFKC: the ligature becomes two letters


def test_a_quote_counts_if_it_lies_in_the_passage_or_contains_it():
    assert score.quote_matches("respectent également les droits de l’homme", PASSAGE)
    assert score.quote_matches(f"Page 4. {PASSAGE} And more of the page.", PASSAGE)  # it contains the passage
    assert not score.quote_matches("Le Conseil fédéral recommande de rejeter l’initiative.", PASSAGE)
    assert not score.quote_matches("", PASSAGE)


def test_a_quote_longer_than_5000_characters_does_not_count():
    assert not score.quote_matches(PASSAGE + " x" * 5000, PASSAGE)


def test_only_the_first_five_items_count():
    miss = {"page": 1, "text": "Le Conseil fédéral recommande de rejeter l’initiative."}
    hit = {"page": 4, "text": "les normes environnementales à l’étranger"}
    assert score.found([hit], PASSAGE)
    assert score.found([miss] * 4 + [hit], PASSAGE)
    assert not score.found([miss] * 5 + [hit], PASSAGE)
    assert not score.found(["not an item"] * 4 + [miss, hit], PASSAGE)  # a wrong item still takes its place

"""Tests for sentences, search and editions in src/strategies/_gied/index.py."""

from src.strategies._gied.index import Booklet, Sentence, sentences, split, terms


def test_sentences_end_at_full_stops_but_not_after_abbreviations_or_numbers():
    text = ("Der Bund zahlt z. B. jährlich rund 600 Millionen Franken für die Biodiversität. Die Initiative "
            "verlangt ab dem 1. Januar 2030 mehr Geld, gemäss Art. 78a Abs. 2 der Bundesverfassung. Der Bundesrat "
            "und das Parlament empfehlen, die Initiative abzulehnen!")
    assert split(text) == [
        "Der Bund zahlt z. B. jährlich rund 600 Millionen Franken für die Biodiversität.",
        "Die Initiative verlangt ab dem 1. Januar 2030 mehr Geld, gemäss Art. 78a Abs. 2 der Bundesverfassung.",
        "Der Bundesrat und das Parlament empfehlen, die Initiative abzulehnen!",
    ]


def test_a_short_sentence_is_joined_to_the_next_or_the_one_before():
    assert split("Nein. Der Bundesrat empfiehlt, die Initiative abzulehnen.") == [
        "Nein. Der Bundesrat empfiehlt, die Initiative abzulehnen."]
    assert split("Der Bundesrat empfiehlt, die Initiative abzulehnen. Nein.") == [
        "Der Bundesrat empfiehlt, die Initiative abzulehnen. Nein."]


def test_terms_are_folded_and_cut():
    assert terms("Lebensräume, Énergies 6,8 % ß") == ["lebensr", "energie", "6", "8", "ss"]


def test_a_sentence_knows_its_page_and_the_two_headings_above_it():
    blocks = [(1, 2, "Im Detail"), (1, 3, "Biodiversitätsinitiative"), (1, 4, "Ausgangslage"),
              (1, 0, "Die Biodiversität ist in der Schweiz zurückgegangen."), (2, 3, "Argumente"),
              (2, 0, "Der Bund investiert jährlich 600 Millionen Franken.")]
    assert sentences(blocks) == [
        Sentence(1, "Biodiversitätsinitiative › Ausgangslage", "Die Biodiversität ist in der Schweiz zurückgegangen."),
        Sentence(2, "Im Detail › Argumente", "Der Bund investiert jährlich 600 Millionen Franken."),
    ]


def booklet(*pages: str) -> Booklet:
    """A booklet with one block of body text per page."""
    return Booklet([(page, 0, text) for page, text in enumerate(pages, 1)], len(pages))


def test_search_finds_the_sentence_about_the_claim():
    found = booklet("Die Biodiversität ist zurückgegangen. Der Bund investiert jährlich 600 Millionen Franken.",
                    "Die Reform senkt den Umwandlungssatz von 6,8 auf 6 Prozent. Das betrifft viele Versicherte.")
    best = found.search("Der Umwandlungssatz sinkt auf 6 Prozent", 1)
    assert [(s.page, s.text[:16]) for s in best] == [(2, "Die Reform senkt")]
    assert found.search("Kernkraftwerke", 3) == []  # no word in common, nothing found


def test_search_can_keep_to_some_pages():
    found = booklet("Der Bund investiert jährlich 600 Millionen Franken in die Biodiversität.",
                    "Die Reform kostet den Bund jährlich 600 Millionen Franken mehr als heute.")
    assert {s.page for s in found.search("600 Millionen Franken", 5)} == {1, 2}
    assert {s.page for s in found.search("600 Millionen Franken", 5, pages={2})} == {2}


def test_a_sentence_has_its_counterpart_at_the_same_place_in_the_other_edition():
    german = booklet("Erster Satz auf der ersten Seite des Büchleins. Zweiter Satz auf der ersten Seite des Büchleins.",
                     "Ein Absatz auf Seite zwei mit genug Text, um ein Satz zu sein.")
    french = booklet("Première phrase de la première page de la brochure. Deuxième phrase de la première page de la brochure.",
                     "Un paragraphe de la page deux avec assez de texte pour une phrase.")
    assert [german.same_place(sentence, french).text[:12] for sentence in french.sentences] == [
        "Erster Satz ", "Zweiter Satz", "Ein Absatz a"]


def test_around_a_sentence_are_its_neighbours_on_the_same_page():
    found = booklet("Erster Satz auf der ersten Seite des Büchleins. Zweiter Satz auf der ersten Seite des Büchleins.",
                    "Der erste Satz auf der zweiten Seite steht hier. Und der zweite Satz auf der zweiten Seite.")
    second, third = found.sentences[1], found.sentences[2]
    assert found.around(second) == found.sentences[:2]  # the next sentence is on another page
    assert found.around(third) == found.sentences[2:]

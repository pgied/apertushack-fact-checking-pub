
import json

import pytest

from src.evaluate import evidence_found
from src.strategies import LLM
from src.strategies._booklet import find_quote, vote_pages
from src.strategies.vote_quotes import VoteQuotes, parse_answer
from tests.test_strategies import Client

PAGES = [
    "Volksabstimmung vom 27. September 2020\nErste Vorlage\nZweite Vorlage",
    "Erste Vorlage: Jagdgesetz\nÄnderung des Jagdgesetzes\nDer Wolf soll reguliert werden können.",
    "Die Kantone können Bestände regulieren.\nDer Bundesrat empfiehlt ein Ja.",
    "Zweite Vorlage: Kampfflugzeuge\nBundesbeschluss über die Beschaffung neuer Kampfflugzeuge",
    "Die Beschaffung kostet höchstens 6 Milliarden Franken.\nDie Mehrwert-\nsteuer wird nicht erhöht.",
]
CASE = {"id": "v1.1-row-1-A", "vote": "Bundesbeschluss über die Beschaffung neuer Kampfflugzeuge",
        "claim": {"text": "L'achat coûtera au maximum 6 milliards de francs."},
        "booklet": {"path": "booklets/b.pdf"}}


def strategy(**settings):
    s = VoteQuotes(VoteQuotes.Settings(name="vote_quotes", **settings), "advanced", {"booklets/b.pdf": PAGES})
    s.setup([CASE], None)
    return s


def test_vote_pages_picks_the_votes_block():
    assert vote_pages(PAGES, CASE["vote"]) == [4, 5]
    assert vote_pages(PAGES, "Änderung des Jagdgesetzes") == [2, 3]
    assert vote_pages(PAGES[3:], CASE["vote"]) == [1, 2]


def test_find_quote_exact_and_fuzzy():
    assert find_quote("die beschaffung kostet höchstens 6 Milliarden Franken", PAGES, [4, 5]) == (
        5, "Die Beschaffung kostet höchstens 6 Milliarden Franken")
    page, text = find_quote("Die Beschaffung kostet maximal 6 Milliarden Franken.", PAGES, [4, 5])
    assert page == 5 and text.startswith("Die Beschaffung")
    assert find_quote("Völlig andere Aussage über ein anderes Thema hier.", PAGES, [4, 5]) is None


def test_only_the_vote_pages_are_sent_and_quotes_are_verified():
    answer = json.dumps({"label": "entailment", "quotes": [
        {"page": 4, "text": "Die Beschaffung kostet höchstens 6 Milliarden Franken."},
        {"page": 5, "text": "This sentence is not in the booklet at all, sorry."}]})
    client = Client(answer)
    verdict = strategy().predict(CASE, LLM(client, "m", {}))
    prompt = client.requests[0]["messages"][1]["content"]
    assert "[Page 4]" in prompt and "[Page 5]" in prompt and "[Page 2]" not in prompt
    assert verdict.label == 0
    assert verdict.evidence[0] == {"page": 5, "text": "Die Beschaffung kostet höchstens 6 Milliarden Franken"}
    assert verdict.evidence[1]["text"].startswith("This sentence")


def test_whole_booklet_setting_sends_every_page():
    client = Client('{"label": "neutral", "quotes": []}')
    verdict = strategy(whole_booklet=True).predict(CASE, LLM(client, "m", {}))
    assert "[Page 2]" in client.requests[0]["messages"][1]["content"]
    assert verdict.label == 1 and verdict.evidence == []


def test_labels_setting_changes_the_definitions_in_both_tasks():
    beginner_case = {"id": "x-B", "reference": {"text": "Der Bundesrat empfiehlt ein Nein."},
                     "claim": {"text": "Le Conseil fédéral recommande le non."}}
    for labels, phrase in (("strict", "not mention"), ("negation", "explicitly states the opposite"),
                           ("loose", "says the opposite of the claim")):
        beginner = VoteQuotes(VoteQuotes.Settings(name="vote_quotes", labels=labels), "beginner", {})
        client = Client("entailment")
        assert beginner.predict(beginner_case, LLM(client, "m", {})).label == 0
        assert phrase in client.requests[0]["messages"][0]["content"]
        client = Client('{"label": "neutral", "quotes": []}')
        strategy(labels=labels).predict(CASE, LLM(client, "m", {}))
        assert phrase in client.requests[0]["messages"][0]["content"]


def test_parse_answer_is_tolerant():
    assert parse_answer('Sure: {"label": "contradiction", "quotes": [{"page": "7", "text": "x"}]}') == (
        2, [{"page": 7, "text": "x"}])
    assert parse_answer('{"label": "entailment", "quotes": ["just a string"]}') == (0, [{"page": None, "text": "just a string"}])
    assert parse_answer("no json, but entailment") == (0, [])


def test_evidence_score():
    pytest.importorskip("rapidfuzz")
    gold = [{"page": 5, "text": "Die Beschaffung kostet höchstens 6 Milliarden Franken.\nDie Mehrwert-\nsteuer"},
            {"page": 6, "text": "wird nicht erhöht."}]
    assert evidence_found([{"page": 5, "text": "Die Mehrwertsteuer wird nicht erhöht"}], gold)
    assert not evidence_found([{"page": 5, "text": "Etwas völlig anderes steht hier"}], gold)
    assert not evidence_found([{"page": 5, "text": "Die Beschaffung kostet " + "x" * 5000}], gold)
    late = [{"page": 1, "text": "nichts"}] * 5 + [{"page": 5, "text": "Die Beschaffung kostet höchstens"}]
    assert not evidence_found(late, gold)

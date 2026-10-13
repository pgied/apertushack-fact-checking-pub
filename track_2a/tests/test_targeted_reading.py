"""Tests for the strategy targeted_reading: reading the table of contents, and reading the sections one by one."""

import re
from types import SimpleNamespace

from src import nli, strategies
from src.strategies.targeted_reading.strategy import TargetedReading, find_vote, table_of_contents

PATH = "booklets/2031_06_01_de.pdf"
# Two votes. The table of contents is on page 3; page 4 starts the first vote and lists its sections again.
PAGES = [
    "Volksabstimmung vom 1. Juni 2031",
    "Erläuterungen des Bundesrates",
    "3\nVolksinitiative «Mehr Velowege»\nIn Kürze 4 – 5\nIm Detail 8\nArgumente 10\nAbstimmungstext 11\n"
    "Bundesgesetz über die Bahn\nIn Kürze 6 – 7\nVorlage im Detail 12\nArgumente 13\nAbstimmungstext 14\nErste Vorlage",
    "4 Erste Vorlage: Velo-Initiative\nVorlage im Detail 8\nArgumente 10\nAbstimmungstext 11\nVolksinitiative «Mehr Velowege»",
    "Der Bundesrat empfiehlt, die Initiative abzulehnen. Sie kostet zu viel.",
    "Bundesgesetz über die Bahn",
    "Die Bahn bekommt mehr Geld.",
    "Im Detail: Velowege in den Kantonen.",
    "Die Initiative kostet 2 Milliarden Fran-\nken. Der Bund müsste sie über zehn Jahre bezahlen.",
    "Argumente des Initiativkomitees.",
    "Abstimmungstext der Initiative.",
    "Im Detail: die Bahn.",
    "Argumente: die Bahn.",
    "Abstimmungstext: die Bahn.",
    "Die Stimmempfehlungen der Parteien.",
]
CASE = {
    "id": "case-0001",
    "booklet": {"path": PATH},
    "vote": "Volksinitiative «Mehr Velowege»",
    "claim": {"text": "L'initiative coûte 2 milliards de francs."},
}


class ScriptedClient:
    """Stands in for the OpenAI client: gives the answers in turn, and keeps the requests."""

    def __init__(self, *answers: str):
        self.answers = list(answers)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        message = SimpleNamespace(content=self.answers.pop(0))
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=2)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def predict(*answers: str) -> tuple[strategies.Verdict, ScriptedClient]:
    client = ScriptedClient(*answers)
    llm = strategies.LLM(client, "apertus", {"temperature": 0, "max_tokens": 10})
    strategy = TargetedReading(TargetedReading.Settings(name="targeted_reading"), "advanced", {PATH: PAGES})
    strategy.setup([CASE], llm)
    return strategy.predict(CASE, llm), client


QUOTE = "Die Initiative kostet 2 Milliarden Franken."  # page 9, where "Franken" is broken across two lines


def test_the_table_of_contents_gives_each_vote_its_sections_up_to_the_next_one():
    votes = table_of_contents(PAGES)
    assert [{kind: pages for kind, pages in vote.items() if kind != "titles"} for vote in votes] == [
        {"summary": (4, 5), "detail": (8, 9), "arguments": (10, 10), "text": (11, 11)},
        {"summary": (6, 7), "detail": (12, 12), "arguments": (13, 13), "text": (14, 15)},
    ]
    assert find_vote("Bundesgesetz über die Bahn", votes, PAGES)["summary"] == (6, 7)
    assert find_vote("Volksinitiative «Keine Autobahnen»", votes, PAGES) is None


ENTAILMENT, CONTRADICTION = nli.LABELS.index("entailment"), nli.LABELS.index("contradiction")
QUOTED = [{"page": 9, "text": QUOTE}, {"page": 9, "text": PAGES[8].strip()}]  # the quote, then its page


def shown(client: ScriptedClient) -> list[list[int]]:
    """The pages of each section shown, without the follow-up questions."""
    prompts = [request["messages"][1]["content"] for request in client.requests if len(request["messages"]) == 2]
    return [[int(n) for n in re.findall(r"\[Page (\d+)\]", prompt)] for prompt in prompts]


def test_it_reads_each_section_on_its_own_arguments_detail_summary_then_text():
    verdict, client = predict("neutral", "neutral", "neutral", "contradiction 11", "11: Abstimmungstext der Initiative.")
    assert verdict == strategies.Verdict(CONTRADICTION, [{"page": 11, "text": PAGES[10]}])
    assert shown(client) == [[10], [8, 9], [4, 5], [11]]


def test_it_always_reads_arguments_detail_and_summary_and_keeps_the_evidence_of_all_that_agree():
    summary = "Der Bundesrat empfiehlt, die Initiative abzulehnen."  # page 5
    verdict, client = predict("entailment 10", "10: Argumente.", "entailment 9", f"9: {QUOTE}", "entailment 5", f"5: {summary}")
    assert verdict == strategies.Verdict(ENTAILMENT, [  # the quotes first, then the whole pages
        QUOTED[0], {"page": 5, "text": summary}, {"page": 10, "text": PAGES[9]}, QUOTED[1], {"page": 5, "text": PAGES[4]},
    ])
    assert shown(client) == [[10], [8, 9], [4, 5]]  # settled: not the text


def test_the_first_section_that_settles_it_wins_when_the_others_disagree():
    verdict, _ = predict("contradiction 10", "10: Argumente.", "entailment 9", f"9: {QUOTE}", "entailment 5", "5: ...")
    assert verdict == strategies.Verdict(CONTRADICTION, [{"page": 10, "text": PAGES[9]}])


def test_the_detail_decides_when_the_arguments_are_neutral():
    verdict, client = predict("neutral", "entailment 9", f"9: {QUOTE}", "neutral")
    assert verdict == strategies.Verdict(ENTAILMENT, QUOTED)
    assert shown(client) == [[10], [8, 9], [4, 5]]


def test_it_is_neutral_when_no_section_settles_the_claim():
    verdict, client = predict("neutral", "neutral", "neutral 4", "no idea")  # an answer without a label counts as neutral
    assert verdict == strategies.Verdict(nli.LABELS.index("neutral"))
    assert len(client.requests) == 4


def test_the_follow_up_asks_for_a_quote_which_comes_first_followed_by_its_page():
    verdict, client = predict("neutral", "entailment", f"«{QUOTE}»", "neutral")  # no page: the quote finds it
    assert verdict == strategies.Verdict(ENTAILMENT, QUOTED)
    first, follow_up = (request["messages"] for request in client.requests[1:3])
    assert follow_up[:2] == first  # the same conversation, continued
    assert follow_up[2] == {"role": "assistant", "content": "entailment"}
    assert client.requests[2]["max_tokens"] == 300


def test_a_quote_not_on_the_pages_falls_back_to_the_page_the_model_names():
    verdict, _ = predict("neutral", "entailment 8", "8: Die Initiative ist teuer und bringt nichts für die Velofahrer.", "neutral")
    assert verdict == strategies.Verdict(ENTAILMENT, [{"page": 8, "text": PAGES[7]}])


def test_a_page_that_is_not_in_the_section_gives_no_evidence():
    verdict, _ = predict("entailment 14", "14: Abstimmungstext: die Bahn.", "neutral", "neutral")
    assert verdict == strategies.Verdict(ENTAILMENT)


def test_a_vote_not_in_the_table_of_contents_falls_back_to_the_whole_booklet():
    case = {**CASE, "vote": "Volksinitiative «Keine Autobahnen»"}
    client = ScriptedClient("entailment 9")  # the baseline asks once, without a follow-up
    llm = strategies.LLM(client, "apertus", {"temperature": 0, "max_tokens": 10})
    strategy = TargetedReading(TargetedReading.Settings(name="targeted_reading"), "advanced", {PATH: PAGES})
    strategy.setup([case], llm)
    assert strategy.predict(case, llm).label == nli.LABELS.index("entailment")
    assert "[Page 15]" in client.requests[0]["messages"][1]["content"]

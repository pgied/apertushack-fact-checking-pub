"""Tests for the gied_ strategies, on the sample cases and booklet in data/examples/, with a stand-in model."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src import dataset, pdf
from src.llm import LLM
from src.strategies.gied_docling.strategy import GiedDocling
from src.strategies.gied_pdfium.strategy import GiedPdfium
from src.strategies.gied_tags.strategy import GiedTags

EXAMPLES = Path(__file__).resolve().parent.parent / "data" / "examples"
CASES = {case["id"]: case for case in dataset.read_jsonl(EXAMPLES / "cases.jsonl")}
ADVANCED = CASES["v1.1-row-1104-A"]  # a German claim, the booklet in French
SUMMARY = CASES["v1.1-row-357-A"]  # an Italian claim about the summary at the front


class Client:
    """Stands in for the OpenAI client: always gives the same answer, and keeps the requests. A claim
    to translate comes back in French."""

    def __init__(self, answer: str):
        self.answer = answer
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        translate = request["messages"][0]["content"].startswith("Translate")
        message = SimpleNamespace(content="Le Conseil fédéral recommande d’accepter l’initiative." if translate
                                  else self.answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=3))


def decide(cls, case: dict, answer: str, **settings) -> tuple[dict, Client]:
    client = Client(answer)
    task = "advanced" if "booklet" in case else "beginner"
    strategy = cls(cls.Settings(name="gied", **settings), task, {}, cases_dir=EXAMPLES)
    return strategy.answer(case, LLM(client, "some-model", {"max_tokens": 12})), client


@pytest.mark.parametrize("cls", [GiedTags, GiedPdfium, GiedDocling])
def test_the_evidence_is_five_sentences_of_the_booklet_with_their_pages(cls):
    prediction, client = decide(cls, ADVANCED, "contradiction 1")
    booklet = " ".join(" ".join(page.split()) for page in pdf.pages(EXAMPLES / ADVANCED["booklet"]["path"]))

    assert prediction["label_name"] == "contradiction"
    assert 1 <= len(prediction["evidence"]) <= 5
    for item in prediction["evidence"]:
        assert 1 <= item["page"] <= 32
        assert 40 <= len(item["text"]) <= 5000
        assert item["text"].split()[0] in booklet
    assert all(item["text"] != ADVANCED["vote"] for item in prediction["evidence"])
    request = client.requests[-1]
    assert request["max_tokens"] == 12
    assert "EXCERPTS:\n[1] (p. " in request["messages"][1]["content"]
    assert request["messages"][1]["content"].rstrip().endswith("Answer:")


def test_a_cross_lingual_claim_without_its_edition_is_translated_first():
    _, client = decide(GiedTags, ADVANCED, "contradiction 1")
    assert len(client.requests) == 2  # the examples have only the French booklet
    assert client.requests[0]["messages"][0]["content"].startswith("Translate this claim into French.")
    assert ADVANCED["claim"]["text"] in client.requests[0]["messages"][0]["content"]


def test_a_cross_lingual_claim_is_searched_in_the_edition_in_its_language(tmp_path):
    (tmp_path / "booklets").mkdir()
    for language in ("fr", "it"):  # the French booklet stands in for its Italian edition
        (tmp_path / "booklets" / f"2020_11_29_{language}.pdf").write_bytes(
            (EXAMPLES / "booklets" / "2020_11_29_fr.pdf").read_bytes())
    client = Client("entailment 1")
    strategy = GiedTags(GiedTags.Settings(name="gied_tags"), "advanced", {}, cases_dir=tmp_path)
    prediction = strategy.answer(SUMMARY, LLM(client, "some-model", {}))

    assert len(client.requests) == 1  # no translation needed
    assert prediction["evidence"] and all(item["page"] for item in prediction["evidence"])


def test_a_neutral_answer_keeps_its_evidence():
    prediction, _ = decide(GiedTags, ADVANCED, "neutral")
    assert prediction["label_name"] == "neutral" and prediction["evidence"]


def test_the_cases_are_decided_alike_in_any_order():
    client = Client("entailment 2")
    strategy = GiedTags(GiedTags.Settings(name="gied_tags"), "advanced", {}, cases_dir=EXAMPLES)
    cases = [ADVANCED, SUMMARY]

    def run(order):
        return {case["id"]: strategy.answer(case, LLM(client, "m", {}))["evidence"]
                for case in order}

    assert run(cases) == run(cases[::-1])


def test_the_beginner_task_is_the_baselines():
    beginner = CASES["v1.1-row-1104-B"]
    prediction, client = decide(GiedTags, beginner, "contradiction")
    assert prediction["evidence"] == [{"page": None, "text": beginner["reference"]["text"]}]
    assert len(client.requests) == 1

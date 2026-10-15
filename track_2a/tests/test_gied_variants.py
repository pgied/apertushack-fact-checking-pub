"""Tests for the strategies built on gied_tags_prompt_adapted: searching further after a neutral answer,
paraphrases, evidence from two editions, searching only the claim's vote, and the strict strategies, which
read only the booklet the case supplies. With a scripted stand-in model and the example booklet."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import dataset, pdf
from src.llm import LLM
from src.strategies._booklet import vote_pages
from src.strategies._gied.index import Sentence, fuse
from src.strategies.gied_multi_lang_evidence.strategy import SYSTEM_PROMPT as MULTI_LANG_PROMPT
from src.strategies.gied_multi_lang_evidence.strategy import GiedMultiLangEvidence
from src.strategies.gied_multi_lang_paraphrase.strategy import GiedMultiLangParaphrase
from src.strategies.gied_multi_lang_scoped.strategy import RULES_PROMPT, GiedMultiLangScoped
from src.strategies.gied_paraphrase.strategy import GiedParaphrase
from src.strategies.gied_retrieval_improvement.strategy import GiedRetrievalImprovement
from src.strategies.gied_strict_multi_lang_scoped.strategy import GiedStrictMultiLangScoped
from src.strategies.gied_strict_paraphrase.strategy import GiedStrictParaphrase
from src.strategies.gied_strict_reasoned.strategy import GiedStrictReasoned
from src.strategies.gied_strict_reasoned_rules.strategy import GiedStrictReasonedRules

EXAMPLES = Path(__file__).resolve().parent.parent / "data" / "examples"
CASE = next(c for c in dataset.read_jsonl(EXAMPLES / "cases.jsonl") if c["id"] == "v1.1-row-357-A")  # Italian claim, French booklet
PARAPHRASES = "\n".join(f"{n}. Les entreprises suisses doivent respecter les droits humains, version {n}." for n in range(1, 6))


class Scripted:
    """Stands in for the OpenAI client: translations and paraphrases come back in French, and the
    decisions are the given answers, one after the other (the last one repeats)."""

    def __init__(self, *decisions: str):
        self.decisions = list(decisions)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        prompt = request["messages"][0]["content"]
        if prompt.startswith("Translate"):
            content = "Les entreprises suisses doivent respecter les droits humains à l’étranger."
        elif prompt.startswith("Rewrite"):
            content = PARAPHRASES
        else:
            content = self.decisions.pop(0) if len(self.decisions) > 1 else self.decisions[0]
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)

    def decisions_asked(self) -> list[dict]:
        return [r for r in self.requests if r["messages"][0]["role"] == "system"]


@pytest.fixture
def two_editions(tmp_path) -> Path:
    """A cases folder with the example booklet in French and, as a stand-in, in Italian."""
    (tmp_path / "booklets").mkdir()
    for language in ("fr", "it"):
        (tmp_path / "booklets" / f"2020_11_29_{language}.pdf").write_bytes(
            (EXAMPLES / "booklets" / "2020_11_29_fr.pdf").read_bytes())
    return tmp_path


def decide(cls, cases_dir: Path, client: Scripted, case: dict = CASE) -> dict:
    strategy = cls(cls.Settings(name="gied"), "advanced", {}, cases_dir=cases_dir)
    return strategy.answer(case, LLM(client, "some-model", {}))


def test_retrieval_improvement_searches_further_only_after_a_neutral_answer(two_editions):
    client = Scripted("entailment 1")
    assert decide(GiedRetrievalImprovement, two_editions, client)["label_name"] == "entailment"
    assert len(client.decisions_asked()) == 1  # evidence found in the first round: no further search

    client = Scripted("neutral", "contradiction 1")
    prediction = decide(GiedRetrievalImprovement, two_editions, client)
    first, second = (r["messages"][1]["content"] for r in client.decisions_asked())
    assert prediction["label_name"] == "contradiction" and prediction["evidence"]
    assert first != second  # the second round shows other excerpts

    client = Scripted("neutral")
    prediction = decide(GiedRetrievalImprovement, two_editions, client)
    assert prediction["label_name"] == "neutral" and prediction["evidence"]
    assert len(client.decisions_asked()) == 3  # rounds=3, all neutral


def test_paraphrase_writes_five_versions_in_the_booklet_language_and_decides_once(two_editions):
    client = Scripted("entailment 1")
    prediction = decide(GiedParaphrase, two_editions, client)
    rewrite = [r for r in client.requests if r["messages"][0]["content"].startswith("Rewrite")]
    assert len(rewrite) == 1 and "5 times in French" in rewrite[0]["messages"][0]["content"]
    assert len(client.decisions_asked()) == 1
    assert CASE["claim"]["text"] in client.decisions_asked()[0]["messages"][1]["content"]  # the original claim
    assert prediction["label_name"] == "entailment" and prediction["evidence"]


def test_multi_lang_shows_both_editions_and_quotes_the_booklet(two_editions):
    client = Scripted("contradiction 1")
    prediction = decide(GiedMultiLangEvidence, two_editions, client)
    assert [r["messages"][0]["content"][:9] for r in client.requests] == ["Translate", GiedMultiLangEvidence.system_prompt[:9]]
    assert "IN THE ITALIAN EDITION:" in client.decisions_asked()[0]["messages"][1]["content"]
    assert prediction["label_name"] == "contradiction"
    assert prediction["evidence"] and all(1 <= item["page"] <= 32 for item in prediction["evidence"])


def test_multi_lang_decides_a_claim_in_the_booklet_language_as_before(two_editions):
    same = {**CASE, "claim": {**CASE["claim"], "language": "fr"}}
    client = Scripted("entailment 1")
    decide(GiedMultiLangEvidence, two_editions, client, same)
    assert len(client.requests) == 1 and "EDITION:" not in client.requests[0]["messages"][1]["content"]


def test_multi_lang_paraphrase_searches_with_paraphrases_instead_of_a_translation(two_editions):
    client = Scripted("entailment 1")
    prediction = decide(GiedMultiLangParaphrase, two_editions, client)
    assert [r["messages"][0]["content"][:7] for r in client.requests] == ["Rewrite", GiedMultiLangParaphrase.system_prompt[:7]]
    assert "IN THE ITALIAN EDITION:" in client.decisions_asked()[0]["messages"][1]["content"]
    assert prediction["label_name"] == "entailment"
    assert prediction["evidence"] and all(1 <= item["page"] <= 32 for item in prediction["evidence"])


def test_multi_lang_paraphrase_with_a_claim_in_the_booklet_language_shows_one_edition(two_editions):
    same = {**CASE, "claim": {**CASE["claim"], "language": "fr"}}
    client = Scripted("neutral")
    prediction = decide(GiedMultiLangParaphrase, two_editions, client, same)
    assert len(client.requests) == 2 and "EDITION:" not in client.decisions_asked()[0]["messages"][1]["content"]
    assert prediction["evidence"]


def test_scoped_quotes_only_the_pages_of_the_claims_vote(two_editions):
    weapons = next(c for c in dataset.read_jsonl(EXAMPLES / "cases.jsonl") if c["id"] == "v1.1-row-1104-A")
    pages = set(vote_pages(pdf.pages(EXAMPLES / "booklets" / "2020_11_29_fr.pdf"), weapons["vote"]))
    assert 0 < len(pages) < 32  # the booklet has two votes
    prediction = decide(GiedMultiLangScoped, two_editions, Scripted("contradiction 1"), weapons)
    assert prediction["evidence"] and {item["page"] for item in prediction["evidence"]} <= pages


def test_scoped_takes_the_rules_prompt_only_when_told_to(two_editions):
    with_rules = GiedMultiLangScoped(GiedMultiLangScoped.Settings(name="s", rules=True), "advanced", {}, cases_dir=two_editions)
    without = GiedMultiLangScoped(GiedMultiLangScoped.Settings(name="s"), "advanced", {}, cases_dir=two_editions)
    assert with_rules.system_prompt == RULES_PROMPT and without.system_prompt == MULTI_LANG_PROMPT


def test_strict_multi_lang_reads_only_the_supplied_booklet_and_shows_the_translation(two_editions):
    strategy = GiedStrictMultiLangScoped(GiedStrictMultiLangScoped.Settings(name="s"), "advanced", {}, cases_dir=two_editions)
    client = Scripted("contradiction 1")
    prediction = strategy.answer(CASE, LLM(client, "some-model", {}))

    assert list(strategy.read_booklets) == [CASE["booklet"]["path"]]  # not the Italian edition beside it
    assert [r["messages"][0]["content"][:9] for r in client.requests] == ["Translate", strategy.system_prompt[:9]]
    system, user = (message["content"] for message in client.decisions_asked()[0]["messages"])
    assert "EDITION" not in system + user
    assert "THE CLAIM IN FRENCH (a translation, to compare it with the excerpts):\nLes entreprises suisses" in user
    assert prediction["label_name"] == "contradiction" and prediction["evidence"]


def test_strict_multi_lang_decides_a_claim_in_the_booklet_language_with_one_request(two_editions):
    same = {**CASE, "claim": {**CASE["claim"], "language": "fr"}}
    client = Scripted("entailment 1")
    decide(GiedStrictMultiLangScoped, two_editions, client, same)
    assert len(client.requests) == 1 and "a translation" not in client.requests[0]["messages"][1]["content"]


def test_strict_paraphrase_searches_with_the_translation_and_five_paraphrases(two_editions):
    strategy = GiedStrictParaphrase(GiedStrictParaphrase.Settings(name="s"), "advanced", {}, cases_dir=two_editions)
    client = Scripted("entailment 1")
    prediction = strategy.answer(CASE, LLM(client, "some-model", {}))

    assert [r["messages"][0]["content"][:7] for r in client.requests] == ["Transla", "Rewrite", strategy.system_prompt[:7]]
    assert list(strategy.read_booklets) == [CASE["booklet"]["path"]]
    assert "THE CLAIM IN FRENCH (a translation" in client.decisions_asked()[0]["messages"][1]["content"]
    assert prediction["label_name"] == "entailment" and prediction["evidence"]


def test_strict_reasoned_reads_only_the_label_and_excerpts_lines_after_the_reasoning(two_editions):
    reasoning = "The booklet sets 25 years, in 3 places; excerpt 2 says the opposite: « ... ».\nLabel: contradiction\nExcerpts: 1"
    strategy = GiedStrictReasoned(GiedStrictReasoned.Settings(name="s"), "advanced", {}, cases_dir=two_editions)
    client = Scripted(reasoning)
    prediction = strategy.answer(CASE, LLM(client, "some-model", {"max_tokens": strategy.settings.max_tokens}))

    decision = client.decisions_asked()[0]
    assert decision["max_tokens"] == 250 and "First explain" in decision["messages"][0]["content"]
    assert "THE CLAIM IN FRENCH (a translation" in decision["messages"][1]["content"]  # the excerpts as before
    assert strategy.read_answer(CASE, reasoning) == (2, [1])  # not 25, 3 or 2 from the reasoning
    assert prediction["label_name"] == "contradiction" and prediction["evidence"]
    assert list(strategy.read_booklets) == [CASE["booklet"]["path"]]


def test_strict_reasoned_rules_adds_the_four_lines_and_decides_as_before(two_editions):
    strategy = GiedStrictReasonedRules(GiedStrictReasonedRules.Settings(name="s"), "advanced", {}, cases_dir=two_editions)
    for line in ("that party's own statements", "argue against the proposal does not contradict", "one by one",
                 "If your explanation says\nthat the booklet does not mention"):
        assert line in strategy.system_prompt
    client = Scripted("The booklet says the opposite: « ... ».\nLabel: contradiction\nExcerpts: 1")
    prediction = strategy.answer(CASE, LLM(client, "some-model", {"max_tokens": strategy.settings.max_tokens}))
    assert client.decisions_asked()[0]["max_tokens"] == 300
    assert prediction["label_name"] == "contradiction" and prediction["evidence"]


def test_a_gied_strategy_keeps_its_answers_when_told_to(two_editions, tmp_path):
    answers = tmp_path / "answers.jsonl"
    strategy = GiedStrictReasoned(GiedStrictReasoned.Settings(name="s", answers=str(answers)), "advanced", {},
                                  cases_dir=two_editions)
    strategy.answer(CASE, LLM(Scripted("Reasoning.\nLabel: entailment\nExcerpts: 2"), "some-model", {}))

    [record] = [json.loads(line) for line in answers.read_text(encoding="utf-8").splitlines()]
    assert record["id"] == CASE["id"] and record["label"] == "entailment" and record["excerpts"] == [2]
    assert record["answer"].endswith("Excerpts: 2") and "EXCERPTS:" in record["prompt"]


def test_fuse_puts_first_what_several_rankings_put_high():
    a, b, c = (Sentence(1, "", text) for text in ("a", "b", "c"))
    assert fuse([[a, b], [b, c], [b, a]]) == [b, a, c]

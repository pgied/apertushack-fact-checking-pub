"""Tests for the gied_ strategies of the beginner task, built on prompt_lab's best prompt, with a scripted
stand-in model and the sample cases in data/examples/."""

from pathlib import Path
from types import SimpleNamespace

from src import dataset
from src.llm import LLM
from src.strategies.gied_beginner_multi_lang.strategy import GiedBeginnerMultiLang
from src.strategies.gied_beginner_paraphrase.strategy import GiedBeginnerParaphrase
from src.strategies.gied_beginner_scoped.strategy import GiedBeginnerScoped, focus

EXAMPLES = Path(__file__).resolve().parent.parent / "data" / "examples"
CASE = next(c for c in dataset.read_jsonl(EXAMPLES / "cases.jsonl") if c["id"] == "v1.1-row-1104-B")  # German claim, French reference
TRANSLATION = "Le Conseil fédéral recommande d’accepter l’initiative."


class Scripted:
    """Stands in for the OpenAI client: translations and paraphrases come back in French; a decision is
    a short explanation and "Label: contradiction"."""

    def __init__(self):
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        prompt = request["messages"][0]["content"]
        if prompt.startswith("Translate"):
            content = TRANSLATION
        elif prompt.startswith("Rewrite"):
            content = "\n".join(f"{n}. Le Conseil fédéral est pour l’initiative, version {n}." for n in range(1, 6))
        else:
            content = "The reference says the Federal Council recommends a no.\nLabel: contradiction"
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)

    def decision(self) -> str:
        return next(r["messages"][1]["content"] for r in self.requests if r["messages"][0]["role"] == "system")


def decide(cls, case: dict = CASE) -> tuple[dict, Scripted]:
    client = Scripted()
    strategy = cls(cls.Settings(name="gied"), "beginner", {})
    return strategy.answer(case, LLM(client, "some-model", {})), client


def test_multi_lang_shows_the_claim_also_in_the_reference_language():
    prediction, client = decide(GiedBeginnerMultiLang)
    assert prediction["label_name"] == "contradiction"
    assert len(client.requests) == 2 and f"THE CLAIM IN FRENCH (a translation" in client.decision()
    assert TRANSLATION in client.decision() and client.decision().endswith("Answer:")

    same = {**CASE, "claim": {**CASE["claim"], "language": "fr"}}
    _, client = decide(GiedBeginnerMultiLang, same)
    assert len(client.requests) == 1 and "THE CLAIM IN" not in client.decision()


def test_paraphrase_shows_five_versions_and_decides_the_claim():
    prediction, client = decide(GiedBeginnerParaphrase)
    assert prediction["label_name"] == "contradiction"
    assert len(client.requests) == 2 and "5 times in French" in client.requests[0]["messages"][0]["content"]
    assert "THE CLAIM IN OTHER WORDS" in client.decision() and client.decision().count("\n- ") == 5
    assert CASE["claim"]["text"] in client.decision()


def test_scoped_shows_only_the_part_of_a_long_reference_around_the_claim():
    filler = [f"Paragraphe {n} sur un tout autre sujet, avec des mots qui n’ont rien à voir ici." for n in range(60)]
    filler[30] = "Le Conseil fédéral recommande de rejeter l’initiative, comme le Parlement."
    long = {**CASE, "reference": {**CASE["reference"], "text": "\n".join(filler)}}
    prediction, client = decide(GiedBeginnerScoped, long)
    reference = client.decision().split("REFERENCE:\n", 1)[1].split("\n\nCLAIM:", 1)[0]
    assert len(client.requests) == 2  # one translation, used for the search and shown, and the decision
    assert "Le Conseil fédéral recommande de rejeter" in reference and "[…]" in reference
    assert len(reference) < len(long["reference"]["text"]) / 2
    assert prediction["label_name"] == "contradiction"


def test_focus_keeps_a_short_text_whole_and_a_text_without_a_match_too():
    text = "Erster Satz über etwas ganz anderes als die Frage.\nZweiter Satz über noch etwas anderes."
    assert focus(text, "Kernkraftwerke", 20, 3) == text  # nothing matches

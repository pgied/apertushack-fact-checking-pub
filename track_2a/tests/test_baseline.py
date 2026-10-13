"""Tests for everything that runs without an API key or a model."""

import os
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir

from src import device, nli, strategies
from src.backend import load_dotenv
from src.evaluate import macro_f1
from src.strategies.baseline.strategy import Baseline

CONF = Path(__file__).resolve().parent.parent / "conf"
REQUEST = {"temperature": 0, "max_tokens": 10}
CASE = {
    "id": "case-0001",
    "reference": {"text": "Le Conseil fédéral recommande de rejeter l'initiative."},
    "claim": {"text": "Der Bundesrat empfiehlt, die Initiative anzunehmen."},
}
# The advanced task: the same claim, against a booklet of two pages
BOOKLET_CASE = {
    "id": "case-0002",
    "booklet": {"path": "booklets/2031_06_01_fr.pdf"},
    "vote": "Initiative populaire « Plus de pistes cyclables »",
    "claim": {"text": "Der Bundesrat empfiehlt, die Initiative anzunehmen."},
}
BOOKLETS = {
    "booklets/2031_06_01_fr.pdf": [
        "Votation populaire du 1er juin 2031",
        "2 Initiative populaire « Plus de pistes cyclables »\nLe Conseil fédéral recommande de rejeter l'initiative.\n",
    ],
}


class FakeClient:
    """Stands in for the OpenAI client and always gives the same answer."""

    def __init__(self, answer: str):
        self.answer = answer
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.request = request
        message = SimpleNamespace(content=self.answer)
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=3)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def decide(answer: str, case: dict = CASE, task: str = "beginner", request: dict = REQUEST) -> tuple[dict, FakeClient]:
    """Decide the case with the baseline, the model giving `answer`; returns the prediction and the client."""
    client = FakeClient(answer)
    baseline = Baseline(Baseline.Settings(name="baseline"), task, BOOKLETS)
    return strategies.answer(baseline, case, strategies.LLM(client, "some-model", request)), client


def test_parse_label():
    assert nli.parse_label("Entailment") == 0
    assert nli.parse_label("neutral.") == 1
    assert nli.parse_label("The answer is: CONTRADICTION") == 2
    assert nli.parse_label("je ne sais pas") is None


def test_the_baseline_returns_the_challenge_format():
    prediction, client = decide("contradiction")

    assert prediction["id"] == "case-0001"
    assert prediction["label"] == 2
    assert prediction["label_name"] == "contradiction"
    assert prediction["evidence"] == [{"page": None, "text": CASE["reference"]["text"]}]
    assert prediction["metrics"]["input_tokens"] == 120
    assert prediction["metrics"]["output_tokens"] == 3
    assert isinstance(prediction["metrics"]["inference_time_ms"], int)
    assert client.request["model"] == "some-model"
    assert client.request["temperature"] == 0


def test_request_settings_are_sent_with_the_request():
    _, client = decide("entailment", request={**REQUEST, "reasoning_effort": "none"})
    assert client.request["max_tokens"] == 10
    assert client.request["reasoning_effort"] == "none"


def test_neutral_and_unclear_answers_have_no_evidence():
    assert decide("neutral")[0]["evidence"] == []

    unclear, _ = decide("???")
    assert unclear["label_name"] == "neutral"
    assert unclear["evidence"] == []


def test_the_advanced_baseline_reads_the_whole_booklet_and_cites_a_page():
    prediction, client = decide("contradiction 2", BOOKLET_CASE, "advanced")

    assert prediction["label_name"] == "contradiction"
    assert prediction["evidence"] == [{"page": 2, "text": BOOKLETS["booklets/2031_06_01_fr.pdf"][1].strip()}]
    prompt = client.request["messages"][1]["content"]
    assert "[Page 1]\nVotation populaire" in prompt and "[Page 2]\n2 Initiative" in prompt
    assert BOOKLET_CASE["vote"] in prompt and BOOKLET_CASE["claim"]["text"] in prompt


def test_the_advanced_baseline_gives_no_evidence_without_a_page_of_the_booklet():
    assert decide("entailment 7", BOOKLET_CASE, "advanced")[0]["evidence"] == []  # the booklet has 2 pages
    assert decide("entailment", BOOKLET_CASE, "advanced")[0]["evidence"] == []
    assert decide("neutral 2", BOOKLET_CASE, "advanced")[0]["evidence"] == []


def test_macro_f1():
    assert macro_f1([0, 1, 2], [0, 1, 2]) == 1.0
    # per-label F1: entailment 0.5, neutral 0, contradiction 1
    assert macro_f1([0, 0, 1, 2], [0, 1, 0, 2]) == 0.5


def test_every_model_config_uses_a_known_backend():
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        for model_file in sorted((CONF / "model").glob("*.yaml")):
            cfg = compose(config_name="config", overrides=[f"model={model_file.stem}"])
            assert cfg.model.name, model_file.name
            assert cfg.model.backend in cfg.backends, model_file.name


def test_the_picked_device_computes():
    x = torch.ones(2, 2, device=device.pick())
    assert (x @ x).sum().item() == 8


def test_load_dotenv(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {"API_KEY": "already-set"})
    env = tmp_path / ".env"
    env.write_text("# a comment\nexport API_KEY='from-file'\nOTHER=value  # trailing comment\n")

    load_dotenv(env)

    assert os.environ["API_KEY"] == "already-set"  # the environment wins over .env
    assert os.environ["OTHER"] == "value"

"""Tests for the challenge's CLI in src/entrypoint.py: its input and output, without a model or a PDF."""

import json
from types import SimpleNamespace

import pytest

from src import backend, entrypoint, pdf

# The advanced and the beginner case of the same dataset row, in one file
BOOKLET = ["Votation populaire du 1er juin 2031", "2 Le Conseil fédéral recommande de rejeter l'initiative."]
CASES = [
    {
        "id": "v1.1-row-7-A",
        "vote": "Initiative populaire « Plus de pistes cyclables »",
        "claim": {"text": "Der Bundesrat empfiehlt, die Initiative abzulehnen.", "language": "de"},
        "booklet": {"path": "booklets/2031_06_01_fr.pdf", "language": "fr"},
    },
    {
        "id": "v1.1-row-7-B",
        "vote": "Initiative populaire « Plus de pistes cyclables »",
        "claim": {"text": "Der Bundesrat empfiehlt, die Initiative abzulehnen.", "language": "de"},
        "reference": {"text": "Le Conseil fédéral recommande de rejeter l'initiative.", "language": "fr"},
    },
]
ENDPOINT_VARIABLES = ("BASE_URL", "API_KEY", "LLM_BASE_URL", "LLM_API_KEY", "LLM_NAME")


class Client:
    """Stands in for the OpenAI client: always gives the same answer, but fails for claims containing `fail`."""

    def __init__(self, answer: str, fail: str | None = None):
        self.answer = answer
        self.fail = fail
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        if self.fail and self.fail in request["messages"][-1]["content"]:
            raise ConnectionError("the endpoint does not answer")
        message = SimpleNamespace(content=self.answer)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=2)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


@pytest.fixture(autouse=True)
def no_endpoint_variables(monkeypatch):
    """Neither the developer's environment nor .env may change what the tests see."""
    for name in ENDPOINT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(entrypoint, "load_dotenv", lambda path: None)


def run(tmp_path, monkeypatch, client: Client, cases: list[dict] = CASES, pages=lambda path: BOOKLET) -> dict:
    """Run the CLI on the cases in tmp_path/data/cases.jsonl and return its predictions by id."""
    cases_file = tmp_path / "data" / "cases.jsonl"
    cases_file.parent.mkdir()
    cases_file.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    output = tmp_path / "output" / "predictions.jsonl"
    monkeypatch.setattr(entrypoint, "make_client", lambda backend: client)
    monkeypatch.setattr(pdf, "pages", pages)

    entrypoint.main(["--input", str(cases_file), "--output", str(output)])

    predictions = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(predictions) == len(cases)  # one line per case
    return {prediction["id"]: prediction for prediction in predictions}


def test_a_file_with_both_tasks_gets_one_line_per_case(tmp_path, monkeypatch, capsys):
    client = Client("contradiction 2")
    predictions = run(tmp_path, monkeypatch, client)

    # The run's time without the model, which the challenge's time score counts, is printed at the end
    assert "Time without the model: " in capsys.readouterr().err

    assert set(predictions) == {"v1.1-row-7-A", "v1.1-row-7-B"}
    for prediction in predictions.values():
        assert (prediction["label"], prediction["label_name"]) == (2, "contradiction")
        assert prediction["metrics"]["input_tokens"] == 100 and prediction["metrics"]["output_tokens"] == 2
    # The advanced case cites a page of its booklet, counted from 1; the beginner case has no page
    assert predictions["v1.1-row-7-A"]["evidence"] == [{"page": 2, "text": BOOKLET[1]}]
    assert predictions["v1.1-row-7-B"]["evidence"] == [{"page": None, "text": CASES[1]["reference"]["text"]}]
    assert {request["model"] for request in client.requests} == {"swiss-ai/Apertus-v1.5-70B"}  # what we submit


def test_a_failing_case_is_answered_neutral_and_the_others_still_run(tmp_path, monkeypatch):
    failing = {**CASES[1], "id": "v1.1-row-8-B", "claim": {"text": "fail here", "language": "de"}}
    predictions = run(tmp_path, monkeypatch, Client("entailment", fail="fail here"), [*CASES, failing])

    assert predictions["v1.1-row-8-B"]["label_name"] == "neutral"
    assert predictions["v1.1-row-8-B"]["evidence"] == []
    assert predictions["v1.1-row-7-B"]["label_name"] == "entailment"


def test_without_its_booklet_a_case_is_answered_neutral(tmp_path, monkeypatch):
    def missing(path):
        raise FileNotFoundError(path)

    predictions = run(tmp_path, monkeypatch, Client("entailment"), pages=missing)

    assert predictions["v1.1-row-7-A"]["label_name"] == "neutral"  # the advanced task could not be set up
    assert predictions["v1.1-row-7-B"]["label_name"] == "entailment"  # the beginner task still runs


def test_the_endpoint_and_key_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("API_KEY", "team-key")
    cfg = entrypoint.configure([])
    cscs = cfg.backends[cfg.model.backend]
    assert cscs.url == "https://api.inference.cscs.ch/v1"  # the default
    assert backend.make_client(cscs).api_key == "team-key"

    # At the evaluation: the organisers' proxy; LLM_NAME, the template's variable, picks the model
    monkeypatch.setenv("BASE_URL", "http://proxy:8000/v1")
    monkeypatch.setenv("LLM_NAME", "swiss-ai/Apertus-v1.5-8B")
    cfg = entrypoint.configure([])
    assert cfg.backends[cfg.model.backend].url == "http://proxy:8000/v1"
    assert cfg.model.name == "swiss-ai/Apertus-v1.5-8B"


def test_the_templates_variable_names_work_too(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm:8000/v1")
    monkeypatch.setenv("LLM_API_KEY", "template-key")

    client = backend.make_client(entrypoint.configure([]).backends.cscs)

    assert str(client.base_url).rstrip("/") == "http://llm:8000/v1"
    assert client.api_key == "template-key"


def test_without_a_key_the_cli_stops():
    with pytest.raises(SystemExit, match="API_KEY or LLM_API_KEY is not set"):
        backend.make_client(entrypoint.configure([]).backends.cscs)


def test_overrides_change_what_we_submit():
    assert entrypoint.configure([]).model.name == "swiss-ai/Apertus-v1.5-70B"
    assert entrypoint.configure(["model=apertus-8b"]).model.name == "swiss-ai/Apertus-v1.5-8B"

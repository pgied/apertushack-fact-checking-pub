"""Tests for the challenge's CLI in src/entrypoint.py: its input and output, without a model or a PDF."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import backend, entrypoint, pdf

EXAMPLES = Path(__file__).resolve().parent.parent / "data" / "examples"
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
    """Run the CLI on the cases in tmp_path/data/cases.jsonl and return its predictions by id. The
    baseline decides both tasks here: it reads the booklet only through `pages`, which stands in for the PDF."""
    cases_file = tmp_path / "data" / "cases.jsonl"
    cases_file.parent.mkdir()
    cases_file.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    output = tmp_path / "output" / "predictions.jsonl"
    monkeypatch.setattr(entrypoint, "make_client", lambda backend: client)
    monkeypatch.setattr(pdf, "pages", pages)

    entrypoint.main(["--input", str(cases_file), "--output", str(output), "strategy=baseline"])

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
    cfg = entrypoint.configure("advanced", [])
    cscs = cfg.backends[cfg.model.backend]
    assert cscs.url == "https://api.inference.cscs.ch/v1"  # the default
    assert backend.make_client(cscs).api_key == "team-key"

    # At the evaluation: the organisers' proxy; LLM_NAME, the template's variable, picks the model
    monkeypatch.setenv("BASE_URL", "http://proxy:8000/v1")
    monkeypatch.setenv("LLM_NAME", "swiss-ai/Apertus-v1.5-8B")
    cfg = entrypoint.configure("advanced", [])
    assert cfg.backends[cfg.model.backend].url == "http://proxy:8000/v1"
    assert cfg.model.name == "swiss-ai/Apertus-v1.5-8B"


def test_the_templates_variable_names_work_too(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm:8000/v1")
    monkeypatch.setenv("LLM_API_KEY", "template-key")

    client = backend.make_client(entrypoint.configure("advanced", []).backends.cscs)

    assert str(client.base_url).rstrip("/") == "http://llm:8000/v1"
    assert client.api_key == "template-key"


def test_without_a_key_the_cli_stops():
    with pytest.raises(SystemExit, match="API_KEY or LLM_API_KEY is not set"):
        backend.make_client(entrypoint.configure("advanced", []).backends.cscs)


def test_each_task_gets_the_strategy_we_submit_for_it_and_overrides_change_both():
    assert entrypoint.configure("beginner", []).strategy.name == "gied_beginner_scoped"
    assert entrypoint.configure("advanced", []).strategy.name == "gied_strict_reasoned"
    for task in ("beginner", "advanced"):
        assert entrypoint.configure(task, []).model.name == "swiss-ai/Apertus-v1.5-70B"
        assert entrypoint.configure(task, ["model=apertus-8b"]).model.name == "swiss-ai/Apertus-v1.5-8B"
        assert entrypoint.configure(task, ["strategy=baseline"]).strategy.name == "baseline"


class Scripted(Client):
    """Stands in for the OpenAI client with the strategies we submit: a translation comes back in French,
    a decision is contradiction, citing the first excerpt."""

    def create(self, **request):
        self.requests.append(request)
        translating = request["messages"][0]["content"].startswith("Translate")
        content = ("Les entreprises suisses doivent respecter les droits humains. Le Conseil fédéral recommande "
                   "de rejeter l'initiative." if translating else "Label: contradiction [1]")
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=2)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)


def test_the_strategies_we_submit_decide_the_examples(tmp_path, monkeypatch):
    """The sample of make run: the advanced and the beginner case of two dataset rows, with a real booklet."""
    client = Scripted("")
    monkeypatch.setattr(entrypoint, "make_client", lambda backend: client)
    output = tmp_path / "predictions.jsonl"
    entrypoint.main(["--input", str(EXAMPLES / "cases.jsonl"), "--output", str(output)])

    predictions = {p["id"]: p for p in map(json.loads, output.read_text(encoding="utf-8").splitlines())}
    cases = {c["id"]: c for c in map(json.loads, (EXAMPLES / "cases.jsonl").read_text(encoding="utf-8").splitlines())}
    assert set(predictions) == set(cases) and all(p["label_name"] == "contradiction" for p in predictions.values())
    for case_id, prediction in predictions.items():
        if "booklet" in cases[case_id]:  # verbatim sentences of the booklet, with their pages
            assert prediction["evidence"] and all(1 <= item["page"] <= 32 for item in prediction["evidence"])
        else:  # the reference, or the part of it the model saw: not scored
            assert prediction["evidence"] and all(item["page"] is None for item in prediction["evidence"])
    asked = [r["messages"][0]["content"] for r in client.requests]
    assert sum(prompt.startswith("Translate") for prompt in asked) == 4  # every claim is in another language
    assert sum("EXCERPTS" in prompt for prompt in asked) == 2 and sum("REFERENCE" in prompt for prompt in asked) == 2

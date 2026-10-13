"""Tests for the strategies: finding them, their settings, and measuring what they do."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException, MissingConfigException
from omegaconf import OmegaConf

from src import nli, pdf
from src.__main__ import run_settings
from src.catalog import Catalog
from src.llm import LLM
from src.strategy import TASKS, Strategy, Verdict

CONF = Path(__file__).resolve().parent.parent / "conf"
CASE = {
    "id": "case-0001",
    "reference": {"text": "Le Conseil fédéral recommande de rejeter l'initiative."},
    "claim": {"text": "Der Bundesrat empfiehlt, die Initiative anzunehmen."},
}


class Client:
    """Stands in for the OpenAI client: always gives the same answer, and keeps the requests."""

    def __init__(self, answer: str = "neutral"):
        self.answer = answer
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        message = SimpleNamespace(content=self.answer)
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=2)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class AskTwice(Strategy):
    """Asks the model twice, the second time with more tokens, and goes by the second answer."""

    tasks = ("beginner",)

    def predict(self, case, llm):
        llm.chat([{"role": "user", "content": "first"}])
        return Verdict(nli.parse_label(llm.chat([{"role": "user", "content": "second"}], max_tokens=50)))


class NoSuchLabel(Strategy):
    tasks = ("beginner",)

    def predict(self, case, llm):
        return Verdict(3)


class Indexed(Strategy):
    """Asks the model once in setup(), and counts the pages of the booklets there."""

    tasks = ("advanced",)

    def setup(self, cases, llm):
        llm.chat([{"role": "user", "content": "once per run"}])
        self.index = {path: len(pages) for path, pages in self.booklets.items()}


def compose_config(*overrides: str):
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


def test_every_strategy_loads_and_says_which_tasks_it_does():
    catalog = Catalog()
    assert "baseline" in catalog.names()
    for name in catalog.names():
        strategy = catalog.load(name)
        assert strategy.tasks and set(strategy.tasks) <= set(TASKS), name


def test_every_strategy_composes_with_its_settings_for_each_of_its_tasks():
    catalog = Catalog()
    for name in catalog.names():
        strategy = catalog.load(name)
        for task in strategy.tasks:
            cfg = compose_config(f"strategy={name}", f"task={task}")
            settings = OmegaConf.to_object(cfg.strategy)
            assert isinstance(settings, strategy.Settings), name
            assert settings.name == name


def test_strategy_settings_are_checked():
    assert compose_config("strategy.max_tokens=20").strategy.max_tokens == 20
    with pytest.raises(ConfigCompositionException):
        compose_config("strategy.max_tokenz=20")  # misspelled
    with pytest.raises(ConfigCompositionException):
        compose_config("strategy.max_tokens=many")  # not a number
    with pytest.raises(MissingConfigException):
        compose_config("strategy=no_such_strategy")


def test_a_run_is_continued_only_with_the_same_task_model_strategy_and_limit():
    settings = run_settings(compose_config("task=advanced", "strategy.max_tokens=20"))

    assert set(settings) == {"task", "model", "strategy", "limit"}
    assert settings["strategy"]["max_tokens"] == 20
    assert run_settings(compose_config("task=advanced")) != settings


def test_a_prediction_counts_all_requests_of_its_case():
    client = Client("entailment")
    strategy = AskTwice(AskTwice.Settings(name="ask_twice"), "beginner", {})
    prediction = strategy.answer(CASE, LLM(client, "some-model", {"max_tokens": 10}))

    assert prediction["label_name"] == "entailment"
    assert prediction["metrics"]["input_tokens"] == 200
    assert prediction["metrics"]["output_tokens"] == 4
    assert [request["max_tokens"] for request in client.requests] == [10, 50]  # the second request overrides it
    assert all(request["model"] == "some-model" for request in client.requests)


def test_a_label_that_does_not_exist_is_refused():
    strategy = NoSuchLabel(NoSuchLabel.Settings(name="no_such_label"), "beginner", {})
    with pytest.raises(ValueError):
        strategy.answer(CASE, LLM(Client(), "some-model", {}))


def test_start_reads_each_booklet_once_and_measures_the_setup(monkeypatch, tmp_path):
    read = []

    def pages(path: Path) -> list[str]:
        read.append(path)
        return [f"the text of {path.name}"]

    monkeypatch.setattr(pdf, "pages", pages)
    cases = [{"id": f"case-{i}", "booklet": {"path": f"booklets/{name}.pdf"}} for i, name in enumerate("aba")]

    strategy, setup = Indexed.start(Indexed.Settings(name="indexed"), "advanced", cases, tmp_path,
                                    LLM(Client(), "some-model", {}))

    assert read == [tmp_path / "booklets/a.pdf", tmp_path / "booklets/b.pdf"]  # relative to the cases' folder
    assert strategy.pages(cases[2]) == ["the text of a.pdf"]
    assert strategy.index == {"booklets/a.pdf": 1, "booklets/b.pdf": 1}
    assert setup["cases"] == 3 and setup["booklets"] == 2
    assert setup["input_tokens"] == 100 and setup["output_tokens"] == 2
    assert isinstance(setup["setup_time_ms"], int)


def test_the_beginner_task_reads_no_booklets(monkeypatch, tmp_path):
    monkeypatch.setattr(pdf, "pages", lambda path: pytest.fail("no booklet to read"))
    strategy, setup = AskTwice.start(AskTwice.Settings(name="ask_twice"), "beginner", [CASE], tmp_path,
                                     LLM(Client(), "some-model", {}))
    assert strategy.booklets == {} and setup["booklets"] == 0

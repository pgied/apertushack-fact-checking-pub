"""Strategies: the ways we try to decide a case, one file or folder each, for the beginner and the advanced task.

A strategy is a file src/strategies/<name>.py, or a folder src/strategies/<name>/ for a bigger one,
that defines one subclass of Strategy. It is found by its name, so nothing else has to change:

    uv run python -m src strategy=<name> task=advanced limit=30

Files and folders whose names start with _ are not strategies: they can hold code that strategies
share. The baseline in baseline.py is the template to copy.
"""

import importlib
import pkgutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from src import pdf
from src.nli import LABELS

TASKS = ("beginner", "advanced")


@dataclass
class Verdict:
    """What a strategy decides for one case."""

    label: int  # the position in nli.LABELS: 0 entailment, 1 neutral, 2 contradiction
    # [{"page": n, "text": "verbatim passage"}]: required for entailment and contradiction, may be empty for neutral
    evidence: list[dict] = field(default_factory=list)


class LLM:
    """The model, for one case: sends chat requests and adds up their tokens.

    Every case gets its own, so its prediction counts the tokens of all its requests. `request` holds
    what is sent with every request: temperature and max_tokens from the strategy's settings, plus
    the backend's extra fields.
    """

    def __init__(self, client, model: str, request: dict):
        self.client = client
        self.request = {"model": model, **request}
        self.requests = self.input_tokens = self.output_tokens = 0
        self.lock = threading.Lock()  # a strategy may send several requests at once

    def chat(self, messages: list[dict], **request) -> str:
        """Send the messages and return the answer. `request` overrides settings for this request, e.g. max_tokens=50."""
        response = self.client.chat.completions.create(messages=messages, **{**self.request, **request})
        with self.lock:
            self.requests += 1
            self.input_tokens += response.usage.prompt_tokens
            self.output_tokens += response.usage.completion_tokens
        return response.choices[0].message.content or ""


class Strategy:
    """The base class of the strategies: subclass it, say which tasks it does, and write predict()."""

    tasks: tuple[str, ...] = ()  # the tasks it can do: "beginner", "advanced" or both

    @dataclass
    class Settings:
        """The strategy's settings, overridden on the command line with strategy.<setting>=...

        A strategy adds its own in `class Settings(Strategy.Settings)`, each with a default.
        """

        name: str = MISSING  # the strategy's file or folder name, filled in by register()
        temperature: float = 0.0  # sent with every request
        max_tokens: int = 10  # sent with every request; the baseline answers with a word and a page

    def __init__(self, settings: Settings, task: str, booklets: dict[str, list[str]]):
        self.settings = settings
        self.task = task
        self.booklets = booklets  # advanced task: the booklets of the run's cases, by path, page by page

    def setup(self, cases: list[dict], llm: LLM) -> None:
        """Called once per run, before the cases, with all of them: e.g. split the booklets into chunks
        and index them. Its time and tokens are measured apart from the cases'."""

    def predict(self, case: dict, llm: LLM) -> Verdict:
        """Decide one case. Several cases run at once, in threads: only read what setup() built."""
        raise NotImplementedError

    def pages(self, case: dict) -> list[str]:
        """Advanced task: the text of the case's booklet, page by page: page n is pages(case)[n - 1]."""
        return self.booklets[case["booklet"]["path"]]


def names() -> list[str]:
    """The strategies: the files and folders here whose names do not start with _."""
    return sorted(module.name for module in pkgutil.iter_modules(__path__) if not module.name.startswith("_"))


def load(name: str) -> type[Strategy]:
    """The subclass of Strategy that src/strategies/<name> defines, in the file or anywhere in the folder."""
    module = importlib.import_module(f"{__name__}.{name}")
    found = [
        value for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, Strategy)
        and (value.__module__ == module.__name__ or value.__module__.startswith(module.__name__ + "."))
    ]
    if len(found) != 1:
        raise TypeError(f"src/strategies/{name} has to define one subclass of Strategy, not {len(found)}")
    return found[0]


def register() -> None:
    """Make the strategies known to Hydra, as config group `strategy` with one option per strategy."""
    store = ConfigStore.instance()
    for name in names():
        try:
            strategy = load(name)
        except Exception as error:  # one broken strategy must not stop the others
            print(f"Strategy {name} is left out, it does not load: {error!r}", file=sys.stderr)
            continue
        store.store(group="strategy", name=name, node=strategy.Settings(name=name))


def start(cls: type[Strategy], settings: Strategy.Settings, task: str, cases: list[dict], cases_dir: Path,
          llm: LLM) -> tuple[Strategy, dict]:
    """Set up the strategy for a run: read the booklets of the cases (advanced task), then call its setup().

    Returns the strategy, and what it took to set it up: that is done once per run, not per case.
    """
    begin = time.perf_counter()
    booklets = {}
    if task == "advanced":  # booklet paths are relative to the folder of cases.jsonl
        for path in sorted({case["booklet"]["path"] for case in cases}):
            booklets[path] = pdf.pages(cases_dir / path)
    strategy = cls(settings, task, booklets)
    strategy.setup(cases, llm)
    return strategy, {
        "cases": len(cases),
        "booklets": len(booklets),
        "input_tokens": llm.input_tokens,
        "output_tokens": llm.output_tokens,
        "setup_time_ms": round((time.perf_counter() - begin) * 1000),
    }


def answer(strategy: Strategy, case: dict, llm: LLM) -> dict:
    """Decide one case with the strategy and return the prediction in the challenge's output format."""
    begin = time.perf_counter()
    verdict = strategy.predict(case, llm)
    inference_time_ms = round((time.perf_counter() - begin) * 1000)
    if verdict.label not in range(len(LABELS)):
        raise ValueError(f"label {verdict.label!r} is none of 0, 1 and 2")
    label = int(verdict.label)  # also when the strategy hands back a numpy or torch integer
    return {
        "id": case["id"],
        "label": label,
        "label_name": LABELS[label],
        "evidence": verdict.evidence,
        "metrics": {
            "input_tokens": llm.input_tokens,
            "output_tokens": llm.output_tokens,
            "inference_time_ms": inference_time_ms,  # all of predict(): every request, and whatever else it does
        },
    }

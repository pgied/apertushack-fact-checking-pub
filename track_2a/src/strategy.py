"""What a strategy is: the base class that every strategy subclasses, and the verdict it returns.

A strategy subclasses Strategy, says which tasks it does and writes predict(); setup() is optional.
The command lines, src/__main__.py and src/entrypoint.py, call start() once per run and answer()
for every case. Where the strategies live and how they are found: src/catalog.py.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING

from src import pdf
from src.llm import LLM
from src.nli import LABELS

TASKS = ("beginner", "advanced")


@dataclass
class Verdict:
    """What a strategy decides for one case."""

    label: int  # the position in nli.LABELS: 0 entailment, 1 neutral, 2 contradiction
    # [{"page": n, "text": "verbatim passage"}]: required for entailment and contradiction, may be empty for neutral
    evidence: list[dict] = field(default_factory=list)


class Strategy:
    """The base class of the strategies: subclass it, say which tasks it does, and write predict()."""

    tasks: tuple[str, ...] = ()  # the tasks it can do: "beginner", "advanced" or both

    @dataclass
    class Settings:
        """The strategy's settings, overridden on the command line with strategy.<setting>=...

        A strategy adds its own in `class Settings(Strategy.Settings)`, each with a default.
        """

        name: str = MISSING  # the strategy's folder name, filled in by Catalog.register()
        temperature: float = 0.0  # sent with every request
        max_tokens: int = 10  # sent with every request; the baseline answers with a word and a page

    def __init__(self, settings: Settings, task: str, booklets: dict[str, list[str]], cases_dir: Path | None = None):
        self.settings = settings
        self.task = task
        self.booklets = booklets  # advanced task: the booklets of the run's cases, by path, page by page
        self.cases_dir = cases_dir  # the folder of cases.jsonl: booklet paths are relative to it

    # What a strategy writes: setup() and predict()

    def setup(self, cases: list[dict], llm: LLM) -> None:
        """Called once per run, before the cases, with all of them: e.g. split the booklets into chunks
        and index them. Its time and tokens are measured apart from the cases'."""

    def predict(self, case: dict, llm: LLM) -> Verdict:
        """Decide one case. Several cases run at once, in threads: only read what setup() built."""
        raise NotImplementedError

    def pages(self, case: dict) -> list[str]:
        """Advanced task: the text of the case's booklet, page by page: page n is pages(case)[n - 1]."""
        return self.booklets[case["booklet"]["path"]]

    # What the command lines call: start() and answer()

    @classmethod
    def start(cls, settings: Settings, task: str, cases: list[dict], cases_dir: Path, llm: LLM) -> tuple["Strategy", dict]:
        """Set up the strategy for a run: read the booklets of the cases (advanced task), then call its setup().

        Returns the strategy, and what it took to set it up: that is done once per run, not per case.
        """
        begin = time.perf_counter()
        booklets = {}
        if task == "advanced":  # booklet paths are relative to the folder of cases.jsonl
            for path in sorted({case["booklet"]["path"] for case in cases}):
                booklets[path] = pdf.pages(cases_dir / path)
        strategy = cls(settings, task, booklets, cases_dir=cases_dir)
        strategy.setup(cases, llm)
        return strategy, {
            "cases": len(cases),
            "booklets": len(booklets),
            "input_tokens": llm.input_tokens,
            "output_tokens": llm.output_tokens,
            "setup_time_ms": round((time.perf_counter() - begin) * 1000),
        }

    def answer(self, case: dict, llm: LLM) -> dict:
        """Decide one case and return the prediction in the challenge's output format."""
        begin = time.perf_counter()
        verdict = self.predict(case, llm)
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

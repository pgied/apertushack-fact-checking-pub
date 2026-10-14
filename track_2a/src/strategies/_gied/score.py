"""Score runs the way the challenge does, and measure the retrieval of the gied_ strategies alone.

    uv run python -m src.strategies._gied.score run data/runs/<run>                # a run, with the official evidence score
    uv run python -m src.strategies._gied.score retrieval strategy=gied_tags       # retrieval only, no model
    uv run python -m src.strategies._gied.score retrieval strategy=gied_tags strategy.pivot=false limit=200

The evidence check is the one in the challenge's evaluate.py (gitlab.com/ifsoftware/hackapertus-starter).
An evidence item counts if, normalised, it has at most 5,000 characters and lies inside the gold
passage (the dataset's reference text) or contains it: rapidfuzz's partial_ratio is 90 or more. A
case whose gold label is entailment or contradiction is found if one of its first five items counts,
whatever label was predicted.

`retrieval` takes the evidence that the strategy has before it asks the model: the sentences it
found. A claim that would need translating is searched as it is, since that needs the model.
"""

import json
import re
import sys
import time
import unicodedata
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src import dataset, evaluate
from src.catalog import Catalog
from src.nli import LABELS
from src.strategies._gied.retrieve import evidence

ROOT = Path(__file__).resolve().parents[3]  # track_2a/
NEUTRAL = LABELS.index("neutral")
MAX_QUOTES, MAX_QUOTE_CHARS, MIN_RATIO = 5, 5000, 90


def normalize(text: str) -> str:
    """As the challenge normalises evidence and gold passages."""
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    text = re.sub(r"(\w) ?-\s*\n\s*(\w)", r"\1\2", text)  # words hyphenated across lines
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_matches(quote: str, passage: str) -> bool:
    """Whether one evidence item counts for the gold passage."""
    from rapidfuzz import fuzz  # a dev dependency: not in the Docker image

    quote, passage = normalize(quote), normalize(passage)
    if not quote or len(quote) > MAX_QUOTE_CHARS:
        return False
    return fuzz.partial_ratio(quote, passage) >= MIN_RATIO


def found(evidence: list, passage: str) -> bool:
    """Whether one of the first five evidence items counts."""
    quotes = [item["text"] for item in evidence[:MAX_QUOTES] if isinstance(item, dict) and isinstance(item.get("text"), str)]
    return any(quote_matches(quote, passage) for quote in quotes)


def references() -> dict[str, str]:
    """The gold passage of each dataset row: the reference text of its beginner case, which has the
    same row number (v1.1-row-42-B for v1.1-row-42-A)."""
    return {row(case["id"]): case["reference"]["text"]
            for case in dataset.read_jsonl(ROOT / "data" / "beginner" / "cases.jsonl")}


def row(case_id: str) -> str:
    return case_id.rsplit("-", 1)[0]


def hit_rates(results: list[tuple[dict, bool]]) -> str:
    """The share of cases whose evidence was found: in all, for claims in the booklet's language, and
    for claims in another language."""
    lines = []
    for name, kept in [("all", lambda gold: True),
                       ("same language", lambda gold: gold["claim_language"] == gold["reference_language"]),
                       ("cross-lingual", lambda gold: gold["claim_language"] != gold["reference_language"])]:
        hits = [hit for gold, hit in results if kept(gold)]
        if hits:
            lines.append(f"  {name:<15} {sum(hits) / len(hits):.3f}   ({len(hits)} cases)")
    return "\n".join(lines)


def score_run(run: Path) -> None:
    saved = OmegaConf.load(run / "config.yaml")
    task = saved.get("task", "beginner")
    gold = dataset.read_jsonl(ROOT / saved.data_dirs[task] / "gold.jsonl")
    predictions = {p["id"]: p for p in dataset.read_jsonl(run / "predictions.jsonl")}
    setup, timing = run / "setup.json", run / "time.json"
    print(f"{run}\n{task.capitalize()} task, strategy {saved.strategy.name}, model {saved.model.name}")
    print(evaluate.report(gold, list(predictions.values()), json.loads(setup.read_text()) if setup.exists() else None,
                          json.loads(timing.read_text()) if timing.exists() else None))
    if task == "advanced":
        passages = references()
        results = [(g, found(predictions[g["id"]]["evidence"], passages[row(g["id"])]))
                   for g in gold if g["label"] != NEUTRAL and g["id"] in predictions]
        print("\nEvidence, as the challenge scores it (Hit@5 of the gold entailment and contradiction cases)")
        print(hit_rates(results))


class ClaimAsIs:
    """Stands in for the model in `retrieval`: asked to translate a claim, it gives the claim back."""

    def chat(self, messages: list[dict], **request) -> str:
        return messages[-1]["content"].rsplit("\n\n", 1)[-1]


def score_retrieval(overrides: list[str]) -> None:
    catalog = Catalog()
    catalog.register()
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=[*overrides, "task=advanced"])
    cases_dir = ROOT / cfg.data_dirs.advanced
    strategy = catalog.load(cfg.strategy.name)(OmegaConf.to_object(cfg.strategy), "advanced", {}, cases_dir=cases_dir)

    gold = {g["id"]: g for g in dataset.read_jsonl(cases_dir / "gold.jsonl")}
    cases = [c for c in dataset.read_jsonl(cases_dir / "cases.jsonl") if gold[c["id"]]["label"] != NEUTRAL]
    if cfg.limit:
        cases = cases[:: max(1, len(cases) // cfg.limit)][: cfg.limit]
    passages = references()

    begin = time.perf_counter()
    results, page_hits, characters = [], [], []
    for case in cases:
        booklet = strategy.booklet(case["booklet"]["path"])
        sentences = strategy.find(case, booklet, ClaimAsIs())
        items = evidence(sentences, [], case["vote"])
        results.append((gold[case["id"]], found(items, passages[row(case["id"])])))
        page_hits.append(bool({i["page"] for i in items} & {e["page"] for e in gold[case["id"]]["evidence"]}))
        characters.append(sum(len(s.text) for excerpt in strategy.excerpts(booklet, sentences) for s in excerpt))
    seconds = time.perf_counter() - begin

    print(f"Retrieval of {cfg.strategy.name} on {len(cases)} gold entailment and contradiction cases; "
          f"settings: {OmegaConf.to_container(cfg.strategy)}")
    print(f"Evidence found, as the challenge scores it (Hit@5)\n{hit_rates(results)}")
    print(f"A gold page among the evidence pages: {sum(page_hits) / len(page_hits):.3f}")
    print(f"Excerpts for the model: {sum(characters) / len(characters):,.0f} characters per case on average")
    print(f"Time without a model: {seconds:.1f} s for {len(strategy.read_booklets)} booklets read and "
          f"{len(cases)} cases searched")


def main(argv: list[str]) -> None:
    if len(argv) >= 2 and argv[0] == "run":
        score_run(Path(argv[1]))
    elif argv and argv[0] == "retrieval":
        score_retrieval(argv[1:])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])

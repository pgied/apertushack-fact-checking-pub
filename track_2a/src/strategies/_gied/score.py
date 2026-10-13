"""Score runs the way the challenge does, and measure the retrieval of the gied_ strategies alone.

    uv run python -m src.strategies._gied.score run data/runs/<run>                # a run, with the official evidence score
    uv run python -m src.strategies._gied.score retrieval strategy=gied_tags       # retrieval only, no model
    uv run python -m src.strategies._gied.score retrieval strategy=gied_pdfium strategy.pivot=false limit=200

The evidence check is the one in the challenge's evaluate.py (gitlab.com/ifsoftware/hackapertus-starter).
An evidence item counts if, normalised, it has at most 5,000 characters and lies inside the gold
passage (the dataset's reference text) or contains it: rapidfuzz's partial_ratio is 90 or more. A
case whose gold label is entailment or contradiction is found if one of its first five items counts,
whatever label was predicted.

`retrieval` takes the evidence that the strategy would return without asking the model: its best
sentences. A cross-lingual claim without the booklet's edition in its language is searched as it is,
since the translation would need the model.
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
    """The gold passage of every dataset row: the reference text of its beginner case, which has the
    same row number (v1.1-row-42-B for v1.1-row-42-A)."""
    cases = dataset.read_jsonl(ROOT / "data" / "beginner" / "cases.jsonl")
    return {row(case["id"]): case["reference"]["text"] for case in cases}


def row(case_id: str) -> str:
    return case_id.rsplit("-", 1)[0]


def hit_rates(cases: list[tuple[dict, bool]]) -> list[str]:
    """Lines with the share of cases found, overall, by language pair kind, and by claim language."""
    groups = {"all": lambda g: True,
              "same language": lambda g: g["claim_language"] == g["reference_language"],
              "cross-lingual": lambda g: g["claim_language"] != g["reference_language"],
              **{f"claim in {lang}": lambda g, lang=lang: g["claim_language"] == lang for lang in ("de", "fr", "it")}}
    lines = []
    for name, member in groups.items():
        hits = [hit for gold, hit in cases if member(gold)]
        if hits:
            lines.append(f"  {name:<15} {sum(hits) / len(hits):.3f}   ({len(hits)} cases)")
    return lines


def score_run(run: Path) -> None:
    saved = OmegaConf.load(run / "config.yaml")
    task = saved.get("task", "beginner")
    gold = dataset.read_jsonl(ROOT / saved.data_dirs[task] / "gold.jsonl")
    predictions = dataset.read_jsonl(run / "predictions.jsonl")
    setup = run / "setup.json"
    print(f"{run}\n{task.capitalize()} task, strategy {saved.strategy.name}, model {saved.model.name}")
    print(evaluate.report(gold, predictions, json.loads(setup.read_text()) if setup.exists() else None))
    if task != "advanced":
        return
    passages, by_id = references(), {p["id"]: p for p in predictions}
    scored = [(g, found(by_id[g["id"]]["evidence"], passages[row(g["id"])]))
              for g in gold if g["label"] != NEUTRAL and g["id"] in by_id]
    long = sum(len(normalize(item.get("text", ""))) > MAX_QUOTE_CHARS
               for p in predictions for item in p["evidence"][:MAX_QUOTES] if isinstance(item, dict))
    print("\nEvidence, as the challenge scores it (Hit@5 of the gold entailment and contradiction cases)")
    print("\n".join(hit_rates(scored)))
    print(f"  items over {MAX_QUOTE_CHARS:,} characters: {long}")


class NoModel:
    """Stands in for the model in `retrieval`: a claim to translate comes back as it is."""

    def __init__(self):
        self.asked = 0

    def chat(self, messages: list[dict], **request) -> str:
        self.asked += 1
        return messages[-1]["content"].rsplit("\n\n", 1)[-1]


def score_retrieval(overrides: list[str]) -> None:
    catalog = Catalog()
    catalog.register()
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=[*overrides, "task=advanced"])
    settings = OmegaConf.to_object(cfg.strategy)
    cls = catalog.load(cfg.strategy.name)
    cases_dir = ROOT / cfg.data_dirs.advanced
    strategy = cls(settings, "advanced", {}, cases_dir=cases_dir)
    if not hasattr(strategy, "search"):
        raise SystemExit(f"{cfg.strategy.name} is not a gied_ retrieval strategy")

    gold = {g["id"]: g for g in dataset.read_jsonl(cases_dir / "gold.jsonl")}
    cases = [c for c in dataset.read_jsonl(cases_dir / "cases.jsonl") if gold[c["id"]]["label"] != NEUTRAL]
    if cfg.limit:
        cases = cases[:: max(1, len(cases) // cfg.limit)][: cfg.limit]
    passages, model = references(), NoModel()

    begin = time.perf_counter()
    for path in sorted({c["booklet"]["path"] for c in cases}):
        strategy.prepared.get(path)
    prepared = time.perf_counter() - begin
    scored, page_hits, chars = [], [], []
    begin = time.perf_counter()
    for case in cases:
        booklet = strategy.prepared.get(case["booklet"]["path"])
        hits = strategy.search(case, booklet, model)
        excerpts = strategy.excerpts(booklet, hits)
        evidence = strategy.evidence(booklet, hits, excerpts, [], case["vote"])
        g = gold[case["id"]]
        scored.append((g, found(evidence, passages[row(case["id"])])))
        page_hits.append(bool({e["page"] for e in evidence} & {e["page"] for e in g["evidence"]}))
        chars.append(sum(len(booklet.sentences[j].text) for run in excerpts for j in run))
    searched = time.perf_counter() - begin

    print(f"Retrieval of {cfg.strategy.name} on {len(cases)} gold entailment and contradiction cases, "
          f"{len({c['booklet']['path'] for c in cases})} booklets; settings: "
          f"{OmegaConf.to_container(cfg.strategy)}")
    print("Evidence found, as the challenge scores it (Hit@5)")
    print("\n".join(hit_rates(scored)))
    print(f"A gold page among the evidence pages: {sum(page_hits) / len(page_hits):.3f}")
    print(f"Excerpts for the model: {sum(chars) / len(chars):,.0f} characters per case on average")
    print(f"Time: {prepared:.1f} s to read and index the booklets (also the other editions as they are needed), "
          f"{searched:.1f} s to search; {model.asked} claims would have been translated by the model")


def main(argv: list[str]) -> None:
    if len(argv) >= 2 and argv[0] == "run":
        score_run(Path(argv[1]))
    elif argv and argv[0] == "retrieval":
        score_retrieval(argv[1:])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])

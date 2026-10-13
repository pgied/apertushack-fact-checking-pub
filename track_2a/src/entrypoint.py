"""The challenge's CLI, which the Docker image runs: decide every case of a JSONL file, one line each.

    python -m src.entrypoint --input /data/cases.jsonl --output /output/predictions.jsonl [overrides]

It follows the contract of the challenge's Solution API Guide, https://hackapertus.notion.site/ef5b4fec112a834da43101ede56300f5:

- The input can mix both tasks: a case with a `booklet` is the advanced task, one with a `reference`
  the beginner task. Booklet paths are relative to the folder of the input, /data in the container.
- Every case gets one line in the output, in the order the cases finish. A case that fails, even
  after the client's retries, is answered neutral: a missing line would count as wrong anyway.
- The model and the strategy are the ones we submit, SUBMITTED below. Hydra overrides after the
  arguments change them, e.g. model=apertus-8b; LLM_NAME in the environment sets the model name.
  The endpoint and its key come from BASE_URL and API_KEY, see conf/config.yaml.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src import dataset
from src.backend import load_dotenv, make_client
from src.catalog import Catalog
from src.llm import LLM
from src.nli import LABELS
from src.strategy import Strategy

ROOT = Path(__file__).resolve().parent.parent  # relative paths in the config start here

# What we submit. Change it here when another model or strategy has proven better in the measurements.
SUBMITTED = ["model=apertus-70b", "strategy=baseline"]

NEUTRAL = LABELS.index("neutral")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Decide claims against their voting booklet or reference text.")
    parser.add_argument("--input", type=Path, required=True, help="the cases, one JSON object per line")
    parser.add_argument("--output", type=Path, required=True, help="where to write the predictions, one per line")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model=apertus-8b")
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("--input and --output must be different files")

    load_dotenv(ROOT / ".env")
    catalog = Catalog()
    catalog.register()
    cfg = configure(args.overrides)
    cases = dataset.read_jsonl(args.input)
    by_task: dict[str, list[dict]] = {}
    for case in cases:
        by_task.setdefault(task_of(case), []).append(case)

    strategy_class = catalog.load(cfg.strategy.name)
    if missing := sorted(set(by_task) - set(strategy_class.tasks)):
        raise SystemExit(f"The strategy {cfg.strategy.name} does not do the {' and '.join(missing)} task")
    backend = cfg.backends[cfg.model.backend]
    client = make_client(backend)
    settings = OmegaConf.to_object(cfg.strategy)  # the strategy's own Settings class, with the overrides
    request = {"temperature": settings.temperature, "max_tokens": settings.max_tokens,
               **OmegaConf.to_container(backend.options)}

    def llm() -> LLM:  # every case gets its own, to count its tokens
        return LLM(client, cfg.model.name, request)

    # Once per task, for all its cases: read the booklets and set up the strategy. If that fails, the
    # task's cases are answered neutral, and the other task still runs.
    started = {}
    for task, task_cases in by_task.items():
        try:
            started[task], setup = strategy_class.start(settings, task, task_cases, args.input.parent, llm())
            print(f"{task}: {len(task_cases)} cases, set up in {setup['setup_time_ms']:,} ms, "
                  f"{setup['booklets']} booklets read", file=sys.stderr)
        except Exception as error:
            print(f"{task}: setting up failed, its {len(task_cases)} cases are answered neutral: {error!r}",
                  file=sys.stderr)
    print(f"Deciding {len(cases)} cases with the strategy {cfg.strategy.name} and {cfg.model.name}", file=sys.stderr)

    failed = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out, ThreadPoolExecutor(backend.workers) as pool:
        futures = [pool.submit(decide, started.get(task_of(case)), case, llm()) for case in cases]
        for future in as_completed(futures):
            prediction, ok = future.result()
            failed += not ok
            out.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            out.flush()
    print(f"Wrote {len(cases)} predictions to {args.output}"
          + (f", {failed} of them neutral because they failed" if failed else ""), file=sys.stderr)


def configure(overrides: list[str]) -> DictConfig:
    """The configuration in conf/ with what we submit, then the given overrides, then LLM_NAME."""
    overrides = [*SUBMITTED, *overrides]
    if os.environ.get("LLM_NAME"):  # the project template's variable for the model's name
        overrides.append(f"model.name='{os.environ['LLM_NAME']}'")
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        return compose(config_name="config", overrides=overrides)


def task_of(case: dict) -> str:
    """advanced for a case with a booklet, beginner for one with a reference text."""
    return "advanced" if "booklet" in case else "beginner"


def decide(strategy: Strategy | None, case: dict, llm: LLM) -> tuple[dict, bool]:
    """The prediction for the case, and whether it worked. Without a strategy, or if it fails, the
    answer is neutral without evidence."""
    begin = time.perf_counter()
    try:
        if strategy is None:
            raise RuntimeError("the strategy could not be set up for this task")
        return strategy.answer(case, llm), True
    except Exception as error:
        print(f"  {case['id']} failed, answered neutral: {error!r}", file=sys.stderr)
        return {
            "id": case["id"],
            "label": NEUTRAL,
            "label_name": LABELS[NEUTRAL],
            "evidence": [],
            "metrics": {
                "input_tokens": llm.input_tokens,
                "output_tokens": llm.output_tokens,
                "inference_time_ms": round((time.perf_counter() - begin) * 1000),
            },
        }, False


if __name__ == "__main__":
    main()

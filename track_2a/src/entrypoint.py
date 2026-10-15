"""The challenge's CLI, which the Docker image runs: decide every case of a JSONL file, one line each.

    python -m src.entrypoint --input /data/cases.jsonl --output /output/predictions.jsonl [overrides]

It follows the contract of the challenge's Solution API Guide, https://hackapertus.notion.site/ef5b4fec112a834da43101ede56300f5:

- The input can mix both tasks: a case with a `booklet` is the advanced task, one with a `reference`
  the beginner task. Booklet paths are relative to the folder of the input, /data in the container.
- Every case gets one line in the output, in the order the cases finish. A case that fails, even
  after the client's retries, is answered neutral: a missing line would count as wrong anyway.
- Each task has the model and the strategy we submit for it, SUBMITTED below. Hydra overrides after
  the arguments change them for both tasks, e.g. model=apertus-8b; LLM_NAME in the environment sets
  the model name. The endpoint and its key come from BASE_URL and API_KEY, see conf/config.yaml.
- At the end it prints the run's time without the model, which the challenge's time score counts
  (src/timing.py). The organisers measure their own; `metrics.inference_time_ms` is not scored.
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

from src import STARTED, dataset
from src.backend import load_dotenv, make_client
from src.catalog import Catalog
from src.llm import LLM
from src.nli import LABELS
from src.strategy import Strategy
from src.timing import Stopwatch, describe

ROOT = Path(__file__).resolve().parent.parent  # relative paths in the config start here

# What we submit for each task. Change it here when another model or strategy has proven better in the
# measurements: gied_strict_reasoned reads only the booklet the case supplies.
SUBMITTED = {
    "beginner": ["model=apertus-70b", "strategy=gied_beginner_scoped"],
    "advanced": ["model=apertus-70b", "strategy=gied_strict_reasoned"],
}

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
    cases = dataset.read_jsonl(args.input)
    by_task: dict[str, list[dict]] = {}
    for case in cases:
        by_task.setdefault(task_of(case), []).append(case)

    # Each task its own configuration, strategy and client. A strategy that does not do its task, or a
    # missing key, stops the run before the first case.
    configs = {task: configure(task, args.overrides) for task in by_task}
    classes = {task: catalog.load(cfg.strategy.name) for task, cfg in configs.items()}
    for task, strategy_class in classes.items():
        if task not in strategy_class.tasks:
            raise SystemExit(f"The strategy {configs[task].strategy.name} does not do the {task} task")
    backends = {task: cfg.backends[cfg.model.backend] for task, cfg in configs.items()}
    clients = {task: make_client(backend) for task, backend in backends.items()}
    settings = {task: OmegaConf.to_object(cfg.strategy) for task, cfg in configs.items()}  # each strategy's own Settings
    requests = {task: {"temperature": settings[task].temperature, "max_tokens": settings[task].max_tokens,
                       **OmegaConf.to_container(backends[task].options)} for task in configs}
    workers = max((backend.workers for backend in backends.values()), default=1)
    stopwatch = Stopwatch(STARTED)  # the run's time without the model, which the challenge's time score counts

    def llm(task: str) -> LLM:  # every case gets its own, to count its tokens
        return LLM(clients[task], configs[task].model.name, requests[task], stopwatch.in_flight)

    # Once per task, for all its cases: read the booklets and set up the strategy. If that fails, the
    # task's cases are answered neutral, and the other task still runs.
    stopwatch.lap("startup")  # Python's start, the imports, the configuration and reading the cases
    started = {}
    for task, task_cases in by_task.items():
        name = f"{task}: {len(task_cases)} cases with the strategy {configs[task].strategy.name} and {configs[task].model.name}"
        try:
            started[task], setup = classes[task].start(settings[task], task, task_cases, args.input.parent, llm(task))
            print(f"{name}, set up in {setup['setup_time_ms']:,} ms, {setup['booklets']} booklets read", file=sys.stderr)
        except Exception as error:
            print(f"{name}: setting up failed, they are answered neutral: {error!r}", file=sys.stderr)
    stopwatch.lap("setup")

    failed = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out, ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(decide, started.get(task_of(case)), case, llm(task_of(case))) for case in cases]
        for future in as_completed(futures):
            prediction, ok = future.result()
            failed += not ok
            out.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            out.flush()
    stopwatch.lap("cases")
    print(f"Wrote {len(cases)} predictions to {args.output}"
          + (f", {failed} of them neutral because they failed" if failed else ""), file=sys.stderr)
    print(f"Time without the model: {describe(stopwatch.summary(len(cases), workers))}", file=sys.stderr)


def configure(task: str, overrides: list[str]) -> DictConfig:
    """The configuration in conf/ with what we submit for the task, then the given overrides, then LLM_NAME."""
    overrides = [*SUBMITTED[task], f"task={task}", *overrides]
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

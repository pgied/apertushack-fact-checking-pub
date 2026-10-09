"""Prepare the data of both tasks, decide the claims with a strategy and Apertus, and score the predictions.

The configuration is in conf/ and composed by Hydra; anything can be overridden on the command line:

    uv run python -m src stage=prepare                          # the dataset into data/beginner/ and data/advanced/
    uv run python -m src limit=30                               # predict 30 cases with the default model and strategy
    uv run python -m src model=apertus-8b-local limit=30        # the same with a local Ollama model
    uv run python -m src task=advanced model=apertus-70b        # the advanced task: the claims against the booklet PDFs
    uv run python -m src strategy=baseline strategy.max_tokens=20   # a strategy from src/strategies/, one setting changed
    uv run python -m src stage=evaluate                         # score the latest run again
    uv run python -m src stage=models model=apertus-8b-local    # list the models of that backend

This is for developing. The challenge's CLI, which the Docker image runs, is src/entrypoint.py.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src import dataset, evaluate, strategies
from src.backend import load_dotenv, make_client

ROOT = Path(__file__).resolve().parent.parent  # relative paths in the config start here


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    stages = {"prepare": prepare, "predict": predict, "evaluate": score, "models": list_models}
    if cfg.stage not in stages:
        raise SystemExit(f"Unknown stage {cfg.stage!r}, use one of: {', '.join(stages)}")
    stages[cfg.stage](cfg)


def prepare(cfg: DictConfig) -> None:
    raw = ROOT / "data" / "raw" / dataset.FILENAME
    if not raw.exists():
        print(f"Downloading {dataset.DATASET} ({dataset.FILENAME}) ...")
        dataset.download(raw)
    out = ROOT / cfg.data_dirs.beginner
    n = dataset.convert(raw, out)
    print(f"Beginner task: wrote {n} cases to {out / 'cases.jsonl'} and their labels to {out / 'gold.jsonl'}")

    out = ROOT / cfg.data_dirs.advanced
    print(f"Advanced task: reading the voting booklets in {out / 'booklets'}, downloading the missing ones ...")
    found = dataset.convert_advanced(raw, out)
    print(f"Advanced task: wrote {n} cases to {out / 'cases.jsonl'} and their labels and evidence pages to "
          f"{out / 'gold.jsonl'}. {len(found)} reference texts found in the booklets: "
          f"{sum(found.values()) / len(found):.0%} of their text on average, at least {min(found.values()):.0%}")


def predict(cfg: DictConfig) -> None:
    if cfg.task not in strategies.TASKS:
        raise SystemExit(f"Unknown task {cfg.task!r}, use one of: {', '.join(strategies.TASKS)}")
    strategy_class = strategies.load(cfg.strategy.name)
    if cfg.task not in strategy_class.tasks:
        raise SystemExit(f"The strategy {cfg.strategy.name} does not do the {cfg.task} task, "
                         f"only: {', '.join(strategy_class.tasks)}")
    backend = cfg.backends[cfg.model.backend]
    run = ROOT / cfg.run_dir
    saved = run / "config.yaml"
    if saved.exists() and run_settings(OmegaConf.load(saved)) != run_settings(cfg):
        raise SystemExit(f"{run} was started with another task, model, strategy or limit: start a new run")
    client = make_client(backend)
    run.mkdir(parents=True, exist_ok=True)
    saved.write_text(OmegaConf.to_yaml(cfg, resolve=True))

    cases_file = ROOT / cfg.data_dirs[cfg.task] / "cases.jsonl"
    cases = dataset.read_jsonl(cases_file)
    if cfg.limit:  # the dataset is sorted by label and language, so take an even spread
        cases = cases[:: max(1, len(cases) // cfg.limit)][: cfg.limit]

    # Predictions already in the run folder are kept, so an interrupted run can be continued.
    predictions = run / "predictions.jsonl"
    done = {p["id"] for p in dataset.read_jsonl(predictions)} if predictions.exists() else set()
    todo = [case for case in cases if case["id"] not in done]
    print(f"{len(todo)} cases of the {cfg.task} task to decide with the strategy {cfg.strategy.name} "
          f"and {cfg.model.name} via {cfg.model.backend} into {run}")
    if not todo:
        write_scores(cfg, run)
        return

    settings = OmegaConf.to_object(cfg.strategy)  # the strategy's own Settings class, with the overrides
    request = {"temperature": settings.temperature, "max_tokens": settings.max_tokens,
               **OmegaConf.to_container(backend.options)}

    def llm() -> strategies.LLM:  # every case gets its own, to count its tokens
        return strategies.LLM(client, cfg.model.name, request)

    # Once per run, for all its cases: read the booklets and set up the strategy, measured apart from the cases
    strategy, setup = strategies.start(strategy_class, settings, cfg.task, cases, cases_file.parent, llm())
    (run / "setup.json").write_text(json.dumps(setup) + "\n")
    print(f"Set up in {setup['setup_time_ms']:,} ms, {setup['booklets']} booklets read")

    failed = 0
    with open(predictions, "a", encoding="utf-8") as out, ThreadPoolExecutor(backend.workers) as pool:
        futures = {pool.submit(strategies.answer, strategy, case, llm()): case["id"] for case in todo}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                prediction = future.result()
            except Exception as error:  # keep going: continuing the run retries this case
                failed += 1
                print(f"  {futures[future]} failed: {error}", file=sys.stderr)
                continue
            out.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            out.flush()
            if n % 50 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}")
    if failed:
        print(f"{failed} cases failed. Continue the run to retry them: run_dir={cfg.run_dir}")
    write_scores(cfg, run)


def run_settings(cfg: DictConfig) -> dict:
    """The settings that must not change while a run is continued, or its predictions would mix."""
    return OmegaConf.to_container(OmegaConf.masked_copy(cfg, ["task", "model", "strategy", "limit"]), resolve=True)


def score(cfg: DictConfig) -> None:
    """Score run_dir, or the latest run if run_dir does not exist (e.g. when it was not given)."""
    run = ROOT / cfg.run_dir
    if not run.exists():
        runs = list((ROOT / cfg.runs_dir).glob("*/predictions.jsonl"))
        if not runs:
            raise SystemExit(f"No runs in {cfg.runs_dir} yet")
        run = max(runs, key=lambda p: p.stat().st_mtime).parent
    write_scores(cfg, run)


def write_scores(cfg: DictConfig, run: Path) -> None:
    saved = OmegaConf.load(run / "config.yaml")
    task = saved.get("task", "beginner")  # runs from before the strategies did the beginner task with the baseline
    strategy = OmegaConf.select(saved, "strategy.name", default="baseline")
    gold = dataset.read_jsonl(ROOT / cfg.data_dirs[task] / "gold.jsonl")
    setup = run / "setup.json"
    report = evaluate.report(gold, dataset.read_jsonl(run / "predictions.jsonl"),
                             json.loads(setup.read_text()) if setup.exists() else None)
    report = f"{task.capitalize()} task, strategy {strategy}, model {saved.model.name}\n{report}"
    (run / "scores.txt").write_text(report + "\n")
    print(f"\n{run}\n{report}")


def list_models(cfg: DictConfig) -> None:
    for model in make_client(cfg.backends[cfg.model.backend]).models.list():
        print(model.id)


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    strategies.register()  # before main(), which composes the configuration: strategy=<name> needs them
    main()

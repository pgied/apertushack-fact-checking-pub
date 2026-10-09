"""Score predictions against the gold labels: macro-F1 first, then breakdowns, evidence and efficiency."""

import math
from collections import Counter

from src.nli import LABELS

NEUTRAL = LABELS.index("neutral")


def macro_f1(gold: list[int], pred: list[int]) -> float:
    """Mean of the per-label F1 scores, each label weighted equally."""
    scores = []
    for label in range(len(LABELS)):
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(scores) / len(scores)


def report(gold_records: list[dict], predictions: list[dict], setup: dict | None = None) -> str:
    """The scores of the predictions. `setup` is what setting up the run took, from its setup.json."""
    gold_by_id = {g["id"]: g for g in gold_records}
    preds = [p for p in predictions if p["id"] in gold_by_id]
    gold = [gold_by_id[p["id"]] for p in preds]
    if not preds:
        return "No predictions match the gold ids."

    y_true = [g["label"] for g in gold]
    y_pred = [p["label"] for p in preds]
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(preds)
    lines = [
        f"Scored {len(preds)} of {len(gold_records)} cases",
        f"Macro-F1  {macro_f1(y_true, y_pred):.3f}",
        f"Accuracy  {accuracy:.3f}",
        "",
        "Confusion matrix (rows: gold, columns: predicted)",
        " " * 15 + "".join(f"{name:>15}" for name in LABELS),
    ]
    counts = Counter(zip(y_true, y_pred))
    for t, name in enumerate(LABELS):
        lines.append(f"{name:<15}" + "".join(f"{counts[(t, p)]:>15}" for p in range(len(LABELS))))

    lines += ["", "Macro-F1 by group"]
    groups = {f"claim in {lang}": lambda g, lang=lang: g["claim_language"] == lang for lang in ("de", "fr", "it")}
    groups["same language"] = lambda g: g["claim_language"] == g["reference_language"]
    groups["cross-lingual"] = lambda g: g["claim_language"] != g["reference_language"]
    for name, in_group in groups.items():
        idx = [i for i, g in enumerate(gold) if in_group(g)]
        if idx:
            f1 = macro_f1([y_true[i] for i in idx], [y_pred[i] for i in idx])
            lines.append(f"  {name:<15} {f1:.3f}   ({len(idx)} cases)")

    lines += ["", "Evidence"]
    decided = [p for p in preds if p["label"] != NEUTRAL]  # entailment and contradiction need evidence
    missing = sum(not p["evidence"] for p in decided)
    lines.append(f"  missing         {missing} of {len(decided)} entailment and contradiction predictions have none")
    idx = [i for i, g in enumerate(gold) if g["label"] != NEUTRAL]
    if idx and all("evidence" in g for g in gold):  # the advanced task: the gold says on which pages the evidence is
        hits = sum(bool(pages(preds[i]) & pages(gold[i])) for i in idx)
        lines.append(f"  page hit        {hits / len(idx):.3f}   ({len(idx)} cases with gold entailment or "
                     "contradiction: a predicted page is one of the gold pages)")

    input_tokens = [p["metrics"]["input_tokens"] for p in preds]
    output_tokens = [p["metrics"]["output_tokens"] for p in preds]
    times = sorted(p["metrics"]["inference_time_ms"] for p in preds)
    p95 = times[math.ceil(0.95 * len(times)) - 1]
    lines += [
        "",
        "Efficiency",
        f"  input tokens    {sum(input_tokens):,} in total, {sum(input_tokens) / len(preds):,.0f} per case",
        f"  output tokens   {sum(output_tokens):,} in total",
        f"  inference time  {sum(times) / len(times):,.0f} ms mean, {p95:,} ms p95",
    ]
    if setup:  # done once for all the run's cases: reading the booklets, and the strategy's setup()
        ms, cases = setup["setup_time_ms"], max(1, setup["cases"])
        lines.append(f"  setup           {ms:,} ms once for {cases} cases, {ms / cases:,.0f} ms per case; "
                     f"{setup['input_tokens']:,} input and {setup['output_tokens']:,} output tokens")
    return "\n".join(lines)


def pages(record: dict) -> set:
    """The pages of a prediction's or a gold record's evidence."""
    return {passage["page"] for passage in record["evidence"]}

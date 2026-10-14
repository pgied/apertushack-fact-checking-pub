"""Prompt lab: the baseline's single request for the beginner task, with the prompt built from parts
that are chosen with settings, so prompt variants can be compared on the same cases:

    uv run python -m src strategy=prompt_lab limit=150                                   # the defaults
    uv run python -m src strategy=prompt_lab strategy.labels=negation limit=150          # other label definitions
    uv run python -m src strategy=prompt_lab strategy.reason=true limit=150              # think first, then answer
    uv run python -m src strategy=prompt_lab strategy.order=claim_first strategy.languages=true limit=150
    uv run python -m src strategy=prompt_lab strategy.reason=true strategy.restate=true strategy.reason_tokens=300 limit=150

To read what the model answered, give a file for its answers: strategy.answers=data/runs/<run>/answers.jsonl
(with run_dir=data/runs/<run> to keep them next to the predictions).

To try a new variant, add an entry to LABELS_TEXT or INTRO and pick it with its setting.
"""

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from src.llm import LLM
from src.strategies.baseline.strategy import NEUTRAL, read_label
from src.strategy import Strategy, Verdict

# What the model is told it does
INTRO = {
    "booklet": """You check claims against official Swiss voting booklets.
You get a REFERENCE text from a booklet and a CLAIM. They may be in different languages
(German, French or Italian): compare their meaning, not their wording.""",
    "factcheck": """You are a careful fact-checker. Decide whether a REFERENCE text from an official Swiss voting
booklet supports a CLAIM about the vote. The two may be in different languages (German, French or
Italian): compare their meaning, not their wording.""",
}

# How the three labels are defined
LABELS_TEXT = {
    # the baseline's wording
    "loose": """- entailment: the reference supports the claim
- contradiction: the reference says the opposite of the claim
- neutral: the reference neither supports nor contradicts the claim""",
    # the baseline's wording, neutral first, and neutral as the answer when in doubt
    "neutral_first": """- neutral: the reference neither supports nor contradicts the claim
- contradiction: the reference says the opposite of the claim
- entailment: the reference supports the claim
If there is no clear relation between the claim and the reference, use the "neutral" label.""",
    # neutral_first, and contradiction only with a sentence of the reference to show for it
    "quote": """- neutral: the reference neither supports nor contradicts the claim
- contradiction: the reference says the opposite of the claim, and you can quote the sentence that does
- entailment: the reference supports the claim
If the reference does not mention what the claim says, or is about another topic, the label is "neutral",
not "contradiction": a claim the reference does not support is not contradicted by it.""",
    # the models call too many neutral claims contradictions: say what neutral covers
    "negation": """- entailment: the reference states what the claim says
- contradiction: the reference clearly states the opposite of the claim, and you could point to the sentence
  that does
- neutral: everything else, in particular when the claim adds a detail, a number or a reason that the
  reference does not give, even if the claim sounds wrong
Something the reference does not mention is neutral, not a contradiction.""",
}

ANSWER_ONE_WORD = "Answer with exactly one word:"
ANSWER_REASONED = """First explain in at most three sentences what the reference says about the claim.
Then give the label on a last line of its own, as "Label: <word>", with one of these words:"""
# The model often gave the label first and explained it afterwards, so the explanation could not change it
ANSWER_EXPLAIN_FIRST = """Start your answer with "The reference says" and never with the label.
In at most three sentences, say what the reference says about the claim. For a contradiction, quote the
sentence of the reference that says the opposite.
Then give the label on a last line of its own, as "Label: <word>", with one of these words:"""
# The model reads German best: have it put the claim into the reference's language before comparing
ANSWER_RESTATED = """First write the claim in the language of the reference, on a line of its own, as "Claim: <claim>".
Then explain in at most three sentences what the reference says about it.
Then give the label on a last line of its own, as "Label: <word>", with one of these words:"""

LANGUAGES = {"de": "German", "fr": "French", "it": "Italian"}
ROOT = Path(__file__).resolve().parents[3]  # relative paths in the settings start here, as in the config


class PromptLab(Strategy):
    tasks = ("beginner",)

    @dataclass
    class Settings(Strategy.Settings):
        intro: str = "booklet"  # a key of INTRO
        labels: str = "loose"  # a key of LABELS_TEXT
        order: str = "reference_first"  # or claim_first
        languages: bool = False  # tell the model the languages of the claim and the reference
        reason: bool = False  # let the model explain before the label
        explain_first: bool = False  # with reason=true: the answer starts with the explanation, not the label
        restate: bool = False  # with reason=true: the model first writes the claim in the reference's language
        reason_tokens: int = 200  # max_tokens when reason=true: max_tokens is for the one-word answer
        answers: str = ""  # a JSONL file to append the model's raw answers to, e.g. data/runs/<run>/answers.jsonl

    def setup(self, cases: list[dict], llm: LLM) -> None:
        # A misspelled variant stops the run before the first request, not on every case
        if self.settings.intro not in INTRO:
            raise SystemExit(f"strategy.intro: use one of {', '.join(INTRO)}")
        if self.settings.labels not in LABELS_TEXT:
            raise SystemExit(f"strategy.labels: use one of {', '.join(LABELS_TEXT)}")
        if self.settings.order not in ("reference_first", "claim_first"):
            raise SystemExit("strategy.order: use reference_first or claim_first")
        for setting in ("restate", "explain_first"):
            if getattr(self.settings, setting) and not self.settings.reason:
                raise SystemExit(f"strategy.{setting}=true needs strategy.reason=true")
        if self.settings.restate and self.settings.explain_first:
            raise SystemExit("strategy.restate and strategy.explain_first are two answer formats: pick one")
        self.lock = threading.Lock()  # cases run in threads, and all of them append to the answers file

    def system_prompt(self) -> str:
        answer = ANSWER_ONE_WORD
        if self.settings.reason:
            answer = ANSWER_REASONED
            if self.settings.restate:
                answer = ANSWER_RESTATED
            elif self.settings.explain_first:
                answer = ANSWER_EXPLAIN_FIRST
        return f"{INTRO[self.settings.intro]}\n\n{answer}\n{LABELS_TEXT[self.settings.labels]}"

    def user_prompt(self, case: dict) -> str:
        reference, claim = case["reference"], case["claim"]
        parts = {"reference": f"REFERENCE:\n{reference['text']}", "claim": f"CLAIM:\n{claim['text']}"}
        if self.settings.languages:
            parts["reference"] = f"REFERENCE (in {LANGUAGES.get(reference['language'], reference['language'])}):\n{reference['text']}"
            parts["claim"] = f"CLAIM (in {LANGUAGES.get(claim['language'], claim['language'])}):\n{claim['text']}"
        first, second = ("claim", "reference") if self.settings.order == "claim_first" else ("reference", "claim")
        return f"{parts[first]}\n\n{parts[second]}\n\nAnswer:"

    def predict(self, case: dict, llm: LLM) -> Verdict:
        messages = [{"role": "system", "content": self.system_prompt()}, {"role": "user", "content": self.user_prompt(case)}]
        if self.settings.reason:
            raw = llm.chat(messages, max_tokens=self.settings.reason_tokens)
            # The label is on the last "Label:" line; the explanation before it may name other labels
            found = re.findall(r"label\s*:\s*(.+)", raw, re.IGNORECASE)
            answer = found[-1] if found else raw
        else:
            raw = answer = llm.chat(messages)
        label = read_label(case, answer)
        if self.settings.answers:
            with self.lock, open(ROOT / self.settings.answers, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": case["id"], "label": label, "answer": raw}, ensure_ascii=False) + "\n")
        # As in the baseline: the reference is the evidence, with no page
        return Verdict(label, [] if label == NEUTRAL else [{"page": None, "text": case["reference"]["text"]}])


"""only the pages of the claim's vote go to the model, and the evidence is the sentences it quotes, checked against the PDF"""

import json
import re
from dataclasses import dataclass

from src.nli import LABELS, parse_label
from src.strategies import LLM, Strategy, Verdict
from src.strategies._booklet import also_on, find_quote, vote_pages
from src.strategies.baseline import Baseline, read_label

NEUTRAL = LABELS.index("neutral")

LOOSE = """- entailment: the {text} supports the claim
- contradiction: the {text} says the opposite of the claim
- neutral: the {text} neither supports nor contradicts the claim"""

STRICT = """- entailment: the {text} states the claim, or it follows clearly from what the {text} states.
- contradiction: the {text} states something that cannot be true together with the claim. For example,
  the claim says the Federal Council recommends a yes, and the {text} says it recommends a no; or the
  claim gives a number, a date or an actor that the {text} gives differently.
- neutral: everything else. In particular, when the {text} does not mention what the claim says, the
  answer is neutral, not contradiction, even if the claim seems unlikely or goes beyond the {text}."""

NEGATION = """- entailment: the {text} states the claim, or it follows clearly from what the {text} states.
- contradiction: only when the {text} explicitly states the opposite of the claim, for example the claim
  says the Federal Council recommends a yes and the {text} says it recommends a no, or the claim says
  something will happen and the {text} says it will not. You must be able to point to the sentence.
- neutral: everything else. Most claims that are not true are neutral, not contradictions: a claim that
  adds a detail the {text} does not give (a number, a date, a deadline, a measure, who says or wants
  it), or that the {text} simply does not address, is neutral, even if it sounds wrong."""

DEFINITIONS = {"loose": LOOSE, "strict": STRICT, "negation": NEGATION}

BEGINNER_PROMPT = """You check claims against official Swiss voting booklets.
You get a REFERENCE text from a booklet and a CLAIM. They may be in different languages
(German, French or Italian): compare their meaning, not their wording. Only the reference counts,
not your own knowledge.

Answer with exactly one word:
{labels}"""

PROMPT = """You check claims against an official Swiss voting booklet.
You get the PAGES of the booklet that treat one vote, each marked [Page n], the VOTE, and a CLAIM.
The claim may be in another language than the booklet (German, French or Italian): compare their
meaning, not their wording. Only the booklet counts, not your own knowledge.

Decide:
{labels}

Answer with JSON only:
{"label": "entailment" | "neutral" | "contradiction",
 "quotes": [{"page": <page number>, "text": "<one sentence from the booklet>"}]}

For entailment and contradiction, give 1 to 3 quotes: the sentences that decide it, copied exactly
from the booklet in the booklet's language, never translated or shortened. Prefer the detailed part
of the booklet to its summary. For neutral, give no quotes."""


class VoteQuotes(Strategy):
    tasks = ("beginner", "advanced")

    @dataclass
    class Settings(Strategy.Settings):
        max_tokens: int = 400
        max_evidence: int = 5
        whole_booklet: bool = False
        labels: str = "negation"

    def setup(self, cases: list[dict], llm: LLM) -> None:
        self.selected = {}
        if self.task == "advanced":
            for case in cases:
                key = (case["booklet"]["path"], case["vote"])
                if key not in self.selected:
                    pages = self.pages(case)
                    self.selected[key] = (list(range(1, len(pages) + 1)) if self.settings.whole_booklet
                                          else vote_pages(pages, case["vote"]))

    def predict(self, case: dict, llm: LLM) -> Verdict:
        if self.settings.labels not in DEFINITIONS:
            raise ValueError(f"strategy.labels={self.settings.labels!r}: use one of {', '.join(DEFINITIONS)}")
        labels = DEFINITIONS[self.settings.labels]
        if self.task == "beginner":
            if self.settings.labels == "loose":
                return Baseline.check_reference(self, case, llm)
            return self.check_reference(case, llm, BEGINNER_PROMPT.replace("{labels}", labels.format(text="reference")))
        pages = self.pages(case)
        selected = self.selected[(case["booklet"]["path"], case["vote"])]
        context = "\n\n".join(f"[Page {n}]\n{pages[n - 1].strip()}" for n in selected)
        prompt = f"PAGES:\n{context}\n\nVOTE:\n{case['vote']}\n\nCLAIM:\n{case['claim']['text']}\n\nJSON:"
        system = PROMPT.replace("{labels}", labels.format(text="booklet"))
        answer = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        label, quotes = parse_answer(answer)
        if label is None:
            label = read_label(case, answer)
        if label == NEUTRAL:
            return Verdict(label)
        return Verdict(label, self.evidence(quotes, pages, selected))

    def check_reference(self, case: dict, llm: LLM, system: str) -> Verdict:
        reference = case["reference"]["text"]
        prompt = f"REFERENCE:\n{reference}\n\nCLAIM:\n{case['claim']['text']}\n\nAnswer:"
        answer = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": prompt}], max_tokens=10)
        label = read_label(case, answer)
        return Verdict(label, [] if label == NEUTRAL else [{"page": None, "text": reference}])

    def evidence(self, quotes: list[dict], pages: list[str], selected: list[int]) -> list[dict]:
        found, fallback = [], []
        for quote in quotes:
            page, text = quote.get("page"), str(quote.get("text") or "").strip()
            order = ([page] if page in selected else []) + [n for n in selected if n != page]
            hit = find_quote(text, pages, order) if text else None
            if hit:
                found.append({"page": hit[0], "text": hit[1]})
                found += [{"page": n, "text": hit[1]} for n in also_on(hit[1], pages, selected, skip=hit[0])]
            elif text:
                fallback.append({"page": page if page in selected else None, "text": text[:5000]})
        unique, seen = [], set()
        for item in found + fallback:
            if (item["page"], item["text"]) not in seen:
                seen.add((item["page"], item["text"]))
                unique.append(item)
        return unique[: self.settings.max_evidence]


def parse_answer(answer: str) -> tuple[int | None, list[dict]]:
    match = re.search(r"\{.*\}", answer, re.DOTALL)
    if not match:
        return parse_label(answer), []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return parse_label(answer), []
    if not isinstance(data, dict):
        return parse_label(answer), []
    quotes = []
    for quote in data.get("quotes") or []:
        if isinstance(quote, str):
            quote = {"page": None, "text": quote}
        if isinstance(quote, dict):
            try:
                page = int(float(quote.get("page")))
            except (TypeError, ValueError):
                page = None
            quotes.append({"page": page, "text": quote.get("text")})
    return parse_label(str(data.get("label", ""))), quotes

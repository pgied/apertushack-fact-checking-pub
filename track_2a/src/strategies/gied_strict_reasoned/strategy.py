"""gied_strict_reasoned: gied_strict_multi_lang_scoped, deciding as the beginner task does.

The search, the excerpts and the evidence are gied_strict_multi_lang_scoped's: only the booklet the case
supplies, a claim in another language searched with its translation, within the claim's vote. Only the
decision changes. gied_strict_multi_lang_scoped answers with JSON only, and its neutral includes a claim
with a detail the booklet does not give, such as a number or a date. Its largest error, on all 1,488
advanced cases, was 42 contradictions answered neutral, often with the opposite in the excerpts, and the
model sometimes gave the label first and explained the contradiction after it. Here, as in
gied_beginner_scoped:

- prompt_lab's `quote` definitions: neutral first, contradiction only with a sentence to quote, and a claim
  the booklet does not mention, or about another topic, is neutral;
- Apertus first explains in at most three sentences what the excerpts say about the claim, then gives the
  label and the numbers of the excerpts that decide it, each on a line of its own;
- only those two lines are read: the reasoning before them may name other labels and numbers.

Deciding the same excerpts this way (15 Oct) got 37 of the 42 missed contradictions and 12 of the 19
neutral claims answered contradiction right, and 59 of 60 cases that were right stayed right.
"""

import re
from dataclasses import dataclass

from src.strategies.baseline.strategy import read_label
from src.strategies.gied_strict_multi_lang_scoped.strategy import GiedStrictMultiLangScoped

SYSTEM_PROMPT = """You check claims against an official Swiss voting booklet.
You get numbered EXCERPTS of the booklet, the VOTE that the claim is about, and a CLAIM. The claim may
be in another language than the booklet (German, French or Italian): compare their meaning, not their
wording. Only the booklet counts, not your own knowledge.

First explain in at most three sentences what the excerpts say about the claim. For a contradiction, quote the
sentence of the booklet that says the opposite.
Then give the label on a line of its own, as "Label: <word>", and on a last line the numbers of the excerpts
that decide it, as "Excerpts: <numbers>". The label is one of these words:
- neutral: the booklet neither supports nor contradicts the claim
- contradiction: the booklet says the opposite of the claim, and you can quote the sentence that does
- entailment: the booklet supports the claim
If the booklet does not mention what the claim says, or is about another topic, the label is "neutral", not
"contradiction": a claim the booklet does not support is not contradicted by it."""


class GiedStrictReasoned(GiedStrictMultiLangScoped):
    @dataclass
    class Settings(GiedStrictMultiLangScoped.Settings):
        max_tokens: int = 250  # the reasoning, the label and the excerpts' numbers

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = SYSTEM_PROMPT

    def read_answer(self, case: dict, answer: str) -> tuple[int, list[int]]:
        """The label on the last "Label:" line, and the excerpts on the last "Excerpts:" line."""
        labels = re.findall(r"label\s*:\s*(.+)", answer, re.IGNORECASE)
        cited = re.findall(r"excerpts?\s*:\s*(.+)", answer, re.IGNORECASE)
        numbers = [int(n) for n in re.findall(r"\d+", cited[-1])] if cited else []
        return read_label(case, labels[-1] if labels else answer), numbers

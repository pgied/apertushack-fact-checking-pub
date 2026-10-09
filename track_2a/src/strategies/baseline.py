"""The baseline, and the template for new strategies: the whole context and the claim in one request.

- Beginner task: the context is the reference text, and it is also the evidence. The beginner task
  has no PDF, so the evidence has no page.
- Advanced task: the context is the whole booklet, page by page. This is full-document prompting,
  the challenge's baseline for the advanced task. The model answers with the label and the page it
  based it on, and the text of that page is the evidence.

To start a strategy of your own, copy this file to src/strategies/<name>.py, rename the class and
run it with strategy=<name>.
"""

import re
import sys

from src.nli import LABELS, parse_label
from src.strategies import LLM, Strategy, Verdict

NEUTRAL = LABELS.index("neutral")

SYSTEM_PROMPT = """You check claims against official Swiss voting booklets.
You get a REFERENCE text from a booklet and a CLAIM. They may be in different languages
(German, French or Italian): compare their meaning, not their wording.

Answer with exactly one word:
- entailment: the reference supports the claim
- contradiction: the reference says the opposite of the claim
- neutral: the reference neither supports nor contradicts the claim"""

BOOKLET_PROMPT = """You check claims against official Swiss voting booklets.
You get a whole voting BOOKLET, page by page, the VOTE that the claim is about, and a CLAIM. The claim
may be in another language than the booklet (German, French or Italian): compare their meaning, not
their wording.

Answer with one word and the number of the page your answer is based on, for example "contradiction 12":
- entailment: the booklet supports the claim
- contradiction: the booklet says the opposite of the claim
- neutral: the booklet neither supports nor contradicts the claim"""


class Baseline(Strategy):
    tasks = ("beginner", "advanced")

    def predict(self, case: dict, llm: LLM) -> Verdict:
        if self.task == "beginner":
            return self.check_reference(case, llm)
        return self.check_booklet(case, llm)

    def check_reference(self, case: dict, llm: LLM) -> Verdict:
        reference = case["reference"]["text"]
        prompt = f"REFERENCE:\n{reference}\n\nCLAIM:\n{case['claim']['text']}\n\nAnswer:"
        answer = llm.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        label = read_label(case, answer)
        # The reference is the evidence passage, so it is returned as is. There is no PDF, hence no page.
        return Verdict(label, [] if label == NEUTRAL else [{"page": None, "text": reference}])

    def check_booklet(self, case: dict, llm: LLM) -> Verdict:
        pages = self.pages(case)
        booklet = "\n\n".join(f"[Page {n}]\n{text}" for n, text in enumerate(pages, 1))
        # The booklet comes first, so the prompts of all cases on a booklet start the same: a server can cache that
        prompt = f"BOOKLET:\n{booklet}\n\nVOTE:\n{case['vote']}\n\nCLAIM:\n{case['claim']['text']}\n\nAnswer:"
        answer = llm.chat([{"role": "system", "content": BOOKLET_PROMPT}, {"role": "user", "content": prompt}])
        label = read_label(case, answer)
        number = re.search(r"\d+", answer)
        page = int(number.group()) if number else 0
        if label == NEUTRAL or not 1 <= page <= len(pages):  # neutral needs no evidence, and a page not in the booklet gives none
            return Verdict(label)
        return Verdict(label, [{"page": page, "text": pages[page - 1].strip()}])


def read_label(case: dict, answer: str) -> int:
    """The label in the model's answer. An answer without one counts as neutral, and is logged."""
    label = parse_label(answer)
    if label is None:
        hint = " (empty: a thinking model? pick one that does not think)" if not answer.strip() else ""
        print(f"  {case['id']}: unclear answer {answer!r}{hint}, counted as neutral", file=sys.stderr)
        label = NEUTRAL
    return label

"""gied_beginner_scoped: gied_beginner_multi_lang that shows only the part of the reference around the
claim, to spend fewer tokens.

A reference is a whole booklet section: 3,700 characters in the median, 7,100 on average, up to
25,000; it makes most of the ~2,000 input tokens per case, and tokens count for 15 % of the task's
score. As gied_multi_lang_scoped keeps to the claim's vote, this keeps to the sentences that match the
claim best (BM25, with the claim in the reference's language), with the sentence before and after each,
about `context_chars` characters in the reference's order; "[…]" marks what is left out. A shorter
reference is shown whole.
"""

from dataclasses import dataclass

from src.llm import LLM
from src.strategies._gied.index import Booklet
from src.strategies._gied.retrieve import translate
from src.strategies.gied_beginner_multi_lang.strategy import GiedBeginnerMultiLang
from src.strategy import Verdict


class GiedBeginnerScoped(GiedBeginnerMultiLang):
    @dataclass
    class Settings(GiedBeginnerMultiLang.Settings):
        context_chars: int = 2000  # about how much of the reference is shown
        hits: int = 6  # sentences found, around which the part shown is cut

    def predict(self, case: dict, llm: LLM) -> Verdict:
        reference, claim = case["reference"], case["claim"]
        query = claim["text"]
        if claim["language"] != reference["language"]:
            query = translate(claim["text"], reference["language"], llm)
            case = {**case, "translation": query}
        if len(reference["text"]) > self.settings.context_chars:
            text = focus(reference["text"], query, self.settings.context_chars, self.settings.hits)
            case = {**case, "reference": {**reference, "text": text}}
        return super().predict(case, llm)  # the claim is translated already, so no second request


def focus(text: str, query: str, chars: int, hits: int) -> str:
    """The sentences of the text around the `hits` that match the query best, in the text's order, up to
    about `chars` characters; "[…]" marks what is left out. The whole text if nothing matches."""
    booklet = Booklet([(1, 0, paragraph) for paragraph in text.splitlines() if paragraph.strip()], 1)
    shown, size = set(), 0
    for sentence in booklet.search(query, hits):
        if size >= chars:
            break
        for near in booklet.around(sentence):
            if near not in shown:
                shown.add(near)
                size += len(near.text)
    if not shown:
        return text
    parts, previous = [], -1  # previous: the position of the last sentence shown
    for i, sentence in enumerate(booklet.sentences):
        if sentence in shown:
            if i != previous + 1:  # something left out before it, also at the start
                parts.append("[…]")
            parts.append(sentence.text)
            previous = i
    if previous != len(booklet.sentences) - 1:  # and at the end
        parts.append("[…]")
    return " ".join(parts)

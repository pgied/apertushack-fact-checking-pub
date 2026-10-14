"""gied_retrieval_improvement: gied_tags_prompt_adapted that searches further only when it found nothing.

Apertus first sees the excerpts around the best `hits` sentences, as in gied_tags_prompt_adapted. An
entailment or contradiction there is the answer: the evidence is found, so there is no further search.
A neutral answer may only mean that the deciding passage was not among them, so Apertus gets the
excerpts around the next-best sentences, up to `rounds` rounds in all, and the claim is neutral only
if no round decides it. On all 1,488 advanced cases, gied_tags_prompt_adapted's misses (70B) were
mostly neutral answers: 62 contradictions and 14 entailments.
"""

from dataclasses import dataclass

from src.llm import LLM
from src.nli import LABELS
from src.strategies._gied.index import Sentence
from src.strategies._gied.retrieve import cited, evidence, numbered
from src.strategies.gied_tags_prompt_adapted.strategy import GiedTagsPromptAdapted
from src.strategy import Verdict

NEUTRAL = LABELS.index("neutral")


class GiedRetrievalImprovement(GiedTagsPromptAdapted):
    @dataclass
    class Settings(GiedTagsPromptAdapted.Settings):
        rounds: int = 3  # rounds of excerpts at most: the second and third only after a neutral answer

    def check_booklet(self, case: dict, llm: LLM) -> Verdict:
        booklet, hits = self.booklet(case["booklet"]["path"]), self.settings.hits
        found = self.find(case, booklet, llm, hits * self.settings.rounds)
        shown: set[Sentence] = set()  # the sentences of the excerpts of earlier rounds
        for step in range(self.settings.rounds):
            excerpts = self.excerpts(booklet, found[step * hits:(step + 1) * hits], shown)
            if not excerpts:
                break
            label, numbers = self.ask(case, numbered(excerpts), llm)
            if label != NEUTRAL:
                return Verdict(label, evidence(found[:(step + 1) * hits], cited(excerpts, numbers), case["vote"]))
            shown.update(sentence for excerpt in excerpts for sentence in excerpt)
        return Verdict(NEUTRAL, evidence(found[:hits], [], case["vote"]))

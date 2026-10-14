"""gied_multi_lang_paraphrase: gied_paraphrase and gied_multi_lang_evidence together.

On 150 advanced cases with 70B, the paraphrases gave the best macro-F1 (0.942) and the two editions
the best evidence (Hit@5 0.979; 0.983 for claims in another language than the booklet). Here the five
paraphrases, written in the booklet's language, take the place of gied_multi_lang_evidence's single
translation, so there is no translation request:

1. Apertus writes five paraphrases of the claim in the booklet's language.
2. The booklet is searched with each of them, and with the claim itself if it is in the booklet's
   language; reciprocal rank fusion merges the rankings.
3. A claim in another language is also searched as it is in the booklet's edition in its language,
   and Apertus sees the excerpts of both editions.
4. Apertus decides once, on the original claim. The evidence is the booklet's sentences that both
   searches found, those of the cited excerpts first.
"""

from dataclasses import dataclass

from src.llm import LLM
from src.strategies._gied.index import fuse
from src.strategies._gied.retrieve import cited, evidence, numbered
from src.strategies.gied_multi_lang_evidence.strategy import SYSTEM_PROMPT, in_booklet, two_editions
from src.strategies.gied_paraphrase.strategy import paraphrases
from src.strategies.gied_tags_prompt_adapted.strategy import GiedTagsPromptAdapted
from src.strategy import Verdict


class GiedMultiLangParaphrase(GiedTagsPromptAdapted):
    system_prompt = SYSTEM_PROMPT

    @dataclass
    class Settings(GiedTagsPromptAdapted.Settings):
        versions: int = 5  # paraphrases of the claim to search the booklet with

    def check_booklet(self, case: dict, llm: LLM) -> Verdict:
        booklet, hits, claim = self.booklet(case["booklet"]["path"]), self.settings.hits, case["claim"]["text"]
        rankings = [booklet.search(version, hits) for version in paraphrases(case, llm, self.settings.versions)]
        if case["claim"]["language"] == case["booklet"]["language"]:
            rankings.append(booklet.search(claim, hits))
        here = fuse(rankings)[:hits]
        own = self.excerpts(booklet, here)

        edition = self.edition(case)
        if edition is None:  # a claim in the booklet's language, or no edition in the claim's
            label, numbers = self.ask(case, numbered(own), llm)
            return Verdict(label, evidence(here, cited(own, numbers), case["vote"]))

        there = edition.search(claim, hits)
        other = self.excerpts(edition, there)
        label, numbers = self.ask(case, two_editions(own, other, case["claim"]["language"]), llm)
        found = fuse([here, in_booklet(booklet, edition, there)])
        excerpts = own + [in_booklet(booklet, edition, excerpt) for excerpt in other]
        return Verdict(label, evidence(found, cited(excerpts, numbers), case["vote"]))

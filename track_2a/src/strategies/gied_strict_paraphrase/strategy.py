"""gied_strict_paraphrase: gied_strict_multi_lang_scoped that also searches with five paraphrases of the claim.

gied_paraphrase searched the booklet with five plain paraphrases of the claim, written by Apertus in the
booklet's language (after Hoyle et al., EMNLP 2023), besides the claim itself, which it searched in the
booklet's edition in the claim's language. Here the claim is searched as in gied_strict_multi_lang_scoped,
itself or its translation, and so are the five paraphrases, all within the pages of the claim's vote;
reciprocal rank fusion merges the rankings. Apertus decides once, on the claim and its translation.

That is one more request than gied_strict_multi_lang_scoped: the translation of a claim in another
language, the paraphrases, the decision. The booklet's other editions are never read.
"""

from dataclasses import dataclass

from src.llm import LLM
from src.strategies._gied.index import Booklet, Sentence, fuse
from src.strategies.gied_paraphrase.strategy import paraphrases
from src.strategies.gied_strict_multi_lang_scoped.strategy import GiedStrictMultiLangScoped


class GiedStrictParaphrase(GiedStrictMultiLangScoped):
    @dataclass
    class Settings(GiedStrictMultiLangScoped.Settings):
        versions: int = 5  # paraphrases of the claim to search with

    def find(self, case: dict, booklet: Booklet, llm: LLM, k: int = 0) -> list[Sentence]:
        k, pages = k or self.settings.hits, self.scope(case)
        rankings = [super().find(case, booklet, llm, k)]  # the claim itself, or its translation
        versions = paraphrases(case["claim"]["text"], case["booklet"]["language"], llm, self.settings.versions)
        rankings += [booklet.search(version, k, pages) for version in versions]
        return fuse(rankings)[:k]

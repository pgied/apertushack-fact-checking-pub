"""gied_multi_lang_evidence: gied_tags_prompt_adapted that shows Apertus the booklet in two languages.

A claim in another language than the booklet gets two searches: the booklet, with the claim translated
into the booklet's language by Apertus, and the booklet's edition in the claim's language, with the
claim as it is. Apertus sees the excerpts of both and decides. The evidence stays in the booklet's
language: a cited excerpt of the other edition counts with the sentences at the same place in the booklet.

A claim in the booklet's language, or without the edition in its language, is decided as in
gied_tags_prompt_adapted.
"""

from src.llm import LLM
from src.strategies._gied.index import Booklet, Sentence, fuse
from src.strategies._gied.retrieve import LANGUAGES, cited, evidence, numbered
from src.strategies.gied_tags_prompt_adapted.strategy import SYSTEM_PROMPT as ADAPTED_PROMPT
from src.strategies.gied_tags_prompt_adapted.strategy import GiedTagsPromptAdapted
from src.strategy import Verdict

SYSTEM_PROMPT = ADAPTED_PROMPT.replace("Only the booklet counts", """If the excerpts have a part IN THE <LANGUAGE> EDITION, those come from the booklet's edition in the
claim's language: they say the same as the booklet at the same place.
Only the booklet counts""")


class GiedMultiLangEvidence(GiedTagsPromptAdapted):
    system_prompt = SYSTEM_PROMPT

    def check_booklet(self, case: dict, llm: LLM) -> Verdict:
        edition = self.edition(case)
        if edition is None:
            return super().check_booklet(case, llm)
        booklet, hits, pages = self.booklet(case["booklet"]["path"]), self.settings.hits, self.scope(case)
        here = booklet.search(self.translate(case, llm), hits, pages)  # the booklet, with the translated claim
        there = edition.search(case["claim"]["text"], hits, pages)  # the edition in the claim's language
        own, other = self.excerpts(booklet, here), self.excerpts(edition, there)
        label, numbers = self.ask(case, two_editions(own, other, case["claim"]["language"]), llm)
        found = fuse([here, in_booklet(booklet, edition, there)])
        excerpts = own + [in_booklet(booklet, edition, excerpt) for excerpt in other]
        return Verdict(label, evidence(found, cited(excerpts, numbers), case["vote"]))


def two_editions(own: list[list[Sentence]], other: list[list[Sentence]], claim_language: str) -> str:
    """The excerpts of the booklet, then those of its edition in the claim's language, numbered on."""
    language = LANGUAGES.get(claim_language, "claim's language").upper()
    return numbered(own) + f"\n\nIN THE {language} EDITION:\n" + numbered(other, start=len(own) + 1)


def in_booklet(booklet: Booklet, edition: Booklet, sentences: list[Sentence]) -> list[Sentence]:
    """Sentences of the other edition, swapped for the booklet's sentences at the same place."""
    swapped = (booklet.same_place(sentence, edition) for sentence in sentences)
    return [sentence for sentence in swapped if sentence]

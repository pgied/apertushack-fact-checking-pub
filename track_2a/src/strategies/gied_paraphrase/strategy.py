"""gied_paraphrase: gied_tags_prompt_adapted that also searches with five paraphrases of the claim.

After Hoyle et al., "Natural Language Decompositions of Implicit Content Enable Better Text
Representations" (EMNLP 2023): a text is matched better when it is also written out as several simpler
statements. Unlike the paper, the versions here are plain paraphrases, with no inferences the claim
does not make, since those would change what is checked.

Apertus writes the versions in the booklet's language, so they also bridge a claim in another language.
Each version is searched, and so is the claim itself, as in gied_tags_prompt_adapted; reciprocal rank
fusion merges the rankings. Apertus then decides once, on the original claim.
"""

import re
from dataclasses import dataclass

from src.llm import LLM
from src.strategies._gied.index import Booklet, Sentence, fuse
from src.strategies._gied.retrieve import LANGUAGES
from src.strategies.gied_tags_prompt_adapted.strategy import GiedTagsPromptAdapted

PARAPHRASE_PROMPT = """Rewrite this claim about a Swiss popular vote {versions} times in {language}. Each version says
the same in simpler words: add nothing, leave nothing out. Write one version per line, nothing else.

{claim}"""


class GiedParaphrase(GiedTagsPromptAdapted):
    @dataclass
    class Settings(GiedTagsPromptAdapted.Settings):
        versions: int = 5  # paraphrases of the claim to search with

    def find(self, case: dict, booklet: Booklet, llm: LLM, k: int = 0) -> list[Sentence]:
        k = k or self.settings.hits
        rankings = [super().find(case, booklet, llm, k)]  # the claim itself
        versions = paraphrases(case["claim"]["text"], case["booklet"]["language"], llm, self.settings.versions)
        rankings += [booklet.search(version, k) for version in versions]
        return fuse(rankings)[:k]


def paraphrases(claim: str, language: str, llm: LLM, versions: int) -> list[str]:
    """The claim in simpler words, `versions` times, in `language` (de, fr or it): one request."""
    prompt = PARAPHRASE_PROMPT.format(versions=versions, language=LANGUAGES.get(language, language), claim=claim)
    answer = llm.chat([{"role": "user", "content": prompt}], max_tokens=80 * versions)
    lines = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip() for line in answer.splitlines()]
    return [line for line in lines if line][:versions]

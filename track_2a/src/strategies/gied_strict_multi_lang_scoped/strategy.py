"""gied_strict_multi_lang_scoped: gied_multi_lang_scoped with only the booklet the case supplies.

A case gives one booklet, and the challenge says to judge against the supplied source.
gied_multi_lang_scoped also searched the booklet's edition in the claim's language, which is in the
cases' folder only when another case uses it. Here the translation alone bridges the languages:

1. A claim in another language than the booklet is translated by Apertus into the booklet's language
   (one short request), as before.
2. The booklet is searched with the translation, within the pages of the claim's vote.
3. Apertus sees the excerpts, the claim and its translation, as in gied_beginner_multi_lang, with
   gied_tags_prompt_adapted's prompt: there is no other edition to explain.

A claim in the booklet's language is decided as in gied_multi_lang_scoped. The booklet's other editions
are never read, whatever `pivot` says.
"""

from dataclasses import dataclass

from src.llm import LLM
from src.strategies._gied.index import Booklet, Sentence
from src.strategies._gied.retrieve import LANGUAGES
from src.strategies.gied_multi_lang_scoped.strategy import GiedMultiLangScoped
from src.strategy import Verdict

# What the prompts of gied_multi_lang_scoped say about the other edition, which is never shown here
EDITION = """If the excerpts have a part IN THE <LANGUAGE> EDITION, those come from the booklet's edition in the
claim's language: they say the same as the booklet at the same place.
"""


class GiedStrictMultiLangScoped(GiedMultiLangScoped):
    @dataclass
    class Settings(GiedMultiLangScoped.Settings):
        pivot: bool = False  # never the booklet's edition in the claim's language: edition() is always None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = self.system_prompt.replace(EDITION, "")

    def edition(self, case: dict) -> Booklet | None:
        return None  # only the booklet the case supplies

    def check_booklet(self, case: dict, llm: LLM) -> Verdict:
        if case["claim"]["language"] != case["booklet"]["language"]:
            case = {**case, "translation": self.translate(case, llm)}
        return super().check_booklet(case, llm)  # without an edition, Retrieve's: find, excerpts, ask

    def find(self, case: dict, booklet: Booklet, llm: LLM, k: int = 0) -> list[Sentence]:
        """The k sentences of the claim's vote that match the claim best (k = hits by default): the claim
        itself if it is in the booklet's language, else its translation."""
        query = case["claim"]["text"]
        if case["claim"]["language"] != case["booklet"]["language"]:
            query = case.get("translation") or self.translate(case, llm)
        return booklet.search(query, k or self.settings.hits, self.scope(case))

    def ask(self, case: dict, context: str, llm: LLM) -> tuple[int, list[int]]:
        """As before, with the translation shown under the claim."""
        if "translation" not in case:
            return super().ask(case, context, llm)
        language = LANGUAGES.get(case["booklet"]["language"], "the booklet's language").upper()
        claim = (f"{case['claim']['text']}\n\nTHE CLAIM IN {language} (a translation, to compare it with the "
                 f"excerpts):\n{case['translation']}")
        return super().ask({**case, "claim": {**case["claim"], "text": claim}}, context, llm)

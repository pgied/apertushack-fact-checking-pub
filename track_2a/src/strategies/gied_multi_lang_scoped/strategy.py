"""gied_multi_lang_scoped: gied_multi_lang_evidence with two changes that its 70 errors on all 1,488
advanced cases (14 Oct) called for, each with its own setting so they can be measured apart.

- scope: search only the claim's vote. A booklet covers one to five votes, and the excerpts came from
  the whole booklet; claims about the voting text or the summary, which read alike for every vote, went
  wrong most often (10 % and 8 %), since the model got another vote's voting text. The vote's pages come
  from _booklet.vote_pages, which reads the running headers ("Erste Vorlage: ..."). They hold all gold
  pages in 989 of 990 cases, and a third of the booklet on average.
- rules: three rules in the prompt. A changed detail (another number, age, date or recommendation) is a
  contradiction, not neutral. A claim about what one party says is checked against that party. A claim
  about another proposal is neutral.
"""

from dataclasses import dataclass

from src import pdf
from src.strategies._booklet import vote_pages
from src.strategies.gied_multi_lang_evidence.strategy import SYSTEM_PROMPT as MULTI_LANG_PROMPT
from src.strategies.gied_multi_lang_evidence.strategy import GiedMultiLangEvidence

RULES_PROMPT = """You check claims against an official Swiss voting booklet.
You get numbered EXCERPTS of the booklet, the VOTE that the claim is about, and a CLAIM. The claim may
be in another language than the booklet (German, French or Italian): compare their meaning, not their
wording. If the excerpts have a part IN THE <LANGUAGE> EDITION, those come from the booklet's edition in the
claim's language: they say the same as the booklet at the same place.
Only the booklet counts, not your own knowledge.

Decide:
- entailment: the booklet states the claim, or it follows clearly from what the booklet states.
- contradiction: the booklet states the opposite of the claim, or gives the same fact differently: another
  number, share, age, date, duration or amount, or a recommendation the other way. You must be able to
  point to the sentence.
- neutral: everything else. The booklet does not deal with what the claim says, or does not give the
  detail at all: a claim the booklet does not support is not contradicted by it. A claim about another
  proposal than the VOTE, or about events the booklet does not mention, is neutral, even if you know it
  to be false.

Who says what: a claim often reports what one party or part of the booklet says, such as the Federal
Council and Parliament, the committee, the voting text or the summary. Check it against what that party
or part says. If it says something different or the opposite, that is a contradiction, even if another
party says what the claim says.

Answer with JSON only:
{"label": "entailment" | "neutral" | "contradiction", "excerpts": [<the numbers of the excerpts that decide it>]}"""


class GiedMultiLangScoped(GiedMultiLangEvidence):
    @dataclass
    class Settings(GiedMultiLangEvidence.Settings):
        scope: bool = True  # search only the pages of the claim's vote
        # the prompt with the three rules, else gied_multi_lang_evidence's. Off by default: on all 1,488 cases they
        # traded contradictions answered neutral for neutrals answered contradiction (macro-F1 0.960, without 0.962)
        rules: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = RULES_PROMPT if self.settings.rules else MULTI_LANG_PROMPT

    def scope(self, case: dict) -> set[int] | None:
        """The pages of the claim's vote, found in the text that start() read; all pages without scope."""
        if not self.settings.scope:
            return None
        path = case["booklet"]["path"]
        pages = self.booklets.get(path)
        if pages is None:  # start() did not read it, as in tests
            with self.reading:
                pages = pdf.pages(self.cases_dir / path)
        return set(vote_pages(pages, case["vote"]))

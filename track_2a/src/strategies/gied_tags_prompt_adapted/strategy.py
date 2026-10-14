"""gied_tags_prompt_adapted: gied_tags with the label definitions and the answer format of vote_quotes.

With gied_tags' prompt, Apertus 8B answered "contradiction" for 1,408 of 1,488 cases: it copies the
label of the example answer, "contradiction 2 1". This prompt takes from vote_quotes.py (labels=negation):

- what each label means: contradiction only when the booklet states the opposite of the claim,
  everything else that the booklet does not state is neutral;
- only the booklet counts, not the model's own knowledge;
- an answer in JSON that names all three labels, with no example label to copy.

Everything else is gied_tags, and the beginner task is the baseline's. How it works: _gied/retrieve.py.
"""

from dataclasses import dataclass

from src.strategies._gied.retrieve import Retrieve

SYSTEM_PROMPT = """You check claims against an official Swiss voting booklet.
You get numbered EXCERPTS of the booklet, the VOTE that the claim is about, and a CLAIM. The claim may
be in another language than the booklet (German, French or Italian): compare their meaning, not their
wording. Only the booklet counts, not your own knowledge.

Decide:
- entailment: the booklet states the claim, or it follows clearly from what the booklet states.
- contradiction: only when the booklet explicitly states the opposite of the claim, for example the claim
  says the Federal Council recommends a yes and the booklet says it recommends a no, or the claim says
  something will happen and the booklet says it will not. You must be able to point to the sentence.
- neutral: everything else. Most claims that are not true are neutral, not contradictions: a claim that
  adds a detail the booklet does not give (a number, a date, a deadline, a measure, who says or wants
  it), or that the booklet simply does not address, is neutral, even if it sounds wrong.

Answer with JSON only:
{"label": "entailment" | "neutral" | "contradiction", "excerpts": [<the numbers of the excerpts that decide it>]}"""


class GiedTagsPromptAdapted(Retrieve):
    system_prompt = SYSTEM_PROMPT

    @dataclass
    class Settings(Retrieve.Settings):
        source: str = "tags"
        max_tokens: int = 40  # the JSON answer is longer than "label n m"

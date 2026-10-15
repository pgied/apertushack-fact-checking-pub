"""gied_strict_reasoned_rules: gied_strict_reasoned with four lines in the prompt, one for each of its main errors.

gied_strict_reasoned made 39 errors on all 1,488 advanced cases (15 Oct). Decided again with the reasoning
logged, nearly all came from the decision, not the search:

- 18 neutral claims answered contradiction, mostly about another topic: the reasoning says that none of
  the excerpts mention the claim, and the label is contradiction anyway;
- 10 entailments answered contradiction, most of them claims about what the proposal would do, which the
  excerpts confirm, judged contradicted because the Federal Council and Parliament argue against it;
- about 3 claims about what one party says, judged by what is said, not by who says it;
- 2 claims with swapped numbers (a third and two thirds) read as matching.

The four lines: check a claim about what a party says against that party; arguments against a proposal do
not contradict what it would do; compare every number, share, date and duration; and, with the labels, if
the explanation says the booklet does not mention the claim, the label is neutral. Everything else, the
search, the excerpts, the evidence and how the answer is read, is gied_strict_reasoned's.
"""

from dataclasses import dataclass

from src.strategies.gied_strict_reasoned.strategy import GiedStrictReasoned

SYSTEM_PROMPT = """You check claims against an official Swiss voting booklet.
You get numbered EXCERPTS of the booklet, the VOTE that the claim is about, and a CLAIM. The claim may
be in another language than the booklet (German, French or Italian): compare their meaning, not their
wording. Only the booklet counts, not your own knowledge.

When you compare:
- A claim about what one party says, wants or recommends (the Federal Council, Parliament, a committee) is
  checked against that party's own statements.
- A claim about what the proposal would do is checked against what the booklet says the proposal does. That
  the Federal Council, Parliament or a committee argue against the proposal does not contradict what it
  would do.
- Compare every number, share, date and duration of the claim with the booklet's, one by one.

First explain in at most three sentences what the excerpts say about the claim. For a contradiction, quote the
sentence of the booklet that says the opposite.
Then give the label on a line of its own, as "Label: <word>", and on a last line the numbers of the excerpts
that decide it, as "Excerpts: <numbers>". The label is one of these words:
- neutral: the booklet neither supports nor contradicts the claim
- contradiction: the booklet says the opposite of the claim, and you can quote the sentence that does
- entailment: the booklet supports the claim
If the booklet does not mention what the claim says, or is about another topic, the label is "neutral", not
"contradiction": a claim the booklet does not support is not contradicted by it. If your explanation says
that the booklet does not mention what the claim says, the label is "neutral"."""


class GiedStrictReasonedRules(GiedStrictReasoned):
    @dataclass
    class Settings(GiedStrictReasoned.Settings):
        max_tokens: int = 300  # the reasoning may compare numbers one by one

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = SYSTEM_PROMPT

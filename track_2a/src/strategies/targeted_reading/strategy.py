"""Targeted reading: read the booklet the way a voter would, one section of the vote at a time.

Advanced task only. The code reads the table of contents at the front of the booklet and finds the
vote of the case in it by its title. Then the model gets that vote's sections one at a time, each
on its own, and answers with entailment or contradiction and a page if the section settles the
claim, or with neutral if it does not:

1. Argumente, Im Detail and In Kürze, always all three. The first of them to settle the claim
   decides the label; the others add their evidence if they agree.
2. Abstimmungstext, if none of them settled the claim. If it does not either, the label is neutral.

For entailment and contradiction, a follow-up question asks for the page and a verbatim quote. The
quote is the evidence if it is on one of the section's pages, followed by its whole page; otherwise
the whole page the model names is. The evidence of all sections that agree is kept, the quotes
first, since only the first five items are scored.

A booklet whose table of contents cannot be read, or a vote that is not in it, falls back to the
baseline: the whole booklet in one request.
"""

import re
import sys
from dataclasses import dataclass

from src.llm import LLM
from src.nli import LABELS, parse_label
from src.strategy import Strategy, Verdict
from src.strategies.baseline.strategy import Baseline

NEUTRAL = LABELS.index("neutral")

# The sections of a vote, with their names in the tables of contents.
# Variants within a language: "Im Detail" and "Vorlage im Detail", "Abstimmungstext" and "Abstimmungstexte" ...
SECTIONS = {
    "summary": r"In Kürze|En bref|In breve",
    "detail": r"(Vorlage )?im Detail|(L.objet )?en détail|In dettaglio",
    "arguments": r"Argumente( Initiativkomitt?ee)?|Arguments|Gli argomenti",
    "text": r"Abstimmungstexte?|Textes? soumis au vote|(Il |I )?test[oi] in votazione",
}
# Entries that belong to no vote, but end the section before them
OTHER = r"Übersicht Volksinitiativen|Vue d.ensemble des deux initiatives|Tabella riassuntiva delle iniziative( popolari)?|Débat au Parlement"
ENTRY = re.compile(
    rf"^(?:(?P<section>{'|'.join(f'(?P<{kind}>{names})' for kind, names in SECTIONS.items())})|{OTHER})"
    r"\s+(?P<start>\d{1,3})(?:\s*[–-]\s*(?P<end>\d{1,3}))?$",
    re.IGNORECASE,
)
ALWAYS = ("arguments", "detail", "summary")  # read in every case, in this order: Argumente holds the most gold passages
UNTIL_SETTLED = ("text",)  # then read one after the other until one settles the claim

SYSTEM_PROMPT = """You check claims against official Swiss voting booklets.
You get the pages of one section of the booklet on a VOTE, and a CLAIM about it. The claim may be in
another language than the booklet (German, French or Italian): compare their meaning, not their wording.

Answer with one word, and for entailment and contradiction the number of the page your answer is
based on, for example "contradiction 12":
- entailment: the pages support the claim
- contradiction: the pages say the opposite of the claim
- neutral: the pages neither support nor contradict the claim"""

FOLLOW_UP = """Quote the passage your answer is based on: copy it word for word from the page, in the
language of the booklet, without translating or shortening it. Answer with the page number, a colon
and the quote, for example: 12: <the passage>"""
MAX_EVIDENCE = 5000  # longer evidence items do not count
MIN_QUOTE = 30  # a shorter quote, in letters and digits, is too short to stand for a passage


class TargetedReading(Baseline):
    tasks = ("advanced",)

    @dataclass
    class Settings(Strategy.Settings):
        quote_tokens: int = 300  # max_tokens of the follow-up question, which answers with the page and the quote

    def setup(self, cases: list[dict], llm: LLM) -> None:
        # The sections of every vote of the run, by booklet and vote title: {kind: (first page, last page)}
        self.sections = {}
        for path in sorted({case["booklet"]["path"] for case in cases}):
            votes = table_of_contents(self.booklets[path])
            if not votes:
                print(f"  {path}: no table of contents found, its cases fall back to the baseline", file=sys.stderr)
            for title in sorted({case["vote"] for case in cases if case["booklet"]["path"] == path}):
                if found := find_vote(title, votes, self.booklets[path]):
                    self.sections[path, title] = found

    def predict(self, case: dict, llm: LLM) -> Verdict:
        sections = self.sections.get((case["booklet"]["path"], case["vote"]))
        if not sections:
            print(f"  {case['id']}: vote not in the table of contents, falls back to the baseline", file=sys.stderr)
            return self.check_booklet(case, llm)
        label, items = None, []
        for kind in ALWAYS:
            if kind in sections and (found := self.read(case, llm, sections[kind])):
                if label is None:
                    label = found[0]
                if found[0] == label:  # a section that disagrees with the one before adds nothing
                    items += [item for item in found[1] if item not in items]
        for kind in UNTIL_SETTLED:
            if label is not None:
                break
            if kind in sections and (found := self.read(case, llm, sections[kind])):
                label, items = found
        if label is None:
            return Verdict(NEUTRAL)
        quotes_first = sorted(items, key=lambda item: item["text"] == self.pages(case)[item["page"] - 1].strip())
        return Verdict(label, quotes_first[:5])  # only the first five are scored

    def read(self, case: dict, llm: LLM, section: tuple[int, int]) -> tuple[int, list[dict]] | None:
        """Show the model one section: its label and evidence if the section settles the claim, else None."""
        pages = self.pages(case)
        shown = [n for n in range(section[0], section[1] + 1) if 1 <= n <= len(pages)]
        if not shown:
            return None
        blocks = "\n".join(f"[Page {n}]\n{pages[n - 1]}" for n in shown)
        prompt = f"PAGES:\n{blocks}\n\nVOTE:\n{case['vote']}\n\nCLAIM:\n{case['claim']['text']}\n\nAnswer:"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        answer = llm.chat(messages)
        label = parse_label(answer)
        if label is None or label == NEUTRAL:  # neutral, or an answer without a label: not settled here
            return None
        follow_up = [{"role": "assistant", "content": answer}, {"role": "user", "content": FOLLOW_UP}]
        reply = llm.chat(messages + follow_up, max_tokens=self.settings.quote_tokens)
        # The page in the answer, else the one in the reply: the model often leaves it out of the answer
        page = next((n for n in (read_page(answer), read_page(reply)) if n in shown), 0)
        return label, evidence(reply, page, shown, pages)


def read_page(answer: str) -> int:
    """The first number in the model's answer, 0 without one."""
    number = re.search(r"\d+", answer)
    return int(number.group()) if number else 0


def evidence(reply: str, page: int, shown: list[int], pages: list[str]) -> list[dict]:
    """The quote in the reply, on the page where it is, followed by that whole page.

    The quote counts if it is on one of the pages shown, the named page first: pdfium breaks words
    across lines without a hyphen, so only letters and digits are compared. Without such a quote, the
    whole page named is the evidence, and without a page shown, there is none.
    """
    quote = reply.split(":", 1)[1] if re.match(r"\s*\D{0,10}\d+\s*:", reply) else reply
    quote = quote.strip().strip("\"'«»“”„ ").strip()
    found = None
    if len(squash(quote)) >= MIN_QUOTE and len(quote) <= MAX_EVIDENCE:
        found = next((n for n in sorted(shown, key=lambda n: n != page) if squash(quote) in squash(pages[n - 1])), None)
    page = found or page
    items = [{"page": found, "text": quote}] if found else []
    whole = {"page": page, "text": pages[page - 1].strip()} if page else None
    if whole and whole not in items:  # also a page that is too long to count: entailment and contradiction need evidence
        items.append(whole)
    return items


def squash(text: str) -> str:
    """Only the letters and digits, in lowercase."""
    return re.sub(r"\W+", "", text.lower())


def table_of_contents(pages: list[str]) -> list[dict]:
    """The votes in the booklet's table of contents, in its order: [{"titles": text, kind: (first, last)}].

    The table of contents is on the first pages, somewhere in pages 2 to 4, before the first section
    starts. It lists, for each vote,
    its title and then its sections with their first page, the summary with its first and last.
    A section ends before the next one in the booklet starts; the last one at the end of the booklet.
    """
    votes, starts, title = [], [], []
    for number in range(2, min(5, len(pages)) + 1):
        if starts and number >= min(starts):  # the booklet itself has begun: its first pages list their sections too
            break
        for line in pages[number - 1].splitlines():
            line = " ".join(line.replace("’", "'").split())
            if not (entry := ENTRY.match(line)):
                title.append(line)
                continue
            start = int(entry["start"])
            starts.append(start)
            if not entry["section"]:
                continue
            kind = next(kind for kind in SECTIONS if entry[kind])
            # A vote starts with a title, or with a section it already has
            if title or not votes or kind in votes[-1]:
                votes.append({"titles": " ".join(title)})
                title = []
            votes[-1][kind] = (start, int(entry["end"])) if entry["end"] else start
    starts = sorted(set(starts)) + [len(pages) + 1]
    for vote in votes:
        for kind in SECTIONS:
            if isinstance(vote.get(kind), int):
                vote[kind] = (vote[kind], next(start for start in starts if start > vote[kind]) - 1)
    # In reading order, and only votes with sections
    return [{kind: vote[kind] for kind in ["titles", *SECTIONS] if kind in vote} for vote in votes if len(vote) > 1]


def find_vote(title: str, votes: list[dict], pages: list[str]) -> dict | None:
    """The sections of the vote with this title: the vote whose title in the table of contents, and on
    the first page of its sections, has the most of the title's words. None without a good match."""
    words = set(re.findall(r"\w{3,}", title.lower()))
    best, best_share = None, 0.0
    for vote in votes:
        text = vote["titles"] + " " + " ".join(pages[vote[kind][0] - 1][:500] for kind in SECTIONS if kind in vote)
        share = len(words & set(re.findall(r"\w{3,}", text.lower()))) / max(len(words), 1)
        if share > best_share:
            best, best_share = vote, share
    if best is None or best_share < 0.5:
        return None
    return {kind: best[kind] for kind in SECTIONS if kind in best}

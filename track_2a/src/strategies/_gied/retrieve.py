"""Retrieve, then verify: the strategy that gied_pdfium, gied_tags and gied_docling share. They differ
only in how they read the booklet, the setting `source` (see pdf.py).

Advanced task, for each case:
1. Read the booklet and index its sentences, the first time a case needs it (Booklet, in index.py).
2. Find the sentences that bear on the claim. A claim in another language than the booklet is
   searched in the booklet's edition in the claim's language, which has the same pages, and each
   sentence found there is swapped for the one at the same place in the booklet. Without that
   edition, Apertus translates the claim first.
3. Ask Apertus, with each sentence found and its neighbours as a numbered excerpt, about
   `context_chars` characters in all. It answers with the label and the excerpts it is based on,
   for example "contradiction 2 1".
4. The evidence is five sentences of the booklet, those of the cited excerpts first. It is returned
   for a neutral answer too: the challenge scores the evidence of every case whose gold label is
   entailment or contradiction, whatever we answer.

Beginner task: the baseline's single request, unchanged.
"""

import re
import threading
from dataclasses import dataclass
from pathlib import Path

from src.llm import LLM
from src.strategies._gied.index import MIN_CHARS, Booklet, Sentence, terms
from src.strategies._gied.pdf import page_count, read
from src.strategies.baseline.strategy import Baseline, read_label
from src.strategy import Verdict

ROOT = Path(__file__).resolve().parents[3]  # track_2a/: relative paths in the settings start here
LANGUAGES = {"de": "German", "fr": "French", "it": "Italian"}
EVIDENCE = 5  # sentences: the challenge scores the first five evidence items

SYSTEM_PROMPT = """You check claims against an official Swiss voting booklet.
You get numbered EXCERPTS of the booklet, the VOTE that the claim is about, and a CLAIM. The claim may
be in another language than the booklet (German, French or Italian): compare their meaning, not their
wording.

Answer with one word and the numbers of the excerpts your answer is based on, for example "contradiction 2 1":
- entailment: the booklet supports the claim
- contradiction: the booklet says the opposite of the claim
- neutral: the booklet neither supports nor contradicts the claim"""

TRANSLATE_PROMPT = "Translate this claim into {language}. Answer with the translation only.\n\n{claim}"


class Retrieve(Baseline):
    tasks = ("beginner", "advanced")
    system_prompt = SYSTEM_PROMPT  # what Apertus is told in the advanced task; a strategy can change it

    @dataclass
    class Settings(Baseline.Settings):
        max_tokens: int = 12  # the label and a few excerpt numbers
        source: str = "tags"  # how the booklet is read: pdfium (the text layer), tags or docling
        converted: str = "data/converted/docling"  # where docling's JSON is, for source=docling
        pivot: bool = True  # search a cross-lingual claim in the booklet's edition in the claim's language
        hits: int = 8  # sentences found, around which the excerpts are cut
        context_chars: int = 2000  # about how much text the excerpts have in all

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_booklets: dict[str, Booklet] = {}  # path → booklet, read once
        self.reading = threading.Lock()  # one booklet at a time: PDFium is not thread-safe

    def booklet(self, path: str) -> Booklet:
        """The booklet at `path`, relative to the cases' folder, read the first time a case needs it."""
        with self.reading:
            if path not in self.read_booklets:
                file = self.cases_dir / path
                blocks = read(file, self.settings.source, ROOT / self.settings.converted)
                self.read_booklets[path] = Booklet(blocks, page_count(file))
            return self.read_booklets[path]

    def check_booklet(self, case: dict, llm: LLM) -> Verdict:
        booklet = self.booklet(case["booklet"]["path"])
        found = self.find(case, booklet, llm)
        excerpts = self.excerpts(booklet, found)
        label, numbers = self.ask(case, numbered(excerpts), llm)
        return Verdict(label, evidence(found, cited(excerpts, numbers), case["vote"]))

    def find(self, case: dict, booklet: Booklet, llm: LLM, k: int = 0) -> list[Sentence]:
        """The k sentences of the booklet that bear most on the claim, best first (k = hits by default).

        A claim in another language than the booklet is searched in the booklet's edition in its own
        language, and each sentence found there is swapped for the one at the same place in the
        booklet. Without that edition, Apertus translates the claim first."""
        k, claim, pages = k or self.settings.hits, case["claim"]["text"], self.scope(case)
        if case["claim"]["language"] == case["booklet"]["language"]:
            return booklet.search(claim, k, pages)
        if edition := self.edition(case):
            swapped = [booklet.same_place(sentence, edition) for sentence in edition.search(claim, k, pages)]
            return list(dict.fromkeys(sentence for sentence in swapped if sentence))
        return booklet.search(self.translate(case, llm), k, pages)

    def scope(self, case: dict) -> set[int] | None:
        """The pages of the booklet that are searched for the case: all of them (None). A strategy can
        narrow this down; the editions have the same pages, so it holds for them too."""
        return None

    def edition(self, case: dict) -> Booklet | None:
        """The booklet's edition in the claim's language, if the claim is in another language, `pivot` is
        on, and that edition is in the same folder with the same pages."""
        path, language = case["booklet"]["path"], case["claim"]["language"]
        other = re.sub(r"_[a-z]{2}\.pdf$", f"_{language}.pdf", path)
        if not self.settings.pivot or other == path or not (self.cases_dir / other).exists():
            return None
        edition = self.booklet(other)
        return edition if edition.pages == self.booklet(path).pages else None

    def translate(self, case: dict, llm: LLM) -> str:
        """The claim, translated by Apertus into the booklet's language."""
        language = LANGUAGES.get(case["booklet"]["language"], "the booklet's language")
        prompt = TRANSLATE_PROMPT.format(language=language, claim=case["claim"]["text"])
        return llm.chat([{"role": "user", "content": prompt}], max_tokens=200)

    def excerpts(self, booklet: Booklet, found: list[Sentence], shown=frozenset()) -> list[list[Sentence]]:
        """For each sentence found, best first: it and its neighbours on the page. A sentence is in one
        excerpt only, and in none if it is in `shown`; the excerpts stop at about context_chars characters."""
        excerpts, used, size = [], set(shown), 0
        for sentence in found:
            excerpt = [s for s in booklet.around(sentence) if s not in used]
            if not excerpt:
                continue
            length = sum(len(s.text) for s in excerpt)
            if excerpts and size + length > self.settings.context_chars:
                break
            excerpts.append(excerpt)
            used.update(excerpt)
            size += length
        return excerpts

    def ask(self, case: dict, context: str, llm: LLM) -> tuple[int, list[int]]:
        """Apertus' answer about the claim, given numbered excerpts: the label, and the numbers in the
        answer, which are the excerpts it cites."""
        prompt = f"VOTE:\n{case['vote']}\n\nEXCERPTS:\n{context}\n\nCLAIM:\n{case['claim']['text']}\n\nAnswer:"
        answer = llm.chat([{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}])
        return read_label(case, answer), [int(n) for n in re.findall(r"\d+", answer)]


def numbered(excerpts: list[list[Sentence]], start: int = 1) -> str:
    """The excerpts as the model reads them: "[1] (p. 12, In Kürze › Ausgangslage) text"."""
    lines = []
    for n, excerpt in enumerate(excerpts, start):
        where = f"p. {excerpt[0].page}" + (f", {excerpt[0].section}" if excerpt[0].section else "")
        lines.append(f"[{n}] ({where}) " + " ".join(sentence.text for sentence in excerpt))
    return "\n\n".join(lines) or "(none found)"


def cited(excerpts: list[list[Sentence]], numbers: list[int]) -> list[list[Sentence]]:
    """The excerpts with these numbers, counted from 1; other numbers in the answer are left out."""
    return [excerpts[n - 1] for n in numbers if 0 < n <= len(excerpts)]


def evidence(found: list[Sentence], cited: list[list[Sentence]], vote: str) -> list[dict]:
    """Five sentences as evidence, verbatim with their page: the sentences found that are in the cited
    excerpts, then the other sentences found, then the rest of the cited excerpts. Never a fragment,
    nor the vote's title, which every gold passage starts with."""
    in_cited = [sentence for excerpt in cited for sentence in excerpt]
    title = set(terms(vote))
    items: list[dict] = []
    for sentence in [s for s in found if s in in_cited] + found + in_cited:
        item = {"page": sentence.page, "text": sentence.text}
        words = terms(sentence.text)
        is_title = sum(word in title for word in words) >= 0.8 * len(words)
        if len(sentence.text) >= MIN_CHARS and not is_title and item not in items:
            items.append(item)
    return items[:EVIDENCE]

"""Retrieve, then verify: the strategy that gied_pdfium, gied_tags and gied_docling share; they differ
only in how they read the booklet (`source`, see pdf.py).

Advanced task:
1. The booklet is read and split into sentences, indexed with BM25, the first time a case needs it,
   while other cases' requests are in flight (Lazy, in index.py).
2. A claim in another language than the booklet is searched in the booklet's edition in the claim's
   language, from the same folder, and what is found there is mapped to the same places in the
   booklet. Without that edition, Apertus translates the claim first.
3. The best sentences, with their neighbours, make up the numbered excerpts that Apertus sees, about
   `context_chars` characters, each with its page and section.
4. Apertus answers with the label and the numbers of the excerpts it based it on. The evidence is up
   to five sentences of the booklet, those of the cited excerpts first. It is returned for a neutral
   answer too: the challenge scores the evidence of every case that is entailment or contradiction.

Beginner task: the baseline's single request, unchanged.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from src.nli import LABELS
from src.strategies import LLM, Verdict
from src.strategies._gied import timing
from src.strategies._gied.index import MIN_CHARS, Booklet, Lazy, terms
from src.strategies._gied.pdf import Block, read
from src.strategies.baseline.strategy import Baseline, read_label

NEUTRAL = LABELS.index("neutral")
LANGUAGES = {"de": "German", "fr": "French", "it": "Italian"}
ROOT = Path(__file__).resolve().parents[3]  # track_2a/: relative paths in the settings start here
MAX_QUOTE = 5000  # characters: longer evidence items do not count

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

    @dataclass
    class Settings(Baseline.Settings):
        max_tokens: int = 12  # the label and a few excerpt numbers
        source: str = "tags"  # how the booklet is read: pdfium, tags or docling
        converted: str = "data/converted/docling"  # docling's JSON of the booklets, by the PDF's SHA-256
        pivot: bool = True  # search a cross-lingual claim in the booklet's edition in the claim's language
        translate: bool = True  # without that edition, let Apertus translate the claim first
        hits: int = 8  # the best sentences, around which the excerpts are cut
        window: int = 1  # sentences before and after each one
        context_chars: int = 2000  # about how much text the excerpts have together
        evidence: int = 5  # sentences returned as evidence: the challenge scores the first five
        neutral_evidence: bool = True  # return evidence for a neutral answer too

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepared = Lazy(self.prepare)  # booklet path → its Booklet, made on first use

    def predict(self, case: dict, llm: LLM) -> Verdict:
        llm = timing.Timed(llm)
        if self.task == "beginner":
            return self.check_reference(case, llm)
        return self.check_booklet(case, llm)

    def check_booklet(self, case: dict, llm: LLM) -> Verdict:
        booklet = self.prepared.get(case["booklet"]["path"])
        hits = self.search(case, booklet, llm)
        excerpts = self.excerpts(booklet, hits)
        prompt = self.prompt(case, booklet, excerpts)
        answer = llm.chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        label = read_label(case, answer)
        cited = [int(n) - 1 for n in re.findall(r"\d+", answer) if 0 < int(n) <= len(excerpts)]
        if label == NEUTRAL and not self.settings.neutral_evidence:
            return Verdict(label)
        return Verdict(label, self.evidence(booklet, hits, excerpts, cited, case["vote"]))

    def prepare(self, path: str) -> Booklet:
        """Read the booklet at `path`, relative to the cases' folder, and index its sentences."""
        if self.cases_dir is not None and (self.cases_dir / path).exists():
            return Booklet(read(self.cases_dir / path, self.settings.source, ROOT / self.settings.converted))
        # No folder to read it from, as in tests: the pages that start() read, one block each
        return Booklet([[Block("text", " ".join(text.split()))] for text in self.booklets[path]])

    def edition(self, path: str, language: str) -> Booklet | None:
        """The booklet's edition in another language, if it is in the same folder."""
        other = re.sub(r"_[a-z]{2}\.pdf$", f"_{language}.pdf", path)
        if other == path or self.cases_dir is None or not (self.cases_dir / other).exists():
            return None
        return self.prepared.get(other)

    def search(self, case: dict, booklet: Booklet, llm) -> list[int]:
        """The sentences of the booklet that bear on the claim, best first."""
        claim, language = case["claim"]["text"], case["claim"]["language"]
        if language != case["booklet"]["language"]:
            edition = self.edition(case["booklet"]["path"], language) if self.settings.pivot else None
            if edition is not None and edition.pages == booklet.pages:
                found = (booklet.map(edition, i) for i in edition.top(edition.scores(claim), self.settings.hits))
                return list(dict.fromkeys(i for i in found if i is not None))
            if self.settings.translate:
                prompt = TRANSLATE_PROMPT.format(language=LANGUAGES.get(case["booklet"]["language"], "the "
                                                 "language of the booklet"), claim=claim)
                claim = llm.chat([{"role": "user", "content": prompt}], max_tokens=200)
        return booklet.top(booklet.scores(claim), self.settings.hits)

    def excerpts(self, booklet: Booklet, hits: list[int]) -> list[list[int]]:
        """Runs of sentences around the hits, within their block, best hit first; runs that touch are
        merged. They stop once they have about context_chars characters."""
        excerpts: list[list[int]] = []
        used: set[int] = set()
        chars = 0
        for i in hits:
            sentence = booklet.sentences[i]
            block = booklet.at[sentence.page, sentence.block]
            at = block.index(i)
            run = [j for j in block[max(0, at - self.settings.window):at + self.settings.window + 1] if j not in used]
            if not run:
                continue
            size = sum(len(booklet.sentences[j].text) for j in run)
            if excerpts and chars + size > self.settings.context_chars:
                break
            for excerpt in excerpts:
                if excerpt[0] in block and (min(run) == max(excerpt) + 1 or max(run) + 1 == min(excerpt)):
                    excerpt[:] = sorted(excerpt + run)
                    break
            else:
                excerpts.append(run)
            used.update(run)
            chars += size
        return excerpts

    def prompt(self, case: dict, booklet: Booklet, excerpts: list[list[int]]) -> str:
        parts = []
        for n, run in enumerate(excerpts, 1):
            first = booklet.sentences[run[0]]
            where = f"p. {first.page}" + (f", {first.section[-80:]}" if first.section else "")
            parts.append(f"[{n}] ({where}) " + " ".join(booklet.sentences[j].text for j in run))
        text = "\n\n".join(parts) or "(none found)"
        return f"VOTE:\n{case['vote']}\n\nEXCERPTS:\n{text}\n\nCLAIM:\n{case['claim']['text']}\n\nAnswer:"

    def evidence(self, booklet: Booklet, hits: list[int], excerpts: list[list[int]], cited: list[int],
                 vote: str) -> list[dict]:
        """Up to `evidence` sentences, verbatim with their page: the hits in the excerpts that the answer
        cites first, then the other hits, then the rest of the cited excerpts. Never the vote's title,
        which every reference starts with."""
        order = [i for n in cited for i in hits if i in excerpts[n]] + hits + [i for n in cited for i in excerpts[n]]
        title = set(terms(vote))
        items, seen = [], set()
        for i in order:
            sentence = booklet.sentences[i]
            words = terms(sentence.text)
            if (sentence.text in seen or len(sentence.text) < MIN_CHARS  # a fragment, such as a lone footnote
                    or sum(w in title for w in words) >= 0.8 * len(words)):
                continue
            seen.add(sentence.text)
            items.append({"page": sentence.page, "text": sentence.text[:MAX_QUOTE]})
            if len(items) == self.settings.evidence:
                break
        return items

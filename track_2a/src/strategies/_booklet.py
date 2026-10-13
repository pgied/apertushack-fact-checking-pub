"""Find the pages of a booklet that talk one vote, and find quotes in a booklet."""

import difflib
import math
import re
from collections import Counter
from functools import lru_cache

WORD = re.compile(r"\w+")
MARKER = re.compile(r"^\s*(?:\d+\s+)?(\w+)\s+(?:Vorlage|objet|oggetto)\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
FOLD = str.maketrans({"’": "'", "‘": "'", "«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "\u00a0": " ",
                      "\u00ad": "", "\ufffe": ""})


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text.translate(FOLD))]


def terms(text: str) -> set:
    ws = words(text)
    return set(ws) | set(zip(ws, ws[1:]))


@lru_cache(maxsize=256)
def segment(pages: tuple[str, ...]):
    owner: list[str | None] = []
    current = None
    for text in pages:
        names = {m.group(2).strip().lower() for m in MARKER.finditer(text.translate(FOLD))}
        if len(names) == 1:
            current = names.pop()
        owner.append(current)
    blocks: dict[str, list[int]] = {}
    for n, name in enumerate(owner, 1):
        if name is not None:
            blocks.setdefault(name, []).append(n)
    block_terms = {name: terms(" ".join(pages[n - 1] for n in ns)) for name, ns in blocks.items()}
    df = Counter(t for ts in block_terms.values() for t in ts)
    return blocks, block_terms, df


def vote_pages(pages: list[str], vote: str) -> list[int]:
    blocks, block_terms, df = segment(tuple(pages))
    if len(blocks) < 2:
        return list(range(1, len(pages) + 1))
    title = {t for t in terms(vote) if isinstance(t, tuple) or len(t) >= 3}
    score = {name: sum(math.log(1 + len(blocks) / df[t]) for t in title if t in ts) for name, ts in block_terms.items()}
    best = max(score, key=score.get)
    if score[best] == 0:
        return list(range(1, len(pages) + 1))
    return blocks[best]


def cut(page: str, spans: list[tuple[str, int, int]], i: int, j: int) -> str:
    return re.sub(r"\s+", " ", page[spans[i][1] : spans[j][2]]).strip()


def find_quote(quote: str, pages: list[str], candidates: list[int], min_ratio: float = 0.75):
    q = words(quote)
    if len(q) < 3:
        return None
    spans = {n: [(m.group().lower(), m.start(), m.end()) for m in WORD.finditer(pages[n - 1].translate(FOLD))]
             for n in candidates}

    for n in candidates:
        ws = [w for w, _, _ in spans[n]]
        for i in range(len(ws) - len(q) + 1):
            if ws[i] == q[0] and ws[i : i + len(q)] == q:
                return n, cut(pages[n - 1], spans[n], i, i + len(q) - 1)

    target = set(q)
    overlap = {n: len(target & {w for w, _, _ in spans[n]}) for n in candidates}
    best = (0.0, None, 0, 0)
    matcher = difflib.SequenceMatcher(None, autojunk=False)
    matcher.set_seq2(q)
    for n in sorted(candidates, key=lambda n: -overlap[n])[:3]:
        ws = [w for w, _, _ in spans[n]]
        for i in range(max(1, len(ws) - len(q) + 1)):
            window = ws[i : i + len(q)]
            if window[0] not in target:
                continue
            matcher.set_seq1(window)
            if matcher.real_quick_ratio() <= best[0] or matcher.quick_ratio() <= best[0]:
                continue
            ratio = matcher.ratio()
            if ratio > best[0]:
                best = (ratio, n, i, min(i + len(q), len(ws)) - 1)
    ratio, n, i, j = best
    if n is None or ratio < min_ratio:
        return None
    return n, cut(pages[n - 1], spans[n], i, j)


def also_on(passage: str, pages: list[str], candidates: list[int], skip: int) -> list[int]:
    q = words(passage)
    if len(q) < 6:
        return []
    needle = " ".join(q)
    return [n for n in candidates if n != skip and needle in " ".join(words(pages[n - 1]))]

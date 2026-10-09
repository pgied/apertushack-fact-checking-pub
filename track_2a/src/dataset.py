"""Download the challenge dataset and convert it to the challenge's input formats, for both tasks."""

import functools
import json
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

from src import pdf

DATASET = "OSTswiss/MNLIoverSwissVotingBooklets"
REVISION = "fc2b27600310778da6bbf445651ddbca22d86269"  # pinned, so the case ids stay stable
VERSION = "v1.1"
FILENAME = f"{VERSION}.jsonl"

# bk.admin.ch closes the connection on Python's default user agent
USER_AGENT = "apertushack-fact-checking (dataset preparation)"

# A reference is looked up in the booklet in chunks of this many letters and digits, about 7 words
CHUNK = 40


def download(raw_path: Path) -> None:
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{FILENAME}"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, raw_path)


def convert(raw_path: Path, out_dir: Path) -> int:
    """Beginner task: write cases.jsonl (what the system gets) and gold.jsonl (the answers, for evaluation)."""
    cases, gold = [], []
    for i, row in enumerate(read_jsonl(raw_path)):
        case_id = make_id(i, "beginner")
        cases.append({
            "id": case_id,
            "vote": row["vote"],
            "claim": {"text": row["claim"], "language": row["claim_language"]},
            "reference": {"text": row["reference_string"], "language": row["reference_language"]},
        })
        gold.append({
            "id": case_id,
            "label": row["entailment_label"],
            "claim_language": row["claim_language"],
            "reference_language": row["reference_language"],
        })
    write_jsonl(out_dir / "cases.jsonl", cases)
    write_jsonl(out_dir / "gold.jsonl", gold)
    return len(cases)


def convert_advanced(raw_path: Path, out_dir: Path) -> dict[tuple[str, str], float]:
    """Advanced task: download the booklets into out_dir/booklets/, then write cases.jsonl and gold.jsonl.

    The system gets the booklet and the vote instead of the reference text. The gold evidence is the
    reference text, split by the booklet page it is on. Returns, for every (booklet, reference), the
    share of the reference that was found in the booklet.
    """
    rows = read_jsonl(raw_path)
    urls = {booklet_name(row): row["booklet_url"] for row in rows}
    for name, url in sorted(urls.items()):
        path = out_dir / "booklets" / name
        if not path.exists():
            print(f"  downloading {name}")
            download_booklet(url, path)
    pages = {name: pdf.pages(out_dir / "booklets" / name) for name in urls}

    cases, gold, found = [], [], {}
    for i, row in enumerate(rows):
        case_id = make_id(i, "advanced")  # the same row number as in the beginner task
        name = booklet_name(row)
        key = (name, row["reference_string"])
        if key not in found:
            found[key] = locate(row["reference_string"], pages[name])
        evidence, _ = found[key]
        cases.append({
            "id": case_id,
            "vote": row["vote"],
            "claim": {"text": row["claim"], "language": row["claim_language"]},
            # relative to the folder of cases.jsonl, which is /data in the challenge's container
            "booklet": {"path": f"booklets/{name}", "language": row["reference_language"]},
        })
        gold.append({
            "id": case_id,
            "label": row["entailment_label"],
            "claim_language": row["claim_language"],
            "reference_language": row["reference_language"],
            "evidence": evidence,
        })
    write_jsonl(out_dir / "cases.jsonl", cases)
    write_jsonl(out_dir / "gold.jsonl", gold)
    return {key: share for key, (_, share) in found.items()}


def make_id(row: int, task: str) -> str:
    """The id the organisers give the case of a dataset row: v1.1-row-42-B in the beginner task (B for
    reference), v1.1-row-42-A in the advanced one (A for document). Rows count from 0."""
    return f"{VERSION}-row-{row}-{'B' if task == 'beginner' else 'A'}"


def booklet_name(row: dict) -> str:
    """For example 2024_11_24_de.pdf: there is one booklet per vote date and language."""
    return f"{row['booklet_publish_date'].replace('-', '_')}_{row['reference_language']}.pdf"


def download_booklet(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def locate(reference: str, pages: list[str]) -> tuple[list[dict], float]:
    """Find the booklet pages that the reference text is on.

    Returns the evidence, the reference split into {"page", "text"} passages, one for each stretch
    of it on a page, and the share of the reference that was found in the booklet.

    The reference and the PDF text differ in spacing, punctuation and line breaks, so both are
    compared as letters and digits only, in chunks of about CHUNK letters. A chunk found on a single
    page puts its text there, and a paragraph that runs over a page break is split where its chunks
    move on. A paragraph without such a chunk, like a heading or the vote title that the booklet
    repeats on many pages, goes with the text after it.
    """
    page_keys = [letters(page) for page in pages]
    marks = []  # [position in the reference, its page or None, how many of its chunks are on each page]
    total = hits = 0
    start = 0
    for paragraph in reference.splitlines(keepends=True):
        folded = [fold(c) for c in paragraph]
        where = [i for i, f in enumerate(folded) for _ in f]  # position of each letter in the paragraph
        text = "".join(folded)
        runs, seen, i = [], Counter(), 0  # runs: [page, its first letter, how many chunks]
        for chunk in chunks(text):
            on = [n for n, key in enumerate(page_keys, 1) if chunk in key]
            total += 1
            hits += bool(on)
            seen.update(on)
            if len(on) == 1 and runs and runs[-1][0] == on[0]:
                runs[-1][2] += 1
            elif len(on) == 1:
                runs.append([on[0], i, 1])
            i += len(chunk)
        if any(n > 1 for _, _, n in runs):  # a lone chunk elsewhere is a repeat, say in the summary
            runs = [run for run in runs if run[2] > 1]
        for k, (page, first, _) in enumerate(runs):
            if k and page == runs[k - 1][0]:
                continue
            p = start
            if k:  # the page break is in the chunk before: go back while CHUNK letters from there are on the page
                while first > 0 and text[first - 1:first - 1 + CHUNK] in page_keys[page - 1]:
                    first -= 1
                p = start + where[first]
                while p > start and not reference[p - 1].isspace():  # back to the start of the word
                    p -= 1
            marks.append([p, page, seen])
        if not runs and paragraph.strip():
            marks.append([start, None, seen])
        start += len(paragraph)

    anchor = None
    if not any(page for _, page, _ in marks):
        # Every chunk is on several pages: the reference is only the vote title, or a legal text
        # that the booklet prints twice. Go by the page with most of it, the first one on a tie.
        count = sum((seen for _, _, seen in marks), Counter())
        if not count:
            return [], 0.0
        anchor = min(count, key=lambda p: (-count[p], p))
    known = [page for _, page, _ in marks]
    for k, mark in enumerate(marks):
        if mark[1]:
            continue
        before = next((p for p in reversed(known[:k]) if p), None)
        after = next((p for p in known[k + 1:] if p), None)
        near = [p for p in (before, after) if p] or [anchor]
        # found on several pages: the one nearest the text around it; else go with the text after it
        mark[1] = min(sorted(mark[2]), key=lambda p: min(abs(p - q) for q in near), default=after or before or anchor)

    evidence = []
    for k, (p, page, _) in enumerate(marks):
        text = reference[p:marks[k + 1][0] if k + 1 < len(marks) else len(reference)]
        if evidence and evidence[-1]["page"] == page:
            evidence[-1]["text"] += text
        else:
            evidence.append({"page": page, "text": text})
    for passage in evidence:
        passage["text"] = passage["text"].strip()
    return evidence, hits / total


def letters(text: str) -> str:
    """Only the letters and digits of the text, in lowercase and without accents."""
    return "".join(map(fold, text))


@functools.cache
def fold(c: str) -> str:
    """The letters and digits in one character: 'É' gives 'e', the ligature 'ﬁ' gives 'fi', '’' nothing."""
    return "".join(d for d in unicodedata.normalize("NFKD", c).lower() if d.isalnum())


def chunks(text: str) -> list[str]:
    """Split into pieces of CHUNK to 2 * CHUNK characters. Texts under 15 characters, like a lone
    "NO.", are left out: they are too short to say where they are."""
    if len(text) < 15:
        return []
    n = max(1, len(text) // CHUNK)
    size = len(text) / n
    return [text[round(i * size):round((i + 1) * size)] for i in range(n)]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

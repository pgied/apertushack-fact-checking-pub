"""Read a voting booklet as a list of blocks in reading order: the readers of the gied_ strategies.

A block is a tuple (page, level, text). The page counts from 1. The level is 1 to 6 for a heading
(H1 to H6) and 0 for body text. The text is plain, as printed.

- text_layer: the PDF's text layer, as the baseline reads it. Short lines that repeat on three or
  more pages, the running headers such as "Erste Vorlage: Biodiversitätsinitiative", become headings.
- tags: the PDF's tags. 59 of the 60 booklets are tagged PDFs: their structure tree has the reading
  order, the headings and the paragraphs, and leaves out running headers and page numbers.
- docling: the JSON that docling wrote for the PDF (gied_docling/convert.py), found by the PDF's SHA-256.

PDFium is not thread-safe: read one PDF at a time.
"""

import ctypes
import hashlib
import json
import re
from collections import Counter
from itertools import groupby
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as raw

Block = tuple[int, int, str]  # (page, level, text): level 1 to 6 for a heading, 0 for body text

HEADINGS = {"H": 1, "H1": 1, "H2": 2, "H3": 3, "H4": 4, "H5": 5, "H6": 6}  # tag → heading level
BODY = {"P", "LI", "TR", "Note", "Caption", "BlockQuote"}  # tags whose text is one block of body text
LEFT_OUT = {"TOC", "Figure", "Artifact"}  # the table of contents repeats the headings; figures have no text
INLINE = {"Span", "Link", "Reference", "Lbl", "Quote", "Em", "Strong", "Sub", "Sup"}  # run on with the text around

DOCLING = {"title", "section_header", "text", "paragraph", "list_item", "caption", "footnote", "table"}


def read(path: Path, source: str, converted: Path) -> list[Block]:
    """The booklet's blocks as `source` reads them: pdfium (the text layer), tags or docling. Without
    docling's JSON, docling falls back to the tags, and an untagged PDF to the text layer."""
    if source not in ("pdfium", "tags", "docling"):
        raise ValueError(f"unknown source {source!r}: use pdfium, tags or docling")
    if source == "docling" and (blocks := docling(path, converted)):
        return blocks
    if source in ("docling", "tags") and (blocks := tags(path)):
        return blocks
    return text_layer(path)


def plain(text: str) -> str:
    """Text as it reads: no soft hyphens, no control characters (the booklets' arrows come out as \\x08),
    and single spaces instead of line breaks."""
    return " ".join(re.sub(r"[\x00-\x08\x0b-\x1f\x7f­￾]", "", text).split())


# The text layer

def text_layer(path: Path) -> list[Block]:
    """The PDF's text layer. On each page, a running header becomes a heading, and the lines between
    the headings one block of body text. Page numbers are left out."""
    pdf = pdfium.PdfDocument(path)
    try:
        pages = [page.get_textpage().get_text_range().split("\r\n") for page in pdf]
    finally:
        pdf.close()
    pages = [[line for line in map(plain, lines) if line and not line.isdigit()] for lines in pages]
    headers = running_headers(pages)
    blocks = []
    for page, lines in enumerate(pages, 1):
        for is_header, group in groupby(lines, key=lambda line: shape(line) in headers):
            if is_header:
                blocks += [(page, 2, re.sub(r"^\d+\s+|\s+\d+$", "", line)) for line in group]
            else:
                blocks.append((page, 0, " ".join(group)))
    return blocks


def running_headers(pages: list[list[str]]) -> set[str]:
    """The short lines that are on three or more pages, page numbers aside."""
    count = Counter(shape(line) for lines in pages for line in set(lines) if len(line) < 100)
    return {line for line, pages in count.items() if pages >= 3}


def shape(line: str) -> str:
    """A line with its numbers replaced, so that the running headers of page 8 and page 9 are the same."""
    return re.sub(r"\d+", "#", line)


# The tags

def tags(path: Path) -> list[Block] | None:
    """The blocks of the PDF's tags, or None if it has none.

    The voting text ("Abstimmungstext") is placed untagged, as it was published. A page with much
    text outside the tags keeps the headings from its tags and takes its body text from the text layer.
    """
    blocks, untagged_pages = [], set()
    pdf = pdfium.PdfDocument(path)
    try:
        for page_number, page in enumerate(pdf, 1):
            texts, untagged = marked_content(page)
            blocks += [(page_number, level, text) for level, text in structure(page, texts)]
            if untagged:
                untagged_pages.add(page_number)
    finally:
        pdf.close()
    if not blocks:
        return None
    if untagged_pages:
        body = [block for block in text_layer(path) if block[0] in untagged_pages and block[1] == 0]
        blocks = [block for block in blocks if block[0] not in untagged_pages or block[1] > 0] + body
        blocks.sort(key=lambda block: block[0])  # back into page order; within a page the order stays
    return blocks


def marked_content(page: pdfium.PdfPage) -> tuple[dict[int, str], bool]:
    """The text of each marked-content ID on the page, which the tags refer to, and whether much of the
    page's text has no ID, so is not in the tags. Running headers have none either, but they are short."""
    textpage = page.get_textpage()
    texts: dict[int, str] = {}
    untagged = 0
    for obj, mcid in text_objects(page.raw, raw.FPDFPage_CountObjects, raw.FPDFPage_GetObject, -1):
        text = utf16(raw.FPDFTextObj_GetText, obj, textpage.raw) or ""
        if mcid >= 0:
            texts[mcid] = texts.get(mcid, "") + text
        else:
            untagged += len(text)
    tagged = sum(map(len, texts.values()))
    return texts, untagged > 300 and untagged > 0.2 * (tagged + untagged)


def text_objects(container, count, get, mcid: int) -> list[tuple]:
    """The text objects of a page with their marked-content ID, also those inside form XObjects, which
    take the ID of their form if they have none of their own."""
    found = []
    for i in range(count(container)):
        obj = get(container, i)
        own = raw.FPDFPageObj_GetMarkedContentID(obj)
        if raw.FPDFPageObj_GetType(obj) == raw.FPDF_PAGEOBJ_TEXT:
            found.append((obj, own if own >= 0 else mcid))
        elif raw.FPDFPageObj_GetType(obj) == raw.FPDF_PAGEOBJ_FORM:
            found += text_objects(obj, raw.FPDFFormObj_CountObjects, raw.FPDFFormObj_GetObject, own if own >= 0 else mcid)
    return found


def structure(page: pdfium.PdfPage, texts: dict[int, str]) -> list[tuple[int, str]]:
    """The headings and paragraphs in the page's tags, in reading order, as (level, text)."""
    found = []

    def visit(element) -> None:
        tag = utf16(raw.FPDF_StructElement_GetType, element) or ""
        kids = children(element)
        if tag in LEFT_OUT:
            return
        if tag in HEADINGS or tag in BODY or (tag not in INLINE and kids and not any(kids)):
            # a heading, a paragraph, or a tag of another kind that holds only text
            if text := plain(text_of(element, texts)):
                found.append((HEADINGS.get(tag, 0), text))
            return
        for kid in kids:
            if kid:
                visit(kid)

    tree = raw.FPDF_StructTree_GetForPage(page.raw)
    try:
        for i in range(raw.FPDF_StructTree_CountChildren(tree)):
            visit(raw.FPDF_StructTree_GetChildAtIndex(tree, i))
    finally:
        raw.FPDF_StructTree_Close(tree)
    return found


def text_of(element, texts: dict[int, str]) -> str:
    """The text of a tag: its ActualText if it has one (a hyphen at a line break has a soft hyphen as
    ActualText), else the text of what it contains, in order. Inline tags, such as spans and links,
    run on with the text around them; other tags, such as table cells, are set apart by a space."""
    actual = utf16(raw.FPDF_StructElement_GetActualText, element)
    if actual is not None:
        return actual
    parts = []
    for i, kid in enumerate(children(element)):
        if kid is None:  # marked content on the page
            parts.append(texts.get(raw.FPDF_StructElement_GetChildMarkedContentID(element, i), ""))
        elif (utf16(raw.FPDF_StructElement_GetType, kid) or "") in INLINE:
            parts.append(text_of(kid, texts))
        else:
            parts.append(f" {text_of(kid, texts)} ")
    return "".join(parts)


def children(element) -> list:
    """A tag's children: a tag, or None where the child is marked content on the page."""
    count = max(0, raw.FPDF_StructElement_CountChildren(element))
    return [raw.FPDF_StructElement_GetChildAtIndex(element, i) or None for i in range(count)]


def utf16(function, *args) -> str | None:
    """Call a PDFium function that writes UTF-16LE text into a buffer; None if there is no such text."""
    size = function(*args, None, 0)
    if size <= 0:
        return None
    buffer = (ctypes.c_ushort * (size // 2 + 1))()
    function(*args, ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ushort)), size)
    return ctypes.string_at(buffer, max(0, size - 2)).decode("utf-16-le", "replace")


# docling

def docling(path: Path, converted: Path) -> list[Block] | None:
    """The blocks in docling's JSON for the PDF, converted/<SHA-256 of the PDF>.json, in docling's
    reading order; None if it is not there. Page headers and footers are not in docling's body."""
    file = converted / f"{sha256(path)}.json"
    if not file.exists():
        return None
    document = json.loads(file.read_text(encoding="utf-8"))
    blocks = []

    def visit(ref: str) -> None:
        _, collection, index = ref.split("/")  # "#/texts/12"
        item = document[collection][int(index)]
        label = item.get("label")
        if label in DOCLING and item.get("prov"):
            text = plain(table_text(item) if label == "table" else item.get("text", ""))
            if text:
                level = item.get("level", 1) if label in ("title", "section_header") else 0
                blocks.append((item["prov"][0]["page_no"], level, text))
        for child in item.get("children", []):
            visit(child["$ref"])

    for child in document["body"]["children"]:
        visit(child["$ref"])
    return blocks


def table_text(table: dict) -> str:
    """A table as text: its rows one after the other, the cells of a row separated by spaces."""
    grid = table.get("data", {}).get("grid", [])
    return " ".join(" ".join(cell.get("text", "") for cell in row) for row in grid)


def page_count(path: Path) -> int:
    pdf = pdfium.PdfDocument(path)
    try:
        return len(pdf)
    finally:
        pdf.close()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

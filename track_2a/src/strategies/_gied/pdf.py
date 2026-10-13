"""Read a voting booklet as blocks of plain text, page by page: the readers of the gied_ strategies.

- pdfium: the PDF's text layer, as src/pdf.py reads it. Short lines that repeat on several pages,
  such as the running header "Erste Vorlage: Biodiversitätsinitiative", become headings, and page
  numbers are dropped.
- tags: the PDF's structure tree. The booklets are tagged PDFs (59 of the 60 so far): their tags
  give the reading order, the headings and the paragraphs, mark hyphens at line breaks, and leave
  out running headers and page numbers. An untagged PDF gives None.
- docling: the JSON that docling wrote for the PDF (see _gied_docling), looked up by the PDF's SHA-256.

A booklet is a list of pages, page n at index n - 1 as in src/pdf.py, and a page is a list of blocks.
PDFium is not thread-safe, so every call into it holds LOCK.
"""

import ctypes
import hashlib
import json
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as raw

READERS = ("pdfium", "tags", "docling")
LOCK = threading.Lock()

# Structure elements that hold one block of text, and what kind of block it is
BLOCKS = {"P": "text", "LI": "text", "LBody": "text", "BlockQuote": "text", "Caption": "text", "TR": "text",
          "Note": "footnote", "H": "heading", **{f"H{n}": "heading" for n in range(1, 7)}}
SKIPPED = {"TOC", "TOCI", "Figure", "Artifact"}  # the table of contents repeats headings; figures have no text
INLINE = {"Span", "Link", "Reference", "Quote", "Em", "Strong", "Sub", "Sup", "Code", "BibEntry", "Annot", "Lbl"}
GROUPS = {"Document", "Part", "Art", "Sect", "Div", "NonStruct", "Private", "Table", "THead", "TBody", "TFoot",
          "L", "Form", "Index"}

# docling's labels, and what kind of block they are; page headers and footers are left out
DOCLING = {"title": "heading", "section_header": "heading", "text": "text", "paragraph": "text",
           "list_item": "text", "caption": "text", "footnote": "footnote", "table": "text"}


@dataclass(frozen=True)
class Block:
    kind: str  # heading, text or footnote
    text: str  # plain text as printed, without markup
    level: int = 0  # headings: 1 for H1 to 6 for H6


def plain(text: str) -> str:
    """Text as it reads: without soft hyphens, control characters (the booklets' arrows come out as
    \\x08) or line breaks, the spaces collapsed."""
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f\u00ad\ufffe]", "", text)
    return " ".join(text.split())


def read(path: Path, source: str, converted: Path | None = None) -> list[list[Block]]:
    """The booklet as `source` reads it. docling falls back to tags, and tags to pdfium, when they have nothing."""
    if source not in READERS:
        raise ValueError(f"unknown reader {source!r}, use one of: {', '.join(READERS)}")
    if source == "docling" and converted and (pages := docling(path, converted)) is not None:
        return pages
    if source in ("docling", "tags") and (pages := tags(path)) is not None:
        return pages
    return text_layer(path)


def text_layer(path: Path) -> list[list[Block]]:
    """The PDF's text layer, page by page. Lines that repeat on three or more pages are running headers:
    they become headings, which name the section of the text after them."""
    with LOCK:
        pdf = pdfium.PdfDocument(path)
        try:
            texts = [page.get_textpage().get_text_range() for page in pdf]
        finally:
            pdf.close()
    pages = [[plain(line) for line in text.split("\r\n")] for text in texts]
    pages = [[line for line in lines if line and not line.isdigit()] for lines in pages]
    shape = lambda line: re.sub(r"\d+", "#", line).strip()  # noqa: E731
    seen = Counter(key for lines in pages for key in {shape(line) for line in lines if len(line) < 100})
    booklet = []
    for lines in pages:
        blocks, text = [], []
        for line in lines:
            if len(line) < 100 and seen[shape(line)] >= 3:
                if text:
                    blocks.append(Block("text", " ".join(text)))
                    text = []
                blocks.append(Block("heading", shape(line).lstrip("# ").strip(), 2))
            else:
                text.append(line)
        if text:
            blocks.append(Block("text", " ".join(text)))
        booklet.append([Block(b.kind, plain(b.text), b.level) for b in blocks if plain(b.text)])
    return booklet


def tags(path: Path) -> list[list[Block]] | None:
    """The blocks of the PDF's structure tree, page by page, or None if the PDF is not tagged.

    Some pages have text that is not in the tree: the voting text ("Abstimmungstext") is placed as it
    was published, untagged. Such a page keeps the headings of its tags and takes its text from the
    text layer."""
    with LOCK:
        pdf = pdfium.PdfDocument(path)
        try:
            pages = [page_tags(page) for page in pdf]
        finally:
            pdf.close()
    booklet = [blocks for blocks, _ in pages]
    if not any(booklet):
        return None
    if untagged := [n for n, (_, gap) in enumerate(pages) if gap]:
        text = text_layer(path)
        for n in untagged:
            booklet[n] = [b for b in booklet[n] if b.kind == "heading"] + [b for b in text[n] if b.kind != "heading"]
    return booklet


def page_tags(page: pdfium.PdfPage) -> tuple[list[Block], bool]:
    """The page's blocks from the structure tree, and whether much of its text is not in the tree."""
    textpage = page.get_textpage()
    texts: dict[int, str] = {}  # the text of each marked-content ID on the page
    tagged = untagged = 0  # characters of text in the tree and outside it (running headers, untagged text)
    for obj, mcid in objects(page.raw, raw.FPDFPage_CountObjects, raw.FPDFPage_GetObject, -1):
        if raw.FPDFPageObj_GetType(obj) != raw.FPDF_PAGEOBJ_TEXT:
            continue
        if mcid >= 0:
            text = utf16(raw.FPDFTextObj_GetText, obj, textpage.raw) or ""
            texts[mcid] = texts.get(mcid, "") + text
            tagged += len(text)
        else:
            untagged += max(0, raw.FPDFTextObj_GetText(obj, textpage.raw, None, 0) // 2 - 1)
    blocks: list[Block] = []
    tree = raw.FPDF_StructTree_GetForPage(page.raw)
    try:
        for i in range(raw.FPDF_StructTree_CountChildren(tree)):
            collect(raw.FPDF_StructTree_GetChildAtIndex(tree, i), texts, blocks)
    finally:
        raw.FPDF_StructTree_Close(tree)
    return blocks, untagged > 300 and untagged > 0.2 * (tagged + untagged)


def objects(container, count, get, mcid: int):
    """The page objects with their marked-content ID, also those inside form XObjects."""
    for i in range(count(container)):
        obj = get(container, i)
        own = raw.FPDFPageObj_GetMarkedContentID(obj)
        yield obj, own if own >= 0 else mcid
        if raw.FPDFPageObj_GetType(obj) == raw.FPDF_PAGEOBJ_FORM:
            yield from objects(obj, raw.FPDFFormObj_CountObjects, raw.FPDFFormObj_GetObject, own if own >= 0 else mcid)


def collect(element, texts: dict[int, str], blocks: list[Block]) -> None:
    """Add the blocks of a structure element and its descendants, in reading order."""
    kind = utf16(raw.FPDF_StructElement_GetType, element) or ""
    if kind in SKIPPED:
        return
    children = [raw.FPDF_StructElement_GetChildAtIndex(element, i)
                for i in range(max(0, raw.FPDF_StructElement_CountChildren(element)))]
    if kind not in BLOCKS and kind not in INLINE and kind not in GROUPS and not all(children):
        kind = "P"  # a tag of the PDF's own, not mapped to a standard one, that holds text: a paragraph
    if kind in BLOCKS:
        text = plain(content(element, texts))
        if text:
            level = int(kind[1]) if BLOCKS[kind] == "heading" and kind[1:].isdigit() else 0
            blocks.append(Block(BLOCKS[kind], text, level or (1 if BLOCKS[kind] == "heading" else 0)))
        return
    for child in children:
        if child:
            collect(child, texts, blocks)


def content(element, texts: dict[int, str]) -> str:
    """The text of a structure element: its ActualText if it has one (a hyphen at a line break has a
    soft hyphen there), else the text of its marked content and children, in order."""
    actual = utf16(raw.FPDF_StructElement_GetActualText, element)
    if actual is not None:
        return actual
    parts = []
    for i in range(max(0, raw.FPDF_StructElement_CountChildren(element))):
        child = raw.FPDF_StructElement_GetChildAtIndex(element, i)
        if child:
            inline = (utf16(raw.FPDF_StructElement_GetType, child) or "") in INLINE
            parts.append(content(child, texts) if inline else f" {content(child, texts)} ")
        else:
            parts.append(texts.get(raw.FPDF_StructElement_GetChildMarkedContentID(element, i), ""))
    return "".join(parts)


def utf16(function, *args) -> str | None:
    """Call a PDFium function that writes UTF-16LE text into a buffer; None if there is no such text."""
    size = function(*args, None, 0)
    if size <= 0:
        return None
    buffer = (ctypes.c_ushort * (size // 2 + 1))()
    function(*args, ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ushort)), size)
    return ctypes.string_at(buffer, max(0, size - 2)).decode("utf-16-le", "replace")


def docling(path: Path, converted: Path) -> list[list[Block]] | None:
    """The blocks docling found, from converted/<sha256 of the PDF>.json, or None if it is not there."""
    file = converted / f"{sha256(path)}.json"
    if not file.exists():
        return None
    document = json.loads(file.read_text(encoding="utf-8"))
    booklet: list[list[Block]] = [[] for _ in range(len(document.get("pages", {})) or 1)]

    def item(ref: str) -> dict:
        _, collection, index = ref.split("/")  # "#/texts/12"
        return document[collection][int(index)]

    def visit(node: dict) -> None:
        label = node.get("label", "")
        if node.get("content_layer", "body") == "body" and label in DOCLING and node.get("prov"):
            text = table_text(node) if label == "table" else node.get("text", "")
            text = plain(text)
            page = node["prov"][0]["page_no"]
            while len(booklet) < page:
                booklet.append([])
            if text:
                level = node.get("level", 1) if DOCLING[label] == "heading" else 0
                booklet[page - 1].append(Block(DOCLING[label], text, level))
        for child in node.get("children", []):
            visit(item(child["$ref"]))

    for child in document["body"].get("children", []):
        visit(item(child["$ref"]))
    return booklet


def table_text(table: dict) -> str:
    """A table as text: its rows one after the other, the cells of a row separated by spaces."""
    grid = table.get("data", {}).get("grid", [])
    return " ".join(" ".join(cell.get("text", "") for cell in row) for row in grid)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

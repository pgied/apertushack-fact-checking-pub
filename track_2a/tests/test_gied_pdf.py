"""Tests for the readers in src/strategies/_gied/pdf.py, on the booklet in data/examples/."""

from pathlib import Path

import pypdfium2 as pdfium

from src.strategies._gied import pdf

BOOKLET = Path(__file__).resolve().parent.parent / "data" / "examples" / "booklets" / "2020_11_29_fr.pdf"


def test_tags_give_headings_and_paragraphs_in_reading_order():
    pages = pdf.tags(BOOKLET)
    assert len(pages) == 32
    page4 = pages[3]
    assert [(b.kind, b.level, b.text) for b in page4[1:3]] == [("heading", 2, "En bref"), ("heading", 3, "Contexte")]
    assert page4[3].kind == "text" and page4[3].text.startswith("On attend des entreprises suisses")
    assert not any("­" in b.text or "￾" in b.text for page in pages for b in page)


def test_the_text_layer_drops_page_numbers_and_makes_running_headers_headings():
    pages = pdf.text_layer(BOOKLET)
    assert len(pages) == 32
    assert not any(b.text.isdigit() for page in pages for b in page)
    assert any(b.kind == "heading" for page in pages for b in page)


def test_an_untagged_pdf_has_no_tags_and_is_read_from_its_text_layer(tmp_path):
    untagged = tmp_path / "untagged.pdf"
    copy = pdfium.PdfDocument.new()
    copy.import_pages(pdfium.PdfDocument(BOOKLET), [3, 4])  # the pages come without the structure tree
    copy.save(untagged)

    assert pdf.tags(untagged) is None
    assert pdf.read(untagged, "tags") == pdf.text_layer(untagged)


def test_docling_without_its_json_falls_back_to_tags(tmp_path):
    assert pdf.read(BOOKLET, "docling", tmp_path) == pdf.tags(BOOKLET)


def test_docling_json_is_read_in_its_reading_order(tmp_path):
    document = {
        "body": {"children": [{"$ref": "#/texts/0"}, {"$ref": "#/groups/0"}, {"$ref": "#/tables/0"}]},
        "groups": [{"label": "list", "children": [{"$ref": "#/texts/1"}]}],
        "texts": [
            {"label": "section_header", "level": 1, "text": "En bref", "prov": [{"page_no": 1}]},
            {"label": "list_item", "text": "Les entre­prises respectent les droits.", "prov": [{"page_no": 2}]},
            {"label": "page_header", "text": "4 Premier objet", "prov": [{"page_no": 2}]},
        ],
        "tables": [{"label": "table", "prov": [{"page_no": 2}],
                    "data": {"grid": [[{"text": "Oui"}, {"text": "Non"}], [{"text": "12"}, {"text": "34"}]]}}],
        "pages": {"1": {}, "2": {}},
    }
    (tmp_path / f"{pdf.sha256(BOOKLET)}.json").write_text(__import__("json").dumps(document))
    pages = pdf.read(BOOKLET, "docling", tmp_path)
    assert pages == [[pdf.Block("heading", "En bref", 1)],
                     [pdf.Block("text", "Les entreprises respectent les droits."), pdf.Block("text", "Oui Non 12 34")]]

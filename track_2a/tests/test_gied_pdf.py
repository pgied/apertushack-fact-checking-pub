"""Tests for the readers in src/strategies/_gied/pdf.py, on the booklet in data/examples/."""

import json
from pathlib import Path

import pypdfium2 as pdfium

from src.strategies._gied import pdf

BOOKLET = Path(__file__).resolve().parent.parent / "data" / "examples" / "booklets" / "2020_11_29_fr.pdf"


def test_tags_give_headings_and_paragraphs_in_reading_order():
    blocks = pdf.tags(BOOKLET)
    page4 = [block for block in blocks if block[0] == 4]
    assert page4[1:3] == [(4, 2, "En bref"), (4, 3, "Contexte")]
    assert page4[3][1] == 0 and page4[3][2].startswith("On attend des entreprises suisses")
    assert not any("­" in text or "￾" in text for _, _, text in blocks)


def test_the_text_layer_drops_page_numbers_and_makes_running_headers_headings():
    blocks = pdf.text_layer(BOOKLET)
    assert {page for page, _, _ in blocks} <= set(range(1, 33))
    assert not any(text.isdigit() for _, _, text in blocks)
    assert any(level for _, level, _ in blocks)


def test_an_untagged_pdf_is_read_from_its_text_layer(tmp_path):
    untagged = tmp_path / "untagged.pdf"
    copy = pdfium.PdfDocument.new()
    copy.import_pages(pdfium.PdfDocument(BOOKLET), [3, 4])  # the pages come without the structure tree
    copy.save(untagged)

    assert pdf.tags(untagged) is None
    assert pdf.read(untagged, "tags", tmp_path) == pdf.text_layer(untagged)


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
    }
    (tmp_path / f"{pdf.sha256(BOOKLET)}.json").write_text(json.dumps(document))
    assert pdf.read(BOOKLET, "docling", tmp_path) == [
        (1, 1, "En bref"), (2, 0, "Les entreprises respectent les droits."), (2, 0, "Oui Non 12 34")]

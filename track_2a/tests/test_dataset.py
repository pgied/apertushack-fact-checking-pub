"""Tests for preparing the data: the cases in the challenge's format, and finding the reference texts on the
booklet pages for the advanced task."""

from src import dataset

# Three booklet pages, as pdfium reads them: a page number at the top, lines broken in the middle
# of a sentence, other spacing than in the reference, and the vote title on more than one page.
PAGES = [
    "Volksinitiative «Mehr Velowege»\nVolksabstimmung vom 1. Juni 2031",
    "2 Volksinitiative «Mehr Velowege»\nDer Bundesrat empfiehlt, die Initiative abzulehnen. Sie würde den Bund\n"
    "jedes Jahr 100000 Franken kosten, und dieses Geld fehlte dann an anderen Orten, etwa bei den\n"
    "Strassen und bei den Bahnen, und am Ende auch bei\n"
    "1 Bundesamt für Strassen, Bericht über die Velowege in der Schweiz, 2030",
    "3 den Kantonen, die heute schon sehr viel Geld für ihre eigenen Velowege ausgeben und dafür\n"
    "wirklich keine neue Regel in der Bundesverfassung brauchen.\n"
    "Das Parlament teilt die Haltung des Bundesrates, und zwar ohne eine einzige Gegenstimme.",
]
REFERENCE = (
    "Volksinitiative « Mehr Velowege »\n\n\n"
    "Der Bundesrat empfiehlt, die Initiative abzulehnen. Sie würde den Bund jedes Jahr 100 000 Franken kosten, "
    "und dieses Geld fehlte dann an anderen Orten, etwa bei den Strassen und bei den Bahnen, und am Ende auch "
    "bei den Kantonen, die heute schon sehr viel Geld für ihre eigenen Velowege ausgeben und dafür wirklich "
    "keine neue Regel in der Bundesverfassung brauchen.\n"
    "Das Parlament teilt die Haltung des Bundesrates, und zwar ohne eine einzige Gegenstimme.\n"
    "1 Bundesamt für Strassen, Bericht über die Velowege in der Schweiz, 2030"
)


def test_locate_splits_the_reference_by_page():
    evidence, found = dataset.locate(REFERENCE, PAGES)

    # The paragraph over the page break is split, and the footnote at the end goes back to page 2
    assert [passage["page"] for passage in evidence] == [2, 3, 2]
    assert evidence[0]["text"].startswith("Volksinitiative « Mehr Velowege »")  # with the text after it
    assert evidence[0]["text"].endswith("und am Ende auch bei")  # exactly at the page break
    assert evidence[1]["text"].startswith("den Kantonen, die heute")
    assert evidence[2]["text"].startswith("1 Bundesamt für Strassen")
    # The passages are the reference, cut between words
    assert " ".join(passage["text"] for passage in evidence).split() == REFERENCE.split()
    assert 0.8 < found <= 1


def test_a_reference_on_several_pages_goes_to_the_first():
    evidence, found = dataset.locate("Volksinitiative « Mehr Velowege »", PAGES)

    assert evidence == [{"page": 1, "text": "Volksinitiative « Mehr Velowege »"}]
    assert found == 1


def test_a_reference_not_in_the_booklet_has_no_evidence():
    assert dataset.locate("Ein ganz anderer Text, der in keinem Büchlein steht.", PAGES) == ([], 0.0)


def test_letters_ignore_case_accents_spacing_and_punctuation():
    assert dataset.letters("Élève « 100 000 » ﬁn") == dataset.letters("eleve 100000 fin") == "eleve100000fin"


def test_booklet_name():
    row = {"booklet_publish_date": "2024-11-24", "reference_language": "de"}
    assert dataset.booklet_name(row) == "2024_11_24_de.pdf"


def test_the_cases_have_the_challenges_format_and_ids(monkeypatch, tmp_path):
    row = {
        "claim": "Der Bundesrat empfiehlt, die Initiative abzulehnen.",
        "claim_language": "de",
        "reference_string": REFERENCE,
        "entailment_label": 0,
        "reference_language": "de",
        "booklet_publish_date": "2031-06-01",
        "booklet_url": "https://www.bk.admin.ch/a-booklet.pdf",
        "vote": "Volksinitiative «Mehr Velowege»",
    }
    raw = tmp_path / "raw.jsonl"
    dataset.write_jsonl(raw, [row, {**row, "claim_language": "fr", "entailment_label": 1}])
    booklet = tmp_path / "advanced" / "booklets" / "2031_06_01_de.pdf"
    booklet.parent.mkdir(parents=True)
    booklet.touch()  # already downloaded
    monkeypatch.setattr(dataset.pdf, "pages", lambda path: PAGES)

    dataset.convert(raw, tmp_path / "beginner")
    dataset.convert_advanced(raw, tmp_path / "advanced")

    # As the organisers write them: the dataset row, then B for the beginner task and A for the advanced one
    assert dataset.read_jsonl(tmp_path / "beginner" / "cases.jsonl")[1] == {
        "id": "v1.1-row-1-B",
        "vote": "Volksinitiative «Mehr Velowege»",
        "claim": {"text": row["claim"], "language": "fr"},
        "reference": {"text": REFERENCE, "language": "de"},
    }
    assert dataset.read_jsonl(tmp_path / "advanced" / "cases.jsonl")[1] == {
        "id": "v1.1-row-1-A",
        "vote": "Volksinitiative «Mehr Velowege»",
        "claim": {"text": row["claim"], "language": "fr"},
        "booklet": {"path": "booklets/2031_06_01_de.pdf", "language": "de"},
    }
    gold = dataset.read_jsonl(tmp_path / "advanced" / "gold.jsonl")
    assert [g["id"] for g in gold] == ["v1.1-row-0-A", "v1.1-row-1-A"]
    assert [g["label"] for g in gold] == [0, 1]
    assert [passage["page"] for passage in gold[0]["evidence"]] == [2, 3, 2]

"""gied_tags: retrieve, then verify, on the booklet's tags: the lightweight reader.

The booklets are tagged PDFs. Their structure tree gives the reading order, the headings and the
paragraphs, and leaves out running headers and page numbers; reading it takes about 0.1 s per booklet.
An untagged PDF is read from its text layer instead. How the strategy works: _gied/retrieve.py.
"""

from dataclasses import dataclass

from src.strategies._gied.retrieve import Retrieve


class GiedTags(Retrieve):
    @dataclass
    class Settings(Retrieve.Settings):
        source: str = "tags"

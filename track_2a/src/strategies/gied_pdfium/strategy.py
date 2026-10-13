"""gied_pdfium: retrieve, then verify, on the booklet's text layer as pypdfium2 reads it.

The text layer has no headings or paragraphs: the lines that repeat on several pages, the running
headers, stand in for the headings. It is what the baseline reads, so this strategy shows what the
retrieval alone brings. How the strategy works: _gied/retrieve.py.
"""

from dataclasses import dataclass

from src.strategies._gied.retrieve import Retrieve


class GiedPdfium(Retrieve):
    @dataclass
    class Settings(Retrieve.Settings):
        source: str = "pdfium"

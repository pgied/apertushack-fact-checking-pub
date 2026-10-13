"""gied_docling: retrieve, then verify, on the booklet as docling reads it.

docling's JSON of each booklet is made beforehand with `python -m src.strategies.gied_docling.convert`
and looked up by the PDF's SHA-256 in `converted`. A booklet without it is read from its tags, as in
gied_tags. How the strategy works: _gied/retrieve.py.
"""

from dataclasses import dataclass

from src.strategies._gied.retrieve import Retrieve


class GiedDocling(Retrieve):
    @dataclass
    class Settings(Retrieve.Settings):
        source: str = "docling"

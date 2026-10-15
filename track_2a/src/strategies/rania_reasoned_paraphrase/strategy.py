"""very straightforward, i thoight combining with gied_strict_reasoned (and then searching with five paraphrases of the claim) would be helpful since the reasoned startegy is the strongest so far
"""

from dataclasses import dataclass

from src.strategies.gied_strict_paraphrase.strategy import GiedStrictParaphrase
from src.strategies.gied_strict_reasoned.strategy import GiedStrictReasoned


class RaniaReasonedParaphrase(GiedStrictParaphrase, GiedStrictReasoned):
    @dataclass
    class Settings(GiedStrictReasoned.Settings):
        versions: int = 5

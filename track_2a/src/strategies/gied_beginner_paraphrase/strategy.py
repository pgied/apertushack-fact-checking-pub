"""gied_beginner_paraphrase: prompt_lab's best prompt for the beginner task, with the claim also in
five simpler paraphrases in the reference's language.

The beginner task has nothing to search, so the paraphrases (as in gied_paraphrase, after Hoyle et al.,
EMNLP 2023, without implicit inferences) help the model read the claim instead: written in the
reference's language, they also bridge a claim in another language, a mix of paraphrase and
multi-lang. The label is still for the original claim.
"""

from dataclasses import dataclass

from src.llm import LLM
from src.strategies.gied_paraphrase.strategy import paraphrases
from src.strategies.prompt_lab.strategy import PromptLab
from src.strategy import Verdict


class GiedBeginnerParaphrase(PromptLab):
    @dataclass
    class Settings(PromptLab.Settings):
        labels: str = "quote"  # prompt_lab's best: 0.962 on all 1,488 beginner cases
        reason: bool = True
        versions: int = 5  # paraphrases of the claim

    def predict(self, case: dict, llm: LLM) -> Verdict:
        versions = paraphrases(case["claim"]["text"], case["reference"]["language"], llm, self.settings.versions)
        return super().predict({**case, "paraphrases": versions}, llm)

    def user_prompt(self, case: dict) -> str:
        prompt = super().user_prompt(case)
        if not case.get("paraphrases"):
            return prompt
        listed = "\n".join(f"- {version}" for version in case["paraphrases"])
        added = f"THE CLAIM IN OTHER WORDS (to help you read it; the label is for the CLAIM):\n{listed}"
        return prompt.removesuffix("\n\nAnswer:") + f"\n\n{added}\n\nAnswer:"

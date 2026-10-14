"""gied_beginner_multi_lang: prompt_lab's best prompt for the beginner task, with a claim in another
language than the reference also shown in the reference's language.

prompt_lab with labels=quote and reason=true decided all 1,488 beginner cases with macro-F1 0.962. In
the advanced task, seeing the booklet in the claim's own language was the biggest gain
(gied_multi_lang_evidence). The beginner task has only the reference, so here Apertus translates the
claim into the reference's language instead (one short request), and the model sees both versions.
A claim in the reference's language is decided as in prompt_lab.
"""

from dataclasses import dataclass

from src.llm import LLM
from src.strategies._gied.retrieve import LANGUAGES, translate
from src.strategies.prompt_lab.strategy import PromptLab
from src.strategy import Verdict


class GiedBeginnerMultiLang(PromptLab):
    @dataclass
    class Settings(PromptLab.Settings):
        labels: str = "quote"  # prompt_lab's best: 0.962 on all 1,488 beginner cases
        reason: bool = True

    def predict(self, case: dict, llm: LLM) -> Verdict:
        if case["claim"]["language"] != case["reference"]["language"] and "translation" not in case:
            case = {**case, "translation": translate(case["claim"]["text"], case["reference"]["language"], llm)}
        return super().predict(case, llm)

    def user_prompt(self, case: dict) -> str:
        prompt = super().user_prompt(case)
        if "translation" not in case:
            return prompt
        language = LANGUAGES.get(case["reference"]["language"], "the reference's language").upper()
        added = f"THE CLAIM IN {language} (a translation, to compare it with the reference):\n{case['translation']}"
        return prompt.removesuffix("\n\nAnswer:") + f"\n\n{added}\n\nAnswer:"

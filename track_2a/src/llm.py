"""The model, as a strategy sees it for one case: chat requests, with their tokens added up."""

import threading

from src.timing import InFlight


class LLM:
    """The model, for one case: sends chat requests and adds up their tokens.

    Every case gets its own, so its prediction counts the tokens of all its requests. `request` holds
    what is sent with every request: temperature and max_tokens from the strategy's settings, plus
    the backend's extra fields. `in_flight` is shared by all cases of a run: it times the requests,
    which the challenge's time score leaves out (src/timing.py).
    """

    def __init__(self, client, model: str, request: dict, in_flight: InFlight | None = None):
        self.client = client
        self.request = {"model": model, **request}
        self.in_flight = in_flight or InFlight()
        self.requests = self.input_tokens = self.output_tokens = 0
        self.lock = threading.Lock()  # a strategy may send several requests at once

    def chat(self, messages: list[dict], **request) -> str:
        """Send the messages and return the answer. `request` overrides settings for this request, e.g. max_tokens=50."""
        with self.in_flight:
            response = self.client.chat.completions.create(messages=messages, **{**self.request, **request})
        with self.lock:
            self.requests += 1
            self.input_tokens += response.usage.prompt_tokens
            self.output_tokens += response.usage.completion_tokens
        return response.choices[0].message.content or ""

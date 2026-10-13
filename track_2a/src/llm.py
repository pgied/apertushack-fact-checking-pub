"""The model, as a strategy sees it for one case: chat requests, with their tokens added up."""

import threading


class LLM:
    """The model, for one case: sends chat requests and adds up their tokens.

    Every case gets its own, so its prediction counts the tokens of all its requests. `request` holds
    what is sent with every request: temperature and max_tokens from the strategy's settings, plus
    the backend's extra fields.
    """

    def __init__(self, client, model: str, request: dict):
        self.client = client
        self.request = {"model": model, **request}
        self.requests = self.input_tokens = self.output_tokens = 0
        self.lock = threading.Lock()  # a strategy may send several requests at once

    def chat(self, messages: list[dict], **request) -> str:
        """Send the messages and return the answer. `request` overrides settings for this request, e.g. max_tokens=50."""
        response = self.client.chat.completions.create(messages=messages, **{**self.request, **request})
        with self.lock:
            self.requests += 1
            self.input_tokens += response.usage.prompt_tokens
            self.output_tokens += response.usage.completion_tokens
        return response.choices[0].message.content or ""

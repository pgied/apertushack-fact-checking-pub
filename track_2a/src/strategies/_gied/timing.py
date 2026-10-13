"""The time a run spends without a model request in flight: what the challenge scores as processing time.

The challenge takes the wall-clock time of a run minus the time in which at least one request to the
model is in flight: parsing and indexing count, the model's latency does not. CLOCK measures the same
for the requests of the gied_ strategies, from when this module is first imported (when the strategies
are registered, at the start of a run) to the end of the process, and prints it to stderr at the end.
"""

import atexit
import sys
import threading
import time
from contextlib import contextmanager


class Clock:
    def __init__(self):
        self.lock = threading.Lock()
        self.start = time.perf_counter()
        self.requests = 0
        self.in_flight = 0  # requests on their way right now
        self.since = 0.0  # when the current stretch with requests in flight began
        self.busy = 0.0  # seconds with at least one request in flight, before that stretch

    @contextmanager
    def request(self):
        with self.lock:
            if not self.in_flight:
                self.since = time.perf_counter()
            self.in_flight += 1
            self.requests += 1
        try:
            yield
        finally:
            with self.lock:
                self.in_flight -= 1
                if not self.in_flight:
                    self.busy += time.perf_counter() - self.since

    def report(self) -> str:
        with self.lock:
            now = time.perf_counter()
            busy = self.busy + (now - self.since if self.in_flight else 0.0)
            requests = self.requests
        wall = now - self.start
        return (f"{requests} model requests, {wall:.1f} s wall-clock: {busy:.1f} s with a request in flight, "
                f"{wall - busy:.1f} s without")


CLOCK = Clock()


class Timed:
    """Sends the requests of an LLM through the clock; the LLM still counts their tokens."""

    def __init__(self, llm):
        self.llm = llm

    def chat(self, messages: list[dict], **request) -> str:
        with CLOCK.request():
            return self.llm.chat(messages, **request)


@atexit.register
def report() -> None:
    if CLOCK.requests:
        print(f"gied timing: {CLOCK.report()}", file=sys.stderr)

"""A run's time as the challenge's time score measures it: the wall-clock time of the run minus the time
in which at least one request to the model is in flight (Q&A slide 17).

So the model's latency does not count against us, but everything else does, once per run: Python's
start and the imports, reading the booklets, indexing and whatever predict() computes, unless a request
of another case is in flight meanwhile. The score divides it by the number of cases. Both command lines
measure it with a Stopwatch: src/__main__.py writes it to the run's time.json, src/entrypoint.py prints it.

The organisers measure it themselves, from outside, so ours differs a little: it starts when Python
begins running our code (src/__init__.py), not when the container starts, and a request is in flight
from when we send it until its answer is read, a little longer than their proxy sees it.
"""

import threading
import time


class InFlight:
    """The time in which at least one request to the model is in flight. All requests of a run go
    through the same one, each as `with in_flight:`; requests in parallel count once."""

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.flying = 0  # requests in flight now
        self.since = 0.0  # when the current stretch with requests in flight began
        self.done = 0.0  # the seconds in flight before that stretch
        self.lock = threading.Lock()

    def __enter__(self):
        with self.lock:
            if not self.flying:
                self.since = self.clock()
            self.flying += 1

    def __exit__(self, *error):
        with self.lock:
            self.flying -= 1
            if not self.flying:
                self.done += self.clock() - self.since

    def seconds(self) -> float:
        """The seconds with at least one request in flight so far."""
        with self.lock:
            return self.done + (self.clock() - self.since if self.flying else 0.0)


class Stopwatch:
    """Times a run lap by lap, each lap without the time in which a request is in flight: lap("startup")
    when the setup begins, lap("setup") when the cases begin, lap("cases") after the last one."""

    def __init__(self, started: float, clock=time.perf_counter):
        self.started = started  # when the run began: src.STARTED
        self.clock = clock
        self.in_flight = InFlight(clock)  # for every LLM of the run
        self.laps: dict[str, float] = {}  # name → seconds without a request in flight
        self.mark = (started, 0.0)  # when the current lap began, and the seconds in flight by then

    def lap(self, name: str) -> None:
        """End the current lap and keep its time without a request in flight as `name`."""
        now, flown = self.clock(), self.in_flight.seconds()
        began, flown_before = self.mark
        self.laps[name] = (now - began) - (flown - flown_before)
        self.mark = (now, flown)

    def summary(self, cases: int, workers: int) -> dict:
        """The run's time up to the last lap, in ms, as time.json holds it."""
        return {
            "cases": cases,  # the cases decided
            "workers": workers,  # requests in parallel: with more, predict()'s own work overlaps more with requests
            "wall_ms": ms(self.mark[0] - self.started),
            "in_flight_ms": ms(self.mark[1]),
            "without_model_ms": {name: ms(seconds) for name, seconds in self.laps.items()},  # what the score counts
        }


def describe(timing: dict) -> str:
    """The time without the model per case and in all, lap by lap, from a summary()."""
    laps = timing["without_model_ms"]
    total = sum(laps.values())
    split = ", ".join(f"{name} {lap / 1000:,.1f} s" for name, lap in laps.items())
    per_case = f"{total / timing['cases']:,.1f} ms per case, " if timing["cases"] else ""
    return f"{per_case}{total / 1000:,.1f} s in all for {timing['cases']:,} cases: {split}"


def ms(seconds: float) -> int:
    return round(seconds * 1000)

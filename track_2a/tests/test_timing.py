"""Tests for timing a run as the challenge's time score does: without the time in which requests are in flight."""

from types import SimpleNamespace

from src.evaluate import report
from src.llm import LLM
from src.timing import InFlight, Stopwatch, describe

GOLD = [{"id": f"case-{i}", "label": 1, "claim_language": "de", "reference_language": "de"} for i in range(2)]
PREDICTIONS = [{"id": g["id"], "label": 1, "evidence": [],
                "metrics": {"input_tokens": 100, "output_tokens": 2, "inference_time_ms": 900}} for g in GOLD]


class Clock:
    """Stands in for time.perf_counter: it moves only when the test moves it."""

    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class SlowClient:
    """Stands in for the OpenAI client: every answer takes `seconds` on the clock."""

    def __init__(self, clock: Clock, seconds: float):
        self.clock, self.seconds = clock, seconds
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.clock.now += self.seconds
        message = SimpleNamespace(content="neutral")
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=2)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def test_requests_in_parallel_count_once_and_the_gaps_not_at_all():
    clock = Clock()
    in_flight = InFlight(clock)
    with in_flight:  # 0 to 4
        clock.now = 1
        with in_flight:  # 1 to 2, in parallel with the first
            clock.now = 2
        clock.now = 4
    clock.now = 6
    with in_flight:  # 6 to 7.5, after 2 s without a request
        clock.now = 7
        assert in_flight.seconds() == 5  # the stretch still in flight counts so far
        clock.now = 7.5
    assert in_flight.seconds() == 5.5  # 0 to 4 and 6 to 7.5


def test_every_request_of_a_case_is_in_flight_while_the_model_answers():
    clock = Clock()
    in_flight = InFlight(clock)
    llm = LLM(SlowClient(clock, seconds=2), "some-model", {}, in_flight)

    llm.chat([{"role": "user", "content": "first"}])
    llm.chat([{"role": "user", "content": "second"}])

    assert in_flight.seconds() == 4 and llm.requests == 2


def test_a_run_is_timed_lap_by_lap_without_the_requests():
    clock = Clock(10)
    stopwatch = Stopwatch(started=9, clock=clock)  # Python began running our code 1 s ago
    stopwatch.lap("startup")  # the imports: no request yet
    with stopwatch.in_flight:  # setup() asks the model for 3 s, then reads booklets for 1 s
        clock.now = 13
    clock.now = 14
    stopwatch.lap("setup")
    with stopwatch.in_flight:  # the cases: 6 s with requests in flight, then 0.5 s without
        clock.now = 20
    clock.now = 20.5
    stopwatch.lap("cases")

    timing = stopwatch.summary(cases=10, workers=8)

    assert timing == {"cases": 10, "workers": 8, "wall_ms": 11_500, "in_flight_ms": 9_000,
                      "without_model_ms": {"startup": 1_000, "setup": 1_000, "cases": 500}}
    assert describe(timing) == "250.0 ms per case, 2.5 s in all for 10 cases: startup 1.0 s, setup 1.0 s, cases 0.5 s"


def test_the_scores_show_the_time_without_the_model_and_say_the_inference_time_is_not_scored():
    timing = {"cases": 2, "workers": 8, "wall_ms": 3_000, "in_flight_ms": 2_000,
              "without_model_ms": {"startup": 800, "setup": 100, "cases": 100}}

    scores = report(GOLD, PREDICTIONS, timing=timing)

    assert "processing time  500.0 ms per case, 1.0 s in all for 2 cases: startup 0.8 s, setup 0.1 s, cases 0.1 s" in scores
    assert "8 in parallel)" in scores
    assert "inference time   900 ms mean, 900 ms p95 (not scored" in scores
    assert "not measured: the run has no time.json" in report(GOLD, PREDICTIONS)  # a run from before
    continued = report(GOLD, PREDICTIONS, timing={**timing, "cases": 1})
    assert "measured when the run was last continued, on 1 of its 2 cases" in continued

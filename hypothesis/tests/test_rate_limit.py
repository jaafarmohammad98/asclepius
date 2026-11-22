"""Requests to PubMed stay under NCBI's limit: 3 a second, or 10 with an API key."""

from hypothesis_engine.pubmed import RateLimiter, rate_for


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.now += seconds


def _most_in_any_second(times: list[float]) -> int:
    return max(sum(1 for t in times if start <= t < start + 1.0) for start in times)


def _request_times(per_second: int, count: int, clock: FakeClock) -> list[float]:
    limiter = RateLimiter(per_second, clock=clock.time, sleep=clock.sleep)
    times = []
    for _ in range(count):
        limiter.wait()
        times.append(clock.now)
    return times


def test_without_a_key_no_second_holds_more_than_three_requests():
    assert _most_in_any_second(_request_times(3, 20, FakeClock())) <= 3


def test_with_a_key_no_second_holds_more_than_ten_requests():
    assert _most_in_any_second(_request_times(10, 50, FakeClock())) <= 10


def test_the_first_request_goes_at_once():
    clock = FakeClock()
    RateLimiter(3, clock=clock.time, sleep=clock.sleep).wait()
    assert clock.now == 1000.0


def test_time_already_spent_between_requests_counts():
    clock = FakeClock()
    limiter = RateLimiter(3, clock=clock.time, sleep=clock.sleep)
    limiter.wait()
    clock.now += 5.0  # parsing a big response took a while
    limiter.wait()
    assert clock.now == 1005.0


def test_rate_depends_on_whether_an_api_key_is_given():
    assert rate_for(None) == 3
    assert rate_for("") == 3
    assert rate_for("abc123") == 10

import threading

from ucc.shard_queue import CapacityGauge


def test_cap_blocks_and_release_frees():
    stop = threading.Event()
    gauge = CapacityGauge(2)
    assert gauge.acquire(stop)
    assert gauge.acquire(stop)
    assert gauge.occupied == 2
    # Queue full: acquire blocks; with stop set it returns False.
    stop.set()
    assert not gauge.acquire(stop, poll_s=0.01)
    stop.clear()
    gauge.release()
    assert gauge.acquire(stop)
    assert gauge.occupied == 2


def test_prime_accounts_for_crash_leftovers():
    stop = threading.Event()
    gauge = CapacityGauge(3)
    gauge.prime(2)  # two raw shards survived a crash
    assert gauge.occupied == 2
    assert gauge.acquire(stop)   # one slot left
    stop.set()
    assert not gauge.acquire(stop, poll_s=0.01)


def test_prime_over_capacity_drains_before_new_downloads():
    stop = threading.Event()
    gauge = CapacityGauge(2)
    gauge.prime(4)  # backlog above the cap
    assert gauge.occupied == 4
    stop.set()
    assert not gauge.acquire(stop, poll_s=0.01)  # nothing free
    stop.clear()
    gauge.release()  # 3 left — still over cap
    gauge.release()  # 2 left — at cap
    stop.set()
    assert not gauge.acquire(stop, poll_s=0.01)
    stop.clear()
    gauge.release()  # 1 left — below cap, a slot opens
    assert gauge.acquire(stop)

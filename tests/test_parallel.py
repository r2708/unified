"""processing.cpu_workers process-pool helper: serial/parallel parity."""

from functools import partial

from ucc.parallel import MIN_PARALLEL_ITEMS, parallel_map, resolve_cpu_workers
from ucc.processing.quality_filter import _content_signals
from ucc.processing.secrets_scan import _scan_content


def test_resolve_cpu_workers():
    assert resolve_cpu_workers(1) == 1
    assert resolve_cpu_workers(3) == 3
    assert resolve_cpu_workers(0) >= 1      # auto
    assert resolve_cpu_workers(None) == 1
    assert resolve_cpu_workers(-2) == 1
    assert resolve_cpu_workers("junk") == 1


def test_small_inputs_run_serially():
    items = [("def f():\n    return 1\n", True)] * 10
    fn = partial(_content_signals, base64_run=2048)
    assert parallel_map(fn, items, cpu_workers=8) == [fn(it) for it in items]


def test_parallel_map_matches_serial_and_preserves_order():
    """Above the size threshold the pool is used; results must be identical
    to serial, in input order. Uses the real quality/secrets workers."""
    n = MIN_PARALLEL_ITEMS + 64
    q_items = [
        (f"import os\nvalue_{i} = {i}\nprint(value_{i})\n", i % 3 != 0)
        for i in range(n)
    ]
    q_fn = partial(_content_signals, base64_run=2048)
    assert parallel_map(q_fn, q_items, cpu_workers=2) == [q_fn(it) for it in q_items]

    s_items = [
        f"# maintainer {i}: dev{i}@example-corp.net\nx = {i}\n" for i in range(n)
    ]
    s_fn = partial(_scan_content, redact_emails=True, redact_ips=True)
    assert parallel_map(s_fn, s_items, cpu_workers=2) == [s_fn(it) for it in s_items]

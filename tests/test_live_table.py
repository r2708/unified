"""Live status table: rendering content and inert fallback."""

from __future__ import annotations

import logging

import pytest

from ucc.live_table import LiveStatusTable, render_status_table
from ucc.logging_utils import set_console_log_level, setup_logging

rich = pytest.importorskip("rich")
from rich.console import Console  # noqa: E402


SNAP = {
    "counts": {"completed": 3, "pending": 7, "downloading": 2},
    "pending": 7,
    "in_flight": 2,
    "retryable": 1,
    "given_up": 0,
    "active_shards": "starcoderdata-000002 (downloading)",
    "done": 3,
    "total": 12,
    "records_out": 123_456,
    "tokens": 9_876_543,
    "slots_used": 2,
    "slots_max": 5,
}


def _render_text(snap, tmp_path, elapsed=100.0) -> str:
    console = Console(record=True, width=100)
    console.print(render_status_table(snap, tmp_path, elapsed))
    return console.export_text()


def test_table_shows_key_metrics(tmp_path):
    text = _render_text(SNAP, tmp_path)
    assert "Unified Code Corpus Pipeline" in text
    assert "3/12 (25.0%)" in text
    assert "starcoderdata-000002 (downloading)" in text
    assert "123,456" in text
    assert "9,876,543" in text
    assert "2/5" in text
    # 123456 records / 100s elapsed
    assert "1,234.6 rec/s" in text


def test_table_renders_empty_snapshot(tmp_path):
    text = _render_text({}, tmp_path, elapsed=0.0)
    assert "0/0" in text
    assert "n/a" in text  # ETA unknown at start


def test_disabled_table_is_inert(tmp_path):
    live = LiveStatusTable(tmp_path, enabled=False)
    assert not live.enabled
    live.start()
    live.update(SNAP)  # must not raise
    live.stop()


def test_set_console_log_level_targets_only_console(tmp_path):
    setup_logging(tmp_path / "logs")
    root = logging.getLogger("ucc")
    console = [h for h in root.handlers if getattr(h, "_ucc_console", False)]
    files = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    assert console and files
    set_console_log_level(logging.WARNING)
    assert all(h.level == logging.WARNING for h in console)
    assert all(h.level != logging.WARNING for h in files)
    set_console_log_level(logging.INFO)
    assert all(h.level == logging.INFO for h in console)

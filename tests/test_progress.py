from ucc.progress import Progress, fmt_eta


class CaptureLog:
    def __init__(self):
        self.lines: list[str] = []

    def info(self, fmt, *args):
        self.lines.append(fmt % args)


def test_percentage_lines_and_final_close():
    log = CaptureLog()
    p = Progress(log, "stage_x", total=200, min_interval_s=0.0, check_every=1)
    p.update(50)
    p.update(50)
    p.update(100)
    p.close()
    assert any("25.0%" in line for line in log.lines)
    assert any("50.0%" in line for line in log.lines)
    assert log.lines[-1].startswith("stage_x: 100.0%")
    assert "(200/200 records" in log.lines[-1].replace("  ", " ") or "200/200" in log.lines[-1]


def test_count_only_mode_when_total_unknown():
    log = CaptureLog()
    p = Progress(log, "normalize", total=None, min_interval_s=0.0, check_every=1)
    p.update(1234)
    p.close()
    assert any("1,234" in line and "%" not in line for line in log.lines)


def test_throttling_suppresses_spam():
    log = CaptureLog()
    p = Progress(log, "busy", total=100_000, min_interval_s=3600.0, check_every=1)
    for _ in range(1000):
        p.update(1)
    assert log.lines == []          # inside the throttle window: silent
    p.close()                        # ...but close always emits
    assert len(log.lines) == 1


def test_fmt_eta():
    assert fmt_eta(42) == "42s"
    assert fmt_eta(125) == "2m05s"
    assert fmt_eta(7300) == "2h01m"

"""
Duplicate stdout/stderr to a UTF-8 log file while a step script runs.

Each pipeline step wraps `main()` in `tee_stdio_to_file(...)` so you get live console output
plus a persistent transcript under `webp/logs/{DOH|PhilGEPS}/`. That file is what you attach
when debugging “it failed on step 05” without rerunning everything.

Implementation: temporarily replace `sys.stdout` / `sys.stderr` with `_Tee`, which forwards
each `write()` to both the original stream and the open log handle. Restored in `finally`
so a crash still puts the console back to normal.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Generator, TextIO


class _Tee:
    """Write-through to multiple text streams (console + log file)."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        n = 0
        for s in self._streams:
            n = s.write(data)
            s.flush()
        return n

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


@contextmanager
def tee_stdio_to_file(log_path: str) -> Generator[None, None, None]:
    """Context manager: tee stdout/stderr to `log_path` until the block exits."""
    import os

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_fh = open(log_path, "w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = _Tee(old_out, log_fh)  # type: ignore[assignment]
        sys.stderr = _Tee(old_err, log_fh)  # type: ignore[assignment]
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        log_fh.close()

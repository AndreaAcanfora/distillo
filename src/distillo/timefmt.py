"""Timestamp formatting and parsing (``mm:ss`` or ``h:mm:ss``)."""

from __future__ import annotations

import re

_CLOCK = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d+))?\s*$")


def format_clock(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_clock(value: str) -> float:
    """Parse ``mm:ss``, ``h:mm:ss`` and subtitle forms like ``00:01:02,500``."""
    match = _CLOCK.match(value)
    if not match:
        raise ValueError(f"not a timestamp: {value!r}")
    hours, minutes, secs, fraction = match.groups()
    result = int(hours or 0) * 3600 + int(minutes) * 60 + int(secs)
    if fraction:
        result += int(fraction) / 10 ** len(fraction)
    return float(result)

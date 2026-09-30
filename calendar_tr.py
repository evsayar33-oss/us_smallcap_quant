"""NYSE session calendar (module name kept as calendar_tr for engine compatibility).

Rules (NYSE full-day closures): New Year's Day, MLK Day (3rd Mon Jan), Presidents' Day
(3rd Mon Feb), Good Friday, Memorial Day (last Mon May), Juneteenth (Jun 19, from 2022),
Independence Day, Labor Day (1st Mon Sep), Thanksgiving (4th Thu Nov), Christmas.
Saturday holidays -> Friday before, Sunday -> Monday after (New Year on Saturday is NOT moved).
Half days are trading sessions. Add one-off closures (e.g. national mourning days) to EXTRA_CLOSURES.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, List

import numpy as np
import pandas as pd

EXTRA_CLOSURES: List[str] = ["2018-12-05", "2025-01-09"]   # Bush Sr. and Carter national days of mourning


def _easter(y: int) -> date:
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(y, month, day)


def _nth_weekday(y, m, wd, n):
    d = date(y, m, 1)
    d += timedelta(days=(wd - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(y, m, wd):
    d = date(y, m + 1, 1) - timedelta(days=1) if m < 12 else date(y, 12, 31)
    return d - timedelta(days=(d.weekday() - wd) % 7)


def _observed(d: date, allow_back_to_friday: bool = True) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1) if allow_back_to_friday else None
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _holiday_set() -> set:
    out = set()
    for y in range(2005, 2032):
        ny = _observed(date(y, 1, 1), allow_back_to_friday=False)
        if ny:
            out.add(ny)
        out.add(_nth_weekday(y, 1, 0, 3))                 # MLK
        out.add(_nth_weekday(y, 2, 0, 3))                 # Presidents
        out.add(_easter(y) - timedelta(days=2))           # Good Friday
        out.add(_last_weekday(y, 5, 0))                   # Memorial
        if y >= 2022:
            out.add(_observed(date(y, 6, 19)))            # Juneteenth
        out.add(_observed(date(y, 7, 4)))                 # Independence
        out.add(_nth_weekday(y, 9, 0, 1))                 # Labor
        out.add(_nth_weekday(y, 11, 3, 4))                # Thanksgiving
        out.add(_observed(date(y, 12, 25)))               # Christmas
    for s_ in EXTRA_CLOSURES:
        out.add(pd.Timestamp(s_).date())
    return out


HOLIDAYS = _holiday_set()


def _d(x) -> date:
    return pd.Timestamp(x).date()


def is_session(x) -> bool:
    d = _d(x)
    return d.weekday() < 5 and d not in HOLIDAYS


def sessions_between(start, end) -> List[pd.Timestamp]:
    """All sessions in [start, end]."""
    s, e = _d(start), _d(end)
    out = []
    cur = s
    while cur <= e:
        if is_session(cur):
            out.append(pd.Timestamp(cur))
        cur += timedelta(days=1)
    return out


def add_sessions(x, n: int) -> pd.Timestamp:
    """The n-th session after x (n>=1)."""
    cur = _d(x)
    k = 0
    while k < n:
        cur += timedelta(days=1)
        if is_session(cur):
            k += 1
    return pd.Timestamp(cur)


def sessions_after_count(d0, d1) -> int:
    """Number of sessions in (d0, d1]."""
    a, b = _d(d0), _d(d1)
    if b <= a:
        return 0
    return len(sessions_between(a + timedelta(days=1), b))


def missing_sessions(dates: Iterable) -> List[pd.Timestamp]:
    """Sessions between the first and last stored date that have no snapshot."""
    ds = sorted({_d(x) for x in dates})
    if len(ds) < 2:
        return []
    have = set(ds)
    return [t for t in sessions_between(ds[0], ds[-1]) if t.date() not in have]


def today_tr() -> pd.Timestamp:
    """Today's date in New York (the exchange's calendar day)."""
    return pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)

#!/usr/bin/env python3
"""
hek_date.py — Date and time utilities
======================================
A collection of helpers built on top of the standard ``datetime``, ``calendar``,
and ``zoneinfo`` modules, plus optional ``dateutil`` support.

Parsing helpers
---------------
``Date("26-01-2016")``              → ``datetime.date``
``Datetime("26-01-2016 09:43:12")`` → ``datetime.datetime``
``Time("17:24:55")``                → ``datetime.datetime`` (date part is epoch)
``TimeWithMs("17:24:55,765")``      → ``datetime.datetime`` (with milliseconds)
``Timedelta("17:24:55")``           → ``datetime.timedelta``

Timedelta unit literals
-----------------------
Arithmetic with natural English notation::

    3*DAYS + 7*HOURS + 45*MINUTES   → datetime.timedelta(days=3, seconds=27900)
    2*WEEKS                         → datetime.timedelta(weeks=2)
"""

import calendar
import datetime
import time as _time
from zoneinfo import ZoneInfo


# ---------------------------------------------------------------------------
# Epoch — used as a zero reference for timedelta formatting
# ---------------------------------------------------------------------------

_EPOCH = datetime.datetime(1900, 1, 1)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def Date(d: str) -> datetime.date:
    """Parse ``'DD-MM-YYYY'`` into a ``datetime.date``.

    Example::

        Date("26-01-2016")  # → datetime.date(2016, 1, 26)
    """
    return datetime.datetime.strptime(d, "%d-%m-%Y").date()


def Datetime(d: str) -> datetime.datetime:
    """Parse ``'DD-MM-YYYY HH:MM:SS'`` into a ``datetime.datetime``.

    Example::

        Datetime("26-01-2016 09:43:12")  # → datetime.datetime(2016, 1, 26, 9, 43, 12)
    """
    return datetime.datetime.strptime(d, "%d-%m-%Y %H:%M:%S")


def Time(d: str) -> datetime.datetime:
    """Parse ``'HH:MM:SS'`` into a ``datetime.datetime`` (date part is the epoch).

    Example::

        Time("17:24:55")
    """
    return datetime.datetime.strptime(d, "%H:%M:%S")


def TimeWithMs(d: str) -> datetime.datetime:
    """Parse ``'HH:MM:SS,fff'`` (comma-separated milliseconds) into a ``datetime.datetime``.

    Example::

        TimeWithMs("17:24:55,765")
    """
    return datetime.datetime.strptime(d, "%H:%M:%S,%f")


def Timedelta(d: str) -> datetime.timedelta:
    """Parse a ``'HH:MM:SS'`` string into a ``datetime.timedelta``.

    Example::

        Timedelta("17:24:55")  # → datetime.timedelta(seconds=62695)
    """
    return Time(d) - _EPOCH


def FormattableTimedelta(d: datetime.timedelta) -> datetime.datetime:
    """Return a ``datetime.datetime`` so that a ``timedelta`` can be formatted with ``strftime``.

    ``datetime.timedelta`` has no ``strftime`` method.  Adding it to the epoch
    produces a ``datetime.datetime`` that can be formatted normally::

        FormattableTimedelta(Timedelta("17:24:55")).strftime("%H:%M:%S")
    """
    return _EPOCH + d


def unixtime(d: datetime.datetime) -> float:
    """Return the Unix timestamp for a ``datetime`` object."""
    return _time.mktime(d.timetuple())


# ---------------------------------------------------------------------------
# hDate — date whose + operator adds whole years
# ---------------------------------------------------------------------------

class hDate(datetime.date):
    """A ``datetime.date`` subclass where ``date + n`` adds *n* whole years.

    Note: adding a year to Feb 29 raises ``ValueError`` because the target year
    may not be a leap year.  Use ``try/except`` or adjust the day beforehand.
    """

    def __add__(self, years: int) -> 'hDate':
        return self.replace(year=self.year + years)


# ---------------------------------------------------------------------------
# Timedelta unit literals
# ---------------------------------------------------------------------------
# Each unit object implements __rmul__ so that ``n * UNIT`` returns a timedelta.

class _Unit:
    """Helper: supports ``n * UNIT`` syntax returning a ``datetime.timedelta``."""

    def __init__(self, **timedelta_kwargs):
        self._kwargs = timedelta_kwargs

    def __rmul__(self, n: float) -> datetime.timedelta:
        # Scale the keyword argument values by n
        return datetime.timedelta(**{k: v * n for k, v in self._kwargs.items()})


DAY = DAYS       = _Unit(days=1)
HOUR = HOURS     = _Unit(seconds=3600)
MIN = MINUTE = MINUTES = _Unit(seconds=60)
WEEK = WEEKS     = _Unit(weeks=1)
MONTH            = 30 * DAYS    # approximate; use dateutil.relativedelta for exact months
YEAR             = _Unit(days=365.2425)   # mean Gregorian year


# ---------------------------------------------------------------------------
# Convenience snapshots (evaluated at import time)
# ---------------------------------------------------------------------------

TODAY = datetime.date.today()
NOW   = datetime.datetime.now()


# ---------------------------------------------------------------------------
# Date utility functions
# ---------------------------------------------------------------------------

def day_of_year(d: datetime.date) -> int:
    """Return the day-of-year (1–366) for *d*."""
    return d.timetuple().tm_yday


def week_number(d: datetime.date) -> int:
    """Return the ISO week number for *d*."""
    return d.isocalendar()[1]


def seconds_of_year(d: datetime.datetime) -> float:
    """Return the number of seconds elapsed since the start of *d*'s year."""
    return (d - datetime.datetime(d.year, 1, 1)).total_seconds()


def days_in_month(month: int, year: int) -> int:
    """Return the number of days in *month* of *year*.

    Uses the bit-trick ``1 << m & 5546`` to identify 31-day months without a
    lookup table.  ``5546 == 0b1010110101010`` encodes months 1–12 with a ``1``
    bit for each month that has 31 days::

        bin(5546) = '0b1010110101010'
        positions (1-indexed from right): 1,3,5,7,8,10,12  → Jan,Mar,May,Jul,Aug,Oct,Dec

    February is handled separately to account for leap years.
    """
    if month == 2:
        return 29 if calendar.isleap(year) else 28
    return 30 + bool(1 << month & 5546)


# ---------------------------------------------------------------------------
# nth-weekday-in-month finder
# ---------------------------------------------------------------------------

# Ordinal selectors for use with dow_date_finder
FIRST  = 0
SECOND = 1
THIRD  = 2
FOURTH = FORTH = 3   # "FORTH" kept as alias for the typo-prone
FIFTH  = 4
LAST   = -1


def dow_date_finder(which: int, weekday: int, month: int, year: int) -> int:
    """Return the day-of-month for the *which*-th occurrence of *weekday* in *month*/*year*.

    Args:
        which:   Ordinal selector — use ``FIRST``, ``SECOND``, …, ``LAST``.
        weekday: ``calendar.MONDAY`` … ``calendar.SUNDAY``.
        month:   1–12.
        year:    Four-digit year.

    Example::

        # Second Thursday of each month in 2010:
        dow_date_finder(SECOND, calendar.THURSDAY, 3, 2010)  # → 11
    """
    first_weekday_of_month, days_in_m = calendar.monthrange(year, month)
    first_match = (weekday - first_weekday_of_month) % 7 + 1
    return list(range(first_match, days_in_m + 1, 7))[which]


# ---------------------------------------------------------------------------
# Week boundaries
# ---------------------------------------------------------------------------

def week_boundaries(year: int, week: int) -> tuple[datetime.date, datetime.date]:
    """Return the Monday and the following Monday for ISO *week* of *year*.

    Example::

        week_boundaries(2010, 13)
        # → (datetime.date(2010, 3, 29), datetime.date(2010, 4, 5))
    """
    start_of_year = datetime.date(year, 1, 1)
    week0 = start_of_year - datetime.timedelta(days=start_of_year.isoweekday())
    monday = week0 + datetime.timedelta(weeks=week, days=1)
    return monday, monday + datetime.timedelta(days=7)


# ---------------------------------------------------------------------------
# Human-readable elapsed time
# ---------------------------------------------------------------------------

_DEFAULT_SUFFIXES = ['y', 'w', 'd', 'h', 'm', 's']

_UNIT_SECONDS = [
    60 * 60 * 24 * 7 * 52,   # year  (52 weeks)
    60 * 60 * 24 * 7,         # week
    60 * 60 * 24,             # day
    60 * 60,                  # hour
    60,                       # minute
    1,                        # second
]


def elapsed_time(
    seconds: int,
    suffixes: list[str] | None = None,
    add_s: bool = False,
    separator: str = ' ',
) -> str:
    """Format *seconds* as a human-readable elapsed-time string.

    Args:
        seconds:   Total number of seconds (integer).
        suffixes:  Six labels for [year, week, day, hour, minute, second].
                   Defaults to ``['y', 'w', 'd', 'h', 'm', 's']``.
        add_s:     When ``True``, append ``'s'`` to plural values.
        separator: String placed between time components.

    Example::

        elapsed_time(90061)              # → '1d 1h 1m 1s'
        elapsed_time(90061, add_s=True)  # → '1d 1h 1m 1s'  (no s on single-char suffixes)
    """
    if suffixes is None:
        suffixes = _DEFAULT_SUFFIXES

    parts = list(zip(suffixes, _UNIT_SECONDS))
    result = []
    for suffix, length in parts:
        value = seconds // length
        if value > 0:
            seconds %= length
            plural_suffix = (suffix + 's') if (add_s and value > 1) else suffix
            result.append(f'{value}{plural_suffix}')
        if seconds < 1:
            break
    return separator.join(result)


# ---------------------------------------------------------------------------
# Date range generator
# ---------------------------------------------------------------------------

def daterange(
    start: datetime.date,
    end: datetime.date,
    delta: datetime.timedelta = datetime.timedelta(days=1),
):
    """Yield dates from *start* up to (but not including) *end*, stepping by *delta*.

    Example::

        list(daterange(datetime.date(2007, 3, 30), datetime.date(2007, 4, 3)))
        # → [date(2007,3,30), date(2007,3,31), date(2007,4,1), date(2007,4,2)]
    """
    current = start
    while current < end:
        yield current
        current += delta


# ---------------------------------------------------------------------------
# Timezone utilities  (uses stdlib zoneinfo, Python 3.9+)
# ---------------------------------------------------------------------------

def get_timezone_offset(tz_name: str, d: datetime.datetime | None = None) -> float:
    """Return the UTC offset in hours for timezone *tz_name* at datetime *d*.

    Uses ``zoneinfo.ZoneInfo`` (stdlib since Python 3.9) rather than ``pytz``.
    Handles DST automatically.

    Args:
        tz_name: IANA timezone name, e.g. ``'Europe/Brussels'``.
        d:       The datetime to evaluate at.  Defaults to now.

    Example::

        get_timezone_offset('Europe/Brussels')  # → 1.0 or 2.0 depending on DST
        get_timezone_offset('US/Pacific')        # → -8.0 or -7.0
    """
    if d is None:
        d = datetime.datetime.now()
    aware = d.replace(tzinfo=ZoneInfo(tz_name))
    return aware.utcoffset().total_seconds() / 3600


# ---------------------------------------------------------------------------
# Hour — time-of-day arithmetic using HH.MM float encoding
# ---------------------------------------------------------------------------

class Hour(float):
    """A time-of-day value stored as a ``float`` in ``HH.MM`` format.

    The integer part is hours; the fractional part (× 100) is minutes.
    For example ``Hour(9.30)`` represents 09:30, and ``Hour(92.14)``
    represents 92 hours and 14 minutes (useful for weekly totals).

    Arithmetic preserves the HH.MM encoding::

        Hour(9.30) + Hour(1.45)   # → Hour(11.15)  i.e. 11:15
        Hour(38.45) / 7.75        # → 5.0  (decimal hours, plain float)
    """

    def __new__(cls, value: float) -> 'Hour':
        hours = int(value)
        minutes = round((value - hours) * 100)   # e.g. 0.14 × 100 = 14 min
        # Re-encode as HH + MM/100 for clean float storage
        return float.__new__(cls, hours + minutes / 100)

    def get_hour(self) -> int:
        """Return the whole-hours component."""
        return int(self)

    def get_min(self) -> int:
        """Return the minutes component (0–59)."""
        return round((float(self) - self.get_hour()) * 100)

    def __add__(self, other: 'Hour') -> 'Hour':
        h = self.get_hour() + other.get_hour()
        m = self.get_min()  + other.get_min()
        carry, m = divmod(m, 60)
        return Hour(h + carry + m / 100)

    def __sub__(self, other: 'Hour') -> 'Hour':
        h = self.get_hour() - other.get_hour()
        m = self.get_min()  - other.get_min()
        if m < 0:
            m += 60
            h -= 1
        return Hour(h + m / 100)

    def __truediv__(self, other: float) -> float:
        """Divide this duration by *other*, returning a plain float (decimal hours)."""
        return (self.get_hour() + self.get_min() / 60) / other

    def __str__(self) -> str:
        return f'{self.get_hour()}:{self.get_min():02d}'

    def __repr__(self) -> str:
        return f'Hour({self.get_hour()}:{self.get_min():02d})'


# ---------------------------------------------------------------------------
# YearMonthDate — a date where the day is optional / ambiguous
# ---------------------------------------------------------------------------

class YearMonthDate(datetime.date):
    """A ``datetime.date`` subclass where the day is optional.

    Useful when only year and month are known (e.g. a billing period).
    The stored day defaults to 1 but is considered ambiguous.

    Example::

        ym = YearMonthDate(2024, 3)       # March 2024, day=1 (ambiguous)
        ym = YearMonthDate.from_date(datetime.date.today())
    """

    ambiguous_day: bool = True

    def __new__(cls, year: int, month: int, day: int | None = None) -> 'YearMonthDate':
        return super().__new__(cls, year, month, day if day is not None else 1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.year!r}, {self.month!r})'

    @classmethod
    def from_date(cls, date: datetime.date) -> 'YearMonthDate':
        """Construct from an existing ``datetime.date``."""
        return cls(date.year, date.month, date.day)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # -- hDate + timedelta units ---------------------------------------------
    print("=== hDate and timedelta units ===")
    BIRTHDATE = hDate(1963, 12, 28)
    age = TODAY - BIRTHDATE
    print(f"TODAY          : {TODAY}")
    print(f"TODAY + 2*DAYS : {TODAY + 2*DAYS}")
    print(f"Next 5 days    : {[TODAY + i*DAYS for i in range(5)]}")
    print(f"Age            : {age}")
    print(f"Age in years   : {age.total_seconds() / (365.2425 * 24 * 3600):.2f}")
    print(f"Retire at 66   : {BIRTHDATE + 66}")

    delta = 3*DAYS + 7*HOURS + 45*MINUTES
    assert str(delta) == '3 days, 7:45:00', str(delta)

    a = Datetime("02-10-2009 17:54:12")
    b = Datetime("26-10-2009 17:54:12")
    assert str(b - a) == '24 days, 0:00:00'

    c = Date("26-10-2009")
    d = Date("26-12-2009")
    assert str(d - c) == '61 days, 0:00:00'

    a = Datetime('20-10-2009 10:30:00')
    b = Datetime('20-10-2009 10:34:30')
    assert str(b - a) == '0:04:30'

    a = TimeWithMs("13:50:59,1")
    b = TimeWithMs("13:51:08,5")
    print(f"TimeWithMs diff: {b - a}")
    print("hDate + units: OK")

    # -- Parsing helpers -----------------------------------------------------
    print("\n=== Parsing helpers ===")
    iso_dt = datetime.datetime.strptime("2007-03-04T21:08:12", "%Y-%m-%dT%H:%M:%S")
    assert iso_dt == datetime.datetime(2007, 3, 4, 21, 8, 12)

    a = datetime.datetime.strptime('8/18/2008', "%m/%d/%Y")
    b = datetime.datetime.strptime('9/26/2008', "%m/%d/%Y")
    assert (b - a).days == 39

    a = datetime.datetime.strptime('17:05:41', "%H:%M:%S")
    b = datetime.datetime.strptime('18:08:42', "%H:%M:%S")
    assert str(b - a) == '1:03:01'

    _, _, _, hour, minute, sec, *_ = _time.strptime('17:05:41', "%H:%M:%S")
    assert (hour, minute, sec) == (17, 5, 41)
    print("Parsing helpers: OK")

    # -- Date formatting -----------------------------------------------------
    print("\n=== Date formatting ===")
    d = datetime.date(2011, 12, 16)
    print(f"  strftime : {d.strftime('%A %d %B %Y')}")        # locale-dependent
    dt = datetime.datetime(2010, 7, 4, 12, 15, 58)
    print(f"  f-string : {dt:%Y-%m-%d %H:%M:%S}")

    # -- days_in_month -------------------------------------------------------
    print("\n=== days_in_month ===")
    assert [days_in_month(m, 2012) for m in range(1, 13)] == \
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    assert [calendar.monthrange(2012, m)[1] for m in range(1, 13)] == \
        [days_in_month(m, 2012) for m in range(1, 13)]
    print("days_in_month: OK")

    # -- calendar display ----------------------------------------------------
    print("\n=== Calendar display ===")
    print(calendar.month(2008, 1))
    cal = calendar.monthcalendar(2012, 1)
    assert cal == [
        [0,  0,  0,  0,  0,  0,  1],
        [2,  3,  4,  5,  6,  7,  8],
        [9,  10, 11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20, 21, 22],
        [23, 24, 25, 26, 27, 28, 29],
        [30, 31, 0,  0,  0,  0,  0],
    ]
    print("monthcalendar: OK")

    # -- dow_date_finder -----------------------------------------------------
    print("\n=== dow_date_finder ===")
    assert calendar.THURSDAY == 3
    # Second Thursday of each month in 2010 (locale-independent, day numbers only)
    second_thursdays = [dow_date_finder(SECOND, calendar.THURSDAY, m, 2010)
                        for m in range(1, 13)]
    assert second_thursdays == [14, 11, 11, 8, 13, 10, 8, 12, 9, 14, 11, 9], second_thursdays
    print(f"Second Thursdays 2010: {second_thursdays}")
    print("dow_date_finder: OK")

    # -- week_boundaries -----------------------------------------------------
    print("\n=== week_boundaries ===")
    assert week_boundaries(2010, 13) == (
        datetime.date(2010, 3, 29),
        datetime.date(2010, 4, 5),
    )
    print("week_boundaries: OK")

    # -- elapsed_time --------------------------------------------------------
    print("\n=== elapsed_time ===")
    seconds = (
        60*60*24*7*52*2   # 2 years
        + 60*60*24*7*1    # 1 week
        + 60*60*24*6      # 6 days
        + 60*60*2         # 2 hours
        + 60*59           # 59 minutes
        + 23              # 23 seconds
    )
    assert elapsed_time(seconds) == '2y 1w 6d 2h 59m 23s'
    assert elapsed_time(seconds, [' year',' week',' day',' hour',' minute',' second']) == \
        '2 year 1 week 6 day 2 hour 59 minute 23 second'
    assert elapsed_time(seconds, [' year',' week',' day',' hour',' minute',' second'], add_s=True) == \
        '2 years 1 week 6 days 2 hours 59 minutes 23 seconds'
    assert elapsed_time(seconds, [' year',' week',' day',' hour',' minute',' second'],
                        add_s=True, separator=', ') == \
        '2 years, 1 week, 6 days, 2 hours, 59 minutes, 23 seconds'
    print("elapsed_time: OK")

    # -- daterange -----------------------------------------------------------
    print("\n=== daterange ===")
    days = list(daterange(datetime.date(2007, 3, 30), datetime.date(2007, 4, 3)))
    assert days == [
        datetime.date(2007, 3, 30),
        datetime.date(2007, 3, 31),
        datetime.date(2007, 4, 1),
        datetime.date(2007, 4, 2),
    ]
    print(f"daterange: {days}")

    # -- Hour ----------------------------------------------------------------
    print("\n=== Hour ===")
    a = Hour(92.14)
    b = Hour(2.56)
    print(f"a         = {a}")          # 92:14
    print(f"b         = {b}")          # 2:56
    print(f"a + b     = {a + b}")      # 95:10
    print(f"a - b     = {a - b}")      # 89:18
    week = Hour(38.45)
    print(f"week / 5  = {week / 5:.2f} decimal hours")
    assert str(Hour(9, 30) if False else Hour(9.30)) == '9:30'

    # -- YearMonthDate -------------------------------------------------------
    print("\n=== YearMonthDate ===")
    ym = YearMonthDate(2024, 3)
    assert ym.year == 2024 and ym.month == 3 and ym.day == 1
    assert repr(ym) == 'YearMonthDate(2024, 3)'
    ym2 = YearMonthDate.from_date(datetime.date(2024, 6, 15))
    assert ym2.month == 6
    print(f"YearMonthDate: {ym!r}")

    # -- get_timezone_offset -------------------------------------------------
    print("\n=== Timezone offsets ===")
    d_summer = datetime.datetime(2024, 7, 1, 12, 0, 0)
    d_winter = datetime.datetime(2024, 1, 1, 12, 0, 0)
    bru_summer = get_timezone_offset('Europe/Brussels', d_summer)
    bru_winter = get_timezone_offset('Europe/Brussels', d_winter)
    assert bru_summer == 2.0, bru_summer   # CEST
    assert bru_winter == 1.0, bru_winter   # CET
    print(f"Brussels summer UTC offset: {bru_summer:+.0f}h")
    print(f"Brussels winter UTC offset: {bru_winter:+.0f}h")
    print("Timezone offsets: OK")

    # -- dateutil (optional) -------------------------------------------------
    try:
        from dateutil.parser import parse
        from dateutil.relativedelta import relativedelta
        time_today = parse("08:00")
        required_time = time_today - relativedelta(minutes=35)
        assert required_time.hour == 7 and required_time.minute == 25
        print("\ndateutil: OK")
    except ImportError:
        print("\ndateutil not installed — skipping dateutil tests")

    print("\nAll assertions passed.")

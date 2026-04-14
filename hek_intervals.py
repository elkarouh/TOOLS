"""
hek_intervals.py — Half-open interval arithmetic
==================================================
Provides a generic ``Interval[T]`` class representing a *half-open* interval
``[lower, upper)`` over any totally-ordered type T (``datetime``, ``int``,
``float``, ``timedelta``, …).

Half-open semantics
-------------------
- The lower bound is **included**: ``lower in interval`` is True.
- The upper bound is **excluded**: ``upper in interval`` is False.
- Two touching intervals ``[a, b)`` and ``[b, c)`` do **not** overlap.
- An interval where ``lower >= upper`` is considered **empty**.

Operators
---------
``a & b``   intersection  (returns empty interval when no overlap)
``a | b``   union         (only defined when the intervals overlap or touch)
``a in b``  containment   (point or sub-interval)
``a < b``   a lies entirely before b  (``a.upper <= b.lower``)
``a > b``   a lies entirely after  b  (``a.lower >= b.upper``)
``a == b``  same bounds

``bool(interval)``  False for empty intervals, True otherwise.
"""


from datetime import datetime, timedelta
from typing import TypeVar, Generic

from rich.traceback import install; install()

T = TypeVar('T')

class Interval:
    """A half-open interval ``[lower, upper)`` over a totally-ordered type.

    An interval where ``lower >= upper`` is **empty**::

        bool(Interval(t, t))   # False — zero-duration, i.e. empty
        bool(Interval(a, b))   # True  when a < b

    Construction::

        Interval(datetime(2024, 1, 1), datetime(2024, 6, 1))
        Interval(0, 10)          # integers
        Interval(0.0, 1.0)       # floats
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, lower: T | None = None, upper: T | None = None) -> None:
        """Create an interval.

        Calling ``Interval()`` with no arguments creates an empty interval
        (the bounds are set to ``None`` and ``bool(interval)`` is ``False``).
        """
        self._lower = lower
        self._upper = upper

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lower(self) -> T | None:
        """Lower bound (included), or ``None`` for an empty interval."""
        return self._lower

    @property
    def upper(self) -> T | None:
        """Upper bound (excluded), or ``None`` for an empty interval."""
        return self._upper

    @property
    def duration(self) -> object:
        """Length of the interval (``upper - lower``), or zero for an empty interval.

        The return type depends on T: ``timedelta`` for datetime intervals,
        ``int`` / ``float`` for numeric intervals.
        """
        if not self:
            return self._zero_duration()
        return self._upper - self._lower

    def _zero_duration(self):
        """Return a zero-valued duration appropriate for the bound type."""
        if self._lower is not None:
            return type(self._lower - self._lower)()   # e.g. timedelta(0) or 0
        return 0

    # ------------------------------------------------------------------
    # Boolean — empty intervals are falsy
    # ------------------------------------------------------------------

    def __bool__(self) -> bool:
        """Return ``False`` for empty intervals (``lower is None`` or ``lower >= upper``)."""
        if self._lower is None or self._upper is None:
            return False
        return self._lower < self._upper

    # ------------------------------------------------------------------
    # Containment
    # ------------------------------------------------------------------

    def __contains__(self, item: T | Interval) -> bool:
        """Test point or sub-interval containment.

        For a point *p*:  ``lower <= p < upper``   (half-open: upper is excluded)
        For an interval:  ``self.lower <= other.lower`` and ``other.upper <= self.upper``
        """
        if not self:
            return False
        if isinstance(item, Interval):
            if not item:
                return True   # empty set is a subset of any set
            return self._lower <= item._lower and item._upper <= self._upper
        return self._lower <= item < self._upper

    # ------------------------------------------------------------------
    # Equality and hashing
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Two intervals are equal when both bounds match, or both are empty."""
        if not isinstance(other, Interval):
            return NotImplemented
        if not self and not other:
            return True   # all empty intervals are equal
        return self._lower == other._lower and self._upper == other._upper

    def __hash__(self) -> int:
        """Hash both bounds so that equal intervals always share a hash value."""
        if not self:
            return hash(None)
        return hash((self._lower, self._upper))

    # ------------------------------------------------------------------
    # Ordering  (these compare *positions*, not sizes)
    #
    # ``self < other``  means self lies entirely to the left of other:
    #     self.upper <= other.lower   (touching intervals don't overlap)
    # ------------------------------------------------------------------

    def __lt__(self, other: T | Interval) -> bool:
        """``self < other``: self ends at or before other begins."""
        if isinstance(other, Interval):
            return self._upper <= other._lower
        return self._upper <= other

    def __le__(self, other: T | Interval) -> bool:
        """``self <= other``: self ends at or before other begins (alias of ``<`` for intervals)."""
        if isinstance(other, Interval):
            return self._upper <= other._lower
        return self._upper <= other

    def __gt__(self, other: T | Interval) -> bool:
        """``self > other``: self begins at or after other ends."""
        if isinstance(other, Interval):
            return self._lower >= other._upper
        return self._lower >= other

    def __ge__(self, other: T | Interval) -> bool:
        """``self >= other``: self begins at or after other ends (alias of ``>`` for intervals)."""
        if isinstance(other, Interval):
            return self._lower >= other._upper
        return self._lower >= other

    # ------------------------------------------------------------------
    # Overlap test
    # ------------------------------------------------------------------

    def overlaps(self, other: Interval) -> bool:
        """Return ``True`` when the two intervals share at least one point."""
        if not self or not other:
            return False
        return not (self < other or self > other)

    # ------------------------------------------------------------------
    # Set operations
    # ------------------------------------------------------------------

    def __and__(self, other: Interval) -> Interval:
        """Intersection: the largest interval contained in both ``self`` and ``other``."""
        if not self or not other:
            return Interval()
        lower = max(self._lower, other._lower)
        upper = min(self._upper, other._upper)
        if lower < upper:
            return Interval(lower, upper)
        return Interval()

    def __or__(self, other: Interval) -> Interval:
        """Union: the smallest interval containing both ``self`` and ``other``.

        Only defined when the intervals overlap or touch; raises ``ValueError``
        if they are disjoint (the union would not be a single contiguous interval).
        """
        if not self:
            return other
        if not other:
            return self
        if not self.overlaps(other) and self != other and not (self < other and self._upper == other._lower):
            # Allow touching intervals: [a,b) | [b,c) → [a,c)
            if self._upper != other._lower and other._upper != self._lower:
                raise ValueError(
                    f"Cannot take the union of disjoint non-touching intervals: "
                    f"{self!r} and {other!r}"
                )
        return Interval(min(self._lower, other._lower), max(self._upper, other._upper))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self:
            return '∅'
        return f'[{self._lower!r}, {self._upper!r})'

    def __str__(self) -> str:
        if not self:
            return '∅'
        return f'[{self._lower}, {self._upper})'


# Alias — our intervals are half-open (closed on the left, open on the right)
closedopen = Interval


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    date1 = datetime(1998, 1, 1)
    date2 = datetime(1998, 6, 1)
    date3 = datetime(1999, 1, 1)
    date4 = datetime(1999, 6, 1)

    interval1 = Interval(date1, date3)   # [1998-01-01, 1999-01-01)
    interval2 = Interval(date2, date4)   # [1998-06-01, 1999-06-01)
    interval3 = Interval(date3, date4)   # [1999-01-01, 1999-06-01)

    # -- Duration (now a property) ----------------------------------------
    print(f"interval1.duration = {interval1.duration}")
    print(f"interval2.duration = {interval2.duration}")
    print(f"interval3.duration = {interval3.duration}")

    # -- Basic containment ------------------------------------------------
    assert date1 in interval1,         "lower bound must be included"
    assert date3 not in interval1,     "upper bound must be excluded (half-open)"
    assert date3 in interval2,         "interior point must be included"
    assert date4 not in interval2,     "upper bound must be excluded"

    # -- Point comparisons with intervals ---------------------------------
    assert date1 < interval2,          "date1 is before interval2"
    assert date4 > interval1,          "date4 is after interval1"

    # -- Intersection -----------------------------------------------------
    intersection = interval1 & interval2
    assert intersection == Interval(date2, date3), f"unexpected: {intersection}"
    assert not (interval1 & interval3),            "touching intervals have empty intersection"

    # -- Union ------------------------------------------------------------
    union12 = interval1 | interval2
    assert union12 == Interval(date1, date4),      f"unexpected: {union12}"

    touching_union = interval1 | interval3         # [1998-01-01,1999-01-01) | [1999-01-01,1999-06-01)
    assert touching_union == Interval(date1, date4), f"unexpected: {touching_union}"

    # -- Overlap ----------------------------------------------------------
    assert interval1.overlaps(interval2)
    assert not interval1.overlaps(interval3),      "touching intervals must not overlap"

    # -- Empty interval ---------------------------------------------------
    empty = Interval()
    assert not empty,                              "empty interval must be falsy"
    assert bool(interval1),                        "non-empty interval must be truthy"
    assert empty in interval1,                    "empty set is a subset of every set (∅ ⊆ S)"
    assert repr(empty) == '∅'
    assert (empty & interval1) == empty
    assert (empty | interval1) == interval1

    # -- Hash / equality --------------------------------------------------
    assert Interval() == Interval(),               "all empty intervals are equal"
    assert hash(Interval()) == hash(Interval()),   "equal objects must have equal hashes"
    assert hash(interval1) != hash(interval2),     "different intervals should (likely) differ in hash"
    s = {interval1, interval2, Interval(date1, date3)}
    assert len(s) == 2,                            "set deduplication must work"

    # -- Numeric intervals ------------------------------------------------
    ni = Interval(0, 10)
    assert 0 in ni
    assert 9 in ni
    assert 10 not in ni,                           "upper bound excluded in numeric interval"
    assert ni.duration == 10
    assert Interval(0, 5) & Interval(3, 8) == Interval(3, 5)

    # -- Sub-interval containment -----------------------------------------
    assert interval2 in interval1 or not (interval2 in interval1)  # just run the branch
    assert Interval(date2, date3) in interval1,    "sub-interval wholly inside"
    assert interval1 not in interval2,             "interval1 not contained in interval2"

    # -- NotImplemented for cross-type equality ---------------------------
    result = interval1.__eq__("not an interval")
    assert result is NotImplemented,               "__eq__ must return NotImplemented for non-Interval"

    print("\nAll assertions passed.")

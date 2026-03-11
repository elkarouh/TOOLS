#!/usr/bin/env python3
"""
hek_rxe.py — Composable regex builder
======================================
Wraps Python's `regex` library (drop-in for `re`) with an operator-based
API so that complex patterns can be assembled from named, readable pieces
instead of raw strings.

Key concepts
------------
- An ``rxe`` instance holds a regex *pattern string* and exposes the standard
  ``re`` operations (``search``, ``match``, ``findall``, …) directly on the object.
- Patterns are composed with Python operators:

    ``p + q``          concatenation
    ``p | q``          character-class union  (builds ``[…]``)
    ``Either(p, q)``   alternation            (builds ``(?:p|q)``)
    ``~p``             negation of a character class or shorthand
    ``p[1:4]``         repetition  {1,4}  (slice syntax)
    ``3 * p``          exact repetition  {3}

- ``Group('name', p)`` / ``.as_('name')`` create named capture groups;
  ``MatchObject`` exposes captured groups as attributes.

Character-class constants (``DIGIT``, ``LETTER``, …) are ``rxe`` instances.
String constants (``DQ``, ``SQ``, ``COLON``) are plain strings and can be
added to patterns with ``+`` like any literal.

See the ``__main__`` block at the bottom for usage examples.
"""

import regex as re

re.DEFAULT_VERSION = re.VERSION1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_pattern(s: 're.Pattern | rxe | str') -> str:
    """Return the raw pattern string for *s*, escaping it if it is a plain string."""
    if isinstance(s, str):
        return re.escape(s)
    return s.pattern


def _escape_for_character_class(s: str) -> str:
    """Escape characters that are special *inside* a ``[…]`` character class."""
    return ''.join(re.escape(c) if c in r'\^[]' else c for c in s)


def _add_grouping(pattern: 'str | rxe') -> str:
    """Wrap *pattern* in a non-capturing group unless it is already atomic.

    A pattern is considered atomic (needs no wrapping) when it is:
    - a single (optionally escaped) character: ``a``, ``\\.``
    - already a character class: ``[a-z]``
    - already a group: ``(…)``
    """
    _no_grouping_needed = (
        r"(?P<char>^\\?.$)"
        r"|(?P<square>^\[(?:[^\n\[]|\\\[)*[^\\]\]$)"
        r"|(?P<braces>^\((?:[^\n\(]|\\\()*[^\\]\)$)"
    )
    s = str(pattern)
    if re.match(_no_grouping_needed, s):
        return s
    return '(?:' + s + ')'


# ---------------------------------------------------------------------------
# MatchObject — thin wrapper that exposes named groups as attributes
# ---------------------------------------------------------------------------

class MatchObject:
    """Wraps a ``re.Match`` and exposes named capture groups as attributes.

    Usage::

        if m := my_pattern.search(text):
            print(m.name_of_group)   # attribute access
            print(m[1])              # positional group
    """

    def __init__(self, match: 're.Match | None') -> None:
        self._match = match

    def __bool__(self) -> bool:
        return self._match is not None

    def __str__(self) -> str:
        return str(self._match)

    def __repr__(self) -> str:
        return repr(self._match)

    @property
    def raw(self):
        """The underlying ``re.Match`` object.

        Use this to access match methods such as ``.start()``,
        ``.end()``, or ``.string``.
        """
        return self._match

    def __getitem__(self, group: 'int | str') -> str:
        """Return a captured group by position (int) or name (str)."""
        return self._match[group]

    def __getattr__(self, name: str) -> object:
        """Expose named capture groups and ``re.Match`` attributes transparently."""
        match = object.__getattribute__(self, '_match')
        # Try as a named capture group first
        try:
            return match.group(name)
        except IndexError:
            pass
        # Fall back to an attribute on the underlying Match object
        try:
            return getattr(match, name)
        except AttributeError:
            raise AttributeError(f"No capture group or match attribute named {name!r}")


# ---------------------------------------------------------------------------
# rxe — the core pattern builder
# ---------------------------------------------------------------------------

class rxe:
    """A composable regular-expression pattern.

    Construct from a raw pattern string::

        p = rxe(r'\\d+')

    or use the provided constants and combinators::

        p = OneOrMore(DIGIT) + '.' + OneOrMore(DIGIT)
    """

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern

    # -- re operations -------------------------------------------------------

    def compile(self, *flags) -> 're.Pattern':
        """Return a compiled ``re.Pattern``."""
        return re.compile(self.pattern, *flags)

    def search(self, string: str, *args) -> MatchObject:
        """Search *string* for the pattern; return a :class:`MatchObject`."""
        return MatchObject(re.search(self.pattern, string, *args))

    def match(self, string: str, *args) -> MatchObject:
        """Match the pattern at the start of *string*; return a :class:`MatchObject`."""
        return MatchObject(re.match(self.pattern, string, *args))

    def split(self, string: str, *args) -> list[str]:
        """Split *string* by the pattern."""
        return re.split(self.pattern, string, *args)

    def findall(self, string: str, *args) -> list:
        """Return all non-overlapping matches in *string*."""
        return re.findall(self.pattern, string, *args)

    def finditer(self, string: str, *args):
        """Return an iterator of Match objects over *string*."""
        return re.finditer(self.pattern, string, *args)

    def sub(self, repl, string: str, *args) -> str:
        """Replace matches of the pattern in *string* with *repl*."""
        return re.sub(self.pattern, repl, string, *args)

    def subn(self, repl, string: str, *args) -> tuple[str, int]:
        """Like ``sub``, but also return the number of substitutions made."""
        return re.subn(self.pattern, repl, string, *args)

    # -- composition ---------------------------------------------------------

    def _raw_add(self, extra_pattern: str, left: bool = False) -> 'rxe':
        """Concatenate *extra_pattern* on the right (or left) of this pattern."""
        assert isinstance(extra_pattern, str)
        combined = extra_pattern + self.pattern if left else self.pattern + extra_pattern
        return rxe(combined)

    def add(self, s: 'rxe | str', left: bool = False) -> 'rxe':
        """Concatenate *s* (escaping it if it is a plain string)."""
        return self._raw_add(_get_pattern(s), left)

    def __add__(self, other: 'rxe | str') -> 'rxe':
        """``p + q`` — concatenation."""
        return self.add(other)

    def __radd__(self, other: 'rxe | str') -> 'rxe':
        """``q + p`` — concatenation (reflected)."""
        return self.add(other, left=True)

    def __rmul__(self, n: int) -> 'rxe':
        """``n * p`` — exact repetition ``{n}``."""
        return rxe(_add_grouping(self.pattern) + f'{{{n}}}')

    def __or__(self, other: 'rxe | str') -> 'rxe':
        """``p | q`` — character-class union, producing ``[…]``.

        Both operands must be single-character patterns or character classes.
        For alternation between multi-character patterns use :func:`Either`.
        """
        if isinstance(other, str):
            other_part = _escape_for_character_class(other)
        elif other.pattern.startswith('\\'):
            other_part = other.pattern          # e.g. \w, \d
        elif other.pattern.startswith('['):
            other_part = other.pattern[1:-1]    # strip outer brackets
        else:
            raise TypeError(
                f"Cannot build a character class from pattern {other.pattern!r}. "
                "Use Either() for alternation between multi-character patterns."
            )

        self_part = self.pattern[1:-1] if self.pattern.startswith('[') else self.pattern
        return rxe('[' + self_part + other_part + ']')

    def __invert__(self) -> 'rxe':
        """``~p`` — negate a character class or shorthand.

        Supports ``\\d``, ``\\w``, ``\\s``, ``\\b`` and any ``[…]`` pattern.
        """
        _negations = {r'\d': r'\D', r'\w': r'\W', r'\s': r'\S', r'\b': r'\B'}
        if self.pattern in _negations:
            return rxe(_negations[self.pattern])
        if self.pattern.startswith('['):
            if self.pattern.startswith('[^'):
                # Double-negate: remove the ^ to restore the positive class
                return rxe('[' + self.pattern[2:])
            return rxe('[^' + self.pattern[1:])
        raise NotImplementedError(
            f"Cannot negate pattern {self.pattern!r}. "
            "Only shorthand classes (\\d, \\w, \\s, \\b) and […] patterns are supported."
        )

    def __getitem__(self, item: 'int | slice') -> 'rxe':
        """Repetition via slice/index syntax.

        =========  ==============================
        Syntax     Meaning
        =========  ==============================
        ``p[n]``   exactly n  →  ``{n}``
        ``p[:]``   zero or more  →  ``*``
        ``p[1:]``  one or more  →  ``+``
        ``p[:1]``  optional  →  ``?``
        ``p[:n]``  at most n  →  ``{,n}``
        ``p[n:]``  at least n  →  ``{n,}``
        ``p[m:n]``  between m and n  →  ``{m,n}``
        =========  ==============================
        """
        if isinstance(item, int):
            return item * self
        start = 0 if item.start is None else item.start
        stop  = item.stop
        if stop is None:                         # p[n:]
            if start == 0:   return ZeroOrMore(self)
            if start == 1:   return OneOrMore(self)
            return AtLeast(start, self)
        if start == 0:                           # p[:n]
            if stop == 1:    return Optional(self)
            return AtMost(stop, self)
        if start == stop:                        # p[n:n]
            return start * self
        return Between(start, stop, self)        # p[m:n]

    # -- named / unnamed groups ----------------------------------------------

    def as_(self, name: str) -> 'rxe':
        """Wrap this pattern in a named capture group ``(?P<name>…)``."""
        return rxe(f'(?P<{name}>{self.pattern})')

    # -- lookahead / lookbehind ----------------------------------------------

    def followed_by(self, s: 'rxe | str') -> 'rxe':
        """Positive lookahead: match only when followed by *s*."""
        return self._raw_add('(?=' + _get_pattern(s) + ')')

    def not_followed_by(self, s: 'rxe | str') -> 'rxe':
        """Negative lookahead: match only when *not* followed by *s*."""
        return self._raw_add('(?!' + _get_pattern(s) + ')')

    def preceded_by(self, s: 'rxe | str') -> 'rxe':
        """Positive lookbehind: match only when preceded by *s*."""
        return self._raw_add('(?<=' + _get_pattern(s) + ')', left=True)

    def not_preceded_by(self, s: 'rxe | str') -> 'rxe':
        """Negative lookbehind: match only when *not* preceded by *s*."""
        return self._raw_add('(?<!' + _get_pattern(s) + ')', left=True)

    def filter_matches(self, s: 'rxe | str') -> 'rxe':
        """Skip (filter out) all occurrences of *s* using ``(*SKIP)(*F)``."""
        return self._raw_add(_get_pattern(s) + '(*SKIP)(*F)|', left=True)

    # -- quantifier modifiers ------------------------------------------------

    def lazy(self) -> 'rxe':
        """Make the preceding quantifier lazy (non-greedy)  ``…?``."""
        return self._raw_add('?')

    non_greedy = lazy   # alias

    def possessive(self) -> 'rxe':
        """Make the preceding quantifier possessive  ``…+``."""
        return self._raw_add('+')

    # -- display -------------------------------------------------------------

    def __str__(self) -> str:
        return self.pattern

    def __repr__(self) -> str:
        return self.pattern  # intentionally terse for snoop / debug output


# ---------------------------------------------------------------------------
# Combinators and quantifiers
# ---------------------------------------------------------------------------

#: ``L`` is an alias for constructing a literal raw-pattern rxe.
L = rxe

def AtLeast(n: int, s: 'rxe | str') -> rxe:
    """Match *s* at least *n* times: ``{n,}``."""
    return rxe(_add_grouping(_get_pattern(s)) + '{%d,}' % n)

def AtMost(n: int, s: 'rxe | str') -> rxe:
    """Match *s* at most *n* times: ``{,n}``."""
    return rxe(_add_grouping(_get_pattern(s)) + '{,%d}' % n)

def Between(lo: int, hi: int, s: 'rxe | str') -> rxe:
    """Match *s* between *lo* and *hi* times (inclusive): ``{lo,hi}``."""
    return rxe(_add_grouping(_get_pattern(s)) + '{%d,%d}' % (lo, hi))

def ZeroOrMore(s: 'rxe | str') -> rxe:
    """Match *s* zero or more times: ``*``."""
    return rxe(_add_grouping(_get_pattern(s)) + '*')

def OneOrMore(s: 'rxe | str') -> rxe:
    """Match *s* one or more times: ``+``."""
    return rxe(_add_grouping(_get_pattern(s)) + '+')

def Optional(s: 'rxe | str') -> rxe:
    """Match *s* zero or one time: ``?``."""
    return rxe(_add_grouping(_get_pattern(s)) + '?')

def Either(*patterns: 'rxe | str') -> rxe:
    """Match any one of the given patterns (alternation): ``(?:p1|p2|…)``."""
    return rxe(_add_grouping('|'.join(_get_pattern(p) for p in patterns)))

def Group(name: str, s: 'rxe | str') -> rxe:
    """Wrap *s* in a named capture group ``(?P<name>…)``."""
    return rxe(f'(?P<{name}>{_get_pattern(s)})')

def UnnamedGroup(s: 'rxe | str') -> rxe:
    """Wrap *s* in an unnamed capture group ``(…)``."""
    return rxe(f'({_get_pattern(s)})')

#: Short alias for :func:`UnnamedGroup`.
G = UnnamedGroup

def AtomicGroup(s: 'rxe | str') -> rxe:
    """Wrap *s* in an atomic (possessive) group ``(?>…)``."""
    return rxe(f'(?>{_get_pattern(s)})')

def Backref(name: str) -> rxe:
    """Insert a back-reference to the named group: ``(?P=name)``."""
    return rxe(f'(?P={name})')

#: Recursive pattern reference ``(?R)``.
RecursiveGroup = rxe('(?R)')


# ---------------------------------------------------------------------------
# Character-set helpers
# ---------------------------------------------------------------------------

def OneOf(*args: str) -> rxe:
    """Match any single character in the given string(s): ``[…]``.

    Accepts one or more strings whose characters are combined into a single
    character class.  Example: ``OneOf('aeiou')`` matches any vowel.
    """
    chars = ''.join(args)
    return rxe('[' + _escape_for_character_class(chars) + ']')

#: Alias for :func:`OneOf`.
AnyOf = OneOf

def AnythingBut(s: 'rxe | str') -> rxe:
    """Match any single character *not* in *s*.

    If *s* is a string, builds a negated character class ``[^…]``.
    If *s* is an ``rxe`` character class or shorthand, applies ``~`` instead.
    """
    if isinstance(s, str):
        return rxe('[^' + _escape_for_character_class(s) + ']')
    return ~s

#: Alias for :func:`AnythingBut`.
AnyBut = AnythingBut


# ---------------------------------------------------------------------------
# Position / zero-width assertions
# ---------------------------------------------------------------------------

WB = WORDBOUNDARY   = rxe(r'\b')
SOL = START_OF_LINE = rxe(r'^')
EOL = END_OF_LINE   = rxe(r'$')
NL                  = rxe(r'\n')
START_OF_STRING     = rxe(r'\A')
START_OF_STRING2    = rxe(r'\G')
END_OF_STRING       = rxe(r'\Z')


# ---------------------------------------------------------------------------
# Single-character patterns
# ---------------------------------------------------------------------------

ANYCHAR             = L('.')
DIGIT = DIGITS      = rxe(r'\d')
ALPHANUMERIC        = rxe(r'\w')
WS = WHITESPACE     = rxe(r'\s')
LETTER = LETTERS    = rxe(r'[a-zA-Z]')
UPPERCASE           = rxe(r'[A-Z0-9]')
DOT                 = rxe(r'\.')

#: Plain-string constants — usable directly in ``+`` concatenation.
DQ = DOUBLEQUOTE    = '"'
SQ = SINGLEQUOTE    = "'"
COLON               = ":"


# ---------------------------------------------------------------------------
# Multi-character patterns
# ---------------------------------------------------------------------------

SOMECHARS           = rxe('.+')
ANYCHARS            = rxe('.*')
ANYLAZY             = rxe('.*?')
ANYSPACE            = rxe(r'\s*')
SOMESPACE           = rxe(r'\s+')
GG                  = G(ANYCHARS)   # unnamed group wrapping .*
GGL                 = G(ANYLAZY)    # unnamed group wrapping .*?
WW = WHOLEWORD      = rxe(r'\b\w+\b')
UPPERCASE_WORD      = rxe(r'\b[A-Z]+\b')
HEX_NUMBER          = rxe(r'\b0[xX][0-9A-Fa-f]+\b')

FILENAME = (
    WORDBOUNDARY
    + OneOrMore(ALPHANUMERIC | '.!_-')
    + WORDBOUNDARY
)
LINUXPATH = OneOrMore('/' + OneOrMore(ALPHANUMERIC | '.!_-'))

IDENTIFIER = (
    WORDBOUNDARY
    + (LETTER | '_')
    + ZeroOrMore(ALPHANUMERIC | '_')
    + WORDBOUNDARY
)
UPPERCASE_IDENTIFIER = (
    WORDBOUNDARY
    + (UPPERCASE | '_')
    + ZeroOrMore(UPPERCASE | '_')
    + WORDBOUNDARY
)

SOMELETTERS = ANYSPACE + OneOrMore(ALPHANUMERIC | '.').lazy()

# ---------------------------------------------------------------------------
# Composite domain patterns
# ---------------------------------------------------------------------------

EXECUTED_TEST = (
    Group('STATUS', Either('SUCCESS', 'FAILED'))
    + WHITESPACE
    + 'FILE:'
    + Group('FILE', FILENAME)
)  # usage: pyrg EXECUTED_TEST -or '$FILE --> $STATUS' tacot1_out_01

FAILED_TEMPLATES = (
    'FAILED'
    + SOMECHARS.lazy()
    + Group('f', FILENAME + 'templ')
    + WORDBOUNDARY
)  # usage: pyrg "FAILED_TEMPLATES" -or '$f'  mono.out | uniq

SC = 'SC_' + 6 * DIGITS  # matches an SC number like SC_123456

TRACE = IDENTIFIER.preceded_by('Debug.Trace' + WS + '(')

FUNC_PROC = (
    G(Either(
        IDENTIFIER.preceded_by('function' + WS),
        IDENTIFIER.preceded_by('procedure' + WS),
    ))
    + GG + WB + 'is' + WB
)

RAISED_EXCEPTION = (
    (UPPERCASE_IDENTIFIER + '.')[:1] + UPPERCASE_IDENTIFIER
).preceded_by('raised' + WS)

EDIFF_FILES = (
    '(ediff-files'
    + SOMESPACE
    + DOUBLEQUOTE + Group('TEMPL', LINUXPATH) + DOUBLEQUOTE
    + SOMESPACE
    + DOUBLEQUOTE + Group('OUT', LINUXPATH) + DOUBLEQUOTE
)  # usage: pyrg EDIFF_FILES -or 'cp \$OUT \$TEMPL'  mono.out

# ---------------------------------------------------------------------------
# Advanced combinators
# ---------------------------------------------------------------------------

def Contains(char: str) -> rxe:
    """Match a string that contains *char* somewhere inside it."""
    return rxe(f'(?:.+?{re.escape(char)}.+?)')

def BetweenMatchingBrackets(pair: str) -> rxe:
    """Match a balanced bracketed expression for the given bracket *pair*.

    Example::

        BetweenMatchingBrackets('()').findall('a(b(c)d)e(f)')
        # → ['(b(c)d)', '(f)']
    """
    return (
        pair[0]
        + ZeroOrMore(Either(AtomicGroup(AnythingBut(pair)), RecursiveGroup))
        + pair[1]
    )

BETWEENQUOTES = (
    Group('quote', AnyOf("'\""))
    + OneOrMore(ANYCHAR).lazy()
    + Backref('quote')
)

def Anything_Outside_Brackets(pair: str) -> rxe:
    """Match any non-empty run of characters *outside* balanced brackets."""
    return OneOrMore(AnythingBut(pair)).filter_matches(BetweenMatchingBrackets(pair))


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------

def _split_on(delimiter: 'rxe | str', line: str, exclusions: list[rxe]) -> list[str]:
    """Split *line* on *delimiter*, ignoring occurrences inside *exclusions*.

    Each element of *exclusions* is a pattern whose matches are skipped over
    using ``(*SKIP)(*F)`` before the delimiter is tried.
    """
    skip = Either(*exclusions) if len(exclusions) > 1 else exclusions[0]
    pattern = (rxe(_get_pattern(delimiter)) + ZeroOrMore(WHITESPACE)).filter_matches(skip)
    return [el for el in pattern.split(line) if el is not None]


def split_on_comma(line: str) -> list[str]:
    """Split *line* on commas, ignoring commas inside brackets or quotes."""
    exclusions = [
        BetweenMatchingBrackets('[]'),
        BetweenMatchingBrackets('()'),
        BETWEENQUOTES,
    ]
    return _split_on(',', line, exclusions)


def split_on_colon(line: str) -> list[str]:
    """Split *line* on colons, ignoring colons inside brackets or quotes."""
    exclusions = [
        BetweenMatchingBrackets('[]'),
        BetweenMatchingBrackets('()'),
        BETWEENQUOTES,
    ]
    return _split_on(':', line, exclusions)


# ---------------------------------------------------------------------------
# Variable / URL patterns
# ---------------------------------------------------------------------------

VAR_DECL = (
    G(ANYSPACE + IDENTIFIER + ANYSPACE)
    + ':'
    + G(AnythingBut('=')[1:])
)

http_protocol = 'http' + Optional('s') + '://'
domain_name   = AnyBut(WS | DQ)[1:]
tld           = '.' + LETTERS[2:3]
URL           = Optional(http_protocol) + Optional('www.') + domain_name


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # -- Repetition ----------------------------------------------------------
    print("=== Repetition ===")
    print(DIGIT[55])    # \d{55}
    print(DIGIT[:])     # \d*
    print((~DIGIT)[:])  # \D*
    print(DIGIT[1:])    # \d+
    print(DIGIT[4:])    # \d{4,}
    print(DIGIT[:1])    # \d?
    print(DIGIT[:4])    # \d{,4}
    print(DIGIT[4:4])   # \d{4}
    print(DIGIT[2:5])   # \d{2,5}

    # -- Inversion -----------------------------------------------------------
    assert (~DIGIT).pattern        == r'\D'
    assert (~ALPHANUMERIC).pattern == r'\W'
    assert (~WHITESPACE).pattern   == r'\S'
    assert (~WORDBOUNDARY).pattern == r'\B'
    print("Inversion: OK")

    # -- Named groups + MatchObject ------------------------------------------
    print("\n=== Named groups ===")
    string_re = (
        Group('quote',    AnyOf("'\""))
        + Group('mystring', ZeroOrMore(LETTERS))
        + Group('mydigits', 3 * DIGITS)
        + Backref('quote')
    )
    m = string_re.match("'hello765'")
    assert m, f"No match: {string_re}"
    assert m.mystring == 'hello'
    assert m.mydigits == '765'
    print(f"{string_re} matches: mystring={m.mystring!r}  mydigits={m.mydigits!r}")

    # -- Email pattern -------------------------------------------------------
    print("\n=== Email ===")
    username  = OneOrMore(ALPHANUMERIC | '.%+-')
    domain    = (ALPHANUMERIC | '.-')[1:]
    tld_local = AnyOf('a-z')[2:6]
    email     = Group('NAME', username) + '@' + Group('DOMAIN', domain) + '.' + tld_local
    print(email.pattern)
    m = email.match('elkarouh@gmail.com')
    assert m, "Email pattern did not match"
    assert m.NAME   == 'elkarouh'
    assert m.DOMAIN == 'gmail'
    print(f"NAME={m.NAME!r}  DOMAIN={m.DOMAIN!r}  group(1)={m[1]!r}")

    # -- IF condition --------------------------------------------------------
    print("\n=== IF condition ===")
    comment_re   = '#' + ZeroOrMore(ANYCHAR)
    if_construct = (
        ZeroOrMore(WHITESPACE)
        + 'if'
        + Group('CONDITION', OneOrMore(AnythingBut(':')))
        + ':'
        + ZeroOrMore(WHITESPACE)
        + Optional(Group('COMMENT', comment_re))
    )
    m = if_construct.match(' if this is true: # ffff')
    assert m, f"if_construct did not match; pattern: {if_construct}"
    assert m.CONDITION.strip() == 'this is true'
    assert m.COMMENT.strip()   == '# ffff'
    print(f"CONDITION={m.CONDITION!r}  COMMENT={m.COMMENT!r}")

    # -- Balanced brackets ---------------------------------------------------
    print("\n=== Balanced brackets ===")
    LINE = 'a(bcd(e)f)g(h)'
    assert BetweenMatchingBrackets('()').findall(LINE) == ['(bcd(e)f)', '(h)']
    mrepl = lambda m: 'X' * len(m[0])
    print(BetweenMatchingBrackets('()').sub(mrepl, LINE))

    LINE = 'dsdsjfklsfd{NOT THIS} dsdjsiodj{NOT THIS EITHER}'
    assert BetweenMatchingBrackets('{}').findall(LINE) == ['{NOT THIS}', '{NOT THIS EITHER}']
    assert Anything_Outside_Brackets('{}').findall(LINE) == ['dsdsjfklsfd', ' dsdjsiodj']
    print("Balanced brackets: OK")

    # -- filter_matches / (*SKIP)(*F) ----------------------------------------
    print("\n=== filter_matches ===")
    LINE = 'tiger imp goat eagle rat'
    imp_or_rat = WORDBOUNDARY + Either('imp', 'rat') + WORDBOUNDARY
    filtered   = OneOrMore(LETTER).filter_matches(imp_or_rat)
    assert filtered.sub(r'(\g<0>)', LINE) == '(tiger) imp (goat) (eagle) rat'
    print(f"filter_matches sub: {filtered.sub(r'(\g<0>)', LINE)!r}")

    # -- split helpers -------------------------------------------------------
    print("\n=== split_on_comma ===")
    LINE = '1,(cat,12),nice,two,(dog,5)'
    comma_re = rxe(',').filter_matches(BetweenMatchingBrackets('()'))
    assert comma_re.sub('|', LINE) == '1|(cat,12)|nice|two|(dog,5)'
    print(split_on_comma(LINE))

    LINE = '1,"cat,12",nice,two,"dog,5"'
    comma_re2 = rxe(',').filter_matches(BETWEENQUOTES)
    assert comma_re2.sub('|', LINE) == '1|"cat,12"|nice|two|"dog,5"'
    print(split_on_comma(LINE))

    # -- FILENAME / LINUXPATH ------------------------------------------------
    print("\n=== FILENAME / LINUXPATH ===")
    assert FILENAME.match('full-simca_l.log')
    assert LINUXPATH.match('/dshujdhs/full-simca_l.log')
    print("FILENAME / LINUXPATH: OK")

    # -- Return type / argument parsing --------------------------------------
    print("\n=== Return type + split_on_comma ===")
    LINE      = 'def hello(x:[]str,y:[][]int) -> []str:'
    ret_type  = '->' + G(AnythingBut(':')[1:]) + ':'
    m         = ret_type.search(LINE)
    assert m, "ret_type not found"
    print(f"return type: {m[1]!r}")
    print(ret_type.sub(lambda m: 'X' * len(m[1]), LINE))

    m = BetweenMatchingBrackets('()').search(LINE)
    assert m
    inner = m.raw.string[m.raw.start() + 1 : m.raw.end() - 1]
    for arg in split_on_comma(inner):
        print(f"  arg: {arg!r}")

    # -- VAR_DECL ------------------------------------------------------------
    print("\n=== VAR_DECL ===")
    LINE = "x :[][]str=['w','e']"
    m = VAR_DECL.match(LINE)
    assert m, "VAR_DECL did not match"
    print(f"name={m[1]!r}  type={m[2]!r}")
    print(VAR_DECL.sub(lambda m: f'{m[1].rstrip()}:"{m[2].lstrip()}"', LINE))

    print("\nAll assertions passed.")

""" inspired from pipe21 and improved
TODO: add windows
"""
import functools
import itertools
import operator
import re
import sys
import fileinput
class Pipe:
    def __init__(self, f=None, *args, **kw):
        self.f=f
        self.args=args 
        self.kw=kw
    def __ror__(self, left):
        return self.f(left)
    def __call__(self, *args, **kwargs):
        return Pipe(lambda x: self.f(x, *args, **kwargs))
    __rrshift__=__ror__ # >>
    def __mul__(self,other):
        return Pipe(lambda x: x >> self | other)
class Map          (Pipe): __ror__ = lambda self, x: map   (self.f, x)
class Filter       (Pipe): __ror__ = lambda self, x: filter(self.f, x)
class Reduce       (Pipe): __ror__ = lambda self, it: functools.reduce(self.f, it, *self.args)
class MapKeys      (Pipe): __ror__ = lambda self, it: ((self.f(k), v) for k, v in it)
class MapValues    (Pipe): __ror__ = lambda self, it: ((k, self.f(v)) for k, v in it)
class FilterFalse  (Pipe): __ror__ = lambda self, it: itertools.filterfalse(self.f, it)
class FilterKeys   (Pipe): __ror__ = lambda self, it: (kv for kv in it if (self.f or bool)(kv[0]))
class FilterValues (Pipe): __ror__ = lambda self, it: (kv for kv in it if (self.f or bool)(kv[1]))
class FlatMap      (Pipe): __ror__ = lambda self, it: itertools.chain.from_iterable(self.f(x) for x in it)
class FlatMapValues(Pipe): __ror__ = lambda self, it: ((k, v) for k, vs in it for v in self.f(vs))
class KeyBy        (Pipe): __ror__ = lambda self, it: ((self.f(x), x) for x in it)
class ValueBy      (Pipe): __ror__ = lambda self, it: ((x, self.f(x)) for x in it)
class Append       (Pipe): __ror__ = lambda self, it: ((*x, self.f(x)) for x in it)
class Keys         (Pipe): __ror__ = lambda self, it: (k for k, v in it)
class Values       (Pipe): __ror__ = lambda self, it: (v for k, v in it)
class Grep         (Pipe): __ror__ = lambda self, it: it | (FilterFalse if self.kw.get('v', False) else Filter)(re.compile(self.f, flags=re.IGNORECASE if self.kw.get('i', False) else 0).search)
class IterLines    (Pipe): __ror__ = lambda self, f: (x.strip() if self.kw.get('strip', True) else x for x in open(f))
class Count        (Pipe): __ror__ = lambda self, it: sum(1 for _ in it)
class Slice        (Pipe): __ror__ = lambda self, it: itertools.islice(it, self.f, *self.args)
class Take         (Pipe): __ror__ = lambda self, it: it | Slice(self.f) | Pipe(list)
class Sorted       (Pipe): __ror__ = lambda self, it: sorted(it, **self.kw)
class GroupBy      (Pipe): __ror__ = lambda self, it: itertools.groupby(sorted(it, key=self.f), key=self.f)
class ReduceByKey  (Pipe): __ror__ = lambda self, it: it | GroupBy(lambda kv: kv[0]) | MapValues(lambda kv: kv | Values() | Reduce(self.f)) | Pipe(list)
class Apply        (Pipe): __ror__ = lambda self, x: x | Exec(self.f, x)
class StarPipe     (Pipe): __ror__ = lambda self, x: self.f(*x)
class StarMap      (Pipe): __ror__ = lambda self, x: itertools.starmap(self.f, x)
class StarFlatMap  (Pipe): __ror__ = lambda self, x: itertools.starmap(self.f, x) | Pipe(itertools.chain.from_iterable)
class MapApply     (Pipe): __ror__ = lambda self, it: (x | Apply(self.f) for x in it)
class Switch       (Pipe): __ror__ = lambda self, x: next((v(x) for k, v in self.f if k(x)), x)
class MapSwitch    (Pipe): __ror__ = lambda self, it: (x | Switch(self.f) for x in it)
class YieldIf      (Pipe): __ror__ = lambda self, it: ((self.f or (lambda y: y))(x) for x in it if self.kw.get('key', bool)(x))
class Join         (Pipe): __ror__ = lambda self, it: it | FlatMap(lambda x: ((x, y) for y in self.f if self.kw.get('key', operator.eq)(x, y)))
class GetItem        (Pipe): __ror__ = lambda self, x: operator.getitem(x, self.f)
class SetItem        (Pipe): __ror__ = lambda self, x: x | Exec(operator.setitem, x, self.f, self.args[0])
class DelItem        (Pipe): __ror__ = lambda self, x: x | Exec(operator.delitem, x, self.f)
class GetAttr        (Pipe): __ror__ = lambda self, x: getattr(x, self.f)
class SetAttr        (Pipe): __ror__ = lambda self, x: x | Exec(setattr, x, self.f, self.args[0])
class DelAttr        (Pipe): __ror__ = lambda self, x: x | Exec(delattr, x, self.f)
class MapGetItem     (Pipe): __ror__ = lambda self, it: (kv | GetItem(self.f) for kv in it)
class MapSetItem     (Pipe): __ror__ = lambda self, it: (kv | SetItem(self.f, self.args[0]) for kv in it)
class MapDelItem     (Pipe): __ror__ = lambda self, it: (kv | DelItem(self.f) for kv in it)
class MapGetAttr     (Pipe): __ror__ = lambda self, it: (kv | GetAttr(self.f) for kv in it)
class MapSetAttr     (Pipe): __ror__ = lambda self, it: (kv | SetAttr(self.f, self.args[0]) for kv in it)
class MapDelAttr     (Pipe): __ror__ = lambda self, it: (kv | DelAttr(self.f) for kv in it)
class MethodCaller   (Pipe): __ror__ = lambda self, x: operator.methodcaller(self.f, *self.args, **self.kw)(x)
class MapMethodCaller(Pipe): __ror__ = lambda self, it: (x | MethodCaller(self.f, *self.args, **self.kw) for x in it)
class Unique(Pipe):
    def __ror__(self, it):
        key = self.f or (lambda x: x)
        seen = set()
        for item in it:
            k = key(item)
            if k in seen:
                continue
            seen.add(k)
            yield item
class Exec(Pipe):
    def __ror__(self, x):
        self.f(*self.args, **self.kw)
        return x

if sys.version_info >= (3, 12):  # pragma: no cover
    class Chunked(Pipe): __ror__ = lambda self, it: itertools.batched(it, self.f)
else:  # pragma: no cover
    class Chunked(Pipe): __ror__ = lambda self, it: iter(functools.partial(lambda n, i: tuple(i | Take(n)), self.f, iter(it)), ())

if __name__=="__main__":
    @Pipe 
    def processing1(lines):
        for line in lines:
            yield "1"+str(line)

    @Pipe 
    def processing2(predictions):
      for pred in predictions:
          yield "2"+str(pred) 

    cxx_processor=(processing1 * processing2)

    #for line in fileinput.input(encoding="utf-8") >> cxx_processor:
    for line in "abcdef" >> cxx_processor:
      print(line)
    print(range(5) | Pipe(list))
    print(range(5) | Map(str) | Pipe(''.join)) # '01234'
    print(range(5) | Filter(lambda x: x % 2 == 0) | Pipe(list)) # [0, 2, 4]


    #print(range(1_000_000) | Map(chr) | Filter(str.isdigit) | Pipe(''.join))
    print(range(1, 100) | MapSwitch([
        (lambda i: i % 3 == i % 5 == 0, lambda x: 'FizzBuzz'),
        (lambda i: i % 3 == 0, lambda x: 'Fizz'),
        (lambda i: i % 5 == 0, lambda x: 'Buzz'),
    ])  | Pipe(list)
    )
    print(range(5) | Reduce(lambda a, b: a + b)) # 10
    print(range(5) | Chunked(2) | Pipe(list)) #[(0, 1), (2, 3), (4,)]  
    print([1,2,2,3,3] | Unique() | Pipe(list)) #[1, 2, 3]  
    raise SystemExit
    import pathlib, operator, webbrowser, re, random
    (
    pathlib.Path.home() / 'docs/knowledge/music'               # take a directory
    | MethodCaller('rglob', '*.md')                          # find all markdown files
    | FlatMap(lambda p: p | IterLines())                   # read all lines from all files and flatten into a single iterable
    | FlatMap(lambda l: re.findall(r'\[(.+)\]\((.+)\)', l))  # keep only lines with a markdown link
    | Map(operator.itemgetter(1))                            # extract a link
    | Pipe(list)                                             # convert iterable of links into a list
    | Pipe(random.choice)                                    # choose random link
    | Pipe(webbrowser.open)                                  # open link in browser
    )    

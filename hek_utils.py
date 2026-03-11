import tokenize, io
from io import StringIO
import itertools
############################# DEBUG TOOLS !!!!! ################################
import snoop # decorator or wuse 'with snoop' for a part of the code
from snoop import spy # decorator
from snoop import pp as pp2# pretty print function
from rich.pretty import pprint as pp # prettier then snoop.pp !!!
from rich import inspect as see
from rich import print
from rich.traceback import install;install()
#from birdseye import eye # decorator
#import heartrate; heartrate.trace(browser=True)
################# better exceptions
from loguru import logger
# in the except clause of a try statement, add:
#logger.exception("SOMETHING WRONG HAPPENED")
# for line in fileinput.input(filename,inplace=True,backup='.bak'):
     # line=line.replace("20210625",datum)
     # line=line.replace("7TVNC",flight)
     # sys.stderr.write(line)
################################################################################
tokenize_string=lambda s:tokenize.tokenize(io.BytesIO(s.encode('utf-8')).readline)
PROCESSED,UNPROCESSED,NOT_PROCESSED="PROCESSED","UNPROCESSED","NOT_PROCESSED"
class Stack(list):
    push=list.append
    peek = lambda self: self[-1]
def sliding_window(seq,n=3):
    """ in case n=3, yields prev,cur,next
    The first prev is None
    The last prev is the last element
    """
    it = itertools.chain(iter([None]),iter(seq), iter([None]))
    result = tuple(itertools.islice(it, n))
    if len(result) == n: yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result # TO DO: replace None by NullObject !!!
    for _ in range(n-2):# dont lose the n-3 tokens at the end of the stream ! change to n-1, to have the last el as current
        result = result[1:]+ (None,)
        yield result
class Pipe:
    def __init__(self, function):
        self.function = function
    def __ror__(self, left):
        return self.function(left)
    def __call__(self, *args, **kwargs):
        return Pipe(lambda x: self.function(x, *args, **kwargs))
    __rrshift__=__ror__ # >>
    def __mul__(self,other):
        return Pipe(lambda x: x >> self | other)
class SymbolTable:
    def __init__(self):
        self.list=[]
    def push(self,item):
        self.list.append(item)
    def pop(self):
        return self.list.pop()
    def enter_scope(self):
        self.push({})
    def leave_scope(self):
        self.pop()
    def add_symbol(self,symbol,declared_type):
        self.list[-1][symbol]=declared_type
    def get_symbol(self,sym):
        for sym_table in reversed(self.list):
            if sym in sym_table:
                return sym_table[sym]
        return None
    def __str__(self):
        return str(self.list)
if 0 and __name__=="__main__":
    print("Test symbol table")
    a=SymbolTable()
    a.enter_scope()
    a.add_symbol('sym1',int)
    a.add_symbol('sym2',float)
    a.enter_scope()
    a.add_symbol('sym1',"user-defined")
    a.new_Attr=777
    #print(a)
    assert a.get_symbol('sym1')=="user-defined"
    assert a.get_symbol('sym2')==float
    assert a.get_symbol('sym3')==None
    a.leave_scope()
    assert a.get_symbol('sym1')==int
    assert a.get_symbol('sym2')==float
    b=copy.deepcopy(a)
    b.add_symbol('sym3','unsigned')
    print(b)
    print(a)
    raise SystemExit

def get_new_line(fin):
    indent_level=0
    incomplete_line=""
    previous_token=None
    for token in tokenize_string(fin.read()):
        #print(token)
        curr_line_number=token.start[0]
        match token.type:
            case tokenize.NL:
                if token.line.lstrip().startswith('#'):
                    yield indent_level,incomplete_line+token.line.rstrip(),UNPROCESSED
                elif previous_token and previous_token.type==tokenize.COMMENT:
                    incomplete_line+=token.line.rsplit("#",1)[0] # CAVEAT we lose the comment !!!
                else:
                    if curr_line_number!=previous_line_number and not token.line.strip():
                        yield indent_level,"NEWLINE",UNPROCESSED # a empty newline (empty NL token)
                        incomplete_line=""
                    else:
                        incomplete_line+=token.line.rstrip() # CAVEAT empty lines disappear !!!
            case tokenize.NEWLINE:
                yield indent_level,incomplete_line+token.line.rstrip(),UNPROCESSED
                incomplete_line=""
            case tokenize.INDENT:
                indent_level +=1
            case tokenize.DEDENT:
                indent_level -=1
                yield indent_level,"",UNPROCESSED  # we want one DEDENT per line  !!!
            case tokenize.ENDMARKER:
                yield indent_level,"",UNPROCESSED
        previous_token=token
        previous_line_number=curr_line_number

if __name__=="__main__":
    example="""\
for i in range:
  print
# comment1
def function(arg1:int, # COMMENT INSIDE PARENTHESES !!!
  arg2:float):
  instruction1
  # comment2
  instruction2
"""
    test_code="""\
class Parent:
  def __del__(self):
    print "deleting"
class Child(Parent):
  age : int
"""
    test_code="""\
def hello(a: var []str,
          b: int) -> float:
  if x[:dd] > "dd:ss": # if loop
    print "inside main branch"

  elif d>3: #  elif branch
    print "inside elif branch"
  else: # else comment
    print "inside else branch"
"""
    for indent,line,status in get_new_line(StringIO(test_code)):
        #print(f"{line:60} INDENT:{indent}")
        print(f"{line:60}")
    raise SystemExit
####################################################################################
from pathlib import Path
from typing import Iterable
def walk(p: Path) -> Iterable[Path]:
    for item in p.iterdir():
        yield item
        if item.is_dir():
            yield from walk(item)
###################################################################################
from quickstruct import *
import struct
class Struct(DataStruct, flags=StructFlags.FixedSize|StructFlags.NoAlignment):
    @classmethod
    def decode(cls,data):
        mydata,rest= data[:cls.size],data[cls.size:]
        try:
            thing = cls.from_bytes(mydata)
        except struct.error:
            return None,data
        else:
            return thing, rest
    def to_binary(self):
        return self.to_bytes()
##################################################################################
class L(list):
    """
    A subclass of list that can accept additional attributes.
    Should be able to be used just like a regular list.

    The problem:
    a = [1, 2, 4, 8]
    a.x = "Hey!" # AttributeError: 'list' object has no attribute 'x'

    The solution:
    a = L(1, 2, 4, 8)
    a.x = "Hey!"
    print a       # [1, 2, 4, 8]
    print a.x     # "Hey!"
    print len(a)  # 4

    You can also do these:
    a = L( 1, 2, 4, 8 , x="Hey!" )                 # [1, 2, 4, 8]
    a = L( 1, 2, 4, 8 )( x="Hey!" )                # [1, 2, 4, 8]
    a = L( [1, 2, 4, 8] , x="Hey!" )               # [1, 2, 4, 8]
    a = L( {1, 2, 4, 8} , x="Hey!" )               # [1, 2, 4, 8]
    a = L( [2 ** b for b in range(4)] , x="Hey!" ) # [1, 2, 4, 8]
    a = L( (2 ** b for b in range(4)) , x="Hey!" ) # [1, 2, 4, 8]
    a = L( 2 ** b for b in range(4) )( x="Hey!" )  # [1, 2, 4, 8]
    a = L( 2 )                                     # [2]
    """
    def __new__(self, *args, **kwargs):
        return super(L, self).__new__(self, args, kwargs)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

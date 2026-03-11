#!/usr/bin/env -S uv run --script
# -*- coding: utf-8 -*-
#pydeps for graph dependencies !!!
from rich import print
from rich.console import Console; console=Console(record=True)
console.log("whatever", log_locals=True)
console.save_html("demo.html")
from rich.traceback import install; install()
import snoop # THE BEST DEBUGGING TOOL !!! use as decorator on functions or with-context on blocks of code
# optional args: @snoop(depth=2, watch=['foo.bar','self.x'],watch_explode=['mydict'])
from snoop import spy # also a decorator for visual debugging !!!
from snoop import pp # THE BEST PRETTPRINT !!
# other debugging tool
from objexplore import explore # pip install objexplore
import wat # pip install wat
######################################
# from justhtml import JustHTML
####################################
import lovely_logger as log
log.FILE_FORMAT = "[%(asctime)s] [%(levelname)-8s] - %(message)s (%(filename)s:%(lineno)s)"
log.CONSOLE_FORMAT = "[%(levelname)-8s] - %(message)s (%(filename)s:%(lineno)s)"
log.DATE_FORMAT = '%Y-%m-%d %H:%M:%S.uuu%z'
log.init('./my_log_file.log')
log.init(filename, to_console=True, level=DEBUG, max_kb=1024, max_files=5)
try:
    1/0
except:
    log.exception("YOU DIVIDED BY ZERO")
###################################################
from typing import Protocol
class Animal(Protocol):
   def eat(self, food) -> float:
       ...
   def sleep(self, hours) -> float:
       ...
######################################
from prettyprinter import cpprint as pprint # pip install prettyprinter
import pandas as pd
df=pd.Dataframe(data, index=..., columns=...)
# df['col1'] or df[['col1','col2']]  filter on columns
# df[condition] ==> filter on rows
# pd.Series = Dataframe with one column !!!
table = pd.pivot_table(df,index=['Sex','Pclass'],aggfunc={'Age':np.mean,'Survived':np.sum})
table = pd.pivot_table(df,index=['Sex','Pclass'],values=['Survived'], aggfunc=np.mean)
table = pd.pivot_table(df,index=['Sex'],columns=['Pclass'],values=['Survived'],aggfunc=np.sum)
for row in df.itertuples(name=None):
	print(row[3])
#################################### file or stdin as arg ###############
import sys
f = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
for line in f:
    pass  # process the line
# the user now can do
# 1. python do-my-stuff.py infile.txt
# 2. cat infile.txt | python do-my-stuff.py
# 3. python do-my-stuff.py < infile.txt
###########################################
from pipepy import ls, grep
from pipe import select, map
from placeholder import _
# https://www.mslinn.com/blog/2020/10/22/scala-style-functional-programming-in-python-3.html
from sspipe import p, px, px as _
from rich.console import Console
console = Console()
######################
# to work with dates: arrow, pendulum, delorean
# DATES: pip install saturn !!!!! (other: arrow, pendulum, delorean, udatetime)
# to replace BeautifulSoup: from requests_html import HTMLSession; session = HTMLSession();r=session.get("google.com")
####################
import networkx as nx; import pyvis
import pygraphina as pg # GRAPH LIBRARY !!!
###################### SCRAPING ################
from requests_html import HTMLSession # pip install requests-html
session = HTMLSession()
r = session.get('https://python.org/')
#####################
######################
def coroutine(fn):
    def wrapper(*args, **kwargs):
        v = fn(*args, **kwargs)
        v.send(None)
        return v
    return wrapper
if __name__=="__main__":
    @coroutine
    def Grep(substr): # CONVENTION: always use capital letter for generators!!!
        while True:
            line = yield
            if substr in line:
                print(f"found {substr}")
    gen=Grep("hello")
    gen.send("this is a line with hello")
###############################################
from addict import Dict
from box import Box # pip install python-box
###########################################
from watchpoints import watch
from easierlog import log
###########################
a=adict.get("hello",missing:=object())
if a is missing:
    pass  # do whatever
    # breakpoint()
################################################################################
# GOOD !!! py2 and py3
def run_command(cmd_lst): # run a external program and retrieve output line by line
    p=subprocess.Popen(command_as_list,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    return iter(p.stdout.readline, b'')
for line in run_command("ls -rtl".split()):
    print(line)
############################################################################
# HOW TO USE send TO EMULATE A STOPPABLE FUNCTION
# @coroutine
def Process_chunks():
    while True:
        try:
            chunk = yield
        except GeneratorExit:
            finish_up()
            return
        else:
            do_whatever_with(chunk)
with closing(Process_chunks()) as coro:
    FTP.retrbinary(command, callback=coro)
########################### COROUTINES AND PIPES AT THE SAME TIME !!!############
# https://gist.github.com/zacharyvoase/119665
# see pipes.py !!!! (ported to python3)
###############################################################
from typing import Mapping, OrderedDict, Sequence, TypedDict
class Point2D(TypedDict):
    x: int
    y: int
    label: str
# how to emulation builtin enums in Python
# PRIORITY:type= HIGH, MEDIUM, LOW = range(3)
# now you can do as in ada
# ACTION:type = Mapping[PRIORITY,Point2D] # type ACTION is [PRIORITY]Point2D

############### solve your path problems ##########################
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0,os.path.dirname(sys.path[0]))
    # the following is sometimes necessary because files are relative to where
    # the script is executed not to where it exists
    current_dir = os.path.dirname(os.path.abspath(__file__))

if __name__=="__main__"  and __package__ is None:
    __package__="name of your package" # allows unit-testing with relative imports

def add_coconut_to_path():
    """Adds coconut to sys.path if it isn't there already."""
    try:
        import coconut  # NOQA
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
###############
#remove duplicates and keep Ordering
list(dict.fromkeys(LIST_WITH_DUPLICATES))
################################################################################
# useful tools: uv, pdmm pyupgrade
# useful packages: addict, justpy
body = Dict()
body.query.filtered.query.match.description = 'addictive'

#Python dictionaries with recursive dot notation access.
my_box.to_json() == {"owner": "Mr. Powers",
                     "affiliates": {
                         "Dr Evil": "Not groovy",
                         "Scott Evil": "Doesn't want to take over family business"
                     }
}
#############################################################################
import platform; print(f"Python version: {platform.python_version()}")
##############################################################################
#-->IF YOU HAVE NO PLAN TO SUBCLASS, USE NamedTuple
from typing import NamedTuple
class Base(NamedTuple):
    x: int
    y: float

a=Base(x=5,y=7.5)
# TRICK: tuples are immutable in theory but you can do a._replace(x=8) !!
#-->IF YOU WANT TO SUBCLASS, USE dataclass INSTEAD
from dataclasses import dataclass

@dataclass(frozen=True)
class Base:
    x: int
    y: float

@dataclass(frozen=True)
class BaseExtended(Base):
    z: str
"""
The classes are not full named tuples, as they don't have a length or support indexing,
but that's trivially added by creating a baseclass that inherits from collections.abc.Sequence,
adding two methods to access fields by index. If you add order=True to the @dataclass() decorator
then your instances become fully orderable the same way (named) tuples are:
"""
from collections.abc import Sequence
from dataclasses import dataclass, fields
class DataclassSequence(Sequence):
    # make a dataclass tuple-like by accessing fields by index
    def __getitem__(self, i):
        return getattr(self, fields(self)[i].name)
    def __len__(self):
        return len(fields(self))

@dataclass(frozen=True, order=True)
class Base(DataclassSequence):
    x: int
    y: int
########################
from functools import total_ordering
@total_ordering
class Number:
    def __init__(self, value):
        self.value = value
    # only define these 2
    def __lt__(self, other):
        return self.value < other.value
    def __eq__(self, other):
        return self.value == other.value
from functools import lru_cache, cached_property
############################# debugging #############################################
try:
    from snoop import snoop, pp, spy # !!!! pip install snoop, birdseye
    import snoop; snoop.install(enabled=True) # BETTER THAN ABOVE LINE
    from q import q, t  # pip install q
    import pretty_errors # pip install pretty-errors
    import better_exceptions # pip install better_exceptions
    import stackprinter;stackprinter.set_excepthook(style='color') # pip install stackprinter
except ImportError:
    pass
###################### STRING MANIPULATION GOODIES###########################
import re
find_quoted_strings=lambda txt: re.findall('''["']([^"']*)["']''',txt)
split_on_comma_but_not_within_quotes=lambda txt: re.findall(r'''(?:["'][^"']*["']|[^,"'])+''', txt)
remove_comments_but_not_within_quotes=lambda line:re.sub(r'''(["'](?:[^"']+|(?<=\\)")*["'])|#[^\n]*''', lambda m: m.group(1) or '',line)
def get_contents_within_braces(string, braces='{}'):
    """Generate parenthesized contents in string as pairs (level, contents).
    https://stackoverflow.com/questions/4284991/parsing-nested-parentheses-in-python-grab-content-by-level
    """
    stack=[]
    for i,c in enumerate(string):
        if c==braces[0]:
            stack.append(i)
        elif c==braces[1] and stack:
            start=stack.pop()
            if len(stack)==0: # here we only need the FIRST LEVEL !!!
                yield string[start+1:i]
            #yield (len(stack), string[start + 1: i]) IF YOU NEED ALL LEVELS
########################################################################
class Stack(list):
    push=list.append
    peek = lambda self: self[-1]

WINDOWS, LINUX, MAC = "Windows", "Linux", "Darwin"
import platform
assert platform.system() != WINDOWS
assert platform.system() != LINUX
assert platform.system() == MAC
PYTHON2=False
################ LOCAL PYPI FOR REPRODUCEABLE BUILDS ############# http://9tabs.com/snippets/2018/01/21/back-up-your-pypy-deps.html
#pip2tgz "/var/www/localpackages" mypackage  # pip install py2pi
#pip install --index-url="file:///var/www/localpackages" mypackage
############################# UNICODE ###################################
# pip install win_unicode_console # DO THIS ON WINDOWS and adapt sitecustomize.py
# with the following contents
#import win_unicode_console; win_unicode_console.enable()
if PYTHON2:	import sys;reload(sys);sys.setdefaultencoding('UTF-8') # THE MAGIC LINE !!!
# in python3, use 'cp437' encoding, it is more universal than 'utf8'
#################################################################################
#################################################
# first start redis server with
# /src/redis-server
import redis
r = redis.Redis(host='localhost',port=6379)
import dill
# at the server side
# serialize a function with its arguments
for i in range(10):
    data = dill.dumps((a_function, [a1, a2]))
    r.lpush('task',data)
# at the client side
while True:
    # Wait until there's an element in the 'tasks' queue
    key, data = r.brpop('task')
    # Deserialize the task
    d_fun, d_args = dill.loads(data)
    # Run the task
    d_fun(*d_args)

##################################################################
# sheebang for windows (don't forget to change the extension to .cmd !!!)
## @setlocal enableextensions & python -x "%~f0" %* & goto :EOF
############################################################
from see import see   # pip install see (look at rich.inspect(all=True) !!!)
#############################################################
if PYTHON2: # backport of python3 subprocess improvements!!! See python3 docs!!
    import subprocess32 as subprocess # ALREADY INSTALLED WITH ANACONDA2 !!!
########################### f-strings #########################
if PYTHON2: # f-string in python2
    import fmt as f  # pip install fmt (only python2)
    # ANOTHER POSSIBILITY
    from say import fmt as f # pip install say (only python2)
    # YET ANOTHER
    from fstrings import f # pip install fstrings
########################## which ##############################
if PYTHON2: # pip install shutilwhich (only python2)
    import shutilwhich;from shutil import which
else:
    from shutil import which
#==> this allows to do:
p=subprocess.Popen([which('npm'), arg1, arg2])
p.wait
###########################################################
from pathlib import Path
with tempfile.NamedTemporaryFile() as temp:
  path = Path(temp.name)
  assert path.exists()
  assert path.is_file()
  path.write_text(data)
  with path.open(mode='w') as afile:
    afile.write(text)
from io import open as File
# a=Path("hello.py") ==> a.absolute().as_posix() for the complete path OR a.resolve()
# Path(".").glob("**/*.py")
# Path(".").iterdir()
# the sitecutomize.py file should contain
#import win_unicode_console
#win_unicode_console.enable()
from enum import Enum  # only python2: pip install enum34

#######################################################################
from tomorrow import threads # DECORATOR TO RUN THREADS (pip install tomorrow)
# @threads(5)
def download(url):
    return requests.get(url)
# BETTER
from deco import concurrent, synchronized # pip install deco | LOOK AT 'pip install pebble'
# @concurrent # We add this for the concurrent function
def process_url (url, data):
  #Does some work which takes a while (more than 1 msec !!!)
  return result
# @synchronized # And we add this for the function which calls the concurrent function
def process_data_set (data):
  results = {}
  for url in urls:
    results[url] = process_url(url, data)
  return results

###########################################################################
import selectors
############################# DICT (alternative to addict)#####################
#from addict import Dict as Tree (Useful for probabilities !!!)
class DottedMixin: # PYTHON3 !!!
    __getattr__= lambda self,key: super().__getitem__(key)
    __setattr__= lambda self,key,val: super().__setitem__(key,val)
class DottedDefaultDict(DottedMixin,defaultdict): pass
Tree = lambda: DottedDefaultDict(Tree)
# a= Tree()
# a.b.c.d.e.f="whatever"
####################################################################

########################### USEFUL STUFF ##############################
ANY=type('',(),{'__eq__':lambda i,_:True})()
Null=type('',(),{'__bool__':lambda s:False,'__call__':lambda s,*a,**k: s,'__getitem__':lambda s,k:s,'__getattr__':lambda s,n:s})()
jsdict=type('jsdict',(dict,),{'__getattr__':lambda s, i: s[i]}) # a javascript dict
MAXIMUM = Ellipsis # not in python3
MINIMUM = None     # not in python3
###################

###########################################################################
# list all files in the directory containing the executable
#which youtube-dl | python -c "import pathlib;print([p.name for p in pathlib.Path(input()).parent.glob('*.py')])"
####################################################
# COMMAND-LINE
import typer # pip install typer
def main (name: str, title: bool = False):
    typer.echo (f"Hello {name}")
if __name__ == "__main__":
    typer.run (main)

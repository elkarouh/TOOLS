import os, sys
import gdb
###################################### gdb utils #################################
def print_symbols_in_frame(frame):
    print("Variables in frame:")
    for symbol in frame.block():
        print(f"{symbol.name} = {frame.read_var(symbol)}")
def local_variables(frame):
    print("Local variables:")
    for sym in frame.block():
        if sym.is_argument or sym.is_variable:
            print(f"{sym.name}={frame.read_var(sym)}")
# Define a function to retrieve the list of functions
def get_function_list():
    # Run the "info functions" command in gdb
    output = gdb.execute("info functions", to_string=True)
    # Split the output into lines
    lines = output.split("\n")
    import pprint as pp 
    # pp.pprint(lines)
    # Extract the function names from the output
    functions = [line.split()[2].split('(')[0] for line in lines if re.match("^\d+:", line)]
    pp.pprint(functions)
    # Return the list of function names
    return functions

# Define a function to retrieve the filename and line number of a function
def get_function_info(function_name):
    # Run the "info symbol" command in gdb to get the address of the function
    output = gdb.execute("info symbol " + function_name, to_string=True)
    # Extract the address from the output
    address = output.split()[0]
    # Run the "info line" command in gdb to get the filename and line number
    output = gdb.execute("info line *{}".format(address), to_string=True).split()
    # Extract the filename and line number from the output
    filename, line_number = output[3], output[1]
    print(f"{function_name} is defined in {filename} at line {line_number}")
    # Return the filename (with quotes removed) and line number
    return filename[1:-1], line_number

def print_backtrace():
    def function_arguments(frame):
        print("Function arguments:")
        arguments=[]
        for symbol in frame.block():
            if symbol.is_argument:
                value = frame.read_var(symbol)
                arguments.append(f"{symbol.name} = {value}")
        return ", ".join(arguments)
    print("Backtrace:")
    frame = gdb.newest_frame()
    frame_count = 0
    while frame:
        print(f"#{frame_count} {frame.name()} ({function_arguments(frame)})")
        block = frame.block()
        if block.is_global:
            print(f"    [non-debug info]")
        else:
            sal = frame.find_sal()
            if sal.is_valid():
                print(f"    at {sal.symtab.filename}:{sal.line}")
            else:
                print("    [source information not available]")
        frame = frame.older()
        frame_count += 1
#####################################################################################
import subprocess, re
import os.path
from datetime import datetime, timedelta
from collections import OrderedDict
import builtins

def print(*args, **kwargs):
    builtins.print(" "*G.nesting_level, *args, **kwargs)

#If passing a single string, either shell must be True or else the string must simply name the program to be executed without specifying any arguments.
def run_command(command_as_list):
    p = subprocess.Popen(
        command_as_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return iter(p.stdout.readline, b'')

def UnpackGdbVal(self, gdb_value):
    """Unpacks gdb.Value objects and returns the best-matched python object."""
    val_type = gdb_value.type.code
    if val_type == gdb.TYPE_CODE_INT or val_type == gdb.TYPE_CODE_ENUM:
      return int(gdb_value)
    if val_type == gdb.TYPE_CODE_VOID:
      return None
    if val_type == gdb.TYPE_CODE_PTR:
      return long(gdb_value)
    if val_type == gdb.TYPE_CODE_ARRAY:
      # This is probably a string
      return str(gdb_value)
    # I'm out of ideas, let's return it as a string
    return str(gdb_value)
# https://stackoverflow.com/questions/34190596/setting-a-breakpoint-in-a-specific-line-inside-a-function-with-gdb
class RelativeFunctionBreakpoint (gdb.Breakpoint):
    """
    Sets a breakpoint relative to 
    """
    def __init__(self, functionName, lineOffset=0):
        location=self.calculate(functionName, int(lineOffset))
        gdb.Breakpoint.__init__(self, location)
    def calculate(self,functionName, lineOffset):
        """
        Calculates an absolute breakpoint location (file:linenumber)
        based on functionName and lineOffset
        """
        # get info about the file and line number where the function is defined
        info_lines = gdb.execute("info line "+functionName, to_string=True) # can return several lines !!!
        filename = functionName.split(':')[0]
        for info_line in info_lines.splitlines():
            if filename in info_line:
                line_number=info_line.split()[1] # WE NAIVELY USE THE FIRST HIT !!!
                break
        else: 
            print(f"{functionName} NOT FOUND")
            return ""
        line = int(line_number)+lineOffset # add the lineOffset
        return f"{filename}:{line:d}"

class PrettyPrintCursor (gdb.Command):
    "This is the help i still need to write"
    def __init__(self, cmd_name):
        gdb.Command.__init__(self, cmd_name, gdb.COMMAND_USER)
    def invoke(self, args, tty):
        print(f"{args}")
        myargs=args.split()
        info = gdb.execute("x/200c Cursor.definition.sql_def.stream.First_Pos", to_string=True)
        print(self.decode(info))
    def decode(self, sql_str):
        sql_res="".join([(el.strip("'") if not '[' in el and "'" in el and el!="'2'" else (", " if el=='3' else '')) for line in sql_str.splitlines() for el in line.split()])
        #print(sql_res.replace('select,','select').replace('from,','from,').replace('where,','where'))
        #sql_res="select, TYPE_ID, CLASSIFICATION, TIME_STAMP, SEQUENCE_NUMBER OBJECT, SENDER_RE from, where,"
        return re.sub('(?<=(lect|from|here)),','',sql_res, flags=re.IGNORECASE)
PrettyPrintCursor('c_pp') # commands need instantiation. User shall use 'c_pp'

class Decode(gdb.Function):
    """Return a sql string. Takes a Cursor as argument.
        called as
        gdb> print $decode("Cursor")
        you need the double quotes for the arguments !!!
"""
    def __init__ (self, cmd_name):
        super (Decode, self).__init__ (cmd_name)
    def invoke (self, cursor, debug_start=0, debug_end=-1):
        print(f"==============>{str(cursor)}")
        sql_str = gdb.execute(f"x/5000c {str(cursor)[1:-1]}.definition.sql_def.stream.First_Pos", to_string=True)
        print(sql_str[debug_start:debug_end])
        sql_res="".join([(el.strip("'") if not '[' in el and "'" in el and el!="'2'" else (", " if el=='3' else (' ' if el=='0' else ''))) for line in sql_str.splitlines() for el in line.split()])
        #print(sql_res.replace('select,','select').replace('from,','from,').replace('where,','where'))
        #sql_res="select, TYPE_ID, CLASSIFICATION, TIME_STAMP, SEQUENCE_NUMBER OBJECT, SENDER_RE from, where,"
        return re.sub('(?<=(lect|from|here)),','',sql_res, flags=re.IGNORECASE)
Decode('decode')

class BreakpointCommand(gdb.Command):
    def __init__(self, cmd_name):
        gdb.Command.__init__(self, cmd_name, gdb.COMMAND_USER)
    def invoke(self, args, tty):
        myargs=args.split()
        RelativeFunctionBreakpoint(*myargs)
        #flight=gdb.parse_and_eval('for_flight.fixed_info')
        #print flight.type.keys()
        print(f"{args}")
BreakpointCommand('pybr') # e.g. pybr operational_log.adb:Insert

class PrintCommand(gdb.Command):
    def __init__(self, cmd_name):
        gdb.Command.__init__(self, cmd_name, gdb.COMMAND_USER)
    def invoke(self, args, tty):
        #gdb.execute('call (void)ConExecCommandLine("level {}")'.format(param))
        #flight=gdb.parse_and_eval('for_flight.fixed_info')
        #print flight.type.keys()
        print(f"============================> {args} <======================================")
PrintCommand('pr')

class BugReport (gdb.Command):
    """Collect required info for a bug report"""
    def __init__(self, cmd_name):
        super(BugReport, self).__init__(cmd_name, gdb.COMMAND_USER)
    def invoke(self, arg, from_tty):
        pagination = gdb.parameter("pagination")
        if pagination: gdb.execute("set pagination off")
        with open("/tmp/bugreport.txt", "w") as f:
            f.write(gdb.execute("thread apply all backtrace full", to_string=True))
        f.write(gdb.execute("thread apply all backtrace full", to_string=False))
        os.system("uname -a >> /tmp/bugreport.txt")
        if pagination: 
            gdb.execute("set pagination on")
BugReport('bugreport')
########################################################################
def run(command):
    return gdb.execute(command, to_string=True)
####################################################################################################
def get_var(varname):
    "a generic getter"
    return gdb.parse_and_eval(varname)

def get_struct(varname):
    text=get_var(varname)
    text=re.sub(r'\{',r'OP_BRA',str(text)) # use convenience function $_as_string(text) ???
    text=re.sub(r'\}',r'CLO_BRA',text)
    for m in re.finditer('OP_BRA.*?CLO_BRA',text):
        ini_text=m.group()
        if '(' in ini_text:
            group_text=re.sub(r'\(','OP_PAR',ini_text)
            group_text=re.sub(r'\)','CLO_PAR',group_text)
            group_text=re.sub(r',','COMMA',group_text)
            text=text.replace(ini_text,group_text)
    text=re.sub(r'\(','{',text)
    text=re.sub(r'\)','}',text)
    text=re.sub(r'=>\s*([^{]*?)([,}])',r': "\1"\2',text)
    text=re.sub(r'""',r'"',text)
    text=re.sub(r'=>',':',text)
    text=re.sub(r'([{,])\s*([^\s]*?)\s*:',r'\1"\2":',text)
    text=re.sub(r'OP_BRA','{',text)
    text=re.sub(r'CLO_BRA','}',text)
    text=re.sub(r'OP_PAR','(',text)
    text=re.sub(r'CLO_PAR',')',text)
    text=re.sub(r'COMMA',',',text)
    try:
        return eval(text)
    except SyntaxError:
        return {}
    except NameError:
        raise

def get_integer(varname):
    return int(gdb.parse_and_eval(varname))

def get_time(varname):
    return get_parsed_time(gdb.parse_and_eval(varname))

def get_duration(varname):
    return get_parsed_duration(gdb.parse_and_eval(varname))

def get_flight_fixed_info(for_flight_varname):
    flight = gdb.parse_and_eval('{}.fixed_info'.format(for_flight_varname))
    flight_as_dict=get_struct(flight)
    #rint(flight.type.keys())
    #ircraft_id = str(flight["aircraft_id"])[1:-1].strip()
    #act_id=flight['tact_id']

def get_airspace(varname):
    fullname = gdb.parse_and_eval(varname)
    parts = str(fullname).split()
    name = parts[0][1:]
    iden = parts[-1][1:-1]
    return name, iden

def get_parsed_time(time):
    "Wed 1998/11/11 14:01:48 {910792908} ==> datetime object"
    nr_seconds_since_epoch = int(str(time).split()[-1][1:-1])
    return datetime.fromtimestamp(nr_seconds_since_epoch)
    #datetime.fromtimestamp(910742400).strftime("%A, %B %d, %Y %H:%M:%S")

def get_parsed_duration(time):
    "0D00H21M57S {1317} ==> timedelta object"
    if 'zero' in str(time).lower():
        return timedelta()
    nr_seconds = int(str(time).split()[-1][1:-1])
    return timedelta(seconds=nr_seconds)
#####################################################################################
class Breakpoint_With_Callback(gdb.Breakpoint):
    def __init__(self, filename, function_name, line_number, nesting_level, callback, condition=None):
        self.function_name = function_name
        self.my_condition=condition
        self.nesting_level = nesting_level
        self.line_number = line_number
        self.filename = os.path.basename(filename)
        gdb.Breakpoint.__init__(self, f'{self.filename}:{line_number}', gdb.BP_BREAKPOINT, internal=False)
        self.callback = callback
        #self.silent= False

    def stop(self):
        G.nesting_level = self.nesting_level
        # Check the condition if provided
        if self.my_condition:
            if not gdb.parse_and_eval(self.my_condition):
                return False  # Condition not met, just continue
        print("_"*40)
        print(f"Breakpoint {self.number}: {self.filename} {self.function_name} at line {self.line_number}")
        print(f"hit #{self.hit_count+1}")
        return self.callback()  # make the callback return True if you want to stop

def retrieve_breakpoints(filename):
    """returns a list of (function,line_number,callback,nesting_level)
    This assumes that you added '--py <your_python_callback>' to the 
    lines you are interested in.
    """
    breakpoints = []
    current_function = None
    for line_number, line in enumerate(open(filename, "r"), 1):
        if filename.endswith('adb'): # ADA program
            if line.lstrip().startswith(("procedure", "function")):
                current_function = line.split()[1]
                nesting_level = len(line) - len(line.lstrip(' ')) # number of leading spaces !!!
        else: # c program
            if line.startswith('void'): # HACK !!!
                current_function=line.split()[1].split('(')[0]
                nesting_level = len(line) - len(line.lstrip(' ')) # number of leading spaces !!!
        if "--py" in line:
            py_func = line.partition("--py")[-1].strip()
            breakpoints.append(
                (current_function, line_number, py_func, nesting_level))
    return breakpoints

def register_breakpoints_from_annotated_files(instrumented_source_filenames, global_dict):
    for filename in instrumented_source_filenames:
        for function_name, line_number, full_callback, level in retrieve_breakpoints(filename):
            #print("2----------",function_name, line_number, full_callback)
            components=full_callback.split(' ',1)
            if len(components)==1:
                callback, condition=components[0], None
            else:
                callback, condition=components[0], components[1].split(' ',1)[-1]
            Breakpoint_With_Callback(filename, function_name, line_number, level, global_dict[callback], condition)

def break_in_all_functions(source_filenames, callback):
    for function_name in get_function_list():
        filename, line_number= get_function_info(function_name)
        if filename in source_filenames and function_name != 'main':
            Breakpoint_With_Callback(filename, function_name, line_number, 0, callback)
            print(f"Breakpoint set at function '{function_name}' in '{filename}'")
            print(f"{filename}:{function_name}")
#########################################################################
def remove_instrumentation(filename):
    import fileinput
    for line in fileinput.input(filename, inplace=True):
        if "--py" in line:
            line, sep, comm = line.partition("--py")
            print(line)
        else:
            print(line, end='')

def save_instrumentation(filename):
    backup_filename = filename+".INSTRUMENTED"

def restore_instrumentation(filename):
    "should use MELD"

#########################################################################
def setup(register_callback):
    if os.environ.get("ATTACH"):
        print("Attaching")
        G.REMOTE = True
    else:
        #print("NOT ATTACHING !!!!")
        pass
    G.my_executable = os.environ.get("PROGRAM_PATH")
    G.my_params = os.environ.get("PARAMS")
    G.my_process_name = os.environ.get("PROCESS_NAME")
    G.old_stdout = sys.stdout
    sys.stdout = G.my_ram_log_file
    G.my_pid=os.environ.get("ATTACH_PID")
    if G.REMOTE:
        if not G.my_pid:
            G.my_pid = next(run_command(["pgrep", G.my_process_name])).strip()
    # print('HEK running gdb command: ------ >'+ 'file ' + G.my_executable)
    if not G.my_pid:
        gdb.execute('file ' + G.my_executable)
    register_callback()

def run():
    if G.REMOTE:
        print("ATTACHING TO PID", G.my_pid)
        gdb.execute('attach ' + G.my_pid)
        gdb.execute('continue')
    else:
        #print(f'HEK running gdb command: ------ > run {G.my_params}')
        # gdb.execute('run {}'.format(G.my_params))
        gdb.execute(f'run {G.my_params}')
    print("#################################################################################")


def tear_down(postprocessing_callback):
    G.my_ram_log_file.close()
    sys.stdout = G.old_stdout
    #print("HEK:done")
    with open(G.my_persistent_log_filename, "w") as f:
        for line in open(G.ram_filename, "r"):
            print(line, end="")
            f.write(line)
    postprocessing_callback()
    gdb.execute('quit')

def error_message(filename):
    print("This program is not to be run directly")
    process_name = os.path.basename(filename)[4:-4]
    print("drun {} params".format(process_name))
    print("OR")
    print("drun attach {} PID?".format(process_name))
    sys.exit()
#########################################################################################
class G:
    REMOTE = False
    my_persistent_log_filename = "MYLOG.log"
    ram_filename = "/dev/shm/mylogfile"
    my_ram_log_file = open(ram_filename, "w")
    nesting_level = 0
    ################
    my_params=""

def main(register_callback, postprocessing_callback=lambda:None):
    setup(register_callback)
    run()
    tear_down(postprocessing_callback)

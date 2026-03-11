#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, sys
from collections import defaultdict
from pathlib import Path
#from rich import print
from typing import Hashable, Dict, List
import array
import networkx as nx
import pickle
####################################################################################
class Style: # won't work if you do  'from rich import print'
    def __init__(self, code):
        prefix,suffix='\x1b[','m'
        self.on, self.off = f"{prefix}{code}{suffix}", f"{prefix}0{suffix}"
    def __call__(self, *args):
        return f"{''.join([f'{self.on}{arg}' for arg in args])}{self.off}"
    def __ror__(self,other):
        return self(other)
reset,bold,dim,_,underscore,blink,_,inverted = [Style(0+i) for i in range(8)]
black,red,green,yellow,blue,magenta,cyan,white = [Style(30+i) for i in range(8)]
bg_black,bg_red,bg_green,bg_yellow,bg_blue,bg_magenta,bg_cyan,bg_white,_,bg_default = [Style(40+i) for i in range(10)]
align = lambda length: lambda fun: lambda string: f"{fun(string):<{length}}" # decorator for function returning strings
############################
def detect_file_encoding():
    with open(mutable_file, 'rb') as f:
        content = f.read()
        import chardet
        encoding = chardet.detect(content)
        print("ENCODING=",encoding)
#######################################################################################
def ptree(tree_dict, reverse=False, prefix: str='', checked_out_files=[], failed_tests=[]):
    """A recursive generator.
    will yield a visual tree structure line by line with each line prefixed by the same characters
    """
    # prefix components:
    space =  '    '
    branch = '│   '
    # pointers:
    if reverse:
        tee =    '├<─ '
        last =   '└<─ '
    else:
        tee =    '├── '
        last =   '└── '
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(tree_dict)-1) + [last]
    for pointer, path in zip(pointers, tree_dict):
        ann_path=path
        if path in checked_out_files:
            ann_path= path + " -> CO"| bold | red
        if path in failed_tests:
            ann_path= path + " -> FAILED"| bold | red | underscore
        #print(f"==== {prefix=} {pointer=} {path=}")
        if not prefix:
            yield ann_path | bg_blue | bold | white | underscore
        else:
            yield prefix + pointer + ann_path
        if tree_dict[path]: # extend the prefix and recurse:
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            yield from ptree(tree_dict[path], reverse, prefix=prefix+extension, checked_out_files=checked_out_files, failed_tests=failed_tests)
######################################################################################################################
##################################################################################################################
if 0:
    import graph_force
    # we have to map the names to integers
    # as graph_force only supports integers as node ids at the moment
    edges = []
    mapping = {n: i for i, n in enumerate(G.nodes)}
    i = 0
    for edge in G.edges:
        edges.append((mapping[edge[0]], mapping[edge[1]]))
    positions = graph_force.layout_from_edge_list(len(G.nodes), edges, iter=1000,
                                        model="spring_model") # model to use, default "spring_model", other option is "networkx_model"


# LAYOUT ALGORITHM
class LayoutedGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        # Call the __init__ method of the parent class (nx.DiGraph)
        super().__init__(*args, **kwargs)
        self.initialize_positions()
    def initialize_positions(self):
        positions = {}
        for node in self.nodes():
            positions[node] = (random.random(), random.random())
        return positions
    def update_positions(self, positions):
        new_positions = positions.copy()
        for node in self.nodes():
            total_force = (0, 0)
            for other_node in self.nodes():
                if node != other_node:
                    # Calculate and accumulate attractive force
                    repulsive_force = calculate_force(positions[node], positions[other_node], repulsive_force=False)
                    total_force = (total_force[0] + attractive_force[0], total_force[1] + attractive_force[1])

                    # Calculate and accumulate repulsive force
                    repulsive_force = calculate_force(positions[node], positions[other_node], repulsive_force=True)
                    total_force = (total_force[0] - repulsive_force[0], total_force[1] - repulsive_force[1])
            # Update node position based on the total force
            new_positions[node] = (positions[node][0] + total_force[0], positions[node][1] + total_force[1])
        return new_positions
    def graph_layout(self, iterations):
        positions = initialize_positions(self)
        for iteration in range(iterations):
            # Update node positions iteratively
            positions = update_positions(self, positions)
        return positions
    def visualize_dag(self, round_angle: bool = False) -> str:
        """
        Copied from https://github.com/ctongfei/py-dagviz/blob/main/dagviz.py
        Creates a text rendering of a directed acyclic graph (DAG) for visualization purposes in a terminal.

        :param round_angle: Whether to use a round-angled box drawing character or not
        :return: A multi-line string representation of the directed acyclic graph, each line corresponding to a node
        """
        assert nx.is_directed_acyclic_graph(self), "Graph contains cycles"

        rows: List[Hashable] = []
        node_to_row: Dict[Hashable, int] = {}
        indents: List[int] = []

        def _process_dag(self, indent: int):
            for sg in nx.weakly_connected_components(self):
                _process_component(self.subgraph(sg), indent=indent)

        def _process_component(self, indent: int):
            sources = [v for v in g.nodes if g.in_degree(v) == 0]
            for i in range(len(sources)):
                node_to_row[sources[i]] = len(rows)
                rows.append(sources[i])
                indents.append(indent + i)
            _process_dag(
                g.subgraph(set(self.nodes).difference(sources)),
                indent=indent + len(sources)
            )

        _process_dag(self, indent=0)
        a = [
            array.array('u', [u' '] * indents[i] * 2)
            for i in range(len(rows))
        ]
        for i, u in enumerate(rows):
            successors = sorted(self.successors(u), key=lambda v: node_to_row[v])
            if len(successors) == 0:
                continue
            l = node_to_row[successors[-1]]
            for j in range(i + 1, l):
                a[j][indents[i] * 2] = u'│'
            for v in successors[:-1]:
                j = node_to_row[v]
                a[j][indents[i] * 2] = u'┼' \
                    if indents[i] > 0 and a[j][indents[i] * 2 - 1] == u'─' \
                    else u'├'
                for k in range(indents[i] * 2 + 1, indents[j] * 2):
                    a[j][k] = u'─'
            a[l][indents[i] * 2] = u'┴' \
                if indents[i] > 0 and a[l][indents[i] * 2 - 1] == u'─' \
                else (u'╰' if round_angle else u'└')
            for k in range(indents[i] * 2 + 1, indents[l] * 2):
                a[l][k] = u'─'

        lines: List[str] = [l.tounicode() + "• " + str(i).replace('\n', ' ') for l, i in zip(a, rows)]
        return '\n'.join(lines)

def calculate_force(position1, position2, repulsive_force=True):
    # Simple inverse distance law for repulsion/attraction
    dx = position2[0] - position1[0]
    dy = position2[1] - position1[1]
    distance = max(math.sqrt(dx**2 + dy**2), 0.0001)  # Avoid division by zero
    force_magnitude = (1.0 if repulsive_force else 0.1)/ distance
    force_vector = (force_magnitude * dx / distance, force_magnitude * dy / distance)
    return force_vector

#####################################################################################################################
#from joblib import Memory
#mem=Memory('/tmp/cache')
#import scrat as sc
class DependencyGraph(nx.DiGraph):
    def __init__(self, *args, **kwargs):
        # Call the __init__ method of the parent class (nx.DiGraph)
        super().__init__(*args, **kwargs)
        #self.d2_connections=[] # for the d2 graph
        self.dependency_graph_computed=False
        self.checked_out_files=set()
        self.failed_tests=set()
        self.files_not_causing_failures=set()
        self.failed_templates=set()
        #self.compute_vob_dependency_graph=mem.cache(self.compute_vob_dependency_graph)
        #self.compute_vob_dependency_graph=sc.stash()(self.compute_vob_dependency_graph)
    def compute_vob_dependency_graph(self):
        if Path("/tmp/graph.gpickle").exists():
            print("retrieving dependency graph from /tmp/graph.gpickle")
            # G= nx.read_gpickle("/tmp/graph.gpickle")
            with open("/tmp/graph.gpickle",'rb') as f:
                G= pickle.load(f)
            self.add_nodes_from(G)
            self.add_edges_from(G.edges)
            self.dependency_graph_computed=True
            return
        print("computing dependency graph")
        directory_path=os.environ['CM_VOB']
        directory_path="/auto/home/ekhassan/LOCAL_SSD/ETFMS"
        for mutable_file in Path(directory_path).rglob("*.el"):
            if DEBUG:
                print("Processing {}".format(mutable_file))
            if "remote_opticon" in mutable_file.name: # unicode errors
                continue
            if "hot_replay_scenario_warmup" in mutable_file.name: # unicode errors
                continue
            self.get_file_dependencies(mutable_file)
        for mutable_file in Path(directory_path).rglob("*.ksh"):
            #print("Processing {}".format(mutable_file))
            if "do_runtime_compare_database.CONFIG" in mutable_file.name:
                continue
            if "demo_zenity" in mutable_file.name:
                continue
            if "ospace" in mutable_file.name:
                continue
            self.get_file_dependencies(mutable_file)
        self.dependency_graph_computed=True
        # Pickle the graph to a file
        #nx.write_gpickle(self, "/tmp/graph.gpickle")
        with open("/tmp/graph.gpickle",'wb') as f:
            pickle.dump(self,f, pickle.HIGHEST_PROTOCOL)
    def get_file_dependencies(self, mutable_file):
        def check_file_path_syntax(filepath):
            filepath = filepath.strip()
            if filepath.startswith('.'):
                return False
            if ')' in filepath:
                if DEBUG:
                    print(f"1. SYNTAX ERROR in {mutable_file} : {filepath}")
                return False
            if '(' in filepath:
                if DEBUG:
                    print(f"2. SYNTAX ERROR in {mutable_file} : {filepath}")
                return False
            try:
                os.path.normpath(filepath)
                return True
            except ValueError:
                if DEBUG:
                    print(f"3. SYNTAX ERROR in {mutable_file} : {filepath}")
                return False
        with open(mutable_file, 'r') as f:
            pattern = r'(call|cpe.receive_msg|check_printer_equal_template|generate_flights|unix_command) "?(.*?)"?\)?'
            file_to_modify=os.path.basename(mutable_file)
            DEBUG=file_to_modify=="XXXan2_fltgen_all_flights.el"
            self.add_node(file_to_modify)
            for line in f:
                line=line.split(';')[0] # remove comments
                matches = re.findall(pattern, line)
                if DEBUG:
                    print("LINE=",line,end="")
                for m in matches:
                    command_type, extracted_text = m
                    if DEBUG:
                        print(f"{command_type=}")
                    if command_type=="generate_flights":
                        parts=line.split()
                        generated_file=parts[2]
                        dep_template=parts[3]
                        #print(f"{file_to_modify=} {generated_file=} {dep_template=}")
                        self.add_edge(generated_file,file_to_modify) # if the generated file is modified, it will impact the file_to_modify
                        self.add_edge(dep_template, generated_file) # if the template is modified, it will impact the generated_file
                        continue
                    if file_to_modify.endswith(".ksh") and command_type=="call":
                        continue
                    if command_type=="unix_command":
                        parts=line.split()
                        if len(parts)==2: # (call inject_9999)
                            dependency=parts[1].strip(')')+".ksh"
                            if DEBUG:
                                print(f"{command_type=} {file_to_modify=} {dependency=}")
                            self.add_edge(dependency, file_to_modify) # if the called file is modified, it will impact 'file_to_modify'
                        continue
                    extracted_text=line.split()[-1].strip(')')
                    if DEBUG and 0:
                        print("1", extracted_text)
                        print("2", extracted_text.strip('"\n'))
                    extracted_text=extracted_text.strip('"\n').split()
                    if not extracted_text:
                        continue
                    if DEBUG:
                        print(f"3: {extracted_text=}")
                        print("4", extracted_text[-1].strip('"\n').split()[-1])
                    path=extracted_text[-1].strip('"\n')
                    if check_file_path_syntax(path): # the path is syntactically valid
                        dependency=os.path.basename(path)
                        if dependency=='printer_1hdr.out':
                            continue
                        if dependency=="tacot.cleanup_tact_data.el": # reduce graph size (read-only file?)
                            continue
                        # print(f"Command type: {command_type}, dependency: {dependency}")
                        match command_type:
                            case "call":
                                self.add_edge(dependency, file_to_modify) # if the called file is modified, it will impact 'file_to_modify'
                            case "cpe.receive_msg":
                                dependency+=".templ"
                                self.add_edge(file_to_modify, dependency) # if the file_to_modify is modified, it will impact the template
                                self.add_edge(dependency, file_to_modify) # if the template is modified, it will impact 'file_to_modify'
                            case "check_printer_equal_template":
                                #print(f"{command_type=} {file_to_modify=} {dependency=}")
                                self.add_edge(file_to_modify, dependency) # if the file_to_modify is modified, it will impact the template
                                if not dependency.endswith('gif'):
                                    self.add_edge(dependency, file_to_modify) # if the template is modified, it will impact 'file_to_modify'
                            case _:
                                print(f"THIS SHOULD NOT HAPPEN: {file_to_modify=} {command_type=} {line=}")
                                raise SystemExit

            # if mutable_file.endswith("test_simul_load_intentions.el"):
            #     raise SystemExit
    def build_checked_out_files_list(self):
        """the CHECKED_OUT_FILES file is produced by running:
        hsb > CHECKED_OUT_FILES
        the file contains one filename per line
        """
        if not os.path.exists('CHECKED_OUT_FILES'):
            #print('CHECKED_OUT_FILES not file found !!!!!!!!!!!!!!!!!\nPlease do: hsb > CHECKED_OUT_FILES')
            print('CHECKED_OUT_FILES not file found !!!!!!!!!!!!!!!!!\nPlease do:Cshow_branches -f -noc > CHECKED_OUT_FILES')
            return
        with open('CHECKED_OUT_FILES', 'r') as file:
            for line in file:
                self.checked_out_files.add(os.path.basename(line.strip()))
    def build_failed_tests_list(self):
        """the FAILED_TESTS file is produced by running:
        failed_tests > FAILED_TESTS
        the file contains one filename per line
        """
        if not os.path.exists('FAILED_TESTS'):
            #print('FAILED_TESTS file not found !!!!!!!!!!!!!!!!!\nPlease do: failed_tests > FAILED_TESTS')
            print('FAILED_TESTS file not found !!!!!!!!!!!!!!!!!\nPlease provide a list of failed tests in a file called FAILED_TESTS')
            return
        with open('FAILED_TESTS', 'r') as file:
            for line in file:
                self.failed_tests.add(os.path.basename(line.strip()))
    def build_failed_templates_list(self):
        """the FAILED_TEMPLATES file is produced by running:
        failed_templates > FAILED_TEMPLATES
        the file contains one filename per line
        """
        if not os.path.exists('FAILED_TEMPLATES'):
            #print('FAILED_TEMPLATES file not found !!!!!!!!!!!!!!!!!\nPlease do: failed_templates > FAILED_TEMPLATES')
            return
        with open('FAILED_TEMPLATES', 'r') as file:
            for line in file:
                self.failed_templates.add(line.strip())
    ###############################################################
    def print_all_paths_from(self,start_node, reverse=False):
        if not self.dependency_graph_computed:
            self.compute_vob_dependency_graph()
        tree = lambda: defaultdict(tree)
        if reverse:
            print("computing reverse graph")
            G=nx.reverse_view(self)
        else:
            G=self
        # Use depth-first search to get edges in DFS traversal order
        dfs_edges = nx.dfs_edges(G, source=start_node)
        paths=[]
        current_path=[start_node]
        previous_depth=0
        for edge in dfs_edges:
            depth=nx.shortest_path_length(G,start_node, edge[1])
            if depth > previous_depth:
                current_path.append(edge[1])
            else:
                paths.append(tuple(current_path))
                current_path=current_path[:depth]+[edge[1]]
            previous_depth=depth
            #print(f"{edge[0]} -> {edge[1]}")
        else:
            paths.append(current_path)
        mydict = tree()
        for path in paths:
            current = mydict
            for node in path:
                current = current[node]
        for line in ptree(mydict,reverse, prefix="", checked_out_files=self.checked_out_files, failed_tests=self.failed_tests):
            print(line)

    def print_all_paths_to(self,start_node):
        self.print_all_paths_from(start_node, reverse=True)
    def nodes_affected_by(self, start_node, reverse=False):
        if not self.dependency_graph_computed:
            self.compute_vob_dependency_graph()
        if reverse:
            G=nx.reverse_view(self)
        else:
            G=self
        # Use depth-first search to get edges in DFS traversal order
        dfs_edges = nx.dfs_edges(G, source=start_node)
        # Extract nodes from the edges
        visited_nodes = set()
        for edge in dfs_edges:
            visited_nodes.add(edge[1])  # Add the target node of each edge
            # self.d2_connections.append(D2Connection(shape_1=edge[0], shape_2=edge[1]))
        return visited_nodes
    def nodes_affecting(self, start_node):
        return self.nodes_affected_by(start_node, reverse=True)
    def connected_components(self, start_node):
        if not self.dependency_graph_computed:
            self.compute_vob_dependency_graph()
        return set(nx.ancestors(self, start_node)) | set(nx.descendants(self, start_node))
    ##############.0
    ###################################################
    def plot_reachable_subgraph(self, impacted_nodes):
        # Create a subgraph induced by reachable nodes
        reachable_subgraph = self.subgraph(impacted_nodes)
        # Plot the reachable subgraph
        nx.draw(reachable_subgraph, with_labels=True, font_weight='bold', node_color='lightcoral')
        plt.title('Reachable Subgraph')
        fig = plt.figure()
        mpld3.save_html(fig, "interactive_graph.html")
        nx.drawing.nx_pydot.write_dot(reachable_subgraph,'path_graph.dot')
        # dot -Tsvg path_graph.dot > path_graph.svg
        # mpld3.show()
        return
        # Save the interactive plot as an SVG file
        # Save the plot as an SVG file
        #plt.savefig("large_graph.svg", format="svg")
        #plt.show()

    def make_dependency_list(self):  # make a dependency list, sorted in ascending order of out_degree
        # Get a list of nodes sorted by out-degree in asscending order
        sorted_nodes = sorted(G.nodes(), key=self.out_degree, reverse=False)
        # Now, sorted_nodes contains the nodes of G sorted by the out-degree of their out-edges
        for node in sorted_nodes:
            print(self.out_degree(node), node)
    def list_checked_out_files_not_causing_failures(self):
        for i,f in enumerate(self.checked_out_files):
            DEBUG=f=="sssssssssinject_40.el"
            if DEBUG:
                print("Processing-- ",f)
            if f in self.failed_tests:
                continue
            affecting_nodes=self.nodes_affecting(f)
            if DEBUG:
                print(f"{affecting_nodes=}")
            for file in {f} | affecting_nodes: # itself and all ancestors
                if DEBUG:
                    print("\t===>",file)
                if file not in self.checked_out_files: # on the condition that they are checked out
                    if DEBUG:
                        print(f"skipping {file}")
                    continue
                if DEBUG:
                    print(f"affecting CO node={file}")
                affected_nodes=self.nodes_affected_by(file)
                if not affected_nodes.isdisjoint(self.failed_tests):
                    # intersection not empty, we break out of this inner loop
                    break
            else:
                 self.files_not_causing_failures.add(f)
                 print(f"================================> {f} didn't cause any failure")
    def list_failed_tests_with_checkedout_dependencies(self):
        for f in self.failed_tests:
            if f in self.checked_out_files:
                print(f"FAILED TEST: {f} (was modified)")
            else:
                print(f"FAILED TEST: {f} untouched")
            for file in self.nodes_affecting(f): # all ancestors
                if file in self.checked_out_files: # on the condition that they are checked out
                    print(f"\t{file} (was modified)")
                elif file.endswith("templ"):
                    print(f"\t{file} (untouched but may have caused the test to fail )")
    def list_failed_tests_connected_components(self):
        for f in self.failed_tests:
            print(f"FAILED TEST: {f}")
            l=list(self.connected_components(f))
            for file in self.connected_components(f): # weakly connected components
                print(f"\t{file}")
    def list_failed_tests_without_dependencies(self):
        for f in self.failed_tests:
            l=list(self.connected_components(f))
            if len(l)==0:
                print(f"FAILED TEST: {f} has no dependencies")
    def list_failed_tests_because_of_templates(self):
        for f in self.failed_tests:
            failed_templates=[template for template in self.nodes_affecting(f) if template in self.failed_templates]
            if failed_templates:
                print(f"{f} failed")
                for templ in failed_templates:
                    print(f"\t--> {templ}")

if __name__ == "__main__":
    DEBUG=False
    G=DependencyGraph()
    G.compute_vob_dependency_graph()
    print("Done computing dependencies")
    G.build_checked_out_files_list()
    print("built co filelist")
    G.build_failed_tests_list()
    print("built failed filelist")
    G.build_failed_templates_list()
    print("GRAPH=",G)
    from IPython import embed
    if 0:
        if 0:
            G.make_dependency_list()
        if 0:
            # Cremove_view_branch $(hsb |  while read f;do cc_has_not_changed $f;done)
            G.list_checked_out_files_that_have_not_changed()
        if 1:
            print("CHECKED OUT FILES NOT CAUSING TESTS FAILURE")
            G.list_checked_out_files_not_causing_failures()
        if 0:
            G.list_failed_tests_with_checkedout_dependencies()
        if 0:
            G.list_failed_tests_without_dependencies()
        if 1:
            print("FAILED TESTS BECAUSE OF TEMPLATES")
            G.list_failed_tests_because_of_templates()
        if 0:
            G.list_failed_tests_connected_components()
        #from ptpython.ipython import embed
        embed()
        raise SystemExit

    if sys.argv[1:]:
        the_file_to_modify=sys.argv[1]
        print("IF YOU MODIFY THIS FILE, THE FOLLOWING FILES WILL BE IMPACTED")
        G.print_all_paths_from(the_file_to_modify)
        print("="*80)
        print("FILES THAT COULD IMPACT THIS FILE")
        the_failed_tests=the_file_to_modify
        G.print_all_paths_to(the_failed_tests)
        print("----------")
        print(G.nodes_affecting(the_failed_tests))
        to=G.print_all_paths_to
        fr=G.print_all_paths_from
        if 0:
            from IPython import embed
            #from ptpython.ipython import embed
            embed()
        if sys.argv[2:]:
            the_target_file=sys.argv[2]
            # Find the shortest path from file_to_modity to one of the impacted files
            try:
                path = nx.shortest_path(G, source=the_file_to_modify, target=the_target_file)
            except Exception as e:
                print(e)
                raise SystemExit
            # Print the path
            print(" -> ".join(f for f in path))
            print("ALL PATHS:")
            paths = nx.all_simple_paths(G, source=the_file_to_modify, target=the_target_file, cutoff=None)
            # paths = nx.all_shortest_paths(G, source=the_file_to_modify, target=the_target_file)
            # You can then iterate over the paths and work with them as needed
            for path in paths:
                # Do something with each path
                print(" -> ".join(f for f in path))
    else:
        embed()
        # the_file_to_modify="set_deviation_parameters.el"
        the_file_to_modify="set_ops_parameters.el"
        the_file_to_modify="inject_40.el"
        the_file_to_modify="inject_3.el"
        the_file_to_modify="make_flight_f103.el"
        the_file_to_modify="aowir_shift_del_to_shift.el"

        impacted_nodes=G.nodes_affected_by(the_file_to_modify)
        if 0:
            G.plot_reachable_subgraph(impacted_nodes)
            # PRODUCE THE d2 file
            diagram = D2Diagram(connections=connections)
            with open("graph.d2", "w", encoding="utf-8") as f:
                f.write(str(diagram))

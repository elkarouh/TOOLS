# -*- coding: utf-8 -*-
"""
Most dynamic programming problems can be solved more elegantly by
searching for a shortest path. Look at the file hek_shortest_path_TEST.py
for a reimplementation of these problems
"""
# TODO, implement stagecoach problem
# TODO, Weighted Interval Scheduling=max independent set in graph theory
# (https://en.wikipedia.org/wiki/Interval_scheduling)
# (http://farazdagi.com/blog/2013/weighted-interval-scheduling/)
# REMARK, better use ShortestPath than this as it is more intuitive, see HEK_SHORTEST_PATH.py
"""
Common Characteristics

The problem can be divided into stages with a decision required at each stage.
There are as many stages as there as decisions to be taken.
Each stage has a number of states associated with it.
The decision at one stage transforms one state into a state in the next stage.
Given the current state, the optimal decision for each of the remaining states
does not depend on the previous states or decisions.
There exists a recursive relationship that identifies the optimal decision for
stage j, given that stage j+1 (or j-1 depending on the direction of recursion)
has already been solved.
The final stage must be solvable by itself (all decisions should finally bring
you in a state that is trivially solvable).
The last two properties are tied up in the recursive relationships given above.
The big skill in dynamic programming, and the art involved, is to take a
problem and determine stages and states so that all of the above hold.
If you can, then the recursive relationship makes finding the values
relatively easy.

"""
#from __future__ import division
import time, sys
from itertools import combinations as xcombinations
def partition(iterable, func):
    result = {}
    for i in iterable:
        result.setdefault(func(i), []).append(i)
    return result
def time_it(f):
    """ Annotate a function with its elapsed execution time. """
    def timed_f(*args, **kwargs):
        t1 = time.time()
        try:
            res=f(*args, **kwargs)
        finally:
            t2 = time.time()
        timed_f.func_time = (t2 - t1, (t2 - t1) * 1000.0)
        if __debug__:
            sys.stdout.write("%s took  %0.3fs %0.3fms\n" % (
                    # f.func_name,
                    f.__name__,
                    timed_f.func_time[0],
                    timed_f.func_time[1],
                    ))
        return res
    return timed_f

class memoize: # A useful decorator
    def __init__(self, function):
        self.function = function
        self.memoized = {}
    def __call__(self, *args):
        try:
            return self.memoized[args]
        except KeyError:
            self.memoized[args] = self.function(*args)
            return self.memoized[args]

class Tuple(tuple):
    def __new__(cls, *values):
        self = tuple.__new__(cls, values)
        return self
    def __add__(self,other):
        return (self[0]+other[0],self[1]+other[1])
    def __radd__(self,other):
        return (other[0]+self[0],other[1]+self[1])
    def __mul__(self,other):
        return (self[0]*other[0],self[1]+other[1])
##############################################################################
def capital_budgetting_problem(): # BINARY KNAPSACK
    # only two decisions at each stage, take it or leave it
    items_values=[15,24,39,63,102,165,267,432,699,1131,1830]
    items_weights=items_values # equal to values in this case
    knapsack_capacity=2498
    # Capital budgetting problem
    items_values=[11000,8000,6000,4000]
    items_weights=[7000,5000,4000,3000]
    knapsack_capacity=14000
    bias=10 # we don't want too many intervals
    # def V(state) returns max([(cost_of_decision,decision)+V(newstate) for all decisions in state]) #
    # first argument=stage, second argument=state
    def V(item_idx,remaining_capa): # integer binary knapsack
        if remaining_capa==0:
            return 0,[]
        if item_idx<0:
            return 0,[]
        weight=items_weights[item_idx]
        value=items_values[item_idx]
        if weight > remaining_capa: # Only one possible choice, reject
            return V(item_idx-1,remaining_capa)
        else: # two possible choices, the reject choice and the accept choice
            return max(
                V(item_idx-1,remaining_capa),
                Tuple(value-bias,[item_idx])+V(item_idx-1,remaining_capa-weight)
                )
    #return V(len(items_values)-1,knapsack_capacity)
    def V2(item_idx,remaining_capa): # integer binary knapsack
        if remaining_capa==0:
            return 0,[]
        if item_idx==len(items_values):
            return 0,[]
        weight=items_weights[item_idx]
        value=items_values[item_idx]
        if weight > remaining_capa: # Only one possible choice, reject
            return V2(item_idx+1,remaining_capa)
        else: # two possible choices, the reject choice and the accept choice
            return max(
                V2(item_idx+1,remaining_capa),
                Tuple(value-bias,[item_idx])+V2(item_idx+1,remaining_capa-weight)
                )
    #return V2(0,knapsack_capacity) # gives indices==[0, 2, 3] !
    def iterV(item_idx_max,remaining_capa_max):
        P={}
        for item_idx in range(item_idx_max+1):
            for remaining_capa in range(remaining_capa_max+1):
                weight=items_weights[item_idx]
                value=items_values[item_idx]
                if item_idx==0:
                    if remaining_capa < weight: # Only one choice, reject
                        P[item_idx,remaining_capa]= 0,[]
                    else:
                        P[item_idx,remaining_capa]=value-bias,[item_idx]
                else:
                    if remaining_capa < weight: # Only one possible choice, reject
                        P[item_idx,remaining_capa]= P[item_idx-1,remaining_capa]
                    else: # two choices, the reject choice and the accept choice
                        P[item_idx,remaining_capa]= max(
                            P[item_idx-1,remaining_capa],
                            Tuple(value-bias,[item_idx])+P[item_idx-1,remaining_capa-weight]
                            )
        return P[item_idx_max,remaining_capa_max]
    return iterV(len(items_values)-1,knapsack_capacity)

val,indices=capital_budgetting_problem()

#assert indices==[0, 2, 3], indices
assert indices==[3, 2, 0], indices
#assert indices==[10, 7, 5, 3]
#print "true value=",sum(items_values[idx] for idx in indices)
#raise SystemExit
# EQUIPMENT REPLACEMENT PROBLEM
"""
Suppose a shop needs to have a certain machine over the next five year period.
Each new machine costs $1000. The cost of maintaining the machine during its
ith year of operation is as follows:  60 ,80   , and 120  .
A machine may be kept up to three years before being traded in.
The trade in value after i years is 1000, 800, 600 and 500.
How can the shop minimize costs over the five year period?
"""
def equipment_problem():
    market_value=[1000,800,600,500]
    maintenance_cost=[60,80,120]
    # first argument=stage(year) , second argument=state(age)
    def V(year,age):
        if year==5: # the final stage
            return -market_value[age],["sell"]
        if year==0:
            return Tuple(1000+maintenance_cost[0],["buy"])+V(year+1,age+1)
        if age==3: # Only one possible choice, trade (=sell and buy)
            return Tuple(-market_value[age]+1000+maintenance_cost[0],["trade"])+V(year+1,1)
        else: # two possible choices, the keep choice and the trade choice
            return min(
                Tuple(-market_value[age]+1000+maintenance_cost[0],["trade"])+V(year+1,1), # TRADE
                Tuple(maintenance_cost[age],["keep"])+  V(year+1,age+1) # KEEP
                )
    return V(0,0)
assert equipment_problem()==(1280, ['buy', 'keep', 'keep', 'trade', 'trade', 'sell'])

# STUDENT EXAMINATION PROBLEM
"""A student is currently taking three courses. It is important that he not
fail all of them. If the probability of failing French is p1, the probability of
failing English is p2 , and the probability of failing Statistics is p3  ,
 then the probability of failing all of them is p1*p2*p3  .
He has left himself with four hours to study. How should he minimize his
probability of failing all his courses? The following gives the probability
of failing each course given he studies for a certain number of hours on that
subject """
def student_problem():
    french=[0.8,0.7,0.65,0.62,0.6] # FAILURE PROB for 0, 1, 2, ... hours
    english=[0.75,0.7,0.67,0.65,0.62]
    statistics=[0.9,0.7,0.6,0.55,0.5]
    stages=[french,english,statistics]
    # first argument=stage, second argument=state
    def V(stage,hours_to_spend):
        if stage==2: # the final stage, use up all remaining hours
            return statistics[hours_to_spend],[hours_to_spend]
        else: # stages 0 and 1, max 5 possible choices
            return min(
                [Tuple(stages[stage][number_hours],[number_hours])*V(stage+1,hours_to_spend-number_hours)
                    for number_hours in range(hours_to_spend+1)]
                )
    return V(0,4)
assert student_problem() == (0.28875000000000001, [1, 0, 3])
# LEAST SQUARES SEGMENTATION OF A SERIES
# divide a series into  G groups
# there are G decisions==> G stages
"""
TODO: # https://kartikkukreja.wordpress.com/2013/10/21/segmented-least-squares-problem/
from scipy import stats
x = [5.05, 6.75, 3.21, 2.66]
y = [1.65, 26.5, -5.93, 7.96]
gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)
print gradient, intercept, r_value, p_value, std_err
"""
def least_square_segmentation_problem(series,number_segments):
    def sum_sq(arr):
        return sum([x*x for x in arr])
    def SSD(start,end): # divide by  N to get the variance
        """
        Sum of Square Differences
        SSD=Σ(x-µ)²=Σ(x²)-Σx²/n
        """
        sum_end=sum(series[:end])
        sum_start=sum(series[:start])
        sum_start_end=sum_end-sum_start
        sumsq_end=sum_sq(series[:end])
        sumsq_start=sum_sq(series[:start])
        sumsq_start_end=sumsq_end-sumsq_start
        return sumsq_start_end- (sum_start_end*sum_start_end)/(end-start)
    # first argument=stage, second argument=state
    def V(nr_seg,j): # divide series[:j] into nr_seg segments
        #print "===>",nr_seg,j
        if nr_seg==1: # the final stage
            return SSD(0,j),[series[:j]]
        else: # the other stages
            return min(
                [Tuple(SSD(i,j),[series[i:j]])+V(nr_seg-1,i) for i in range(nr_seg-1,j)]
                )
    def iterV(number_segments,length):
        P={}
        for nr_seg in range(1,number_segments+1):
            for j in range(nr_seg,length+1):
                if nr_seg==1: # the final stage
                    P[nr_seg,j]= SSD(0,j),[series[:j]]
                else: # the other stages
                    P[nr_seg,j]= min(
                        [Tuple(SSD(i,j),[series[i:j]])+P[nr_seg-1,i] for i in range(nr_seg-1,j)]
                        )
        return P[number_segments,length]
    return V(number_segments,len(series))
#return iterV(number_segments,len(series)) iterative version
series=[10,20,34,50,60]
print (least_square_segmentation_problem(series,2))
assert least_square_segmentation_problem(series,2)==(340.66666666666674, [[50, 60], [10, 20, 34]])

# BEST ALIGNMENT OF 2 SEQUENCES
"""The Longest Common Subsequence (LCS) problem is as follows. We are
given two strings: string S of length n, and string T of length m.
Our goal is to produce their longest common subsequence: the longest sequence
of characters that appear left-to-right (but not necessarily in a contiguous
block) in both strings"""

def best_alignment(mystring1,mystring2):
    def score(i,j):# first argument=stage, second argument=state
        return 5 if mystring1[i]==mystring2[j] else -2
    def V(i,j): # i=index of first string, j=index of second string
        if i==-1 and j==-1: # the final stage
            return 0,[]
        elif j==-1:
            return -6,[(mystring1[0],"=")]
        elif i==-1:
            return -6,[("=",mystring2[0])]
        else: # the other stages
            return max(
                Tuple(score(i,j),[(mystring1[i],mystring2[j])])+V(i-1,j-1),
                Tuple(-6,[("=",mystring2[j])])+V(i,j-1),
                Tuple(-6,[(mystring1[i],"=")])+V(i-1,j)
                )
    def iterV(i_max,j_max):
        P={}
        P[-1,-1]= 0,[] # the final stage
        for i in range(i_max+1):
            P[i,-1]= -6,[(mystring1[0],"=")]
        for j in range(j_max+1):
            P[-1,j]= -6,[("=",mystring2[0])]
        for i in range(i_max+1):
            for j in range(j_max+1):
                P[i,j]= max(
                    Tuple(score(i,j),[(mystring1[i],mystring2[j])])+P[i-1,j-1],
                    Tuple(-6,[("=",mystring2[j])])+P[i,j-1],
                    Tuple(-6,[(mystring1[i],"=")])+P[i-1,j]
                    )
        return P[i_max,j_max]

    #return V(len(mystring1)-1,len(mystring2)-1) #recursive version
    return iterV(len(mystring1)-1,len(mystring2)-1) # iterative version
str1="TTCATA"
str2="TGCTCGTA"
assert best_alignment(str1,str2)==(11, [('A', 'A'), ('T', 'T'), ('A', 'G'), ('C', 'C'), ('T', 'T'), ('=', 'C'), ('=', 'G'), ('T', 'T')])
str1='GAATTCAGTTA'
str2='GGATCGA'
new1,new2= zip(*list(reversed(best_alignment(str1,str2)[1])))
assert "".join(new1) == "GAATTCAGTTA"
assert "".join(new2) == "GGA=TC=G==A"
#print list(reversed(best_alignment(str1,str2)[1]))
str1='ABAZDC'
str2='BACBAD'
new1,new2= zip(*list(reversed(best_alignment(str1,str2)[1])))
assert "".join(new1) == "=A=BAZDC"
assert "".join(new2) == "BACBA=D="

######################################################################
'''
From SO question
The minimum possible sum|x_i - x_j| using K pairs (2K numbers) from N numbers
'''

N, K = 4, 2
num = sorted([1515, 1520, 1500, 1535])

best = {} #best[(k,n)] = minimum sum using k pairs out of 0 to n
def b(k,n):
    if (k,n) in best:
        return best[(k,n)]
    if k==0:
        return 0
    return float('inf')

for n in range(1,N):
    for k in range(1,K+1):
        best[(k,n)] = min([b(k,n-1),                      #Not using num[n]
                           b(k-1,n-2) + num[n]-num[n-1]]) #Using num[n]

assert best[(K,N-1)]==30
#raise SystemExit
#############################################################################
# THE TSP problem
import math
try:
    from PIL import Image
    Image.Image.__getitem__=Image.Image.getpixel
    Image.Image.__setitem__=Image.Image.putpixel
    PIL_INSTALLED=True
except:
    print('PIL import failed')
    PIL_INSTALLED=False
#import random
#coords=[(random.randint(10,490),random.randint(10,490)) for _ in range(6)]
coords=[(100,100),(100,200),(200,200),(150,300),(200,100)]
def draw_graph(coordinates):
    im=Image.new('RGB',(500,500),'blue')
    from PIL import ImageDraw
    draw=ImageDraw.Draw(im)
    def draw_circle(x,y,text='hassan',color='yellow',size=20):
        bbox=(x-size/2,y-size/2,x+size/2,y+size/2)
        draw.ellipse(bbox,fill=color)
        draw.text((x,y-10),text,fill='blue')
    for i,point in enumerate(coordinates):
        draw_circle(point[0],point[1],str(i))
    draw.polygon(coordinates)
    im.show() #temporary !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if PIL_INSTALLED:
    draw_graph(coords)
def distance(p1,p2):
    return math.sqrt((p2[0]-p1[0])*(p2[0]-p1[0])+(p2[1]-p1[1])*(p2[1]-p1[1]))
def cal_total_distance(coords):
    #print 'coords=',coords,
    dis=0
    for i,point in enumerate(coords):
        if i+1==len(coords):
            i=0
            break # comment out if you want the length of the closed tour !!!!
        next_point=coords[(i+1) if i+1 != len(coords) else 0]
        #print '==>',distance(point,next_point)
        dis+= distance(point,next_point)
    return dis
#print 'INITIAL DISTANCE='
assert int(cal_total_distance(coords)) == 517
def tsp1(startcity,tussencities,endcity): # working, recursive version
    def V(yet_to_visit,myendcity): # STATE =(yet_to_visit set + current end city)
        if not yet_to_visit:
            return distance(startcity,myendcity),[]
        return min(
            [V(yet_to_visit-set([tussencity]),tussencity)+
                Tuple(distance(tussencity,myendcity),[tussencity])
                for tussencity in yet_to_visit]
            )
    cost,path=V(tussencities,endcity)
    return cost,[startcity]+path+[endcity]
def tsp_iter1(startcity,yet_to_visit,endcity): # working, iterative version
    V={} # STATE =(yet_to_visit set + current end city)
    for myendcity in yet_to_visit:
        V[frozenset([]),myendcity]=distance(startcity,myendcity),[]
    # all other set lengths
    n=len(yet_to_visit)
    for set_len in range(1,n+1):
        for tussen_set in xcombinations(list(yet_to_visit),set_len):
            for myendcity in list(yet_to_visit)+[startcity,endcity]:
                if myendcity in tussen_set:continue
                V[frozenset(tussen_set),myendcity]=min(
                    [V[frozenset(tussen_set)-frozenset([tussencity]),tussencity]+
                        Tuple(distance(tussencity,myendcity),[tussencity])
                        for tussencity in tussen_set]
                    )
    cost,path=  V[yet_to_visit,endcity]
    return cost,[startcity]+path+[endcity]
def tsp2(startcity,tussencities,endcity):  #recursive version, another standpoint
    def V(current_city,already_visited): # STATE=(already visited set+current start city)
        if len(already_visited)==len(tussencities): #
            return distance(current_city,endcity),[endcity]
        return min(
            [V(next_city,already_visited | set([next_city]))+
                Tuple(distance(current_city,next_city),[next_city])
                for next_city in tussencities if next_city not in already_visited]
            )
    cost,path= V(startcity,set([]))
    return cost,path+[endcity]
def tsp_iter2(startcity,tussencities,endcity): # working, iterative version
    V={} # STATE=(already visited set+current start city)
    for mystartcity in tussencities:
        V[mystartcity,tussencities]= distance(mystartcity,endcity),[]
    n=len(tussencities)
    for set_len in range(n-1,0,-1):
        for already_visited_set in xcombinations(list(tussencities),set_len):
            for mystartcity in list(tussencities):
                V[mystartcity,frozenset(already_visited_set)]=min(
                    [V[next_city,frozenset(already_visited_set)|frozenset([next_city])]+
                        Tuple(distance(mystartcity,next_city),[next_city])
                        for next_city in tussencities if next_city not in already_visited_set]
                    )
    # the final stage
    V[startcity,frozenset([])]=min(
        [V[next_city,frozenset([next_city])]+
            Tuple(distance(startcity,next_city),[next_city])
            for next_city in tussencities]
        )
    cost,path= V[startcity,frozenset([])]
    return cost,[startcity]+path+[endcity]
if PIL_INSTALLED:
    start=coords[0]
    end=coords[0]
    #cost,path= tsp1(start,set(coords[1:]),end)
    #cost,path= tsp2(start,set(coords[1:]),end)
    #cost,path= tsp_iter1(start,frozenset(coords[1:]),end)
    cost,path= tsp_iter2(start,frozenset(coords[1:]),end)
    #print '======>>>>',cost,int(cost),path
    assert int(cost),path == (623, [(100, 100), (100, 200), (150, 300), (200, 200), (200, 100), (100, 100)])
    new_coords=path
    assert [coords.index(idx) for idx in new_coords] == [0, 1, 3, 2, 4, 0]
    assert int(cal_total_distance(new_coords))== 523
    draw_graph(new_coords)
#raise SystemExit
# brute force solution, to prove above algorithm
# for per in perm(coords[1:]):
#   print coords.index(start),
#   for idx in per: print coords.index(idx),
#   print coords.index(end),
#   print cal_total_distance([start]+per+[end])
###########################################################################
from networkx import Graph #, gnm_random_graph
#from networkx.drawing.layout import spring_layout, spectral_layout,circular_layout
G=Graph()
G.add_edge("london","paris")
G.add_edge("london","madrid")
G.add_edge("london","rome")
G.add_edge("london","liverpool")
G.add_edge("rome","paris")
#from see import see
print ('----------------',G['london'].keys())
#raise SystemExit
#pos= spring_layout(G)
#pos= spectral_layout(G)
#pos=circular_layout(G)
#print pos
#print [(v[0]*250,v[1]*250) for v in pos.values()]
#draw_graph([(v[0]*250+250,v[1]*250+250) for v in pos.values()])
###########################################################################
def set_cover(graph):  # WORKING  !!!!
    nodes=list(graph.nodes())
    # sorting by descending order of covering power(degree) will improve performance !!!
    nodes.sort(key=graph.degree,reverse=True)
    print (nodes)
    def V(city_index,already_covered): # STATE=(already covered+current city)
        if len(already_covered)==len(nodes): #
            return 0,[]
        if city_index==len(nodes):# we've made all decisions and the problem is not solved
            return 100,[] # this is a failure therefore the high cost
        city_name=nodes[city_index]
        return min( # only two possible decisions
            V(city_index+1,already_covered), # reject
            Tuple(1,[city_name])+V(city_index+1,already_covered | set([city_name]+list(graph[city_name].keys()))) # accept
            )
    cost,path= V(0,set([]))
    return cost,path
def set_cover2(graph):  # WORKING  !!!!
    nodes=list(graph.nodes())
    # sorting by descending order of covering power(degree) will improve performance !!!
    nodes.sort(key=graph.degree,reverse=True)
    def V(city_index,yet_to_cover): # STATE =(yet_to_cover set + current city)
        if not yet_to_cover:
            return 0,[]
        if city_index==len(nodes): # we've made all decisions and the problem is not solved
            return 100,[] # this is a failure therefore the high cost
        city_name=nodes[city_index]
        return min(
            V(city_index+1,yet_to_cover), # reject
            Tuple(1,[city_name])+V(city_index+1,yet_to_cover-set([city_name]+list(graph[city_name])))
            )
    cost,path=V(0,set(nodes))
    return cost,path

GP=Graph(name="my graph")
edges=[('i', 'h'), ('i', 'j'), ('h', 'j'), ('k', 'h'), ('k', 'i'), ('k', 'j'),
    ('a', 'b'), ('a', 'k'), ('a', 'i'), ('a', 'h'), ('a', 'e'), ('c', 'd'),
    ('c', 'b'), ('e', 'f'), ('e', 'g'), ('d', 'a'), ('d', 'b'), ('f', 'g')  ]
GP.add_edges_from(edges)
# print("----->",GP.nodes())
# assert GP.nodes()==['a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j']
print (f"----> {set_cover(GP)=}")
raise SystemExit
# assert set_cover(GP) == (3, ['b', 'e', 'j'])

########################################################################
# Graph coloring has two kind of applications:
# 1. apportion a resource while using as few of it as possible
     # the nodes (user) need a resource (= color). Put an edge between two nodes (users)
     # that are not allowed to share a resource (several nodes may share a resource)
     # Another problem is edge-coloring in bipartite graph.
     # There is an edge between two nodes (a user and a named resource).
     # that together need to share a resource (unnamed) (a named resource cannot be shared!!!)
# 2. partition a set into as few sets as possible
"""
Scheduling
Vertex coloring models to a number of scheduling problems.
In the cleanest form, a given set of jobs need to be assigned to time slots,
each job requires one such slot. Jobs can be scheduled in any order, but pairs
of jobs may be in conflict in the sense that they may not be assigned to the
same time slot, for example because they both rely on a shared resource.
The corresponding graph contains a vertex for every job and an edge for every
conflicting pair of jobs. The chromatic number of the graph is exactly the
minimum makespan, the optimal time to finish all jobs without conflicts.
Details of the scheduling problem define the structure of the graph
For example, when assigning aircraft to flights, the resulting conflict graph
is an interval graph, so the coloring problem can be solved efficiently.
In bandwidth allocation to radio stations, the resulting conflict graph is a
unit disk graph, so the coloring problem is 3-approximable.
Register allocation
Main article: Register allocation
To improve the execution time of the resulting code, one of the
techniques of compiler optimization is register allocation, where the most
frequently used values of the compiled program are kept in the fast processor
registers. Ideally, values are assigned to registers so that they can all
reside in the registers when they are used.

The textbook approach to this problem is to model it as a graph coloring
problem.[29] The compiler constructs an interference graph, where vertices are
symbolic registers and an edge connects two nodes if they are needed at the
same time. If the graph can be colored with k colors then the variables can
be stored in k registers.
"""
@time_it
def graph_coloring(graph): #  WORKING
    FAIL=1000
    # we try to minimize the number of colors used
    max_degree=max(dict(graph.degree()).values())
    color_range=frozenset(range(max_degree+1)) #the max number colors= degree+1
    print ('COLOR RANGE=',color_range)
    nodes=list(graph.nodes())
    nodes.sort(key=graph.degree,reverse=True)# could ordering improve performance ????
    print ('nodes=',nodes)
    VV={} # for memoisation
    def possible_decisions(decisions_uptonow):
        color_range=frozenset(range(MIN))
        if not decisions_uptonow: # first node
            # return color_range WASTEFUL !!!!
            # We impose the color for the first decision !!!
            return frozenset([0])
        if len(decisions_uptonow)==len(nodes): # all decisions taken
            return frozenset([])
        curr_node=nodes[len(decisions_uptonow)]
        neighbours=graph[curr_node]
        colors_used = set([color for i,color in enumerate(decisions_uptonow) if nodes[i] in neighbours])
        return frozenset(color_range-colors_used)
    def possible_decisions_for_future_nodes(decisions_uptonow): # memoize this (and make it recursive)
        curr_index= len(decisions_uptonow)
        poss_decisions=set()
        for future_node_index in range(curr_index,len(nodes)):
            curr_node=nodes[future_node_index]
            neighbours=graph[curr_node]
            colors_used = set([color for i,color in enumerate(decisions_uptonow) if nodes[i] in neighbours])
            possible_decisions=frozenset(color_range-colors_used)
            poss_decisions.add((future_node_index,possible_decisions))
        return frozenset(poss_decisions)
    def V(city_index,decisions_uptonow): # STATE=(current city+decisions up to now)
        global MIN
        if city_index==len(nodes):# we've made all decisions
            if len(set(decisions_uptonow))<MIN:
                MIN=len(set(decisions_uptonow))
            return 0,[]
        if len(set(decisions_uptonow)) >= MIN:
            return FAIL,[]
        poss_dec=possible_decisions(decisions_uptonow)
        if not poss_dec:
            return FAIL,[]
        state=city_index,possible_decisions_for_future_nodes(decisions_uptonow) # USEFUL FOR MEMOIZATION
        if state in VV:
            return VV[state]
        v=VV[state]= min( [Tuple(0 if color in decisions_uptonow else 1,[color])+
                V(city_index+1,decisions_uptonow+[color])
                for color in poss_dec]
            )
        return v
    cost,path= V(0,[])
    if cost>=1000: return cost,[]
    return cost,path,partition(nodes,lambda node: path[nodes.index(node)])
GP=Graph(name="my graph")
#edges=[('i', 'h'), ('i', 'j'), ('h', 'j'), ('k', 'h'), ('k', 'i'), ('k', 'j'),
#   ('a', 'b'), ('a', 'k'), ('a', 'i'), ('a', 'h'), ('a', 'e'), ('c', 'd'),
#   ('c', 'b'), ('e', 'f'), ('e', 'g'), ('d', 'a'), ('d', 'b'), ('f', 'g')  ]
edges=[('v1', 'v5'), ('v1', 'v6'), ('v2', 'v5'), ('v2', 'v3'), ('v3', 'v6'), ('v3', 'v7'),
    ('v4', 'v7'), ('v5', 'v8'), ('v5', 'v9'), ('v5', 'v6'), ('v6', 'v9'), ('v6', 'v10'),
    ('v7', 'v10'), ('v7', 'v11'), ('v7', 'v12'), ('v9', 'v10')  ]
GP.add_edges_from(edges)
# complexity seems to be mostly determined by the number of edges (> 100==> problems)
#GP=gnm_random_graph(100,110) # nr vertices, nr edges
#GP=gnm_random_graph(20,100) # nr vertices, nr edges
#GP=gnm_random_graph(500,100)
max_degree=max(dict(GP.degree()).values())
assert max_degree==5
MIN = max_degree + 1 # global variable !!!!!
#MIN=9
solution=graph_coloring(GP)
print ("S---->", solution)
# assert solution ==(3, [0, 1, 1, 0, 0, 2, 2, 1, 0, 0, 0, 1], {0: ['v5', 'v10', 'v3', 'v12', 'v11', 'v4'], 1: ['v6', 'v7', 'v2', 'v8'], 2: ['v9', 'v1']})

# TO DO, improve by limiting the number of times a color (resource) can be used
######################################################################
def longest_increasing_subsequence_problem(sequence): # WORKING
    def admissible(i):
        empty=True
        for x in range(i):
            if sequence[i]>sequence[x]:
                yield x
                empty=False
        if empty:
            yield -1
    VV={}
    def V(i):
        if i==-1:
            return 0,[]
        v=VV[i]=max([V(j)+Tuple(1,[sequence[i]]) for j in admissible(i) ])
        return v
    #print "V(0)=",V(0)
    #print "V(1)=",V(1)
    #print "V(2)=",V(2)
    #print "V(3)=",V(3)
    #print "V(4)=",V(4)
    #print "V(5)=",V(5)
    #print "V(6)=",V(6)
    #print "V(7)=",V(7)
    #print "V(8)=",V(8)
    return max(V(x) for x in range(len(sequence)))
seq=9,5,2,8,7,3,1,6,4
print (longest_increasing_subsequence_problem(seq))
###############################################################################
# 8 queen problem
@time_it
def eight_queen_problem(): # WORKING
    NR_QUEENS=100
    def conflict(row1, col1, row2, col2):
        "Would putting two queens in (row1, col1) and (row2, col2) conflict?"
        return (row1 == row2 ## same row
                or col1 == col2 ## same column
                or row1-col1 == row2-col2  ## same ' diagonal
                or row1+col1 == row2+col2) ## same / diagonal
    def conflict_any(row1,col1,prev_decisions):
        for row,col in prev_decisions:
            if conflict(row1,col1,row,col):
                return True
        return False
    def possible_decisions(decisions_uptonow):
        if not decisions_uptonow: # first node
            return frozenset([(i,0) for i in range(NR_QUEENS)])
        if len(decisions_uptonow)==NR_QUEENS: # all decisions taken
            return frozenset([])
        current_col=len(decisions_uptonow)
        acceptable_rows=[]
        for next_row in range(NR_QUEENS):
            if not conflict_any(next_row,current_col,decisions_uptonow):
                acceptable_rows.append(next_row)
        return frozenset([(poss_row,current_col) for poss_row in set(acceptable_rows)])
    SUCCESS=1
    FAIL=0
    def V(queen_index,decisions_uptonow): # STATE=(current row+decisions up to now)
        if queen_index==NR_QUEENS:# we've made all decisions without failing !!
            yield SUCCESS,[] # we decide success only at the end !!!
        poss_dec=possible_decisions(decisions_uptonow)
        if not poss_dec:
            yield FAIL,[]
        for beslissing in poss_dec:
            for status,decisions in V(queen_index+1,decisions_uptonow+[beslissing]):
                if status==SUCCESS:
                    yield SUCCESS,[beslissing]+decisions
    for __,path in V(0,[]):
        print ('SOLUTION=',path)
        break
print ('EIGHT QUEEN PROBLEM')
eight_queen_problem()
###############################################################################
# to do: crew scheduling problem
########################################
# THE TOWERS OF HANOI
def transfer(discs, source, destination, storage): # WORKING
    if discs == 1:
        print ("Moving disc from %s to %s" % (source, destination))
    else:
        transfer(discs - 1, source, storage, destination)
        transfer(1,source,destination,storage)
        transfer(discs - 1, storage, destination, source)
number_of_disks=4
transfer(number_of_disks, 'A', 'B', 'C')
###################################################################
def argmax(seq):
    '''Returns the index of the item in a list with the largest value '''
    return seq.index(max(seq))
# TV GAME: madal hayat
def mada(): # NOT WORKING YET
    gains=[0,1,3,6,12,36,60,120,180,300]
    VV={}
    def V(R,W,gain_idx): # number of red balls, white balls and index of gain up to now
        #print 'state=',R,W,gain_uptonow
        if R==0: # all reds are gone, we lose everything
            return 0
        if W==0: # all whites are gone, the game is over
            return gains[gain_idx]
        state=R,W,gain_idx
        if state in VV: return VV[state]
        if gain_idx==0: # no choice, continue
            v=VV[state]=W*1.0/(R+W)*V(R,W-1,1)+R*1.0/(R+W)*V(R-1,W,0)
            print (state,v,'CONTINUE')
        else:
            v=VV[state]=max([gains[gain_idx], # STOP
            W*1.0/(R+W)*V(R,W-1,min(gain_idx+1,9))+R*1.0/(R+W)*V(R-1,W,max(gain_idx-1,0)) # CONTINUE
                ])
            print (state,v,'STOP' if v==gains[gain_idx] else 'CONTINUE')
        return  v
    return V(4,9,0)
solution= mada()
print (solution)
##############################################################################
# maximum subarray problem
def max_subarray(A): # WIKIPEDIA CODE, TO DO: IMPROVE IT TO RETURN SOLUTION !!!
    max_so_far = max_ending_here = 0
    for x in A:
        max_ending_here = max(0, max_ending_here + x)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far
#the contiguous subarray with the largest sum is 4, -1, 2, 1, with sum 6
solution=max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4])
print (solution)



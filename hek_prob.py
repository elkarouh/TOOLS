# TODO implement NaiveBayes classifier (see example in wikipedia)
import sys, math
from decimal import Decimal
from fractions import Fraction
Fraction.__repr__=lambda self:"{0}/{1}".format(self.numerator,self.denominator) if self.denominator>1 else str(self.numerator)
F= Fraction
import itertools as it, random
from collections import Counter

ANY=type('',(),{'__eq__':lambda i,_:True})()
Null=type('',(),{'__bool__':lambda s:False,'__call__':lambda s,*a,**k: s,'__getitem__':lambda s,k:s,'__getattr__':lambda s,n:s})()
jsdict=type('jsdict',(dict,),{'__getattr__':lambda s, i: s[i]}) # a javascript dicta
import pandas as pd
class RandomVariable:
    # key: discrete domain; value: float[0..1]
    def __init__(self, main_variable):
        self.main_variable=main_variable
        self.df= pd.DataFrame()
    def add_evidence_pandas (self, **evidence): # add_evidence (C=True,T='Negative')
        self.df = pd.concat([self.df, pd.DataFrame([evidence])], ignore_index=True)
    def get_prob_pandas (self, hypothesis, **evidence): # using pandas
        mask = pd.Series(True, index=self.df.index)
        for key, value in evidence.items():
            mask &= (self.df[key] == value)
        denom=mask.sum()
        mask &= (self.df[self.main_variable] == hypothesis)
        num=mask.sum()
        #####################
        # filtered_df = self.df.copy()
        # for key, value in evidence.items():
        #     filtered_df = filtered_df[filtered_df[key] == value]
        # denom=filtered_df.shape[0]
        # filtered_df = filtered_df[filtered_df[self.main_variable] == hypothesis]
        # num=filtered_df.shape[0]
        #####################
        theta=(num+2)/(denom+4)
        uncertainty_at95p=2*math.sqrt(theta*(1-theta)/(denom+4)) # 2-sigma 95% confidence
        self.theta=theta
        self.uncertainty_at95p=uncertainty_at95p
        return (theta, uncertainty_at95p)
    def display (self):
        print (f"{self.theta*100:.2f}% \u00B1 {self.uncertainty_at95p*100:.2f}%")

cancer=RandomVariable("Cancer")
for _ in range (20):
    cancer.add_evidence_pandas (Cancer=True, Test='negative')
for _ in range (80):
    cancer.add_evidence_pandas (Cancer=True, Test='positive')
for _ in range (700):
    cancer.add_evidence_pandas (Cancer=False, Test='positive')
for _ in range (9200):
    cancer.add_evidence_pandas (Cancer=False, Test='negative')
#print(cancer.df)
prob=cancer.get_prob_pandas (True, Test='positive') # probability of cancer if test is positive
cancer.display ()
raise SystemExit

def P(args):
    """The probability of an event, given a pmf
    predicate: a predicate that is true if outcome in the event.
    pmf: a probability distribution of {outcome: frequency} pairs."""
    predicate, pmf= args
    return sum(pmf[o] for o in pmf if predicate(o))

class PMF(Counter):
    """A Probability Distribution; an {outcome: probability} mapping
    where probabilities sum to 1.
    CONDITIONAL PROBABILITIES:
    a=PMF(cancer=10)*PMF(positive=F(9,10),negative=F(1,10))
    b=PMF(healthy=990)*PMF(positive=F(1,10),negative=F(9,10))
    experiment= a+b gives the desired combined probability
    Counter({'cancer-negative': 1,
         'cancer-positive': 9,
         'healthy-negative': 891,
         'healthy-positive': 99})
    P(has_cancer | is_positive & c)=9/108=1/12
    """
    def __init__(self, *args, **kwargs):
        Counter.__init__(self, *args, **kwargs)
        self.total=0
        self.recompute()
        self.mutating=False
    def recompute(self):
        self.total = sum(self.values())
        if isinstance(self.total, int):
            self.total = Fraction(self.total, 1)
    def __getitem__(self, key):
        if self.mutating:
            return Counter.__getitem__(self, key)
        else:
            return Counter.__getitem__(self, key)/self.total
    def update(self,*args, **kwds):
        self.mutating=True
        Counter.update(self,*args, **kwds)
        self.recompute()
        self.mutating=False
    def subtract(self,*args, **kwds):
        self.mutating=True
        Counter.subtract(self,*args, **kwds) # BUG IN THE STANDARD LIBRARY
        self.recompute()
        self.mutating=False
    def get_true_counts(self):
        """return the true counts"""
        return {key:self[key]*self.total for key in self}
    def __add__(self,other):
        return PMF(Counter.__add__(self,other))
    def __mul__(self,other):
        """The joint distribution of two independent probability distributions.
            Result is all entries of the form {a+sep+b: P(a)*P(b)}"""
        sep="-"
        return PMF({str(a) + sep + str(b): self.get_true_counts()[a] * other.get_true_counts()[b] for a in self for b in other})
    def __rand__(self,predicate):
        "A new PMF, restricted to the outcomes of this PMF for which the predicate is true."
        return PMF({e:self.get_true_counts()[e] for e in self if predicate(e)})
    def __ror__(self,predicate):
        "a tuple suitable for use by P(predicate,pmf)."
        return (predicate,self)
    def __repr__(self):
        return '{'+", ".join("{0}=>{1}".format(i,self[i]) for i in self)+'}'
    def random( self ):
        """ each call -> a random key with the given probability"""
        #return random.choice(list(self.elements())) # SHORT BUT LESS EFFICIENT
        idx = random.randrange(self.total)
        for i,elem in enumerate(self.elements()):
            if i==idx:
                return elem
    def display(self):
        # %matplotlib inline
        x=np.arange(len(self))
        plt.bar(x ,height=self.values())
        plt.xticks(x, self.keys())
        plt.show()
    def EV(self): #expected value
        return sum(key*count for key,count in self.get_true_counts().items())/self.total
    def variance(self):
        pass
    def median(self):
        pass
    def bayesian_update(self, data):
        """Performs a Bayesian update.
            child classes should provide a likelihood method that evaluates the
            likelihood of the data under a given hypothesis
        """
        for hypo in self:
            self[hypo] *= self.likelihood(data, hypo)
        self.recompute()
    def __hash__(self):
        """Returns an integer hash value."""
        return id(self)
    def __eq__(self, other):
        return self is other
    def render(self):
        """Returns values and their probabilities, suitable for plotting."""
        return zip(*sorted(self.items()))

def choose(n,pmf):
    "All combinations of n items; each combo as a concatenated str."
    return PMF('-'.join(combo) for combo in it.combinations(pmf.elements(), n))

class NormalizedPMF(PMF):
    def __init__(self, *args,**kwargs):
        #values should add up to 1 !!!!
        #PMF.__init__(self,...)
        pass

class GaussianPMF(PMF):
    import random
    def __init__(self, mu, sigma, bins=20):
        values= [random.gauss(mu,sigma) for _ in range(1000)]
        PMF.__init__(self,values)



# from pytexit import py2tex # formula to latex

def compute_size():
    """FIND THE SIZE OF ALL .git directories under the current dir"""
    tsize=0
    if PY3:
        from glob import glob
    else:
        from glob2 import glob
    for f in glob("**/.git"):
        #a=!du -sh "$f"
        size,name= a.s.split(None,1)
        if size.strip().endswith(('K','M')):
            ksize=float(size[:-1]) \
                            if size.endswith("K") else 1024*float(size[:-1])
        else:
            ksize=float(size)/1000
        print (repr(ksize)+"K", name)
        tsize += ksize
    return tsize

############################################################################
class Point(complex):
    x = property(lambda p: p.real)
    y = property(lambda p: p.imag)
    __str__=lambda s:"{}-{}".format(s.x,s.y)
def distance(point1,point2):
    return abs(point2-point1)
import cmath
class Vector(Point):
    def rotate(self,angle_in_degree): # rotate counterclockwise
        self = self*cmath.rect(1,angle_in_degree)
    __str__=lambda s:"mag{}-phase{}".format(cmath.polar(s))
def angle(vector1,vector2):
    return cmath.phase(vector2/vector1) # in radian
    return cmath.phase(vector2/vector1) *180/cmath.pi  # in degree
def get_vector(point1, point2):
    return point2-point1
def get_unit_vector(point1, point2):
    return (point2-point1)/abs(point2-point1)

def regularAngle(a):
    while a < 0: a += 2*cmath.pi
    while a > 2*cmath.pi: a -= 2*cmath.pi
    return a
############################################################################
#rr=lambda x,res:int(round(x/res))*res
#rr=lambda x,res:round(Decimal(round(x/res)*res),6)

# https://gist.github.com/jackiekazil/6201722
# from decimal import ROUND_HALF_UP
# Here are all your options for rounding:
# This one offers the most out of the box control
# ROUND_05UP       ROUND_DOWN       ROUND_HALF_DOWN  ROUND_HALF_UP
# ROUND_CEILING    ROUND_FLOOR      ROUND_HALF_EVEN  ROUND_UP
#our_value = Decimal(16.0/7)
#output = Decimal(our_value.quantize(Decimal('.01'), rounding=ROUND_HALF_UP))

class BayesClassifier:
    def __init__(self, causes, low, high, bins=20):
        self.causes={}
        for cause in causes:
            self.causes[cause]=PMF()
        self.low=low
        self.high=high
        self.bins=bins
        bin_resolution=0.01
        self.resolution= int(round((self.high-self.low)/bins)/bin_resolution)*bin_resolution
    def round_to_resolution(self,value):
        #return int(round(value/self.resolution))*self.resolution
        return round(Decimal(round(value/self.resolution)*self.resolution),6)
    def __getitem__(self, key):
        return PMF.__getitem__(self, self.round_to_resolution(key))
    def __setitem__(self, key,value):
        PMF.__setitem__(self, self.round_to_resolution(key),value)
    def update(self, cause,**kw): # update('male',length=3,weight=4)
        for key in kw:
            kw[key]=self.round_to_resolution(kw[key])
        self.causes[cause].update(kw)
    def bayesian_update(self,**kw):
        pass
    def classify(self,**features):
        best_cause=None
        best_prob=0
        for cause in self.causes:
            prob=1 # COMPUTE HERE
            for feature,val in features.items():
                pmf=self.causes[feature]
                prob= prob*P(val|pmf)
            if prob>best_prob:
                best_prob=prob
                best_cause=cause
        return best_cause

if __name__=="__main__":
    dice= PMF("123456")
    dice.update("123")
    dice.subtract("123")
    even = lambda o: int(o)%2==0
    smaller_than_4= lambda o: int(o)<4
    if 0:
        print(dice)
        print("TRUE COUNTS=",dice.get_true_counts())
        print(even & dice, P(even | dice))
        print(smaller_than_4 & (even & dice), P(smaller_than_4 | even & dice))
        print(P(smaller_than_4 | (even & dice)))
        print(P(smaller_than_4 | dice))
        #
        a=PMF(cancer=10)*PMF(positive=F(9,10),negative=F(1,10))
        b=PMF(healthy=990)*PMF(positive=F(1,10),negative=F(9,10))
        experiment= a+b # gives the desired combined probability
        has_cancer= lambda o: 'cancer' in o
        is_positive= lambda o: 'positive' in o
        assert P(has_cancer | is_positive & experiment)== F(9,108) == F(1,12)
        assert P(is_positive | has_cancer & experiment)== F(9,10)
    if 0:
        employee_birthday=PMF("D{}".format(i+1) for i in range(365))
        def firm(nr_employees):
            a= employee_birthday
            for _ in range(nr_employees-1):
                a = a * employee_birthday
            return a
        size= int(sys.argv[1]) if sys.argv[1:] else 10
        exp=firm(size)
        #print exp
        same_day= lambda o:len(o.split('-')) > len(set(o.split('-')))
        #print same_day & exp
        print(P(same_day | exp))

    if 0:
        height1=PMF(male=4)*PMF([6,5.92,5.58,5.92])
        weight1=PMF(male=4)*PMF([180,190,170,165])
        foot_s1=PMF(male=4)*PMF([12,11,12,10])
        height2=PMF(female=4)*PMF([5,5.5,5.42,5.75])
        weight2=PMF(female=4)*PMF([100,150,130,150])
        foot_s2=PMF(female=4)*PMF([6,8,7,9])

        sample_height=6
        sample_weight=130
        sample_foot_size=8
        #P(male|(sample_height,sample_weight, sample_foot_size))=k*P(male)*P(height|male)*P(weight|male)*P(foot_size|male)
        #P(female|(sample_height,sample_weight, sample_foot_size))=k*P(female)*P(height|female)*P(weight|female)*P(foot_size|female)

    if 0: # EM algorithm
        # http://math.stackexchange.com/questions/25111/how-does-expectation-maximization-work
        rolls = [ "HTTTHHTHTH","HHHHTHHHHH","HTHHHHHTHH","HTHTTTHHTT","THHHTHHHTH"]
        is_head=lambda o:o=='H'
        def likelihoodOfRoll(roll,theta):
            numHeads=roll.count("H")
            n=len(roll)
            return pow(theta,numHeads)*pow(1-theta,n-numHeads)
        def em_single(prior_a,prior_b, observations):
            pmf_a=PMF() # the new pmf for a
            pmf_b=PMF() # the new pmf for b
            # E-step,COMPLETE THE MISSING DATA
            for roll in observations: #
                #Assuming data is from A,compute likelihood from current guess for A
                # complete data with A --> P(A & D)
                likelihood_a= likelihoodOfRoll(roll,prior_a) # P(A & D)
                #Assuming data is from B,compute likelihood from current guess for B
                # complete data with B --> P(B & D)
                likelihood_b= likelihoodOfRoll(roll,prior_b) # P(B & D)
                weight_a=int(likelihood_a/(likelihood_a+likelihood_b)*100)
                weight_b=int(likelihood_b/(likelihood_a+likelihood_b)*100)
                # let's complete the data with the expected number of counts
                pmf_a.update('H'*roll.count("H")*weight_a) # P(A & D)
                pmf_a.update('T'*roll.count("T")*weight_a)
                pmf_b.update('H'*roll.count("H")*weight_b) # P(B & D)
                pmf_b.update('T'*roll.count("T")*weight_b)
            # M-step, compute the new parameters with the completed data
            new_prior_a,new_prior_b= P(is_head | pmf_a),P(is_head| pmf_b)
            return new_prior_a,new_prior_b
        def em(prior_a,prior_b,observations, tol=1e-6, iterations=10000):
            for i in range(iterations):
                #print "iteration", i
                new_prior_a,new_prior_b= em_single(prior_a,prior_b,observations)
                delta=abs(prior_a-new_prior_a)
                if delta<tol:
                    break
                #print new_prior_a,new_prior_b
                prior_a,prior_b= new_prior_a,new_prior_b
            return new_prior_a,new_prior_b, i
        print(em(0.6,0.5,rolls))

    if 1:
        # how the polls went wrong
        import random
        poll_samples=["ST"]*48+["SC"]*52 # SC=say clinton, ST= say trump
        random.shuffle(poll_samples)
        #print poll_samples
        p_T=0.1
        p_C=0.9
        p_ST_if_T=0.94; p_SC_if_T=1-p_ST_if_T
        p_SC_if_C=0.99; p_ST_if_C=1-p_SC_if_C
        for i in range(10):
            likelihood_T=0
            likelihood_C=0
            for poll_sample in poll_samples:
                if poll_sample=="ST": # says T
                    # complete data with T and compute P(T & ST) (likelihood of completed data)
                    likelihood_T+= p_T*p_ST_if_T/(p_T*p_ST_if_T+p_C*p_ST_if_C)
                    # complete data with C and compute P(C & ST) (likelihood of completed data)
                    likelihood_C+= p_C*p_ST_if_C/(p_T*p_ST_if_T+p_C*p_ST_if_C)
                else: # says C
                    # complete data with T and compute P(T & SC) (likelihood of completed data)
                    likelihood_T+= p_T*p_SC_if_T/(p_T*p_SC_if_T+p_C*p_SC_if_C)
                    # complete data with C and compute P(C & SC) (likelihood of completed data)
                    likelihood_C+= p_C*p_SC_if_C/(p_T*p_SC_if_T+p_C*p_SC_if_C)
            print ("TOTAL:",likelihood_T, likelihood_C, likelihood_T+likelihood_C)
            p_T= likelihood_T/(likelihood_T+likelihood_C)
            p_C= likelihood_C/(likelihood_T+likelihood_C)
            #print p_T, p_C

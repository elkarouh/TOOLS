"""
Shortest-path / dynamic-programming optimiser framework.

ALL DYNAMIC PROGRAMMING PROBLEMS CAN BE TRANSFORMED INTO SHORTEST-PATH PROBLEMS
UNDER THE CONDITION THAT for each decision there is only ONE next state.
This is NOT the case for reinforcement-learning problems where a decision can lead
to several states with a probability — VALUE ITERATION must be used there to find
a policy (best action in every state).
"""

import heapq
import itertools
from collections import deque
from math import exp, log
from typing import Any, Generator, Iterable


# ---------------------------------------------------------------------------
# ANY sentinel — matches every value via == (useful as a wildcard end-state)
# ---------------------------------------------------------------------------
class _AnyMeta(type):
    """Metaclass that makes every instance equal to everything."""
    def __instancecheck__(cls, instance):
        return True

class _Any(metaclass=_AnyMeta):
    """Singleton wildcard: ``ANY == x`` is always True."""
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __eq__(self, _other):
        return True
    def __hash__(self):
        return hash("ANY")
    def __repr__(self):
        return "ANY"

ANY = _Any()
ANY=type('',(),{'__eq__':lambda i,_: 1})()

# ---------------------------------------------------------------------------
# Queue implementations
# ---------------------------------------------------------------------------
# class PriorityQueue(list):
#    def push(self, item):
#        heapq.heappush(self,  item)
#    def pop(self):
#        return heapq.heappop(self)
#class FifoQueue(list):
#    def push(self,item):
#        self.append(item)
#    def pop(self):
#        return super(FifoQueue,self).pop(0)
#class LifoQueue(list):
#    def push(self,item):
#        self.append(item)

class PriorityQueue:
    """Min-heap priority queue. Items must support < comparison on their first element."""

    def __init__(self, items: Iterable = ()):
        self._counter = itertools.count()   # tie-breaker — avoids comparing paths
        self._heap: list = []
        for item in items:
            self.push(item)

    def push(self, item) -> None:
        # item[0] is the priority; insert a counter so lists are never compared.
        heapq.heappush(self._heap, (item[0], next(self._counter), item[1:]))

    def pop(self):
        priority, _count, rest = heapq.heappop(self._heap)
        return (priority, *rest)

    def __bool__(self):
        return bool(self._heap)


class FifoQueue:
    """FIFO queue backed by a deque (O(1) append and popleft)."""

    def __init__(self, items: Iterable = ()):
        self._dq: deque = deque(items)

    def push(self, item) -> None:
        self._dq.append(item)

    def pop(self):
        return self._dq.popleft()

    def __bool__(self):
        return bool(self._dq)


class LifoQueue:
    """LIFO (stack) queue."""

    def __init__(self, items: Iterable = ()):
        self._stack: list = list(items)

    def push(self, item) -> None:
        self._stack.append(item)

    def pop(self):
        return self._stack.pop()

    def __bool__(self):
        return bool(self._stack)


# ---------------------------------------------------------------------------
# Core Optimizer
# ---------------------------------------------------------------------------
class Optimizer:
    """
    Generic shortest-path / dynamic-programming optimiser.

    Subclass this and override:
      - ``get_state(past_decisions)``      → current state from the decision history
      - ``get_next_decisions(state)``      → list of ``(decision, cost)`` pairs
      - ``get_heuristic_cost(state)``      → admissible heuristic (default 0, = Dijkstra)
      - ``cost_operator(accumulated, new)`` → how costs combine (default: addition)
    """

    def __init__(self, offset: float = 0):
        """
        Parameters
        ----------
        offset:
            Added to every step cost to handle negative costs.
            CAVEAT: works correctly only when all solutions have the same length.
        """
        self.offset = offset
        self.decision_path: list = []   # last expanded path — available to get_next_decisions
        self.start_state: Any = None

    # ------------------------------------------------------------------
    # Methods to override
    # ------------------------------------------------------------------

    def get_state(self, past_decisions: list) -> Any:
        """Return the state that results from *past_decisions*."""
        raise NotImplementedError("Override get_state()")

    def get_next_decisions(self, current_state: Any) -> list[tuple]:
        """Return a list of ``(decision, cost)`` pairs from *current_state*."""
        raise NotImplementedError("Override get_next_decisions()")

    def get_heuristic_cost(self, current_state: Any) -> float:
        """Admissible estimate of remaining cost. Override for A*; default gives Dijkstra."""
        return 0

    def cost_operator(self, accumulated: float, step_cost: float) -> float:
        """
        Combine accumulated cost with a single step cost.
        Override for multiplicative costs, log-probabilities, etc.
        The result must be >= accumulated (path length must grow).
        """
        step_cost += self.offset
        assert step_cost >= 0, (
            f"Negative step cost {step_cost - self.offset} detected. "
            "Re-create the Optimizer with a suitable offset."
        )
        return accumulated + step_cost

    def hcost_operator(self, past_cost: float, current_state: Any) -> float:
        """f(n) = g(n) + h(n)."""
        return past_cost + self.get_heuristic_cost(current_state)

    def real_cost(self, cost: float) -> float:
        """Undo the offset to recover the true path cost."""
        return cost - self.offset * len(self.decision_path)

    # ------------------------------------------------------------------
    # Shortest-path (A* / Dijkstra)
    # ------------------------------------------------------------------

    def shortest_path(
        self,
        start_state: Any,
        end_state: Any,
        allsolutions: bool = True,
    ) -> Generator[tuple[float, list], None, None]:
        """
        Yield ``(cost, path)`` pairs in non-decreasing cost order.

        Parameters
        ----------
        start_state:
            The starting state.
        end_state:
            The target state (supports ``ANY`` wildcard).
        allsolutions:
            If *False*, stop after the first (optimal) solution.
        """
        self.start_state = start_state

        # Fringe: (heuristic_cost, actual_cost, path, state)
        # Storing the state avoids recomputing it on every pop.
        fringe = PriorityQueue([(0, 0, [], start_state)])
        visited: set = set()

        while fringe:
            _, cost, path, current_state = fringe.pop()

            if not allsolutions and current_state in visited:
                continue

            self.decision_path = path
            visited.add(current_state)

            if current_state == end_state:
                yield self.real_cost(cost), path
                if not allsolutions:
                    return

            for new_decision, step_cost in self.get_next_decisions(current_state):
                new_path = path + [new_decision]
                next_state = self.get_state(new_path)
                if next_state not in visited:
                    new_cost = self.cost_operator(cost, step_cost)
                    hcost = self.hcost_operator(new_cost, next_state)
                    fringe.push((hcost, new_cost, new_path, next_state))

    # ------------------------------------------------------------------
    # Generic traversal (BFS / DFS / best-first)
    # ------------------------------------------------------------------

    def is_end_state(self, state: Any) -> bool:
        """Override to signal termination in :meth:`traverse`."""
        return False

    def visit_state(self, state: Any) -> None:
        """Called by :meth:`traverse` each time a state is first expanded."""
        print("state =", state)

    def traverse(
        self,
        start_state: Any,
        fringetype=PriorityQueue,
        limit: int = 100_000,
    ):
        """
        Generic graph traversal.  The search strategy is determined by *fringetype*:
          - ``PriorityQueue`` → best-first / Dijkstra
          - ``FifoQueue``     → breadth-first
          - ``LifoQueue``     → depth-first

        Parameters
        ----------
        start_state:
            Root of the search.
        fringetype:
            Queue class controlling the traversal order.
        limit:
            Maximum path length before backtracking (depth limit for DFS).

        Returns
        -------
        ``(real_cost, path)`` for the first goal found, or ``[]`` on failure.
        """
        fringe = fringetype([(0, [], start_state)])
        visited: set = set()

        while fringe:
            cost, path, current_state = fringe.pop()

            if len(path) >= limit:
                return []

            if current_state in visited:
                continue

            self.visit_state(current_state)

            if self.is_end_state(current_state):
                self.decision_path = path
                return self.real_cost(cost), path

            visited.add(current_state)

            for new_decision, step_cost in self.get_next_decisions(current_state):
                new_path = path + [new_decision]
                next_state = self.get_state(new_path)
                if next_state not in visited:
                    fringe.push((cost + step_cost, new_path, next_state))

        return []   # fringe exhausted — failure

    def breadth_first_search(self, start_state: Any):
        """BFS traversal from *start_state*."""
        return self.traverse(start_state, FifoQueue)

    def depth_first_search(self, start_state: Any, limit: int = 100_000):
        """DFS traversal from *start_state* with optional depth *limit*."""
        return self.traverse(start_state, LifoQueue, limit)

    # Keep legacy names as aliases
    breadfirst_search = breadth_first_search
    depthfirst_search = depth_first_search

    def iterative_deepening(self, start_state: Any):
        """Iterative-deepening DFS from *start_state*."""
        for limit in range(1, 100):
            solution = self.traverse(start_state, LifoQueue, limit)
            if solution:
                return solution
        return []

    # ------------------------------------------------------------------
    # Longest-path helpers
    # ------------------------------------------------------------------

    def longest_path_min(
        self,
        end_state: Any,
        excluded_lengths: Iterable[int] = (),
        offset: float = 1_000,
    ):
        """
        Internal helper for :meth:`longest_path`.
        Negates revenues so a shortest-path search finds the maximum.

        Parameters
        ----------
        end_state:
            Target state.
        excluded_lengths:
            Path lengths to skip at the goal (used to iterate over progressively
            longer solutions).
        offset:
            Must exceed the maximum absolute revenue per step.
        """
        excluded = set(excluded_lengths)
        fringe = PriorityQueue([(0, [], self.start_state)])
        visited: dict = {}       # state → best real revenue seen
        solution = (0, None)

        while fringe:
            cost, path, current_state = fringe.pop()
            real_revenue = len(path) * offset - cost

            prev_best = visited.get(current_state)
            if prev_best is not None and real_revenue <= prev_best:
                continue
            visited[current_state] = real_revenue

            if current_state == end_state:
                return real_revenue, path

            for new_decision, revenue in self.get_next_decisions(current_state):
                new_path = path + [new_decision]
                next_state = self.get_state(new_path)
                cost_step = -revenue + offset
                assert cost_step > 0, (
                    f"Step revenue {revenue} >= offset {offset}. "
                    "Increase the offset parameter."
                )
                new_cost = cost + cost_step
                new_real = len(new_path) * offset - new_cost

                penalty = 100_000 if len(new_path) in excluded and next_state == end_state else 0
                new_cost += penalty

                prev = visited.get(next_state)
                if prev is None or new_real > prev:
                    fringe.push((new_cost, new_path, next_state))

        return solution

    def longest_path(
        self,
        start_state: Any,
        end_state: Any,
        max_path_length: int = 1_000,
        offset: float = 1_000,
    ):
        """
        Find the longest (highest-revenue) path from *start_state* to *end_state*.

        Uses repeated calls to :meth:`longest_path_min`, each time excluding
        the previous solution's length, until no longer path exists.

        Parameters
        ----------
        max_path_length:
            Hard cap on path length to prevent infinite loops on cyclic graphs.
        offset:
            Passed to :meth:`longest_path_min`; must exceed max step revenue.
        """
        self.start_state = start_state
        revenue, path = self.longest_path_min(end_state, offset=offset)
        if path is None:
            return 0, None

        excluded = {len(path)}
        best_revenue, best_path = revenue, path

        while True:
            new_revenue, new_path = self.longest_path_min(
                end_state, excluded_lengths=excluded, offset=offset
            )
            if new_path is None or len(new_path) <= len(best_path):
                break
            if len(new_path) > max_path_length:
                break
            excluded.add(len(new_path))
            if new_revenue > best_revenue:
                best_revenue, best_path = new_revenue, new_path

        return best_revenue, best_path


# ===========================================================================
# EXAMPLES / TESTS
# ===========================================================================
if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # Example 1 – simple weighted graph (Dijkstra / longest path)
    # -----------------------------------------------------------------------
    class MyOptimizer(Optimizer):
        G = {
            's': [('u', 10), ('x', 5)],
            'u': [('v', 1), ('x', 2)],
            'v': [('y', 4)],
            'x': [('u', 3), ('v', 9), ('y', 2)],
            'y': [('s', 7), ('v', 6)],
        }

        def get_state(self, past_decisions):
            return past_decisions[-1]

        def get_next_decisions(self, curr_state):
            return self.G.get(curr_state, [])

    op = MyOptimizer()
    solution = op.longest_path('s', 'v', max_path_length=4)
    print("Longest path s→v:", solution)

    # -----------------------------------------------------------------------
    # Example 2 – DP tutorial graph
    # -----------------------------------------------------------------------
    class MyOptimizer2(Optimizer):
        G = {
            'a': [('b', 2), ('c', 4), ('d', 3)],
            'b': [('e', 7), ('f', 4), ('g', 6)],
            'c': [('e', 3), ('f', 2), ('g', 4)],
            'd': [('e', 4), ('f', 1), ('g', 5)],
            'e': [('h', 1), ('i', 4)],
            'f': [('h', 6), ('i', 3)],
            'g': [('h', 3), ('i', 3)],
            'h': [('j', 3)],
            'i': [('j', 4)],
        }

        def get_state(self, past_decisions):
            return past_decisions[-1]

        def get_next_decisions(self, curr_state):
            return self.G.get(curr_state, [])

    print('======= SHORTEST a→j =======')
    op2 = MyOptimizer2()
    for solution in op2.shortest_path('a', 'j'):
        print(solution)
    print('======= LONGEST  a→j =======')
    print(op2.longest_path('a', 'j'))

    # -----------------------------------------------------------------------
    # Example 3 – Rod Cutting
    # https://medium.com/@pratikone/dynamic-programming-for-the-confused-rod-cutting-problem-588892796840
    # -----------------------------------------------------------------------
    ROD_SIZE = 5

    class RodCutting(Optimizer):
        problem = {
            'size 1': (1, 1),
            'size 2': (2, 5),
            'size 3': (3, 8),
            'size 4': (4, 9),
            'size 5': (5, 10),
            'size 6': (6, 17),
            'size 7': (7, 17),
            'size 8': (8, 20),
            'size 9': (9, 24),
            'size 10': (10, 30),
        }

        def get_state(self, past_decisions):
            stage = len(past_decisions)
            remaining = ROD_SIZE - sum(self.problem[d][0] for d in past_decisions)
            return stage, remaining

        def get_next_decisions(self, current_state):
            _stage, remaining = current_state
            return [
                (choice, value[1])
                for choice, value in self.problem.items()
                if value[0] <= remaining
            ]

    print('======= ROD CUTTING =======')
    op = RodCutting()
    print(op.longest_path((0, ROD_SIZE), (ANY, 0)))

    # -----------------------------------------------------------------------
    # Example 4 – Capital Budgeting (Michael Trick)
    # -----------------------------------------------------------------------
    class CapitalBudgeting(Optimizer):
        _choices = {
            0: {'plant1-p1': (0, 0), 'plant1-p2': (1, 5), 'plant1-p3': (2, 6)},
            1: {'plant2-p1': (0, 0), 'plant2-p2': (2, 8), 'plant2-p3': (3, 9), 'plant2-p4': (4, 12)},
            2: {'plant3-p1': (0, 0), 'plant3-p2': (1, 4)},
        }
        _costs = {k: v[0] for stage in _choices.values() for k, v in stage.items()}

        def get_state(self, past_decisions):
            stage = len(past_decisions)
            spent = sum(self._costs[d] for d in past_decisions)
            return stage, 5 - spent

        def get_next_decisions(self, current_state):
            stage, budget = current_state
            choices = self._choices.get(stage, {})
            return [
                (name, vals[1])
                for name, vals in choices.items()
                if vals[0] <= budget
            ]

    print('======= CAPITAL BUDGETING =======')
    op = CapitalBudgeting()
    print(op.longest_path((0, 5), (3, 0)))

    # -----------------------------------------------------------------------
    # Example 5 – Knapsack
    # -----------------------------------------------------------------------
    class Knapsack(Optimizer):
        items = [('item1', 2, 65), ('item2', 3, 80), ('item3', 1, 30)]

        def get_state(self, past_decisions):
            stage = len(past_decisions)
            remaining = 5
            for i, qty in enumerate(past_decisions):
                remaining -= qty * self.items[i][1]
            return stage, remaining

        def get_next_decisions(self, current_state):
            stage, remaining = current_state
            if stage >= len(self.items):
                return []
            _name, weight, benefit = self.items[stage]
            decisions = []
            qty = 0
            while qty * weight <= remaining:
                decisions.append((qty, benefit * qty))
                qty += 1
            return decisions

    print('======= KNAPSACK =======')
    op = Knapsack()
    print(op.longest_path((0, 5), (3, 0)))

    # -----------------------------------------------------------------------
    # Example 6 – Equipment Replacement (Michael Trick)
    # -----------------------------------------------------------------------
    class EquipmentReplacement(Optimizer):
        maintenance_cost = {0: 60, 1: 80, 2: 120}
        market_value = {0: 1000, 1: 800, 2: 600, 3: 500}

        def get_state(self, past_decisions):
            year = len(past_decisions)
            if year == 6:
                return 6, -1
            age = 0
            for decision in past_decisions:
                age = age + 1 if decision == 'keep' else 1
            return year, age

        def get_next_decisions(self, current_state):
            year, age = current_state
            if age == -1:
                return []
            if year == 0:
                return [('buy', 1000 + self.maintenance_cost[0])]
            if year == 5:
                return [('sell', -self.market_value[age])]
            if age == 3:
                return [('trade', -self.market_value[age] + 1000 + self.maintenance_cost[0])]
            return [
                ('keep', self.maintenance_cost[age]),
                ('trade', -self.market_value[age] + 1000 + self.maintenance_cost[0]),
            ]

    print('======= EQUIPMENT REPLACEMENT =======')
    op = EquipmentReplacement(offset=10_000)
    for solution in op.shortest_path((0, 0), (6, -1)):
        print(solution)

    # -----------------------------------------------------------------------
    # Example 7 – Romania map (A* with heuristic)
    # -----------------------------------------------------------------------
    class BookMap(Optimizer):
        G = {
            'arad':      [('sibiu', 140), ('timisoara', 118), ('zerind', 75)],
            'bucharest': [('giurgiu', 90), ('urzineci', 85), ('fagaras', 211), ('pitesti', 101)],
            'craiova':   [('rimnicu', 146), ('pitesti', 138), ('drobeta', 120)],
            'drobeta':   [('craiova', 120), ('mehadia', 75)],
            'eforie':    [('hirsova', 86)],
            'fagaras':   [('sibiu', 99), ('bucharest', 211)],
            'giurgiu':   [('bucharest', 90)],
            'hirsova':   [('eforie', 86), ('urzineci', 98)],
            'lasi':      [('neamt', 87), ('vaslui', 92)],
            'lugoj':     [('mehadia', 70), ('timisoara', 111)],
            'mehadia':   [('drobeta', 75), ('lugoj', 70)],
            'neamt':     [('lasi', 87)],
            'oradea':    [('zerind', 71), ('sibiu', 151)],
            'pitesti':   [('bucharest', 101), ('rimnicu', 97), ('craiova', 138)],
            'rimnicu':   [('pitesti', 97), ('sibiu', 80), ('craiova', 146)],
            'sibiu':     [('rimnicu', 80), ('arad', 140), ('oradea', 151), ('fagaras', 99)],
            'timisoara': [('lugoj', 111), ('arad', 118)],
            'urzineci':  [('bucharest', 85), ('vaslui', 142), ('hirsova', 98)],
            'vaslui':    [('urzineci', 142), ('lasi', 92)],
            'zerind':    [('arad', 75), ('oradea', 71)],
        }
        _heuristic = {
            'arad': 366, 'bucharest': 0, 'craiova': 160, 'drobeta': 242,
            'eforie': 161, 'fagaras': 176, 'giurgiu': 77, 'hirsova': 151,
            'lasi': 226, 'lugoj': 244, 'mehadia': 241, 'neamt': 234,
            'oradea': 380, 'pitesti': 100, 'rimnicu': 193, 'sibiu': 253,
            'timisoara': 329, 'urzineci': 80, 'vaslui': 199, 'zerind': 374,
        }

        def get_state(self, past_decisions):
            return past_decisions[-1]

        def get_next_decisions(self, current_state):
            return self.G.get(current_state, [])

        def get_heuristic_cost(self, city):
            try:
                return self._heuristic[city]
            except KeyError:
                raise ValueError(f"Unknown city: {city!r}")

    op = BookMap()
    print('======= ROMANIA MAP: oradea → bucharest =======')
    for solution in op.shortest_path('oradea', 'bucharest'):
        print(solution)

    # -----------------------------------------------------------------------
    # Example 8 – Least-square segmentation
    # -----------------------------------------------------------------------
    def _sum_sq(arr):
        return sum(x * x for x in arr)

    class LeastSquareSegmenter(Optimizer):
        series = [10, 20, 34, 50, 60, 70, 80]

        def SSD(self, start, end):
            sub = self.series[start:end]
            n = end - start
            if n == 0:
                return 0.0
            return _sum_sq(sub) - (sum(sub) ** 2) / n

        def get_state(self, past_decisions):
            nr_segments = self.start_state[0] - len(past_decisions)
            last_index = sum(len(d) for d in past_decisions)
            return nr_segments, last_index

        def get_next_decisions(self, current_state):
            nr_seg, j = current_state
            if nr_seg == 1:
                return [(self.series[j:], self.SSD(j, len(self.series)))]
            return [
                (self.series[j:i], self.SSD(j, i))
                for i in range(j + 1, len(self.series) - nr_seg + 1)
            ]

    print('======= LEAST-SQUARE SEGMENTATION (2 segments) =======')
    op = LeastSquareSegmenter()
    nr_segments = 2
    for sol in op.shortest_path((nr_segments, 0), (0, ANY)):
        print(sol)

    # -----------------------------------------------------------------------
    # Example 9 – Best string alignment
    # -----------------------------------------------------------------------
    class BestAlignment(Optimizer):
        def get_state(self, past_decisions):
            s1, s2 = self.start_state
            len1 = sum(1 for c1, _ in past_decisions if c1 != '=')
            len2 = sum(1 for _, c2 in past_decisions if c2 != '=')
            return s1[len1:], s2[len2:]

        def get_next_decisions(self, current_state):
            r1, r2 = current_state
            if not r1 and not r2:
                return []
            if not r1:
                return [(('=', r2[0]), 6)]
            if not r2:
                return [((r1[0], '='), 6)]
            ch1, ch2 = r1[0], r2[0]
            return [
                ((ch1, ch2), 0 if ch1 == ch2 else 2),
                (('=', ch2), 6),
                ((ch1, '='), 6),
            ]

    str1, str2 = 'GAATTCAGTTA', 'GGATCGA'
    op = BestAlignment()
    print('======= BEST ALIGNMENT =======')
    for solution in op.shortest_path((str1, str2), ('', '')):
        print(solution)

    # -----------------------------------------------------------------------
    # Example 10 – Hidden Markov Model (Viterbi via shortest-path)
    # http://en.wikipedia.org/wiki/Viterbi_algorithm
    # -----------------------------------------------------------------------
    class HiddenMarkovModel(Optimizer):
        """
        Finds the most probable hidden-state sequence for a sequence of observations.

        Uses the shortest-path framework with log-probability costs so that
        maximising probability becomes minimising (negative log-probability).
        """

        def __init__(self):
            super().__init__(offset=1)
            self.hidden_states = ('Healthy', 'Fever')
            self.start_p  = {'Healthy': 0.6, 'Fever': 0.4}
            self.trans_p  = {
                'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
                'Fever':   {'Healthy': 0.4, 'Fever': 0.6},
            }
            self.emit_p   = {
                'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
                'Fever':   {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
            }

        def get_state(self, past_decisions):
            if not past_decisions:
                return 0, None
            return len(past_decisions), past_decisions[-1]

        def get_next_decisions(self, current_state):
            stage, hidden = current_state
            if stage == len(self.obs):
                return []
            obs = self.obs[stage]
            if stage == 0:
                return [(y, self.start_p[y] * self.emit_p[y][obs]) for y in self.hidden_states]
            return [(y, self.trans_p[hidden][y] * self.emit_p[y][obs]) for y in self.hidden_states]

        def cost_operator(self, accumulated, step_prob):
            # Minimise negative log-probability → maximises probability
            return accumulated + log(self.offset / step_prob)

        def get_probability(self, sequence):
            """Compute the exact joint probability of *sequence* given self.obs."""
            y = sequence[0]
            prob = self.start_p[y] * self.emit_p[y][self.obs[0]]
            for prev, curr, o in zip(sequence, sequence[1:], self.obs[1:]):
                prob *= self.trans_p[prev][curr] * self.emit_p[curr][o]
            return prob

        def get_most_probable_sequences(self, observations):
            """Yield ``(probability, hidden_state_sequence)`` in decreasing probability order."""
            self.obs = observations
            start = (0, None)
            end   = (len(observations), ANY)
            for _cost, seq in self.shortest_path(start, end):
                yield self.get_probability(seq), seq

        def get_next_most_probable_state(self):
            """Given the already-decoded optimal sequence, predict the next state/observation."""
            prev = self.optimal_sequence[-1]
            best_prob, best_state, best_obs = 0, None, None
            for next_obs in ('normal', 'cold', 'dizzy'):
                for next_state in self.hidden_states:
                    prob = self.trans_p[prev][next_state] * self.emit_p[next_state][next_obs]
                    print(f"  next_obs={next_obs!r}, next_state={next_state!r}, prob={prob:.3g}")
                    if prob > best_prob:
                        best_prob, best_state, best_obs = prob, next_state, next_obs
            return best_state, best_obs

    print("#" * 50)
    print("HIDDEN MARKOV MODEL")
    print("#" * 50)
    observations = ('normal', 'cold', 'dizzy')
    print("Observations:", observations)
    hmm = HiddenMarkovModel()
    print("Most probable hidden-state sequences (best first):")
    for i, (prob, seq) in enumerate(hmm.get_most_probable_sequences(observations)):
        if i == 0:
            hmm.optimal_sequence = seq
        print(f"  seq={seq}  prob={prob:.4g}")
    print("Predicting next state/observation:")
    hmm.get_next_most_probable_state()

    # -----------------------------------------------------------------------
    # Image seam-carving (requires PIL; gated so the rest runs without it)
    # -----------------------------------------------------------------------
    try:
        import math
        import os
        from PIL import Image

        Image.Image.__getitem__ = Image.Image.getpixel
        Image.Image.__setitem__ = Image.Image.putpixel

        def get_black_white(im):
            im = im.convert('L')
            im = im.point(lambda x: 255 if x < 155 else 0)
            return im.crop(im.getbbox())

        def enhance_black_white(im):
            im = im.point(lambda x: 0 if x < 155 else 255)
            return im.crop(im.getbbox())

        def get_neighbours(x, y, xmax, ymax):
            if y < ymax:                         yield x,     y + 1
            if y > 0:                            yield x,     y - 1
            if y > 0 and x < xmax:              yield x + 1, y - 1
            if x < xmax:                         yield x + 1, y
            if y < ymax and x < xmax:           yield x + 1, y + 1
            if x > 0 and y > 0:                 yield x - 1, y - 1
            if x > 0:                            yield x - 1, y
            if y < ymax and x > 0:              yield x - 1, y + 1

        class ImageSeamOptimizer(Optimizer):
            def __init__(self, xmax, ymax, im):
                super().__init__()
                self.xmax, self.ymax, self.im = xmax, ymax, im

            def get_state(self, past_decisions):
                return past_decisions[-1]

            def get_next_decisions(self, current_state):
                x, y = current_state
                return [
                    (ns, self._cost(x, y, ns))
                    for ns in get_neighbours(x, y, self.xmax, self.ymax)
                    if ns not in self.decision_path
                ]

            def _left_neighbours(self, x, y):
                if x > 0 and y > 0:       yield x - 1, y - 1
                if x > 0:                  yield x - 1, y
                if x > 0 and y < self.ymax: yield x - 1, y + 1

            def _right_neighbours(self, x, y):
                if y < self.ymax:                      yield x, y + 1
                if y < self.ymax and x < self.xmax:   yield x + 1, y + 1
                if y < self.ymax and x > 0:            yield x - 1, y + 1

            def _cost(self, x, y, next_state):
                xn, yn = next_state
                if self.im[xn, yn] == 255:
                    return 150
                assert self.im[xn, yn] == 0
                if yn > y:
                    nbrs = [self.im[xp, yp] for xp, yp in self._left_neighbours(xn, yn)]
                else:
                    nbrs = [self.im[xp, yp] for xp, yp in self._right_neighbours(xn, yn)]
                if nbrs.count(255) > 1:
                    return 1
                bonus = sum(1 for xp in range(x) if self.im[xp, y] == 255)
                dist = 5 * math.sqrt((xn - x) ** 2 + (yn - y) ** 2)
                return dist if bonus >= 5 else 50

        def find_seam(image_file, output_file, color=False):
            image_file = os.path.abspath(image_file)
            im_orig = Image.open(image_file)
            im = get_black_white(im_orig) if color else enhance_black_white(im_orig)
            im.save(output_file)
            xmax, ymax = im.size[0] - 1, im.size[1] - 1
            for x in range(xmax):
                for y in range(ymax):
                    if im[x, y] == 0:
                        nbrs = [im[xp, yp] for xp, yp in get_neighbours(x, y, xmax, ymax)]
                        if nbrs.count(255) > 5:
                            im[x, y] = 255
            xstart = 5
            while im[xstart, ymax] == 255:
                xstart += 1
            op = ImageSeamOptimizer(xmax, ymax, im)
            return next(op.shortest_path((xstart, ymax), (0, 0), allsolutions=False))

    except ImportError:
        pass  # PIL not available — skip image examples


    # ---------------------------------------------------------------------------
    # HMM demo
    # ---------------------------------------------------------------------------
    def demo_hmm():
        """Run the HMM Viterbi demo and return the most probable sequence."""
        from math import log
        # Inline minimal HMM to avoid dependency on __main__ block
        class _HMM(Optimizer):
            def __init__(self):
                super().__init__(offset=1)
                self.hidden_states = ('Healthy', 'Fever')
                self.start_p  = {'Healthy': 0.6, 'Fever': 0.4}
                self.trans_p  = {
                    'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
                    'Fever':   {'Healthy': 0.4, 'Fever': 0.6},
                }
                self.emit_p   = {
                    'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
                    'Fever':   {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
                }
            def get_state(self, past_decisions):
                if not past_decisions: return 0, None
                return len(past_decisions), past_decisions[-1]
            def get_next_decisions(self, current_state):
                stage, hidden = current_state
                if stage == len(self.obs): return []
                o = self.obs[stage]
                if stage == 0:
                    return [(y, self.start_p[y] * self.emit_p[y][o]) for y in self.hidden_states]
                return [(y, self.trans_p[hidden][y] * self.emit_p[y][o]) for y in self.hidden_states]
            def cost_operator(self, acc, p):
                return acc + log(self.offset / p)
            def get_probability(self, seq):
                prob = self.start_p[seq[0]] * self.emit_p[seq[0]][self.obs[0]]
                for prev, curr, o in zip(seq, seq[1:], self.obs[1:]):
                    prob *= self.trans_p[prev][curr] * self.emit_p[curr][o]
                return prob

        hmm = _HMM()
        obs = ('normal', 'cold', 'dizzy')
        hmm.obs = obs
        first = next(hmm.shortest_path((0, None), (len(obs), ANY)))
        return hmm.get_probability(first[1]), first[1]

    demo_hmm()

# ---------------------------------------------------------------------------
# TODO list
# ---------------------------------------------------------------------------
# TODO: implement the least-squares segmentation problem fully
# TODO: implement Kalman filter (a special case of HMM)
# TODO: detect cycles automatically in longest_path
# TODO: implement Bellman-Ford for graphs with negative-cost edges

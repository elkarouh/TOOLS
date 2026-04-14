"""
Shortest-path / dynamic-programming optimiser framework.

ALL DYNAMIC PROGRAMMING PROBLEMS CAN BE TRANSFORMED INTO SHORTEST-PATH PROBLEMS
UNDER THE CONDITION THAT for each decision there is only ONE next state.
This is NOT the case for reinforcement-learning problems where a decision can lead
to several states with a probability — VALUE ITERATION must be used there to find
a policy (best action in every state).
"""

from math import log
from typing import Any, Generator

from stdlib import ANY, FifoQueue, LifoQueue, PriorityQueue


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
        self.decision_path: list = []  # last expanded path — available to get_next_decisions
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
        fringe = PriorityQueue((0, 0, [], start_state))
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
        fringe = fringetype((0, [], start_state))
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

        return []  # fringe exhausted — failure

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
    # Longest-path
    # ------------------------------------------------------------------

    def longest_path(
        self,
        start_state: Any,
        end_state: Any,
        max_path_length: int = 1_000,
    ) -> tuple[float, list | None]:
        """
        Find the path from *start_state* to *end_state* with the highest
        total revenue (sum of step revenues returned by ``get_next_decisions``).

        Uses a max-revenue-first search.  The visited table is keyed on
        ``(state, path_length)`` so that a state reached at depth *d* with
        inferior revenue is pruned, but the same state reached at a different
        depth is explored independently.  This correctly handles cyclic graphs
        while still pruning dominated sub-paths.

        Parameters
        ----------
        max_path_length:
            Hard cap on path length to prevent infinite expansion on cyclic
            graphs.  Increase if your problem has very long optimal paths.

        Returns
        -------
        ``(revenue, path)`` for the best solution, or ``(0, None)`` if no
        path to *end_state* exists.
        """
        self.start_state = start_state
        best_revenue: float | None = None
        best_path: list | None = None

        # Fringe: (-revenue, path_length, path, state).
        # Negating revenue lets PriorityQueue (min-heap) act as a max-heap.
        fringe = PriorityQueue((0, 0, [], start_state))

        # visited[(state, path_length)] → best revenue seen so far.
        # Prune any revisit that cannot improve on the recorded revenue.
        visited: dict[tuple, float] = {}
        NEG_INF = float("-inf")

        while fringe:
            neg_rev, path_length, path, current_state = fringe.pop()
            revenue = -neg_rev

            key = (current_state, path_length)
            if visited.get(key, NEG_INF) >= revenue:
                continue
            visited[key] = revenue

            if current_state == end_state and path_length > 0:
                if best_revenue is None or revenue > best_revenue:
                    best_revenue, best_path = revenue, path

            if path_length < max_path_length:
                for new_decision, step_rev in self.get_next_decisions(current_state):
                    new_path = path + [new_decision]
                    next_state = self.get_state(new_path)
                    new_rev = revenue + step_rev
                    new_len = path_length + 1
                    if visited.get((next_state, new_len), NEG_INF) < new_rev:
                        fringe.push((-new_rev, new_len, new_path, next_state))

        if best_path is None:
            return 0, None
        return best_revenue, best_path


# ---------------------------------------------------------------------------
# TODO list
# ---------------------------------------------------------------------------
# TODO: implement the least-squares segmentation problem fully
# TODO: implement Kalman filter (a special case of HMM)
# TODO: detect cycles automatically in longest_path
# TODO: implement Bellman-Ford for graphs with negative-cost edges

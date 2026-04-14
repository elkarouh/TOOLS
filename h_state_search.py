"""
State Space Search
------------------

Generic state-space search framework.
No dependency on h_shortest_path.
"""

import itertools

from stdlib import FifoQueue, LifoQueue, PriorityQueue


# ============================================================
# STATE SEARCH BASE CLASS
# ============================================================


class StateSearcher:
    """
    Generic state-space searcher.

    Subclass and override:
        get_next_states(state)  -> list of (state, action, next_state) triples
        is_goal(state)          -> True when the goal is reached
        heuristic(state)        -> estimated moves remaining (default 0)

    Then call one of:
        search(initial)             — BFS (fewest moves, guaranteed optimal)
        search_dfs(initial, limit)  — DFS with optional depth limit
        search_best_first(initial)  — greedy best-first (heuristic only)
        search_astar(initial)       — A* (moves so far + heuristic)
        search_all(initial)         — generator yielding every solution (BFS order)

    Every method returns a list of (state, action, next_state) triples,
    or [] if no solution exists.
    """

    def get_next_states(self, current_state):
        """Return all legal moves as (currnet_state, action, next_state) triples."""
        raise NotImplementedError("Override get_next_states()")

    def is_goal(self, state) -> bool:
        """Return True when state is a goal state."""
        raise NotImplementedError("Override is_goal()")

    def heuristic(self, state) -> float:
        """Admissible estimate of moves still needed. Override for A*/best-first."""
        return 0

    # ------------------------------------------------------------------
    # Internal search engine
    # ------------------------------------------------------------------

    def _search(
        self, initial, queue_type, use_heuristic=False, use_cost=False, limit=None
    ):
        counter = itertools.count()

        def push(fringe, depth, path, state):
            h = self.heuristic(state) if use_heuristic else 0
            g = depth if use_cost else 0
            priority = g + h
            if queue_type is PriorityQueue:
                fringe.push((priority, next(counter), depth, path, state))
            else:
                fringe.push((depth, path, state))

        def pop(fringe):
            item = fringe.pop()
            if queue_type is PriorityQueue:
                _priority, _cnt, depth, path, state = item
            else:
                depth, path, state = item
            return depth, path, state

        fringe = queue_type()
        push(fringe, 0, [], initial)
        visited = set()

        while fringe:
            depth, path, state = pop(fringe)

            if state in visited:
                continue
            visited.add(state)

            if self.is_goal(state):
                return path

            if limit is not None and depth >= limit:
                continue

            for from_state, action, next_state in self.get_next_states(state):
                if next_state not in visited:
                    step = (from_state, action, next_state)
                    push(fringe, depth + 1, path + [step], next_state)

        return []

    # ------------------------------------------------------------------
    # Public search strategies
    # ------------------------------------------------------------------

    def search(self, initial):
        """BFS — always returns the shortest solution (fewest moves)."""
        return self._search(initial, FifoQueue)

    def search_dfs(self, initial, limit=None):
        """DFS — optionally bounded by depth limit. Not guaranteed optimal."""
        return self._search(initial, LifoQueue, limit=limit)

    def search_best_first(self, initial):
        """Greedy best-first — expands lowest heuristic node. Fast, not optimal."""
        return self._search(initial, PriorityQueue, use_heuristic=True, use_cost=False)

    def search_astar(self, initial):
        """A* — optimal when heuristic is admissible."""
        return self._search(initial, PriorityQueue, use_heuristic=True, use_cost=True)

    def search_all(self, initial):
        """Yield every solution in BFS order (shortest first). Use break to stop early."""
        fringe = FifoQueue()
        fringe.push((0, [], initial))
        visited = set()

        while fringe:
            depth, path, state = fringe.pop()

            if state in visited:
                continue
            visited.add(state)

            if self.is_goal(state):
                yield path

            for from_state, action, next_state in self.get_next_states(state):
                if next_state not in visited:
                    fringe.push(
                        (
                            depth + 1,
                            path + [(from_state, action, next_state)],
                            next_state,
                        )
                    )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_solution(self, solution, start):
        if not solution:
            print("No solution")
            return

        for prev, action, next_state in solution:
            print(f"{prev} --{action}--> {next_state}")

        print(f"({len(solution)} moves)")

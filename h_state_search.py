"""
State Space Search Examples
---------------------------

Self-contained state-space search framework and classic AI puzzles.
No dependency on h_shortest_path.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from typing import Any, Generator


# ============================================================
# Queue implementations (self-contained copies)
# ============================================================


class PriorityQueue:
    def __init__(self):
        self._counter = itertools.count()
        self._heap = []

    def push(self, item):
        heapq.heappush(self._heap, (item[0], next(self._counter), item[1:]))

    def pop(self):
        priority, _count, rest = heapq.heappop(self._heap)
        return (priority, *rest)

    def __bool__(self):
        return bool(self._heap)


class FifoQueue:
    def __init__(self):
        self._dq = deque()

    def push(self, item):
        self._dq.append(item)

    def pop(self):
        return self._dq.popleft()

    def __bool__(self):
        return bool(self._dq)


class LifoQueue:
    def __init__(self):
        self._stack = []

    def push(self, item):
        self._stack.append(item)

    def pop(self):
        return self._stack.pop()

    def __bool__(self):
        return bool(self._stack)


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


# ============================================================
# WATER JUG
# ============================================================
# ===========================================================================
# StateSearcher demo — Water-Jug Problem
# ===========================================================================
"""
Water-Jug Problem
-----------------
You have two jugs:
  • Big jug  : capacity 5 litres
  • Small jug: capacity 3 litres

Neither jug has any measurement markings.
You have an unlimited water supply (tap) and a drain.

Legal moves
~~~~~~~~~~~
  1. Fill big   — fill the 5-litre jug to the top from the tap
  2. Fill small — fill the 3-litre jug to the top from the tap
  3. Empty big  — pour all water in the big jug down the drain
  4. Empty small— pour all water in the small jug down the drain
  5. Pour big→small — pour from big into small until small is full or big is empty
  6. Pour small→big — pour from small into big until big is full or small is empty

Goal: have exactly 4 litres in the big jug.

State: (big, small)  where  0 ≤ big ≤ 5,  0 ≤ small ≤ 3.

Solution (BFS — fewest moves):
  (0,0) →fill big→ (5,0) →pour big→small→ (2,3) →empty small→ (2,0)
       →pour big→small→ (0,2) →fill big→ (5,2) →pour big→small→ (4,3) ✓
"""


class WaterJug(StateSearcher):
    def __init__(self, cap_big=5, cap_small=3, target=4):
        self.B = cap_big
        self.S = cap_small
        self.T = target

    def get_next_states(self, current_state):
        b, s = current_state
        B, S = self.B, self.S
        # These 4 moves are always possible !
        next_states = [
            ("fill big", (B, s)),
            ("fill small", (b, S)),
            ("empty big", (0, s)),
            ("empty small", (b, 0)),
        ]
        # pour big → small
        p = min(b, S - s)
        next_states.append(("pour big→small", (b - p, s + p)))
        # pour small → big
        p = min(s, B - b)
        next_states.append(("pour small→big", (b + p, s - p)))
        return [
            (current_state, name, next_state)
            for name, next_state in next_states
            if next_state != current_state
        ]

    def is_goal(self, state):
        b, _ = state
        return b == self.T


# ============================================================
# MISSIONARIES AND CANNIBALS
# ============================================================


class MissionariesCannibals(StateSearcher):
    def get_next_states(self, state):
        M, C, boat = state
        moves = []
        direction = -1 if boat == 0 else 1

        def safe(M, C):
            M_right = 3 - M
            C_right = 3 - C
            if M > 0 and C > M:
                return False
            if M_right > 0 and C_right > M_right:
                return False
            return True

        for m, c in [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]:
            newM = M + direction * m
            newC = C + direction * c
            if 0 <= newM <= 3 and 0 <= newC <= 3:
                if safe(newM, newC):
                    moves.append((state, f"{m}M {c}C cross", (newM, newC, 1 - boat)))

        return moves

    def is_goal(self, state):
        return state == (0, 0, 1)


# ============================================================
# WOLF GOAT CABBAGE
# ============================================================


class WolfGoatCabbage(StateSearcher):
    def safe(self, state):
        f, w, g, c = state
        if w == g and f != w:
            return False
        if g == c and f != g:
            return False
        return True

    def get_next_states(self, state):
        f, w, g, c = state
        other = 1 - f
        moves = []

        def add(name, ns):
            if self.safe(ns):
                moves.append((state, name, ns))

        add("farmer", (other, w, g, c))

        if f == w:
            add("take wolf", (other, other, g, c))

        if f == g:
            add("take goat", (other, w, other, c))

        if f == c:
            add("take cabbage", (other, w, g, other))

        return moves

    def is_goal(self, state):
        return state == (1, 1, 1, 1)


# ============================================================
# TOWER OF HANOI
# ============================================================


class TowerOfHanoi(StateSearcher):
    def __init__(self, n=3):
        self.n = n

    def get_next_states(self, state):
        rods = [list(r) for r in state]
        moves = []

        for i in range(3):
            if not rods[i]:
                continue

            disk = rods[i][-1]

            for j in range(3):
                if i == j:
                    continue

                if not rods[j] or rods[j][-1] > disk:
                    new = [list(r) for r in rods]
                    new[i].pop()
                    new[j].append(disk)

                    moves.append((state, f"{i}->{j}", tuple(tuple(r) for r in new)))

        return moves

    def is_goal(self, state):
        return state == ((), (), tuple(range(self.n, 0, -1)))


# ============================================================
# EIGHT PUZZLE
# ============================================================


class EightPuzzle(StateSearcher):
    goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)

    def get_next_states(self, state):
        i = state.index(0)
        r, c = divmod(i, 3)

        moves = []

        def swap(j, name):
            s = list(state)
            s[i], s[j] = s[j], s[i]
            moves.append((state, name, tuple(s)))

        if r > 0:
            swap(i - 3, "up")
        if r < 2:
            swap(i + 3, "down")
        if c > 0:
            swap(i - 1, "left")
        if c < 2:
            swap(i + 1, "right")

        return moves

    def is_goal(self, state):
        return state == self.goal


# ============================================================
# DEMOS
# ============================================================


def demo_waterjug():
    print("=" * 60)
    print("WATER-JUG PROBLEM  (5-litre big, 3-litre small, target 4 L)")
    print("=" * 60)

    jug = WaterJug(cap_big=5, cap_small=3, target=4)
    initial = (0, 0)  # both jugs empty

    # --- BFS: guaranteed shortest solution ---
    print("\n[BFS — shortest solution]")
    sol_bfs = jug.search(initial)
    jug.print_solution(sol_bfs, initial)

    # --- A* (heuristic is trivial here, same result) ---
    print("\n[A* — optimal solution]")
    sol_astar = jug.search_astar(initial)
    jug.print_solution(sol_astar, initial)

    # --- All solutions (BFS order, stop after the first 3) ---
    print("\n[All solutions — first 3 in BFS order]")
    for i, sol in enumerate(jug.search_all(initial)):
        print(f"\n  Solution #{i + 1}  ({len(sol)} moves):")
        jug.print_solution(sol, initial)
        if i >= 2:
            break


def demo_missionaries():
    print("\n=== MISSIONARIES AND CANNIBALS ===")

    puzzle = MissionariesCannibals()
    initial = (3, 3, 0)

    sol = puzzle.search(initial)
    puzzle.print_solution(sol, initial)


def demo_wolf_goat():
    print("\n=== WOLF GOAT CABBAGE ===")

    puzzle = WolfGoatCabbage()
    initial = (0, 0, 0, 0)

    sol = puzzle.search(initial)
    puzzle.print_solution(sol, initial)


def demo_hanoi():
    print("\n=== TOWER OF HANOI ===")

    hanoi = TowerOfHanoi(n=3)
    initial = (tuple(range(3, 0, -1)), (), ())

    sol = hanoi.search(initial)
    hanoi.print_solution(sol, initial)


def demo_eight_puzzle():
    print("\n=== EIGHT PUZZLE ===")

    puzzle = EightPuzzle()
    initial = (1, 2, 3, 4, 0, 6, 7, 5, 8)

    sol = puzzle.search(initial)
    puzzle.print_solution(sol, initial)


if __name__ == "__main__":
    demo_waterjug()
    demo_missionaries()
    demo_wolf_goat()
    demo_hanoi()
    demo_eight_puzzle()

"""Examples and tests for h_shortest_path.Optimizer."""

from math import log

from h_shortest_path import Optimizer
from stdlib import ANY


# ---------------------------------------------------------------------------
# Example 1 – simple weighted graph (Dijkstra / longest path)
# ---------------------------------------------------------------------------
class MyOptimizer(Optimizer):
    G = {
        "s": [("u", 10), ("x", 5)],
        "u": [("v", 1), ("x", 2)],
        "v": [("y", 4)],
        "x": [("u", 3), ("v", 9), ("y", 2)],
        "y": [("s", 7), ("v", 6)],
    }

    def get_state(self, past_decisions):
        return past_decisions[-1]

    def get_next_decisions(self, curr_state):
        return self.G.get(curr_state, [])


# ---------------------------------------------------------------------------
# Example 2 – DP tutorial graph
# ---------------------------------------------------------------------------
class MyOptimizer2(Optimizer):
    G = {
        "a": [("b", 2), ("c", 4), ("d", 3)],
        "b": [("e", 7), ("f", 4), ("g", 6)],
        "c": [("e", 3), ("f", 2), ("g", 4)],
        "d": [("e", 4), ("f", 1), ("g", 5)],
        "e": [("h", 1), ("i", 4)],
        "f": [("h", 6), ("i", 3)],
        "g": [("h", 3), ("i", 3)],
        "h": [("j", 3)],
        "i": [("j", 4)],
    }

    def get_state(self, past_decisions):
        return past_decisions[-1]

    def get_next_decisions(self, curr_state):
        return self.G.get(curr_state, [])


# ---------------------------------------------------------------------------
# Example 3 – Rod Cutting
# https://medium.com/@pratikone/dynamic-programming-for-the-confused-rod-cutting-problem-588892796840
# ---------------------------------------------------------------------------
ROD_SIZE = 5


class RodCutting(Optimizer):
    problem = {
        "size 1": (1, 1),
        "size 2": (2, 5),
        "size 3": (3, 8),
        "size 4": (4, 9),
        "size 5": (5, 10),
        "size 6": (6, 17),
        "size 7": (7, 17),
        "size 8": (8, 20),
        "size 9": (9, 24),
        "size 10": (10, 30),
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


# ---------------------------------------------------------------------------
# Example 4 – Capital Budgeting (Michael Trick)
# ---------------------------------------------------------------------------
class CapitalBudgeting(Optimizer):
    _choices = {
        0: {"plant1-p1": (0, 0), "plant1-p2": (1, 5), "plant1-p3": (2, 6)},
        1: {
            "plant2-p1": (0, 0),
            "plant2-p2": (2, 8),
            "plant2-p3": (3, 9),
            "plant2-p4": (4, 12),
        },
        2: {"plant3-p1": (0, 0), "plant3-p2": (1, 4)},
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
            (name, vals[1]) for name, vals in choices.items() if vals[0] <= budget
        ]


# ---------------------------------------------------------------------------
# Example 5 – Knapsack
# ---------------------------------------------------------------------------
MAX_CAP = 5


class Knapsack(Optimizer):
    items = [("item1", 2, 65), ("item2", 3, 80), ("item3", 1, 30)]

    def get_state(self, past_decisions):
        stage = len(past_decisions)
        remaining = MAX_CAP
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


# ---------------------------------------------------------------------------
# Example 6 – Equipment Replacement (Michael Trick)
# ---------------------------------------------------------------------------
class EquipmentReplacement(Optimizer):
    maintenance_cost = {0: 60, 1: 80, 2: 120}
    market_value = {0: 1000, 1: 800, 2: 600, 3: 500}

    def get_state(self, past_decisions):
        year = len(past_decisions)
        if year == 6:
            return 6, -1
        age = 0
        for decision in past_decisions:
            age = age + 1 if decision == "keep" else 1
        return year, age

    def get_next_decisions(self, current_state):
        year, age = current_state
        if age == -1:
            return []
        if year == 0:
            return [("buy", 1000 + self.maintenance_cost[0])]
        if year == 5:
            return [("sell", -self.market_value[age])]
        if age == 3:
            return [
                ("trade", -self.market_value[age] + 1000 + self.maintenance_cost[0])
            ]
        return [
            ("keep", self.maintenance_cost[age]),
            ("trade", -self.market_value[age] + 1000 + self.maintenance_cost[0]),
        ]


# ---------------------------------------------------------------------------
# Example 7 – Romania map (A* with heuristic)
# ---------------------------------------------------------------------------
class BookMap(Optimizer):
    G = {
        "arad": [("sibiu", 140), ("timisoara", 118), ("zerind", 75)],
        "bucharest": [
            ("giurgiu", 90),
            ("urzineci", 85),
            ("fagaras", 211),
            ("pitesti", 101),
        ],
        "craiova": [("rimnicu", 146), ("pitesti", 138), ("drobeta", 120)],
        "drobeta": [("craiova", 120), ("mehadia", 75)],
        "eforie": [("hirsova", 86)],
        "fagaras": [("sibiu", 99), ("bucharest", 211)],
        "giurgiu": [("bucharest", 90)],
        "hirsova": [("eforie", 86), ("urzineci", 98)],
        "lasi": [("neamt", 87), ("vaslui", 92)],
        "lugoj": [("mehadia", 70), ("timisoara", 111)],
        "mehadia": [("drobeta", 75), ("lugoj", 70)],
        "neamt": [("lasi", 87)],
        "oradea": [("zerind", 71), ("sibiu", 151)],
        "pitesti": [("bucharest", 101), ("rimnicu", 97), ("craiova", 138)],
        "rimnicu": [("pitesti", 97), ("sibiu", 80), ("craiova", 146)],
        "sibiu": [("rimnicu", 80), ("arad", 140), ("oradea", 151), ("fagaras", 99)],
        "timisoara": [("lugoj", 111), ("arad", 118)],
        "urzineci": [("bucharest", 85), ("vaslui", 142), ("hirsova", 98)],
        "vaslui": [("urzineci", 142), ("lasi", 92)],
        "zerind": [("arad", 75), ("oradea", 71)],
    }
    _heuristic = {
        "arad": 366,
        "bucharest": 0,
        "craiova": 160,
        "drobeta": 242,
        "eforie": 161,
        "fagaras": 176,
        "giurgiu": 77,
        "hirsova": 151,
        "lasi": 226,
        "lugoj": 244,
        "mehadia": 241,
        "neamt": 234,
        "oradea": 380,
        "pitesti": 100,
        "rimnicu": 193,
        "sibiu": 253,
        "timisoara": 329,
        "urzineci": 80,
        "vaslui": 199,
        "zerind": 374,
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


# ---------------------------------------------------------------------------
# Example 8 – Least-square segmentation
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Example 9 – Best string alignment
# ---------------------------------------------------------------------------
class BestAlignment(Optimizer):
    def get_state(self, past_decisions):
        s1, s2 = self.start_state
        len1 = sum(1 for c1, _ in past_decisions if c1 != "=")
        len2 = sum(1 for _, c2 in past_decisions if c2 != "=")
        return s1[len1:], s2[len2:]

    def get_next_decisions(self, current_state):
        r1, r2 = current_state
        if not r1 and not r2:
            return []
        if not r1:
            return [(("=", r2[0]), 6)]
        if not r2:
            return [((r1[0], "="), 6)]
        ch1, ch2 = r1[0], r2[0]
        return [
            ((ch1, ch2), 0 if ch1 == ch2 else 2),
            (("=", ch2), 6),
            ((ch1, "="), 6),
        ]


# ---------------------------------------------------------------------------
# Example 10 – Hidden Markov Model (Viterbi via shortest-path)
# http://en.wikipedia.org/wiki/Viterbi_algorithm
# ---------------------------------------------------------------------------
class HiddenMarkovModel(Optimizer):
    """
    Finds the most probable hidden-state sequence for a sequence of observations.

    Uses the shortest-path framework with log-probability costs so that
    maximising probability becomes minimising (negative log-probability).
    """

    def __init__(self):
        super().__init__(offset=1)
        self.hidden_states = ("Healthy", "Fever")
        self.start_p = {"Healthy": 0.6, "Fever": 0.4}
        self.trans_p = {
            "Healthy": {"Healthy": 0.7, "Fever": 0.3},
            "Fever": {"Healthy": 0.4, "Fever": 0.6},
        }
        self.emit_p = {
            "Healthy": {"normal": 0.5, "cold": 0.4, "dizzy": 0.1},
            "Fever": {"normal": 0.1, "cold": 0.3, "dizzy": 0.6},
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
            return [
                (y, self.start_p[y] * self.emit_p[y][obs])
                for y in self.hidden_states
            ]
        return [
            (y, self.trans_p[hidden][y] * self.emit_p[y][obs])
            for y in self.hidden_states
        ]

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
        end = (len(observations), ANY)
        for _cost, seq in self.shortest_path(start, end):
            yield self.get_probability(seq), seq

    def get_next_most_probable_state(self):
        """Given the already-decoded optimal sequence, predict the next state/observation."""
        prev = self.optimal_sequence[-1]
        best_prob, best_state, best_obs = 0, None, None
        for next_obs in ("normal", "cold", "dizzy"):
            for next_state in self.hidden_states:
                prob = (
                    self.trans_p[prev][next_state]
                    * self.emit_p[next_state][next_obs]
                )
                print(
                    f"  next_obs={next_obs!r}, next_state={next_state!r}, prob={prob:.3g}"
                )
                if prob > best_prob:
                    best_prob, best_state, best_obs = prob, next_state, next_obs
        return best_state, best_obs


# ---------------------------------------------------------------------------
# Image seam-carving (requires PIL; gated so the rest runs without it)
# ---------------------------------------------------------------------------
try:
    import math
    import os
    from PIL import Image

    Image.Image.__getitem__ = Image.Image.getpixel
    Image.Image.__setitem__ = Image.Image.putpixel

    def get_black_white(im):
        im = im.convert("L")
        im = im.point(lambda x: 255 if x < 155 else 0)
        return im.crop(im.getbbox())

    def enhance_black_white(im):
        im = im.point(lambda x: 0 if x < 155 else 255)
        return im.crop(im.getbbox())

    def get_neighbours(x, y, xmax, ymax):
        if y < ymax:
            yield x, y + 1
        if y > 0:
            yield x, y - 1
        if y > 0 and x < xmax:
            yield x + 1, y - 1
        if x < xmax:
            yield x + 1, y
        if y < ymax and x < xmax:
            yield x + 1, y + 1
        if x > 0 and y > 0:
            yield x - 1, y - 1
        if x > 0:
            yield x - 1, y
        if y < ymax and x > 0:
            yield x - 1, y + 1

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
            if x > 0 and y > 0:
                yield x - 1, y - 1
            if x > 0:
                yield x - 1, y
            if x > 0 and y < self.ymax:
                yield x - 1, y + 1

        def _right_neighbours(self, x, y):
            if y < self.ymax:
                yield x, y + 1
            if y < self.ymax and x < self.xmax:
                yield x + 1, y + 1
            if y < self.ymax and x > 0:
                yield x - 1, y + 1

        def _cost(self, x, y, next_state):
            xn, yn = next_state
            if self.im[xn, yn] == 255:
                return 150
            assert self.im[xn, yn] == 0
            if yn > y:
                nbrs = [self.im[xp, yp] for xp, yp in self._left_neighbours(xn, yn)]
            else:
                nbrs = [
                    self.im[xp, yp] for xp, yp in self._right_neighbours(xn, yn)
                ]
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
                    nbrs = [
                        im[xp, yp] for xp, yp in get_neighbours(x, y, xmax, ymax)
                    ]
                    if nbrs.count(255) > 5:
                        im[x, y] = 255
        xstart = 5
        while im[xstart, ymax] == 255:
            xstart += 1
        op = ImageSeamOptimizer(xmax, ymax, im)
        return next(op.shortest_path((xstart, ymax), (0, 0), allsolutions=False))

except ImportError:
    pass  # PIL not available — skip image examples


if __name__ == "__main__":
    op = MyOptimizer()
    solution = op.longest_path("s", "v", max_path_length=4)
    print("Longest path s→v:", solution)

    print("======= SHORTEST a→j =======")
    op2 = MyOptimizer2()
    for solution in op2.shortest_path("a", "j"):
        print(solution)
    print("======= LONGEST  a→j =======")
    print(op2.longest_path("a", "j"))

    print("======= ROD CUTTING =======")
    op = RodCutting()
    print(op.longest_path((0, ROD_SIZE), (ANY, 0)))

    print("======= CAPITAL BUDGETING =======")
    op = CapitalBudgeting()
    print(op.longest_path((0, 5), (3, 0)))

    print("======= KNAPSACK =======")
    op = Knapsack()
    print(op.longest_path((0, 5), (3, 0)))

    print("======= EQUIPMENT REPLACEMENT =======")
    op = EquipmentReplacement(offset=10_000)
    for solution in op.shortest_path((0, 0), (6, -1)):
        print(solution)

    op = BookMap()
    print("======= ROMANIA MAP: oradea → bucharest =======")
    for solution in op.shortest_path("oradea", "bucharest"):
        print(solution)

    print("======= LEAST-SQUARE SEGMENTATION (2 segments) =======")
    op = LeastSquareSegmenter()
    nr_segments = 2
    for sol in op.shortest_path((nr_segments, 0), (0, ANY)):
        print(sol)

    str1, str2 = "GAATTCAGTTA", "GGATCGA"
    op = BestAlignment()
    print("======= BEST ALIGNMENT =======")
    for solution in op.shortest_path((str1, str2), ("", "")):
        print(solution)

    print("#" * 50)
    print("HIDDEN MARKOV MODEL")
    print("#" * 50)
    observations = ("normal", "cold", "dizzy")
    print("Observations:", observations)
    hmm = HiddenMarkovModel()
    print("Most probable hidden-state sequences (best first):")
    for i, (prob, seq) in enumerate(hmm.get_most_probable_sequences(observations)):
        if i == 0:
            hmm.optimal_sequence = seq
        print(f"  seq={seq}  prob={prob:.4g}")
    print("Predicting next state/observation:")
    hmm.get_next_most_probable_state()

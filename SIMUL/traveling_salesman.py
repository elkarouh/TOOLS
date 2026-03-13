"""
Traveling Salesman Problem (TSP) using Simulated Annealing

Demonstrates solving the classic TSP using our optimization framework.
No external dependencies required (replaces simanneal library).

Features:
- Haversine distance calculation for lat/lon coordinates
- Multiple neighborhood strategies (swap, 2-opt, insert)
- Visualization of routes and convergence
- Comparison with nearest neighbor heuristic

Usage:
    python traveling_salesman.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import itertools

from random_variable import RandomVariable
from simulated_annealing import (
    SimulatedAnnealing,
    CoolingSchedule,
    NeighborhoodFunction,
    OptimizationResult,
)


# ============================================================================
# DISTANCE CALCULATIONS
# ============================================================================

def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate great-circle distance between two points on Earth.

    Args:
        coord1: (latitude, longitude) in degrees
        coord2: (latitude, longitude) in degrees

    Returns:
        Distance in miles
    """
    R = 3963  # Earth's radius in miles

    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def euclidean_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)


# ============================================================================
# TSP PROBLEM DEFINITION
# ============================================================================

@dataclass
class TSPProblem:
    """
    Traveling Salesman Problem definition.

    Stores cities and precomputes distance matrix for efficiency.
    """

    cities: Dict[str, Tuple[float, float]]  # {name: (lat, lon)}
    distance_func: Callable = haversine_distance
    _distance_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict, repr=False)
    _city_list: List[str] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Precompute distance matrix."""
        self._city_list = list(self.cities.keys())

        for city_a in self.cities:
            self._distance_matrix[city_a] = {}
            for city_b in self.cities:
                if city_a == city_b:
                    self._distance_matrix[city_a][city_b] = 0.0
                else:
                    self._distance_matrix[city_a][city_b] = self.distance_func(
                        self.cities[city_a], self.cities[city_b]
                    )

    def distance(self, city_a: str, city_b: str) -> float:
        """Get distance between two cities."""
        return self._distance_matrix[city_a][city_b]

    def route_length(self, route: List[str]) -> float:
        """Calculate total length of a route (returns to start)."""
        total = 0.0
        for i in range(len(route)):
            total += self.distance(route[i-1], route[i])
        return total

    def route_length_from_indices(self, indices: List[int]) -> float:
        """Calculate route length from city indices."""
        route = [self._city_list[i] for i in indices]
        return self.route_length(route)

    @property
    def n_cities(self) -> int:
        return len(self.cities)

    @property
    def city_names(self) -> List[str]:
        return self._city_list


# ============================================================================
# TSP NEIGHBORHOOD FUNCTIONS
# ============================================================================

@dataclass
class SwapNeighborhood(NeighborhoodFunction):
    """Swap two random cities in the tour."""

    def __call__(self, route: List[int], bounds=None) -> List[int]:
        if len(route) < 2:
            return route

        new_route = list(route)
        i, j = random.sample(range(len(route)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route

    def __str__(self) -> str:
        return "Swap"


@dataclass
class TwoOptNeighborhood(NeighborhoodFunction):
    """
    2-opt neighborhood: reverse a segment of the tour.

    This is often more effective than simple swaps for TSP.
    """

    def __call__(self, route: List[int], bounds=None) -> List[int]:
        if len(route) < 4:
            return route

        new_route = list(route)

        # Select two cut points
        i, j = sorted(random.sample(range(len(route)), 2))

        # Reverse the segment between i and j
        new_route[i:j+1] = reversed(new_route[i:j+1])

        return new_route

    def __str__(self) -> str:
        return "2-opt"


@dataclass
class InsertNeighborhood(NeighborhoodFunction):
    """Remove a city and insert it at a random position."""

    def __call__(self, route: List[int], bounds=None) -> List[int]:
        if len(route) < 2:
            return route

        new_route = list(route)

        # Remove a random city
        i = random.randrange(len(route))
        city = new_route.pop(i)

        # Insert at a random position
        j = random.randrange(len(new_route) + 1)
        new_route.insert(j, city)

        return new_route

    def __str__(self) -> str:
        return "Insert"


@dataclass
class MixedNeighborhood(NeighborhoodFunction):
    """Mix of swap, 2-opt, and insert moves."""

    swap_prob: float = 0.3
    two_opt_prob: float = 0.5
    # insert_prob = 1 - swap_prob - two_opt_prob

    def __post_init__(self):
        self._swap = SwapNeighborhood()
        self._two_opt = TwoOptNeighborhood()
        self._insert = InsertNeighborhood()

    def __call__(self, route: List[int], bounds=None) -> List[int]:
        r = random.random()
        if r < self.swap_prob:
            return self._swap(route, bounds)
        elif r < self.swap_prob + self.two_opt_prob:
            return self._two_opt(route, bounds)
        else:
            return self._insert(route, bounds)

    def __str__(self) -> str:
        return "Mixed(swap+2opt+insert)"


# ============================================================================
# TSP SOLVER
# ============================================================================

@dataclass
class TSPResult:
    """Results from TSP optimization."""

    route: List[str]
    distance: float
    iterations: int
    acceptance_rate: float
    history: List[float] = field(default_factory=list)

    def __str__(self) -> str:
        route_str = " -> ".join(self.route + [self.route[0]])
        return (
            f"TSP Solution:\n"
            f"  Distance: {self.distance:.1f} miles\n"
            f"  Iterations: {self.iterations:,}\n"
            f"  Acceptance Rate: {self.acceptance_rate*100:.1f}%\n"
            f"  Route: {route_str}"
        )


class TSPSolver:
    """
    Traveling Salesman Problem solver using Simulated Annealing.

    Example:
        >>> cities = {'A': (0, 0), 'B': (1, 0), 'C': (1, 1), 'D': (0, 1)}
        >>> solver = TSPSolver(cities, distance_func=euclidean_distance)
        >>> result = solver.solve()
        >>> print(result.route, result.distance)
    """

    def __init__(
        self,
        cities: Dict[str, Tuple[float, float]],
        distance_func: Callable = haversine_distance,
    ):
        """
        Initialize TSP solver.

        Args:
            cities: Dictionary of {city_name: (lat, lon)}
            distance_func: Function to compute distance between coordinates
        """
        self.problem = TSPProblem(cities, distance_func)

    def _create_objective(self) -> Callable[[List[int]], float]:
        """Create objective function for the optimizer."""
        def objective(route: List[int]) -> float:
            return self.problem.route_length_from_indices(route)
        return objective

    def solve(
        self,
        cooling: Optional[CoolingSchedule] = None,
        neighborhood: Optional[NeighborhoodFunction] = None,
        max_iterations: int = 100000,
        initial_route: Optional[List[str]] = None,
        start_city: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> TSPResult:
        """
        Solve the TSP using Simulated Annealing.

        Args:
            cooling: Cooling schedule (default: geometric)
            neighborhood: Neighborhood function (default: 2-opt)
            max_iterations: Maximum iterations
            initial_route: Starting route (default: random)
            start_city: Rotate solution to start with this city
            seed: Random seed
            verbose: Print progress

        Returns:
            TSPResult with optimal route found
        """
        if seed is not None:
            random.seed(seed)
            RandomVariable.set_global_seed(seed)

        # Default cooling schedule - tuned for TSP
        if cooling is None:
            # Start with high temperature for exploration
            initial_distance = self._estimate_initial_distance()
            cooling = CoolingSchedule.geometric(T0=initial_distance * 0.5, alpha=0.9999)

        # Default neighborhood
        if neighborhood is None:
            neighborhood = TwoOptNeighborhood()

        # Initial route (as indices)
        if initial_route is not None:
            route_indices = [self.problem.city_names.index(c) for c in initial_route]
        else:
            route_indices = list(range(self.problem.n_cities))
            random.shuffle(route_indices)

        # Create and run optimizer
        sa = SimulatedAnnealing(
            objective=self._create_objective(),
            initial_solution=route_indices,
            cooling=cooling,
            neighbor=neighborhood,
            minimize=True,
        )

        result = sa.optimize(
            max_iterations=max_iterations,
            track_history=True,
            verbose=verbose,
        )

        # Convert indices back to city names
        best_route = [self.problem.city_names[i] for i in result.best_solution]

        # Rotate to start with specified city
        if start_city and start_city in best_route:
            while best_route[0] != start_city:
                best_route = best_route[1:] + best_route[:1]

        return TSPResult(
            route=best_route,
            distance=result.best_value,
            iterations=result.iterations,
            acceptance_rate=result.acceptance_rate,
            history=result.history,
        )

    def _estimate_initial_distance(self) -> float:
        """Estimate initial route distance for temperature calibration."""
        # Use nearest neighbor heuristic for rough estimate
        nn_route = self.nearest_neighbor_heuristic()
        return self.problem.route_length(nn_route)

    def nearest_neighbor_heuristic(self, start: Optional[str] = None) -> List[str]:
        """
        Construct a route using nearest neighbor heuristic.

        Good for getting an initial upper bound on solution quality.
        """
        if start is None:
            start = self.problem.city_names[0]

        unvisited = set(self.problem.city_names)
        route = [start]
        unvisited.remove(start)

        while unvisited:
            current = route[-1]
            nearest = min(unvisited, key=lambda c: self.problem.distance(current, c))
            route.append(nearest)
            unvisited.remove(nearest)

        return route

    def random_route(self) -> List[str]:
        """Generate a random route."""
        route = list(self.problem.city_names)
        random.shuffle(route)
        return route

    def brute_force(self, max_cities: int = 10) -> TSPResult:
        """
        Solve by brute force (only for small problems).

        Warning: O(n!) complexity - only practical for n <= 10.
        """
        if self.problem.n_cities > max_cities:
            raise ValueError(f"Brute force only for <= {max_cities} cities")

        cities = self.problem.city_names
        first = cities[0]  # Fix first city to reduce permutations
        rest = cities[1:]

        best_route = None
        best_distance = float('inf')

        for perm in itertools.permutations(rest):
            route = [first] + list(perm)
            dist = self.problem.route_length(route)
            if dist < best_distance:
                best_distance = dist
                best_route = route

        return TSPResult(
            route=best_route,
            distance=best_distance,
            iterations=math.factorial(len(rest)),
            acceptance_rate=0.0,
        )


# ============================================================================
# DEMO DATA
# ============================================================================

# 20 largest US cities
US_CITIES = {
    'New York City': (40.72, -74.00),
    'Los Angeles': (34.05, -118.25),
    'Chicago': (41.88, -87.63),
    'Houston': (29.77, -95.38),
    'Phoenix': (33.45, -112.07),
    'Philadelphia': (39.95, -75.17),
    'San Antonio': (29.53, -98.47),
    'Dallas': (32.78, -96.80),
    'San Diego': (32.78, -117.15),
    'San Jose': (37.30, -121.87),
    'Detroit': (42.33, -83.05),
    'San Francisco': (37.78, -122.42),
    'Jacksonville': (30.32, -81.70),
    'Indianapolis': (39.78, -86.15),
    'Austin': (30.27, -97.77),
    'Columbus': (39.98, -82.98),
    'Fort Worth': (32.75, -97.33),
    'Charlotte': (35.23, -80.85),
    'Memphis': (35.12, -89.97),
    'Baltimore': (39.28, -76.62),
}

# Smaller test set
SMALL_CITIES = {
    'A': (0, 0),
    'B': (0, 1),
    'C': (1, 1),
    'D': (1, 0),
    'E': (0.5, 0.5),
}


# ============================================================================
# DEMONSTRATIONS
# ============================================================================

def demo_basic_tsp():
    """Basic TSP demonstration."""
    print("=" * 70)
    print("TRAVELING SALESMAN PROBLEM - BASIC DEMO")
    print("=" * 70)

    print("\n20 Largest US Cities:")
    print("-" * 50)
    for city, (lat, lon) in list(US_CITIES.items())[:5]:
        print(f"  {city}: ({lat:.2f}°N, {abs(lon):.2f}°W)")
    print(f"  ... and {len(US_CITIES) - 5} more")

    solver = TSPSolver(US_CITIES)

    # Nearest neighbor heuristic for comparison
    nn_route = solver.nearest_neighbor_heuristic(start='New York City')
    nn_distance = solver.problem.route_length(nn_route)
    print(f"\nNearest Neighbor Heuristic: {nn_distance:.0f} miles")

    # Random route for comparison
    random_route = solver.random_route()
    random_distance = solver.problem.route_length(random_route)
    print(f"Random Route: {random_distance:.0f} miles")

    # Simulated Annealing
    print("\nRunning Simulated Annealing...")
    result = solver.solve(
        max_iterations=100000,
        start_city='New York City',
        seed=42,
        verbose=True,
    )

    print(f"\n{result}")

    improvement = (nn_distance - result.distance) / nn_distance * 100
    print(f"\nImprovement over nearest neighbor: {improvement:.1f}%")


def demo_neighborhood_comparison():
    """Compare different neighborhood functions."""
    print("\n" + "=" * 70)
    print("NEIGHBORHOOD FUNCTION COMPARISON")
    print("=" * 70)

    solver = TSPSolver(US_CITIES)

    neighborhoods = [
        ("Swap", SwapNeighborhood()),
        ("2-opt", TwoOptNeighborhood()),
        ("Insert", InsertNeighborhood()),
        ("Mixed", MixedNeighborhood()),
    ]

    print(f"\n{'Neighborhood':<15} {'Distance':>12} {'Iterations':>12} {'Accept Rate':>12}")
    print("-" * 55)

    results = []
    for name, neighborhood in neighborhoods:
        result = solver.solve(
            neighborhood=neighborhood,
            max_iterations=50000,
            seed=42,
        )
        results.append((name, result))
        print(f"{name:<15} {result.distance:>12.0f} {result.iterations:>12,} {result.acceptance_rate*100:>11.1f}%")

    best_name, best_result = min(results, key=lambda x: x[1].distance)
    print(f"\nBest: {best_name} with {best_result.distance:.0f} miles")


def demo_cooling_comparison():
    """Compare different cooling schedules."""
    print("\n" + "=" * 70)
    print("COOLING SCHEDULE COMPARISON")
    print("=" * 70)

    solver = TSPSolver(US_CITIES)
    initial_dist = solver._estimate_initial_distance()

    schedules = [
        ("Geometric (α=0.9999)", CoolingSchedule.geometric(T0=initial_dist*0.5, alpha=0.9999)),
        ("Geometric (α=0.999)", CoolingSchedule.geometric(T0=initial_dist*0.5, alpha=0.999)),
        ("Linear", CoolingSchedule.linear(T0=initial_dist*0.5, delta=0.1)),
        ("Fast", CoolingSchedule.fast(T0=initial_dist*0.5)),
    ]

    print(f"\n{'Schedule':<25} {'Distance':>12} {'Iterations':>12}")
    print("-" * 55)

    for name, schedule in schedules:
        result = solver.solve(
            cooling=schedule,
            max_iterations=50000,
            seed=42,
        )
        print(f"{name:<25} {result.distance:>12.0f} {result.iterations:>12,}")


def demo_small_problem():
    """Demonstrate on a small problem with brute force comparison."""
    print("\n" + "=" * 70)
    print("SMALL PROBLEM (with brute force verification)")
    print("=" * 70)

    # Create a small grid of cities
    cities = {
        'A': (0, 0), 'B': (0, 1), 'C': (0, 2),
        'D': (1, 0), 'E': (1, 1), 'F': (1, 2),
        'G': (2, 0), 'H': (2, 1), 'I': (2, 2),
    }

    print(f"\n{len(cities)} cities in a 3x3 grid")

    solver = TSPSolver(cities, distance_func=euclidean_distance)

    # Brute force optimal
    print("\nFinding optimal solution by brute force...")
    optimal = solver.brute_force()
    print(f"Optimal distance: {optimal.distance:.4f}")
    print(f"Optimal route: {' -> '.join(optimal.route)}")

    # Simulated Annealing
    print("\nRunning Simulated Annealing (10 trials)...")
    sa_distances = []
    for i in range(10):
        result = solver.solve(max_iterations=10000, seed=i)
        sa_distances.append(result.distance)

    avg_distance = sum(sa_distances) / len(sa_distances)
    best_distance = min(sa_distances)
    optimal_found = sum(1 for d in sa_distances if abs(d - optimal.distance) < 0.01)

    print(f"SA average distance: {avg_distance:.4f}")
    print(f"SA best distance: {best_distance:.4f}")
    print(f"Times optimal found: {optimal_found}/10")
    print(f"Gap from optimal: {(avg_distance - optimal.distance) / optimal.distance * 100:.1f}%")


def demo_convergence():
    """Show convergence behavior."""
    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS")
    print("=" * 70)

    solver = TSPSolver(US_CITIES)

    result = solver.solve(
        max_iterations=100000,
        seed=42,
    )

    # Sample history at regular intervals
    history = result.history
    n_points = 20
    step = max(1, len(history) // n_points)

    print(f"\nConvergence over {len(history):,} iterations:")
    print(f"\n{'Iteration':>12} {'Best Distance':>15} {'Progress':>30}")
    print("-" * 60)

    initial = history[0]
    final = history[-1]

    for i in range(0, len(history), step):
        dist = history[i]
        progress = (initial - dist) / (initial - final) * 100 if initial != final else 100
        bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))
        print(f"{i:>12,} {dist:>15.0f} {bar} {progress:5.1f}%")

    # Final
    print(f"{len(history)-1:>12,} {final:>15.0f} {'█' * 20} 100.0%")


def main():
    """Run all demonstrations."""
    random.seed(42)
    RandomVariable.set_global_seed(42)

    demo_basic_tsp()
    demo_neighborhood_comparison()
    demo_cooling_comparison()
    demo_small_problem()
    demo_convergence()

    print("\n" + "=" * 70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)
    print("""
    Key findings:

    1. 2-opt neighborhood generally works best for TSP
    2. Slow cooling (α close to 1) finds better solutions
    3. SA consistently improves on nearest neighbor heuristic
    4. For small problems, SA often finds the optimal solution
    """)


if __name__ == "__main__":
    main()

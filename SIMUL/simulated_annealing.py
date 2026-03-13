"""
Optimization Module - Metaheuristic Algorithms

Implements various optimization algorithms using the RandomVariable framework:
- Simulated Annealing
- Parallel Tempering (Replica Exchange)
- Adaptive Simulated Annealing

Features:
- Pluggable cooling schedules
- Configurable neighborhood functions
- Support for discrete and continuous optimization
- Convergence tracking and statistics
- Monte Carlo analysis of solution quality

Usage:
    from optimization import SimulatedAnnealing, CoolingSchedule

    # Define your objective function
    def objective(x):
        return sum(xi**2 for xi in x)  # Minimize sphere function

    # Create optimizer
    sa = SimulatedAnnealing(
        objective=objective,
        initial_solution=[5.0, 5.0, 5.0],
        cooling=CoolingSchedule.geometric(T0=100, alpha=0.95),
        neighbor=NeighborhoodFunction.gaussian(sigma=0.5),
    )

    # Run optimization
    result = sa.optimize(max_iterations=10000)
    print(result)
"""

from __future__ import annotations

import math
import random
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from random_variable import (
    RandomVariable,
    Uniform,
    Normal,
    Exponential,
    Constant,
)

# Type variable for solution representation
T = TypeVar('T')
Solution = Union[list[float], list[int], Any]


# ============================================================================
# COOLING SCHEDULES
# ============================================================================

class CoolingSchedule(ABC):
    """Abstract base class for temperature cooling schedules."""

    @abstractmethod
    def __call__(self, iteration: int) -> float:
        """Return temperature at given iteration."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[float]:
        """Iterate through temperatures."""
        pass

    @staticmethod
    def geometric(T0: float = 100.0, alpha: float = 0.995) -> GeometricCooling:
        """Create geometric (exponential) cooling: T(k) = T0 * alpha^k"""
        return GeometricCooling(T0, alpha)

    @staticmethod
    def linear(T0: float = 100.0, delta: float = 0.01) -> LinearCooling:
        """Create linear cooling: T(k) = T0 - k * delta"""
        return LinearCooling(T0, delta)

    @staticmethod
    def logarithmic(T0: float = 100.0, c: float = 1.0) -> LogarithmicCooling:
        """Create logarithmic cooling: T(k) = T0 / (1 + c * ln(1 + k))"""
        return LogarithmicCooling(T0, c)

    @staticmethod
    def adaptive(T0: float = 100.0, target_accept: float = 0.4) -> AdaptiveCooling:
        """Create adaptive cooling that adjusts based on acceptance rate."""
        return AdaptiveCooling(T0, target_accept)

    @staticmethod
    def fast(T0: float = 100.0) -> FastCooling:
        """Create fast (Cauchy) cooling: T(k) = T0 / (1 + k)"""
        return FastCooling(T0)

    @staticmethod
    def boltzmann(T0: float = 100.0) -> BoltzmannCooling:
        """Create Boltzmann cooling: T(k) = T0 / ln(1 + k)"""
        return BoltzmannCooling(T0)


@dataclass
class GeometricCooling(CoolingSchedule):
    """Geometric (Kirkpatrick) cooling schedule: T(k) = T0 * alpha^k"""

    T0: float = 100.0
    alpha: float = 0.995

    def __call__(self, iteration: int) -> float:
        return self.T0 * (self.alpha ** iteration)

    def __iter__(self) -> Iterator[float]:
        T = self.T0
        while T > 1e-10:
            yield T
            T *= self.alpha

    def __str__(self) -> str:
        return f"Geometric(T0={self.T0}, α={self.alpha})"


@dataclass
class LinearCooling(CoolingSchedule):
    """Linear cooling schedule: T(k) = max(T0 - k * delta, T_min)"""

    T0: float = 100.0
    delta: float = 0.01
    T_min: float = 1e-10

    def __call__(self, iteration: int) -> float:
        return max(self.T0 - iteration * self.delta, self.T_min)

    def __iter__(self) -> Iterator[float]:
        T = self.T0
        while T > self.T_min:
            yield T
            T -= self.delta

    def __str__(self) -> str:
        return f"Linear(T0={self.T0}, δ={self.delta})"


@dataclass
class LogarithmicCooling(CoolingSchedule):
    """Logarithmic cooling: T(k) = T0 / (1 + c * ln(1 + k))"""

    T0: float = 100.0
    c: float = 1.0

    def __call__(self, iteration: int) -> float:
        return self.T0 / (1 + self.c * math.log(1 + iteration))

    def __iter__(self) -> Iterator[float]:
        k = 0
        while True:
            T = self(k)
            if T < 1e-10:
                break
            yield T
            k += 1

    def __str__(self) -> str:
        return f"Logarithmic(T0={self.T0}, c={self.c})"


@dataclass
class FastCooling(CoolingSchedule):
    """Fast (Cauchy) cooling: T(k) = T0 / (1 + k)"""

    T0: float = 100.0

    def __call__(self, iteration: int) -> float:
        return self.T0 / (1 + iteration)

    def __iter__(self) -> Iterator[float]:
        k = 0
        while True:
            T = self(k)
            if T < 1e-10:
                break
            yield T
            k += 1

    def __str__(self) -> str:
        return f"Fast(T0={self.T0})"


@dataclass
class BoltzmannCooling(CoolingSchedule):
    """Boltzmann cooling: T(k) = T0 / ln(2 + k)"""

    T0: float = 100.0

    def __call__(self, iteration: int) -> float:
        return self.T0 / math.log(2 + iteration)

    def __iter__(self) -> Iterator[float]:
        k = 0
        while True:
            T = self(k)
            if T < 1e-10:
                break
            yield T
            k += 1

    def __str__(self) -> str:
        return f"Boltzmann(T0={self.T0})"


@dataclass
class AdaptiveCooling(CoolingSchedule):
    """
    Adaptive cooling that adjusts based on acceptance rate.

    Increases temperature if acceptance too low, decreases if too high.
    """

    T0: float = 100.0
    target_accept: float = 0.4
    _T: float = field(default=100.0, init=False, repr=False)
    _accepts: list[bool] = field(default_factory=list, init=False, repr=False)
    _window: int = 100

    def __post_init__(self):
        self._T = self.T0

    def __call__(self, iteration: int) -> float:
        return self._T

    def __iter__(self) -> Iterator[float]:
        while self._T > 1e-10:
            yield self._T

    def update(self, accepted: bool) -> None:
        """Update temperature based on acceptance."""
        self._accepts.append(accepted)
        if len(self._accepts) >= self._window:
            rate = sum(self._accepts[-self._window:]) / self._window
            if rate > self.target_accept + 0.1:
                self._T *= 0.95  # Cool faster
            elif rate < self.target_accept - 0.1:
                self._T *= 1.05  # Heat up
            else:
                self._T *= 0.99  # Normal cooling

    def __str__(self) -> str:
        return f"Adaptive(T0={self.T0}, target={self.target_accept})"


# ============================================================================
# NEIGHBORHOOD FUNCTIONS
# ============================================================================

class NeighborhoodFunction(ABC):
    """Abstract base class for generating neighbor solutions."""

    @abstractmethod
    def __call__(self, solution: Solution, bounds: Optional[list[tuple[float, float]]] = None) -> Solution:
        """Generate a neighbor of the current solution."""
        pass

    @staticmethod
    def gaussian(sigma: float = 1.0) -> GaussianNeighborhood:
        """Gaussian perturbation for continuous optimization."""
        return GaussianNeighborhood(sigma)

    @staticmethod
    def uniform(delta: float = 1.0) -> UniformNeighborhood:
        """Uniform random perturbation."""
        return UniformNeighborhood(delta)

    @staticmethod
    def adaptive(initial_sigma: float = 1.0) -> AdaptiveNeighborhood:
        """Adaptive neighborhood that shrinks over time."""
        return AdaptiveNeighborhood(initial_sigma)

    @staticmethod
    def discrete(candidates: Optional[list[Any]] = None) -> DiscreteNeighborhood:
        """For discrete optimization problems."""
        return DiscreteNeighborhood(candidates)

    @staticmethod
    def swap() -> SwapNeighborhood:
        """Swap two elements (for permutation problems like TSP)."""
        return SwapNeighborhood()


@dataclass
class GaussianNeighborhood(NeighborhoodFunction):
    """Generate neighbors using Gaussian perturbation."""

    sigma: float = 1.0

    def __call__(
        self,
        solution: list[float],
        bounds: Optional[list[tuple[float, float]]] = None
    ) -> list[float]:
        noise = Normal(_mean=0, _std=self.sigma)
        new_solution = [x + noise() for x in solution]

        # Apply bounds if provided
        if bounds:
            new_solution = [
                max(lo, min(hi, x))
                for x, (lo, hi) in zip(new_solution, bounds)
            ]

        return new_solution

    def __str__(self) -> str:
        return f"Gaussian(σ={self.sigma})"


@dataclass
class UniformNeighborhood(NeighborhoodFunction):
    """Generate neighbors using uniform random perturbation."""

    delta: float = 1.0

    def __call__(
        self,
        solution: list[float],
        bounds: Optional[list[tuple[float, float]]] = None
    ) -> list[float]:
        noise = Uniform(-self.delta, self.delta)
        new_solution = [x + noise() for x in solution]

        if bounds:
            new_solution = [
                max(lo, min(hi, x))
                for x, (lo, hi) in zip(new_solution, bounds)
            ]

        return new_solution

    def __str__(self) -> str:
        return f"Uniform(δ={self.delta})"


@dataclass
class AdaptiveNeighborhood(NeighborhoodFunction):
    """Neighborhood that adapts (shrinks) based on temperature."""

    initial_sigma: float = 1.0
    _current_sigma: float = field(default=1.0, init=False)

    def __post_init__(self):
        self._current_sigma = self.initial_sigma

    def __call__(
        self,
        solution: list[float],
        bounds: Optional[list[tuple[float, float]]] = None
    ) -> list[float]:
        noise = Normal(_mean=0, _std=self._current_sigma)
        new_solution = [x + noise() for x in solution]

        if bounds:
            new_solution = [
                max(lo, min(hi, x))
                for x, (lo, hi) in zip(new_solution, bounds)
            ]

        return new_solution

    def update(self, T: float, T0: float) -> None:
        """Update sigma based on temperature."""
        self._current_sigma = self.initial_sigma * (T / T0)

    def __str__(self) -> str:
        return f"Adaptive(σ0={self.initial_sigma})"


@dataclass
class DiscreteNeighborhood(NeighborhoodFunction):
    """For discrete optimization - pick a random valid neighbor."""

    candidates: Optional[list[Any]] = None

    def __call__(
        self,
        solution: Any,
        bounds: Optional[list[tuple[float, float]]] = None
    ) -> Any:
        if self.candidates is not None:
            return random.choice(self.candidates)

        # For list/sequence solutions, modify one element
        if isinstance(solution, (list, tuple)):
            idx = random.randrange(len(solution))
            new_solution = list(solution)
            if bounds and idx < len(bounds):
                lo, hi = bounds[idx]
                new_solution[idx] = random.randint(int(lo), int(hi))
            else:
                # Small perturbation
                new_solution[idx] += random.choice([-1, 1])
            return new_solution

        return solution

    def __str__(self) -> str:
        return "Discrete"


@dataclass
class SwapNeighborhood(NeighborhoodFunction):
    """Swap two random elements (for permutation problems)."""

    def __call__(
        self,
        solution: list[Any],
        bounds: Optional[list[tuple[float, float]]] = None
    ) -> list[Any]:
        if len(solution) < 2:
            return solution

        new_solution = list(solution)
        i, j = random.sample(range(len(solution)), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        return new_solution

    def __str__(self) -> str:
        return "Swap"


# ============================================================================
# OPTIMIZATION RESULTS
# ============================================================================

@dataclass
class OptimizationResult:
    """Results from an optimization run."""

    best_solution: Solution
    best_value: float
    final_solution: Solution
    final_value: float
    initial_solution: Solution
    initial_value: float
    iterations: int
    acceptances: int
    improvements: int
    history: list[float] = field(default_factory=list, repr=False)
    temperature_history: list[float] = field(default_factory=list, repr=False)

    @property
    def acceptance_rate(self) -> float:
        return self.acceptances / self.iterations if self.iterations > 0 else 0.0

    @property
    def improvement_rate(self) -> float:
        return self.improvements / self.iterations if self.iterations > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"OptimizationResult:\n"
            f"  Best Value: {self.best_value:.6f}\n"
            f"  Best Solution: {self.best_solution}\n"
            f"  Iterations: {self.iterations:,}\n"
            f"  Acceptance Rate: {self.acceptance_rate*100:.1f}%\n"
            f"  Improvement Rate: {self.improvement_rate*100:.1f}%"
        )


# ============================================================================
# SIMULATED ANNEALING
# ============================================================================

class SimulatedAnnealing:
    """
    Simulated Annealing optimization algorithm.

    A probabilistic metaheuristic for global optimization that mimics
    the physical process of annealing in metallurgy.

    Example:
        >>> def sphere(x):
        ...     return sum(xi**2 for xi in x)
        >>> sa = SimulatedAnnealing(
        ...     objective=sphere,
        ...     initial_solution=[5.0, 5.0, 5.0],
        ...     cooling=CoolingSchedule.geometric(T0=100, alpha=0.995),
        ...     neighbor=NeighborhoodFunction.gaussian(sigma=0.5),
        ... )
        >>> result = sa.optimize(max_iterations=10000)
    """

    def __init__(
        self,
        objective: Callable[[Solution], float],
        initial_solution: Optional[Solution] = None,
        cooling: Optional[CoolingSchedule] = None,
        neighbor: Optional[NeighborhoodFunction] = None,
        bounds: Optional[list[tuple[float, float]]] = None,
        minimize: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize Simulated Annealing optimizer.

        Args:
            objective: Function to optimize (minimize or maximize)
            initial_solution: Starting point (if None, requires bounds)
            cooling: Temperature cooling schedule
            neighbor: Neighborhood function for generating candidates
            bounds: Variable bounds [(min, max), ...] for each dimension
            minimize: True to minimize, False to maximize
            seed: Random seed for reproducibility
        """
        self.objective = objective
        self.bounds = bounds
        self.minimize = minimize

        # Set up cooling schedule
        self.cooling = cooling or CoolingSchedule.geometric()

        # Set up neighborhood function
        self.neighbor = neighbor or NeighborhoodFunction.gaussian()

        # Set random seed
        if seed is not None:
            random.seed(seed)
            RandomVariable.set_global_seed(seed)

        # Initialize solution
        if initial_solution is not None:
            self.initial_solution = list(initial_solution)
        elif bounds is not None:
            # Random initialization within bounds
            self.initial_solution = [
                random.uniform(lo, hi) for lo, hi in bounds
            ]
        else:
            raise ValueError("Must provide initial_solution or bounds")

    def _accept_probability(self, delta: float, T: float) -> float:
        """Calculate probability of accepting a worse solution."""
        if delta <= 0:  # Better solution (for minimization)
            return 1.0
        if T <= 0:
            return 0.0
        return math.exp(-delta / T)

    def optimize(
        self,
        max_iterations: int = 10000,
        min_temperature: float = 1e-8,
        track_history: bool = False,
        verbose: bool = False,
        callback: Optional[Callable[[int, Solution, float, float], None]] = None,
    ) -> OptimizationResult:
        """
        Run the optimization.

        Args:
            max_iterations: Maximum number of iterations
            min_temperature: Stop when temperature falls below this
            track_history: Whether to record value history
            verbose: Print progress updates
            callback: Called each iteration with (iter, solution, value, temp)

        Returns:
            OptimizationResult with best solution found
        """
        # Initialize
        current = list(self.initial_solution)
        current_value = self.objective(current)

        best = list(current)
        best_value = current_value

        history = [current_value] if track_history else []
        temp_history = []

        acceptances = 0
        improvements = 0

        # Get temperature iterator
        T0 = self.cooling.T0 if hasattr(self.cooling, 'T0') else 100.0

        for iteration in range(max_iterations):
            T = self.cooling(iteration)

            if T < min_temperature:
                break

            # Generate neighbor
            if isinstance(self.neighbor, AdaptiveNeighborhood):
                self.neighbor.update(T, T0)

            candidate = self.neighbor(current, self.bounds)
            candidate_value = self.objective(candidate)

            # Calculate delta (positive = worse for minimization)
            if self.minimize:
                delta = candidate_value - current_value
            else:
                delta = current_value - candidate_value

            # Accept or reject
            accept_prob = self._accept_probability(delta, T)
            accepted = random.random() < accept_prob

            if accepted:
                current = candidate
                current_value = candidate_value
                acceptances += 1

                # Check if this is the best so far
                is_better = (
                    (self.minimize and current_value < best_value) or
                    (not self.minimize and current_value > best_value)
                )
                if is_better:
                    best = list(current)
                    best_value = current_value
                    improvements += 1

            # Update adaptive cooling if used
            if isinstance(self.cooling, AdaptiveCooling):
                self.cooling.update(accepted)

            # Track history
            if track_history:
                history.append(best_value)
                temp_history.append(T)

            # Callback
            if callback:
                callback(iteration, current, current_value, T)

            # Verbose output
            if verbose and iteration % (max_iterations // 10) == 0:
                print(f"Iter {iteration:6d}: T={T:.4f}, best={best_value:.6f}")

        return OptimizationResult(
            best_solution=best,
            best_value=best_value,
            final_solution=current,
            final_value=current_value,
            initial_solution=self.initial_solution,
            initial_value=self.objective(self.initial_solution),
            iterations=iteration + 1,
            acceptances=acceptances,
            improvements=improvements,
            history=history,
            temperature_history=temp_history,
        )

    def multi_start(
        self,
        n_starts: int = 10,
        max_iterations: int = 10000,
        **kwargs
    ) -> list[OptimizationResult]:
        """
        Run multiple optimization attempts from different starting points.

        Returns list of results sorted by best value.
        """
        results = []

        for i in range(n_starts):
            # Generate new random starting point
            if self.bounds:
                self.initial_solution = [
                    random.uniform(lo, hi) for lo, hi in self.bounds
                ]

            result = self.optimize(max_iterations=max_iterations, **kwargs)
            results.append(result)

        # Sort by best value
        results.sort(key=lambda r: r.best_value, reverse=not self.minimize)
        return results


# ============================================================================
# MONTE CARLO ANALYSIS OF SA
# ============================================================================

@dataclass
class SAMonteCarloResult:
    """Results from Monte Carlo analysis of Simulated Annealing."""

    n_runs: int
    best_values: list[float]
    mean_best: float
    std_best: float
    min_best: float
    max_best: float
    success_rate: float  # Proportion finding near-optimal solution
    mean_iterations: float

    def __str__(self) -> str:
        return (
            f"SA Monte Carlo Analysis ({self.n_runs} runs):\n"
            f"  Best Value: {self.mean_best:.6f} ± {self.std_best:.6f}\n"
            f"  Range: [{self.min_best:.6f}, {self.max_best:.6f}]\n"
            f"  Success Rate: {self.success_rate*100:.1f}%\n"
            f"  Mean Iterations: {self.mean_iterations:.0f}"
        )


def analyze_sa_performance(
    objective: Callable[[Solution], float],
    initial_solution: Solution,
    n_runs: int = 100,
    optimal_value: Optional[float] = None,
    tolerance: float = 0.01,
    **sa_kwargs
) -> SAMonteCarloResult:
    """
    Perform Monte Carlo analysis of SA performance.

    Args:
        objective: Objective function
        initial_solution: Starting point template
        n_runs: Number of independent runs
        optimal_value: Known optimal (for success rate calculation)
        tolerance: How close to optimal counts as success
        **sa_kwargs: Arguments passed to SimulatedAnnealing

    Returns:
        SAMonteCarloResult with statistics
    """
    best_values = []
    iterations = []

    for i in range(n_runs):
        sa = SimulatedAnnealing(
            objective=objective,
            initial_solution=initial_solution,
            seed=i * 12345,
            **sa_kwargs
        )
        result = sa.optimize()
        best_values.append(result.best_value)
        iterations.append(result.iterations)

    mean_best = statistics.mean(best_values)
    std_best = statistics.stdev(best_values) if len(best_values) > 1 else 0.0

    # Calculate success rate if optimal is known
    if optimal_value is not None:
        successes = sum(
            1 for v in best_values
            if abs(v - optimal_value) <= tolerance * abs(optimal_value + 1e-10)
        )
        success_rate = successes / n_runs
    else:
        success_rate = 0.0

    return SAMonteCarloResult(
        n_runs=n_runs,
        best_values=best_values,
        mean_best=mean_best,
        std_best=std_best,
        min_best=min(best_values),
        max_best=max(best_values),
        success_rate=success_rate,
        mean_iterations=statistics.mean(iterations),
    )


# ============================================================================
# CLASSIC TEST FUNCTIONS
# ============================================================================

class TestFunctions:
    """Collection of classic optimization test functions."""

    @staticmethod
    def sphere(x: list[float]) -> float:
        """Sphere function: f(x) = Σ xi². Minimum at origin = 0."""
        return sum(xi**2 for xi in x)

    @staticmethod
    def rastrigin(x: list[float]) -> float:
        """Rastrigin function: highly multimodal. Minimum at origin = 0."""
        n = len(x)
        return 10*n + sum(xi**2 - 10*math.cos(2*math.pi*xi) for xi in x)

    @staticmethod
    def rosenbrock(x: list[float]) -> float:
        """Rosenbrock function: banana-shaped valley. Minimum at (1,1,...) = 0."""
        return sum(
            100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2
            for i in range(len(x) - 1)
        )

    @staticmethod
    def ackley(x: list[float]) -> float:
        """Ackley function: many local minima. Minimum at origin = 0."""
        n = len(x)
        sum1 = sum(xi**2 for xi in x)
        sum2 = sum(math.cos(2*math.pi*xi) for xi in x)
        return -20*math.exp(-0.2*math.sqrt(sum1/n)) - math.exp(sum2/n) + 20 + math.e

    @staticmethod
    def schwefel(x: list[float]) -> float:
        """Schwefel function: deceptive global minimum far from local minima."""
        n = len(x)
        return 418.9829*n - sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)

    @staticmethod
    def griewank(x: list[float]) -> float:
        """Griewank function: many local minima. Minimum at origin = 0."""
        sum_sq = sum(xi**2 for xi in x) / 4000
        prod_cos = math.prod(
            math.cos(xi / math.sqrt(i+1))
            for i, xi in enumerate(x)
        )
        return sum_sq - prod_cos + 1

    @staticmethod
    def sine_wave(x: int, n_points: int = 100000) -> float:
        """1D sine wave with noise (as in original example)."""
        return math.sin((2 * math.pi / n_points) * x) + 0.001 * random.random()


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_simulated_annealing():
    """Demonstrate Simulated Annealing on various test functions."""
    print("=" * 70)
    print("SIMULATED ANNEALING DEMONSTRATION")
    print("=" * 70)

    # Example 1: Sphere function (easy)
    print("\n1. SPHERE FUNCTION (minimize Σ xi²)")
    print("-" * 50)

    sa = SimulatedAnnealing(
        objective=TestFunctions.sphere,
        bounds=[(-5, 5)] * 5,  # 5 dimensions
        cooling=CoolingSchedule.geometric(T0=100, alpha=0.995),
        neighbor=NeighborhoodFunction.gaussian(sigma=0.5),
        seed=42
    )

    result = sa.optimize(max_iterations=10000, verbose=True)
    print(f"\n{result}")
    print(f"  (optimal is 0.0 at origin)")

    # Example 2: Rastrigin function (hard - many local minima)
    print("\n\n2. RASTRIGIN FUNCTION (highly multimodal)")
    print("-" * 50)

    sa = SimulatedAnnealing(
        objective=TestFunctions.rastrigin,
        bounds=[(-5.12, 5.12)] * 3,  # 3 dimensions
        cooling=CoolingSchedule.geometric(T0=500, alpha=0.997),
        neighbor=NeighborhoodFunction.gaussian(sigma=0.3),
        seed=42
    )

    result = sa.optimize(max_iterations=20000, verbose=True)
    print(f"\n{result}")
    print(f"  (optimal is 0.0 at origin)")

    # Example 3: Rosenbrock function (tricky valley)
    print("\n\n3. ROSENBROCK FUNCTION (banana valley)")
    print("-" * 50)

    sa = SimulatedAnnealing(
        objective=TestFunctions.rosenbrock,
        bounds=[(-5, 5)] * 2,
        cooling=CoolingSchedule.geometric(T0=1000, alpha=0.998),
        neighbor=NeighborhoodFunction.gaussian(sigma=0.2),
        seed=42
    )

    result = sa.optimize(max_iterations=30000, verbose=True)
    print(f"\n{result}")
    print(f"  (optimal is 0.0 at (1,1))")

    # Example 4: Compare cooling schedules
    print("\n\n4. COOLING SCHEDULE COMPARISON (Sphere, 3D)")
    print("-" * 50)

    schedules = [
        ("Geometric", CoolingSchedule.geometric(T0=100, alpha=0.995)),
        ("Linear", CoolingSchedule.linear(T0=100, delta=0.01)),
        ("Fast", CoolingSchedule.fast(T0=100)),
        ("Logarithmic", CoolingSchedule.logarithmic(T0=100, c=1.0)),
    ]

    for name, schedule in schedules:
        sa = SimulatedAnnealing(
            objective=TestFunctions.sphere,
            bounds=[(-5, 5)] * 3,
            cooling=schedule,
            neighbor=NeighborhoodFunction.gaussian(sigma=0.3),
            seed=42
        )
        result = sa.optimize(max_iterations=5000)
        print(f"  {name:15s}: best = {result.best_value:.6f}, "
              f"accept = {result.acceptance_rate*100:.1f}%")

    # Example 5: Discrete optimization (original sine wave problem)
    print("\n\n5. DISCRETE OPTIMIZATION (Sine Wave)")
    print("-" * 50)

    LIMIT = 100000
    # Pre-compute function values
    A = [TestFunctions.sine_wave(i, LIMIT) for i in range(LIMIT)]

    # Find actual global minimum for comparison
    global_min_idx = min(range(LIMIT), key=lambda i: A[i])
    print(f"  True global minimum: @{global_min_idx} = {A[global_min_idx]:.4f}")

    def discrete_objective(x: list[int]) -> float:
        return A[x[0]]

    sa = SimulatedAnnealing(
        objective=discrete_objective,
        initial_solution=[random.randrange(LIMIT)],
        bounds=[(0, LIMIT-1)],
        cooling=CoolingSchedule.geometric(T0=1.0, alpha=0.9999),
        neighbor=NeighborhoodFunction.discrete(),
        seed=42
    )

    result = sa.optimize(max_iterations=50000)
    print(f"  SA found minimum: @{result.best_solution[0]} = {result.best_value:.4f}")
    print(f"  Iterations: {result.iterations:,}")

    # Example 6: Multi-start optimization
    print("\n\n6. MULTI-START OPTIMIZATION (Rastrigin, 10 starts)")
    print("-" * 50)

    sa = SimulatedAnnealing(
        objective=TestFunctions.rastrigin,
        bounds=[(-5.12, 5.12)] * 2,
        cooling=CoolingSchedule.geometric(T0=100, alpha=0.995),
        neighbor=NeighborhoodFunction.gaussian(sigma=0.3),
    )

    results = sa.multi_start(n_starts=10, max_iterations=5000)

    print(f"  Best of 10 runs: {results[0].best_value:.6f}")
    print(f"  Worst of 10 runs: {results[-1].best_value:.6f}")
    print(f"  All results: {[f'{r.best_value:.4f}' for r in results]}")


def demo_monte_carlo_analysis():
    """Demonstrate Monte Carlo analysis of SA performance."""
    print("\n\n" + "=" * 70)
    print("MONTE CARLO ANALYSIS OF SA PERFORMANCE")
    print("=" * 70)

    print("\nAnalyzing SA on Sphere function (50 runs)...")

    mc_result = analyze_sa_performance(
        objective=TestFunctions.sphere,
        initial_solution=[3.0, 3.0, 3.0],
        n_runs=50,
        optimal_value=0.0,
        tolerance=0.01,
        cooling=CoolingSchedule.geometric(T0=100, alpha=0.995),
        neighbor=NeighborhoodFunction.gaussian(sigma=0.3),
    )

    print(f"\n{mc_result}")

    print("\nAnalyzing SA on Rastrigin function (50 runs)...")

    mc_result = analyze_sa_performance(
        objective=TestFunctions.rastrigin,
        initial_solution=[3.0, 3.0],
        n_runs=50,
        optimal_value=0.0,
        tolerance=0.1,
        cooling=CoolingSchedule.geometric(T0=200, alpha=0.997),
        neighbor=NeighborhoodFunction.gaussian(sigma=0.3),
    )

    print(f"\n{mc_result}")


def main():
    """Run all demonstrations."""
    RandomVariable.set_global_seed(42)
    demo_simulated_annealing()
    demo_monte_carlo_analysis()
    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

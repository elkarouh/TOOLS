"""
Bayesian Optimization Module

Implements Bayesian Optimization using Gaussian Process surrogate models
for sample-efficient optimization of expensive black-box functions.

Features:
- Gaussian Process regression with multiple kernels
- Acquisition functions: Expected Improvement, UCB, Probability of Improvement
- Support for noisy observations
- Comparison utilities with other optimizers (SA, random search)

Usage:
    from bayesian_optimization import BayesianOptimizer, AcquisitionFunction

    def expensive_function(x):
        # Simulates expensive evaluation
        return sum(xi**2 for xi in x)

    bo = BayesianOptimizer(
        objective=expensive_function,
        bounds=[(-5, 5), (-5, 5)],
        acquisition=AcquisitionFunction.expected_improvement(),
    )

    result = bo.optimize(n_iterations=50)
    print(result)
"""

from __future__ import annotations

import math
import random
import statistics
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

from random_variable import RandomVariable, Uniform, Normal


# ============================================================================
# KERNEL FUNCTIONS
# ============================================================================

class Kernel(ABC):
    """Abstract base class for GP kernels (covariance functions)."""

    @abstractmethod
    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute covariance between two points."""
        pass

    @abstractmethod
    def matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute covariance matrix between sets of points."""
        pass


@dataclass
class RBFKernel(Kernel):
    """
    Radial Basis Function (Squared Exponential) kernel.

    k(x, x') = σ² * exp(-||x - x'||² / (2 * l²))

    Parameters:
        length_scale: Controls smoothness (larger = smoother)
        variance: Output variance (signal variance)
    """

    length_scale: float = 1.0
    variance: float = 1.0

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        dist_sq = np.sum((x1 - x2) ** 2)
        return self.variance * np.exp(-dist_sq / (2 * self.length_scale ** 2))

    def matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1

        # Compute pairwise squared distances
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        dist_sq = X1_sq + X2_sq.T - 2 * X1 @ X2.T
        dist_sq = np.maximum(dist_sq, 0)  # Numerical stability

        return self.variance * np.exp(-dist_sq / (2 * self.length_scale ** 2))

    def __str__(self) -> str:
        return f"RBF(l={self.length_scale}, σ²={self.variance})"


@dataclass
class Matern52Kernel(Kernel):
    """
    Matérn 5/2 kernel - less smooth than RBF, often more realistic.

    k(x, x') = σ² * (1 + √5*r + 5r²/3) * exp(-√5*r)
    where r = ||x - x'|| / l
    """

    length_scale: float = 1.0
    variance: float = 1.0

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        dist = np.sqrt(np.sum((x1 - x2) ** 2))
        r = dist / self.length_scale
        sqrt5_r = np.sqrt(5) * r
        return self.variance * (1 + sqrt5_r + 5 * r**2 / 3) * np.exp(-sqrt5_r)

    def matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1

        # Compute pairwise distances
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        dist_sq = X1_sq + X2_sq.T - 2 * X1 @ X2.T
        dist = np.sqrt(np.maximum(dist_sq, 0))

        r = dist / self.length_scale
        sqrt5_r = np.sqrt(5) * r
        return self.variance * (1 + sqrt5_r + 5 * r**2 / 3) * np.exp(-sqrt5_r)

    def __str__(self) -> str:
        return f"Matérn5/2(l={self.length_scale}, σ²={self.variance})"


@dataclass
class Matern32Kernel(Kernel):
    """
    Matérn 3/2 kernel - rougher than 5/2.

    k(x, x') = σ² * (1 + √3*r) * exp(-√3*r)
    """

    length_scale: float = 1.0
    variance: float = 1.0

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> float:
        dist = np.sqrt(np.sum((x1 - x2) ** 2))
        r = dist / self.length_scale
        sqrt3_r = np.sqrt(3) * r
        return self.variance * (1 + sqrt3_r) * np.exp(-sqrt3_r)

    def matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        if X2 is None:
            X2 = X1

        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        dist_sq = X1_sq + X2_sq.T - 2 * X1 @ X2.T
        dist = np.sqrt(np.maximum(dist_sq, 0))

        r = dist / self.length_scale
        sqrt3_r = np.sqrt(3) * r
        return self.variance * (1 + sqrt3_r) * np.exp(-sqrt3_r)

    def __str__(self) -> str:
        return f"Matérn3/2(l={self.length_scale}, σ²={self.variance})"


# ============================================================================
# GAUSSIAN PROCESS
# ============================================================================

class GaussianProcess:
    """
    Gaussian Process regression model.

    Provides predictions with uncertainty estimates for Bayesian optimization.
    """

    def __init__(
        self,
        kernel: Optional[Kernel] = None,
        noise_variance: float = 1e-6,
        normalize_y: bool = True,
    ):
        """
        Initialize Gaussian Process.

        Args:
            kernel: Covariance function (default: RBF)
            noise_variance: Observation noise variance (σ_n²)
            normalize_y: Whether to normalize target values
        """
        self.kernel = kernel or RBFKernel()
        self.noise_variance = noise_variance
        self.normalize_y = normalize_y

        # Fitted parameters
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianProcess:
        """
        Fit the GP to training data.

        Args:
            X: Training inputs, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)

        Returns:
            self
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y).flatten()

        self.X_train = X

        # Normalize y
        if self.normalize_y:
            self.y_mean = np.mean(y)
            self.y_std = np.std(y) + 1e-8
            y_normalized = (y - self.y_mean) / self.y_std
        else:
            self.y_mean = 0.0
            self.y_std = 1.0
            y_normalized = y

        self.y_train = y_normalized

        # Compute kernel matrix with noise
        K = self.kernel.matrix(X) + self.noise_variance * np.eye(len(X))

        # Compute inverse (using Cholesky for numerical stability)
        try:
            L = np.linalg.cholesky(K)
            self.K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(X))))
            self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_normalized))
        except np.linalg.LinAlgError:
            # Fall back to direct inverse with more regularization
            K += 1e-4 * np.eye(len(X))
            self.K_inv = np.linalg.inv(K)
            self.alpha = self.K_inv @ y_normalized

        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions at new points.

        Args:
            X: Query points, shape (n_queries, n_features)
            return_std: Whether to return standard deviation

        Returns:
            mean: Predicted means
            std: Predicted standard deviations (if return_std=True)
        """
        X = np.atleast_2d(X)

        if self.X_train is None:
            raise ValueError("GP must be fitted before predicting")

        # Compute cross-covariance
        K_star = self.kernel.matrix(X, self.X_train)

        # Predictive mean
        mean = K_star @ self.alpha
        mean = mean * self.y_std + self.y_mean  # Denormalize

        if not return_std:
            return mean

        # Predictive variance
        K_star_star = self.kernel.matrix(X)
        var = np.diag(K_star_star - K_star @ self.K_inv @ K_star.T)
        var = np.maximum(var, 1e-10)  # Numerical stability
        std = np.sqrt(var) * self.y_std  # Denormalize

        return mean, std

    def sample(self, X: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """
        Draw samples from the GP posterior.

        Args:
            X: Query points
            n_samples: Number of samples to draw

        Returns:
            Samples, shape (n_samples, n_queries)
        """
        X = np.atleast_2d(X)
        mean, std = self.predict(X, return_std=True)

        # For simplicity, draw independent samples (ignoring correlations)
        samples = np.random.randn(n_samples, len(X)) * std + mean
        return samples


# ============================================================================
# ACQUISITION FUNCTIONS
# ============================================================================

class AcquisitionFunction(ABC):
    """Abstract base class for acquisition functions."""

    @abstractmethod
    def __call__(
        self,
        X: np.ndarray,
        gp: GaussianProcess,
        y_best: float,
        minimize: bool = True
    ) -> np.ndarray:
        """
        Compute acquisition function values.

        Args:
            X: Query points
            gp: Fitted Gaussian Process
            y_best: Best observed value so far
            minimize: Whether we're minimizing

        Returns:
            Acquisition values (higher = more promising)
        """
        pass

    @staticmethod
    def expected_improvement(xi: float = 0.01) -> ExpectedImprovement:
        """Expected Improvement acquisition function."""
        return ExpectedImprovement(xi)

    @staticmethod
    def upper_confidence_bound(beta: float = 2.0) -> UpperConfidenceBound:
        """Upper Confidence Bound (GP-UCB) acquisition function."""
        return UpperConfidenceBound(beta)

    @staticmethod
    def probability_improvement(xi: float = 0.01) -> ProbabilityOfImprovement:
        """Probability of Improvement acquisition function."""
        return ProbabilityOfImprovement(xi)

    @staticmethod
    def thompson_sampling() -> ThompsonSampling:
        """Thompson Sampling acquisition function."""
        return ThompsonSampling()


@dataclass
class ExpectedImprovement(AcquisitionFunction):
    """
    Expected Improvement (EI) acquisition function.

    EI(x) = E[max(f_best - f(x), 0)]

    Balances exploitation (mean) and exploration (uncertainty).
    """

    xi: float = 0.01  # Exploration-exploitation trade-off

    def __call__(
        self,
        X: np.ndarray,
        gp: GaussianProcess,
        y_best: float,
        minimize: bool = True
    ) -> np.ndarray:
        X = np.atleast_2d(X)
        mean, std = gp.predict(X, return_std=True)

        # For minimization, we want improvement below y_best
        if minimize:
            improvement = y_best - mean - self.xi
        else:
            improvement = mean - y_best - self.xi

        # Compute EI using the analytical formula
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = improvement / std
            ei = improvement * self._norm_cdf(Z) + std * self._norm_pdf(Z)
            ei = np.where(std > 1e-10, ei, 0.0)

        return ei

    @staticmethod
    def _norm_cdf(x: np.ndarray) -> np.ndarray:
        """Standard normal CDF."""
        return 0.5 * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))

    @staticmethod
    def _norm_pdf(x: np.ndarray) -> np.ndarray:
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def __str__(self) -> str:
        return f"EI(ξ={self.xi})"


@dataclass
class UpperConfidenceBound(AcquisitionFunction):
    """
    Upper Confidence Bound (GP-UCB) acquisition function.

    UCB(x) = μ(x) + β * σ(x)  (for maximization)
    LCB(x) = μ(x) - β * σ(x)  (for minimization)

    Higher beta = more exploration.
    """

    beta: float = 2.0  # Exploration parameter

    def __call__(
        self,
        X: np.ndarray,
        gp: GaussianProcess,
        y_best: float,
        minimize: bool = True
    ) -> np.ndarray:
        X = np.atleast_2d(X)
        mean, std = gp.predict(X, return_std=True)

        if minimize:
            # Lower confidence bound (we want to minimize)
            # Return negative so higher = better for acquisition optimization
            return -(mean - self.beta * std)
        else:
            return mean + self.beta * std

    def __str__(self) -> str:
        return f"UCB(β={self.beta})"


@dataclass
class ProbabilityOfImprovement(AcquisitionFunction):
    """
    Probability of Improvement (PI) acquisition function.

    PI(x) = P(f(x) < f_best - xi)  (for minimization)

    Purely exploitative compared to EI.
    """

    xi: float = 0.01

    def __call__(
        self,
        X: np.ndarray,
        gp: GaussianProcess,
        y_best: float,
        minimize: bool = True
    ) -> np.ndarray:
        X = np.atleast_2d(X)
        mean, std = gp.predict(X, return_std=True)

        if minimize:
            improvement = y_best - mean - self.xi
        else:
            improvement = mean - y_best - self.xi

        with np.errstate(divide='ignore', invalid='ignore'):
            Z = improvement / std
            pi = 0.5 * (1 + np.vectorize(math.erf)(Z / np.sqrt(2)))
            pi = np.where(std > 1e-10, pi, 0.0)

        return pi

    def __str__(self) -> str:
        return f"PI(ξ={self.xi})"


@dataclass
class ThompsonSampling(AcquisitionFunction):
    """
    Thompson Sampling - samples from the posterior and returns that sample.

    Natural exploration through randomization.
    """

    def __call__(
        self,
        X: np.ndarray,
        gp: GaussianProcess,
        y_best: float,
        minimize: bool = True
    ) -> np.ndarray:
        X = np.atleast_2d(X)
        mean, std = gp.predict(X, return_std=True)

        # Sample from posterior
        samples = mean + std * np.random.randn(len(X))

        if minimize:
            return -samples  # Negate so higher = better
        return samples

    def __str__(self) -> str:
        return "ThompsonSampling"


# ============================================================================
# BAYESIAN OPTIMIZER
# ============================================================================

@dataclass
class BOResult:
    """Results from Bayesian Optimization."""

    best_solution: np.ndarray
    best_value: float
    X_observed: np.ndarray
    y_observed: np.ndarray
    n_iterations: int
    n_initial: int
    history: list[float] = field(default_factory=list)

    @property
    def total_evaluations(self) -> int:
        return len(self.y_observed)

    def __str__(self) -> str:
        return (
            f"Bayesian Optimization Result:\n"
            f"  Best Value: {self.best_value:.6f}\n"
            f"  Best Solution: {self.best_solution}\n"
            f"  Total Evaluations: {self.total_evaluations}\n"
            f"  Initial Samples: {self.n_initial}\n"
            f"  BO Iterations: {self.n_iterations}"
        )


class BayesianOptimizer:
    """
    Bayesian Optimization for expensive black-box functions.

    Uses Gaussian Process surrogate model and acquisition function
    to efficiently find optimal solutions with few function evaluations.

    Example:
        >>> def expensive_func(x):
        ...     return (x[0] - 2)**2 + (x[1] + 1)**2
        >>>
        >>> bo = BayesianOptimizer(
        ...     objective=expensive_func,
        ...     bounds=[(-5, 5), (-5, 5)],
        ... )
        >>> result = bo.optimize(n_iterations=30)
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        acquisition: Optional[AcquisitionFunction] = None,
        kernel: Optional[Kernel] = None,
        noise_variance: float = 1e-6,
        minimize: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize Bayesian Optimizer.

        Args:
            objective: Function to optimize
            bounds: Variable bounds [(min, max), ...]
            acquisition: Acquisition function (default: EI)
            kernel: GP kernel (default: Matérn 5/2)
            noise_variance: Observation noise
            minimize: True to minimize, False to maximize
            seed: Random seed
        """
        self.objective = objective
        self.bounds = np.array(bounds)
        self.n_dims = len(bounds)
        self.acquisition = acquisition or ExpectedImprovement()
        self.kernel = kernel or Matern52Kernel()
        self.noise_variance = noise_variance
        self.minimize = minimize

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            RandomVariable.set_global_seed(seed)

        # Storage for observations
        self.X_observed: list[np.ndarray] = []
        self.y_observed: list[float] = []

        # GP model
        self.gp = GaussianProcess(
            kernel=self.kernel,
            noise_variance=self.noise_variance,
        )

    def _sample_initial(self, n_samples: int) -> np.ndarray:
        """Generate initial samples using Latin Hypercube Sampling."""
        # Simple LHS implementation
        samples = np.zeros((n_samples, self.n_dims))

        for i in range(self.n_dims):
            lo, hi = self.bounds[i]
            # Create evenly spaced intervals and sample one point per interval
            intervals = np.linspace(lo, hi, n_samples + 1)
            points = np.array([
                np.random.uniform(intervals[j], intervals[j+1])
                for j in range(n_samples)
            ])
            np.random.shuffle(points)
            samples[:, i] = points

        return samples

    def _optimize_acquisition(
        self,
        n_candidates: int = 1000,
        n_local: int = 5,
    ) -> np.ndarray:
        """
        Find the point that maximizes the acquisition function.

        Uses random sampling + local refinement.
        """
        # Get current best
        if self.minimize:
            y_best = min(self.y_observed)
        else:
            y_best = max(self.y_observed)

        # Random candidate sampling
        candidates = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(n_candidates, self.n_dims)
        )

        # Evaluate acquisition function
        acq_values = self.acquisition(candidates, self.gp, y_best, self.minimize)

        # Get top candidates for local search
        top_indices = np.argsort(acq_values)[-n_local:]

        best_x = candidates[top_indices[-1]]
        best_acq = acq_values[top_indices[-1]]

        # Simple local refinement around best candidates
        for idx in top_indices:
            x_center = candidates[idx]

            # Local random search
            for _ in range(20):
                # Small perturbation
                delta = np.random.randn(self.n_dims) * 0.1 * (self.bounds[:, 1] - self.bounds[:, 0])
                x_new = np.clip(
                    x_center + delta,
                    self.bounds[:, 0],
                    self.bounds[:, 1]
                )

                acq_new = self.acquisition(
                    x_new.reshape(1, -1), self.gp, y_best, self.minimize
                )[0]

                if acq_new > best_acq:
                    best_x = x_new
                    best_acq = acq_new

        return best_x

    def optimize(
        self,
        n_iterations: int = 50,
        n_initial: int = 5,
        verbose: bool = False,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
    ) -> BOResult:
        """
        Run Bayesian Optimization.

        Args:
            n_iterations: Number of BO iterations (after initial sampling)
            n_initial: Number of initial random samples
            verbose: Print progress
            callback: Called each iteration with (iter, x, y)

        Returns:
            BOResult with optimization results
        """
        # Initial sampling
        if verbose:
            print(f"Generating {n_initial} initial samples...")

        X_init = self._sample_initial(n_initial)

        for x in X_init:
            y = self.objective(x)
            self.X_observed.append(x)
            self.y_observed.append(y)

        history = [min(self.y_observed) if self.minimize else max(self.y_observed)]

        if verbose:
            print(f"Initial best: {history[0]:.6f}")

        # BO iterations
        for i in range(n_iterations):
            # Fit GP
            X_train = np.array(self.X_observed)
            y_train = np.array(self.y_observed)
            self.gp.fit(X_train, y_train)

            # Find next point
            x_next = self._optimize_acquisition()

            # Evaluate objective
            y_next = self.objective(x_next)

            # Store observation
            self.X_observed.append(x_next)
            self.y_observed.append(y_next)

            # Track best
            if self.minimize:
                current_best = min(self.y_observed)
            else:
                current_best = max(self.y_observed)
            history.append(current_best)

            if callback:
                callback(i, x_next, y_next)

            if verbose and (i + 1) % max(1, n_iterations // 10) == 0:
                print(f"Iter {i+1:4d}: y={y_next:.6f}, best={current_best:.6f}")

        # Get best solution
        y_array = np.array(self.y_observed)
        if self.minimize:
            best_idx = np.argmin(y_array)
        else:
            best_idx = np.argmax(y_array)

        return BOResult(
            best_solution=np.array(self.X_observed[best_idx]),
            best_value=self.y_observed[best_idx],
            X_observed=np.array(self.X_observed),
            y_observed=y_array,
            n_iterations=n_iterations,
            n_initial=n_initial,
            history=history,
        )

    def suggest_next(self) -> np.ndarray:
        """Suggest the next point to evaluate (for manual control)."""
        if len(self.X_observed) == 0:
            # Return random initial point
            return np.random.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1]
            )

        # Fit GP and optimize acquisition
        X_train = np.array(self.X_observed)
        y_train = np.array(self.y_observed)
        self.gp.fit(X_train, y_train)

        return self._optimize_acquisition()

    def tell(self, x: np.ndarray, y: float) -> None:
        """Record an observation (for manual control)."""
        self.X_observed.append(np.atleast_1d(x))
        self.y_observed.append(y)


# ============================================================================
# COMPARISON UTILITIES
# ============================================================================

@dataclass
class ComparisonResult:
    """Results from optimizer comparison."""

    optimizer_name: str
    best_values: list[float]
    mean_best: float
    std_best: float
    mean_evaluations: float
    mean_time: float

    def __str__(self) -> str:
        return (
            f"{self.optimizer_name}:\n"
            f"  Best: {self.mean_best:.6f} ± {self.std_best:.6f}\n"
            f"  Evaluations: {self.mean_evaluations:.0f}\n"
            f"  Time: {self.mean_time:.3f}s"
        )


def compare_optimizers(
    objective: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    n_evaluations: int = 100,
    n_runs: int = 10,
    optimal_value: Optional[float] = None,
) -> dict[str, ComparisonResult]:
    """
    Compare Bayesian Optimization with other methods.

    Args:
        objective: Function to optimize
        bounds: Variable bounds
        n_evaluations: Budget of function evaluations
        n_runs: Number of independent runs
        optimal_value: Known optimal (for reference)

    Returns:
        Dictionary of results for each optimizer
    """
    import time
    from simulated_annealing import SimulatedAnnealing, CoolingSchedule, NeighborhoodFunction

    results = {}
    bounds_np = np.array(bounds)
    n_dims = len(bounds)

    # Bayesian Optimization
    bo_bests = []
    bo_times = []
    n_bo_iters = max(10, n_evaluations - 5)  # Reserve 5 for initial

    for run in range(n_runs):
        start = time.time()
        bo = BayesianOptimizer(
            objective=objective,
            bounds=bounds,
            seed=run * 123,
        )
        result = bo.optimize(n_iterations=n_bo_iters, n_initial=5)
        bo_times.append(time.time() - start)
        bo_bests.append(result.best_value)

    results['Bayesian (EI)'] = ComparisonResult(
        optimizer_name='Bayesian (EI)',
        best_values=bo_bests,
        mean_best=np.mean(bo_bests),
        std_best=np.std(bo_bests),
        mean_evaluations=n_evaluations,
        mean_time=np.mean(bo_times),
    )

    # Bayesian Optimization with UCB
    bo_ucb_bests = []
    bo_ucb_times = []

    for run in range(n_runs):
        start = time.time()
        bo = BayesianOptimizer(
            objective=objective,
            bounds=bounds,
            acquisition=UpperConfidenceBound(beta=2.0),
            seed=run * 123,
        )
        result = bo.optimize(n_iterations=n_bo_iters, n_initial=5)
        bo_ucb_times.append(time.time() - start)
        bo_ucb_bests.append(result.best_value)

    results['Bayesian (UCB)'] = ComparisonResult(
        optimizer_name='Bayesian (UCB)',
        best_values=bo_ucb_bests,
        mean_best=np.mean(bo_ucb_bests),
        std_best=np.std(bo_ucb_bests),
        mean_evaluations=n_evaluations,
        mean_time=np.mean(bo_ucb_times),
    )

    # Simulated Annealing
    sa_bests = []
    sa_times = []

    def objective_list(x):
        return objective(np.array(x))

    for run in range(n_runs):
        start = time.time()
        sa = SimulatedAnnealing(
            objective=objective_list,
            bounds=bounds,
            cooling=CoolingSchedule.geometric(T0=100, alpha=0.99),
            neighbor=NeighborhoodFunction.gaussian(sigma=0.3),
            seed=run * 123,
        )
        result = sa.optimize(max_iterations=n_evaluations)
        sa_times.append(time.time() - start)
        sa_bests.append(result.best_value)

    results['Simulated Annealing'] = ComparisonResult(
        optimizer_name='Simulated Annealing',
        best_values=sa_bests,
        mean_best=np.mean(sa_bests),
        std_best=np.std(sa_bests),
        mean_evaluations=n_evaluations,
        mean_time=np.mean(sa_times),
    )

    # Random Search
    rs_bests = []
    rs_times = []

    for run in range(n_runs):
        np.random.seed(run * 123)
        start = time.time()

        best_y = float('inf')
        for _ in range(n_evaluations):
            x = np.random.uniform(bounds_np[:, 0], bounds_np[:, 1])
            y = objective(x)
            if y < best_y:
                best_y = y

        rs_times.append(time.time() - start)
        rs_bests.append(best_y)

    results['Random Search'] = ComparisonResult(
        optimizer_name='Random Search',
        best_values=rs_bests,
        mean_best=np.mean(rs_bests),
        std_best=np.std(rs_bests),
        mean_evaluations=n_evaluations,
        mean_time=np.mean(rs_times),
    )

    return results


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

class TestFunctions:
    """Benchmark functions for optimization."""

    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """Sphere: minimum at origin = 0."""
        return float(np.sum(x**2))

    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """Rosenbrock: minimum at (1,1,...) = 0."""
        x = np.atleast_1d(x)
        return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

    @staticmethod
    def branin(x: np.ndarray) -> float:
        """
        Branin function (2D).
        Global minima at (-π, 12.275), (π, 2.275), (9.42478, 2.475) ≈ 0.397887
        """
        x1, x2 = x[0], x[1]
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)

        return float(
            a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t)*np.cos(x1) + s
        )

    @staticmethod
    def hartmann6(x: np.ndarray) -> float:
        """
        Hartmann 6D function.
        Global minimum ≈ -3.32237 at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
        """
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ])
        P = 1e-4 * np.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ])

        outer = 0
        for i in range(4):
            inner = 0
            for j in range(6):
                inner += A[i, j] * (x[j] - P[i, j])**2
            outer += alpha[i] * np.exp(-inner)

        return float(-outer)

    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """Ackley: minimum at origin = 0."""
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(2 * np.pi * x))
        return float(
            -20 * np.exp(-0.2 * np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e
        )

    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """Rastrigin: minimum at origin = 0."""
        n = len(x)
        return float(10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x)))


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_bayesian_optimization():
    """Demonstrate Bayesian Optimization."""
    print("=" * 70)
    print("BAYESIAN OPTIMIZATION DEMONSTRATION")
    print("=" * 70)

    # Example 1: Simple 2D optimization
    print("\n1. BRANIN FUNCTION (2D)")
    print("-" * 50)
    print("   Global minimum ≈ 0.398 at multiple points")

    bo = BayesianOptimizer(
        objective=TestFunctions.branin,
        bounds=[(-5, 10), (0, 15)],
        acquisition=ExpectedImprovement(xi=0.01),
        seed=42,
    )

    result = bo.optimize(n_iterations=30, n_initial=5, verbose=True)
    print(f"\n{result}")

    # Example 2: Different acquisition functions
    print("\n\n2. ACQUISITION FUNCTION COMPARISON (Sphere 3D)")
    print("-" * 50)

    acquisitions = [
        ("Expected Improvement", ExpectedImprovement(xi=0.01)),
        ("UCB (β=2)", UpperConfidenceBound(beta=2.0)),
        ("UCB (β=4)", UpperConfidenceBound(beta=4.0)),
        ("Prob. Improvement", ProbabilityOfImprovement(xi=0.01)),
        ("Thompson Sampling", ThompsonSampling()),
    ]

    for name, acq in acquisitions:
        bo = BayesianOptimizer(
            objective=TestFunctions.sphere,
            bounds=[(-5, 5)] * 3,
            acquisition=acq,
            seed=42,
        )
        result = bo.optimize(n_iterations=30, n_initial=5)
        print(f"  {name:25s}: best = {result.best_value:.6f}")

    # Example 3: Different kernels
    print("\n\n3. KERNEL COMPARISON (Rosenbrock 2D)")
    print("-" * 50)

    kernels = [
        ("RBF (l=1.0)", RBFKernel(length_scale=1.0)),
        ("RBF (l=0.5)", RBFKernel(length_scale=0.5)),
        ("Matérn 5/2", Matern52Kernel(length_scale=1.0)),
        ("Matérn 3/2", Matern32Kernel(length_scale=1.0)),
    ]

    for name, kernel in kernels:
        bo = BayesianOptimizer(
            objective=TestFunctions.rosenbrock,
            bounds=[(-5, 5)] * 2,
            kernel=kernel,
            seed=42,
        )
        result = bo.optimize(n_iterations=40, n_initial=5)
        print(f"  {name:20s}: best = {result.best_value:.6f}")

    # Example 4: 6D Hartmann function
    print("\n\n4. HARTMANN 6D FUNCTION")
    print("-" * 50)
    print("   Global minimum ≈ -3.322")

    bo = BayesianOptimizer(
        objective=TestFunctions.hartmann6,
        bounds=[(0, 1)] * 6,
        acquisition=ExpectedImprovement(xi=0.01),
        seed=42,
    )

    result = bo.optimize(n_iterations=100, n_initial=10, verbose=True)
    print(f"\n{result}")


def demo_comparison():
    """Compare Bayesian Optimization with other methods."""
    print("\n\n" + "=" * 70)
    print("OPTIMIZER COMPARISON")
    print("=" * 70)

    print("\nComparing optimizers on Branin function (2D)")
    print("Budget: 50 evaluations, 10 runs each\n")

    results = compare_optimizers(
        objective=TestFunctions.branin,
        bounds=[(-5, 10), (0, 15)],
        n_evaluations=50,
        n_runs=10,
        optimal_value=0.397887,
    )

    print(f"{'Optimizer':<25} {'Mean Best':>12} {'Std':>10} {'Time':>10}")
    print("-" * 60)
    for name, res in sorted(results.items(), key=lambda x: x[1].mean_best):
        print(f"{name:<25} {res.mean_best:>12.4f} {res.std_best:>10.4f} {res.mean_time:>10.3f}s")

    print("\n\nComparing on Sphere function (5D)")
    print("Budget: 100 evaluations, 10 runs each\n")

    results = compare_optimizers(
        objective=TestFunctions.sphere,
        bounds=[(-5, 5)] * 5,
        n_evaluations=100,
        n_runs=10,
        optimal_value=0.0,
    )

    print(f"{'Optimizer':<25} {'Mean Best':>12} {'Std':>10} {'Time':>10}")
    print("-" * 60)
    for name, res in sorted(results.items(), key=lambda x: x[1].mean_best):
        print(f"{name:<25} {res.mean_best:>12.6f} {res.std_best:>10.6f} {res.mean_time:>10.3f}s")


def demo_sample_efficiency():
    """Show sample efficiency of BO vs other methods."""
    print("\n\n" + "=" * 70)
    print("SAMPLE EFFICIENCY ANALYSIS")
    print("=" * 70)

    print("\nTracking best-so-far vs number of evaluations (Branin)")
    print("Single run comparison\n")

    n_evals = 50
    bounds = [(-5, 10), (0, 15)]

    # Bayesian Optimization
    bo = BayesianOptimizer(
        objective=TestFunctions.branin,
        bounds=bounds,
        seed=42,
    )
    bo_result = bo.optimize(n_iterations=n_evals - 5, n_initial=5)
    bo_history = bo_result.history

    # Random Search
    np.random.seed(42)
    rs_history = []
    best = float('inf')
    for i in range(n_evals):
        x = np.random.uniform([-5, 0], [10, 15])
        y = TestFunctions.branin(x)
        if y < best:
            best = y
        rs_history.append(best)

    print(f"{'Evals':>6} {'BO':>12} {'Random':>12}")
    print("-" * 35)
    for i in [5, 10, 20, 30, 40, 50]:
        bo_val = bo_history[min(i, len(bo_history)-1)]
        rs_val = rs_history[min(i-1, len(rs_history)-1)]
        print(f"{i:>6} {bo_val:>12.4f} {rs_val:>12.4f}")

    print(f"\nOptimal value: 0.3979")
    print(f"BO reached within 1% of optimal after ~{sum(1 for h in bo_history if h > 0.4)} evaluations")


def main():
    """Run all demonstrations."""
    np.random.seed(42)
    RandomVariable.set_global_seed(42)

    demo_bayesian_optimization()
    demo_comparison()
    demo_sample_efficiency()

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

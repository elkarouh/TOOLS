"""
Random Variable Module for Discrete-Event Simulation

Provides a clean abstraction for random number generation with various
probability distributions. Supports reproducible simulations via seed control.

Usage:
    interarrival = Exponential(mean=1.0)
    service_time = Exponential(mean=5.0)
    
    # Sample values
    next_arrival = interarrival.sample()
    processing_time = service_time.sample()
    
    # Or use as callable
    next_arrival = interarrival()
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence


class RandomVariable(ABC):
    """
    Abstract base class for random variables.
    
    Subclasses implement specific probability distributions.
    All random variables can be sampled via sample() or called directly.
    """
    
    _global_rng: Optional[random.Random] = None
    
    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the random variable.
        
        Args:
            seed: Optional seed for this specific random variable.
                  If None, uses the global RNG or system random.
        """
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = None  # Will use global or system random
    
    @classmethod
    def set_global_seed(cls, seed: int) -> None:
        """Set a global seed for all RandomVariables without individual seeds."""
        cls._global_rng = random.Random(seed)
    
    @classmethod
    def reset_global_seed(cls) -> None:
        """Reset the global RNG to use system random."""
        cls._global_rng = None
    
    @property
    def rng(self) -> random.Random:
        """Get the random number generator to use."""
        if self._rng is not None:
            return self._rng
        if RandomVariable._global_rng is not None:
            return RandomVariable._global_rng
        return random._inst  # System random instance
    
    @abstractmethod
    def sample(self) -> float:
        """Generate a random sample from this distribution."""
        pass
    
    def __call__(self) -> float:
        """Allow calling the random variable directly to get a sample."""
        return self.sample()
    
    @abstractmethod
    def mean(self) -> float:
        """Return the theoretical mean of the distribution."""
        pass
    
    @abstractmethod
    def variance(self) -> float:
        """Return the theoretical variance of the distribution."""
        pass
    
    def std(self) -> float:
        """Return the theoretical standard deviation."""
        return math.sqrt(self.variance())
    
    def samples(self, n: int) -> list[float]:
        """Generate n samples from this distribution."""
        return [self.sample() for _ in range(n)]


@dataclass
class Constant(RandomVariable):
    """
    A deterministic "random" variable that always returns the same value.
    
    Useful for testing or when a parameter should be fixed.
    """
    
    value: float
    _rng: Optional[random.Random] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        pass  # No RNG needed for constant
    
    def sample(self) -> float:
        return self.value
    
    def mean(self) -> float:
        return self.value
    
    def variance(self) -> float:
        return 0.0
    
    def __str__(self) -> str:
        return f"Constant({self.value})"


@dataclass
class Exponential(RandomVariable):
    """
    Exponential distribution with specified mean.
    
    Commonly used for:
    - Interarrival times in Poisson processes
    - Service times in queueing systems
    
    PDF: f(x) = (1/μ) * exp(-x/μ) for x >= 0
    where μ is the mean.
    """
    
    _mean: float
    _rng: Optional[random.Random] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        if self._mean <= 0:
            raise ValueError("Mean must be positive for exponential distribution")
    
    def sample(self) -> float:
        return self.rng.expovariate(1.0 / self._mean)
    
    def mean(self) -> float:
        return self._mean
    
    def variance(self) -> float:
        return self._mean ** 2
    
    def rate(self) -> float:
        """Return the rate parameter (λ = 1/mean)."""
        return 1.0 / self._mean
    
    def __str__(self) -> str:
        return f"Exponential(mean={self._mean})"


@dataclass
class Uniform(RandomVariable):
    """
    Uniform distribution over [low, high].
    
    All values in the range are equally likely.
    """
    
    low: float
    high: float
    _rng: Optional[random.Random] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError("low must be less than high")
    
    def sample(self) -> float:
        return self.rng.uniform(self.low, self.high)
    
    def mean(self) -> float:
        return (self.low + self.high) / 2.0
    
    def variance(self) -> float:
        return ((self.high - self.low) ** 2) / 12.0
    
    def __str__(self) -> str:
        return f"Uniform({self.low}, {self.high})"


@dataclass
class Normal(RandomVariable):
    """
    Normal (Gaussian) distribution with specified mean and standard deviation.
    
    Note: Can produce negative values, which may not be suitable for
    modeling times. Consider using Truncated or LogNormal for non-negative values.
    """
    
    _mean: float
    _std: float
    _rng: Optional[random.Random] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        if self._std < 0:
            raise ValueError("Standard deviation must be non-negative")
    
    def sample(self) -> float:
        return self.rng.gauss(self._mean, self._std)
    
    def mean(self) -> float:
        return self._mean
    
    def variance(self) -> float:
        return self._std ** 2
    
    def __str__(self) -> str:
        return f"Normal(mean={self._mean}, std={self._std})"


@dataclass
class TruncatedNormal(RandomVariable):
    """
    Normal distribution truncated to [low, high].
    
    Useful when you need normally-distributed values that must be
    within a certain range (e.g., non-negative service times).
    
    Uses rejection sampling.
    """
    
    _mean: float
    _std: float
    low: float = 0.0
    high: float = float('inf')
    _rng: Optional[random.Random] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        if self._std < 0:
            raise ValueError("Standard deviation must be non-negative")
        if self.low >= self.high:
            raise ValueError("low must be less than high")
    
    def sample(self) -> float:
        while True:
            value = self.rng.gauss(self._mean, self._std)
            if self.low <= value <= self.high:
                return value
    
    def mean(self) -> float:
        # Approximation - exact formula is complex
        return self._mean
    
    def variance(self) -> float:
        # Approximation
        return self._std ** 2
    
    def __str__(self) -> str:
        return f"TruncatedNormal(mean={self._mean}, std={self._std}, [{self.low}, {self.high}])"


@dataclass
class LogNormal(RandomVariable):
    """
    Log-normal distribution.
    
    If X ~ LogNormal(μ, σ), then log(X) ~ Normal(μ, σ).
    Always produces positive values, good for modeling times.
    
    Note: μ and σ are the parameters of the underlying normal distribution,
    not the mean and std of the log-normal itself.
    """
    
    mu: float  # Mean of the underlying normal
    sigma: float  # Std of the underlying normal
    _rng: Optional[random.Random] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("Sigma must be non-negative")
    
    @classmethod
    def from_mean_std(cls, mean: float, std: float) -> LogNormal:
        """Create a LogNormal with desired mean and std of the distribution itself."""
        if mean <= 0:
            raise ValueError("Mean must be positive for log-normal")
        if std < 0:
            raise ValueError("Standard deviation must be non-negative")
        
        variance = std ** 2
        mu = math.log(mean ** 2 / math.sqrt(variance + mean ** 2))
        sigma = math.sqrt(math.log(1 + variance / mean ** 2))
        return cls(mu=mu, sigma=sigma)
    
    def sample(self) -> float:
        return self.rng.lognormvariate(self.mu, self.sigma)
    
    def mean(self) -> float:
        return math.exp(self.mu + self.sigma ** 2 / 2)
    
    def variance(self) -> float:
        return (math.exp(self.sigma ** 2) - 1) * math.exp(2 * self.mu + self.sigma ** 2)
    
    def __str__(self) -> str:
        return f"LogNormal(μ={self.mu}, σ={self.sigma})"


@dataclass
class Triangular(RandomVariable):
    """
    Triangular distribution with mode (most likely value).
    
    Useful when you have minimum, maximum, and most likely estimates.
    """
    
    low: float
    high: float
    mode: float
    _rng: Optional[random.Random] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        if not (self.low <= self.mode <= self.high):
            raise ValueError("Must have low <= mode <= high")
    
    def sample(self) -> float:
        return self.rng.triangular(self.low, self.high, self.mode)
    
    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3.0
    
    def variance(self) -> float:
        a, b, c = self.low, self.high, self.mode
        return (a**2 + b**2 + c**2 - a*b - a*c - b*c) / 18.0
    
    def __str__(self) -> str:
        return f"Triangular({self.low}, {self.mode}, {self.high})"


@dataclass
class Erlang(RandomVariable):
    """
    Erlang distribution (sum of k exponential random variables).
    
    Useful for modeling multi-stage processes where each stage
    has an exponentially distributed duration.
    
    When k=1, this is the exponential distribution.
    """
    
    k: int  # Shape parameter (number of stages)
    _mean: float  # Mean of the entire distribution
    _rng: Optional[random.Random] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be at least 1")
        if self._mean <= 0:
            raise ValueError("Mean must be positive")
    
    def sample(self) -> float:
        # Mean of each exponential stage
        stage_mean = self._mean / self.k
        return sum(self.rng.expovariate(1.0 / stage_mean) for _ in range(self.k))
    
    def mean(self) -> float:
        return self._mean
    
    def variance(self) -> float:
        return (self._mean ** 2) / self.k
    
    def __str__(self) -> str:
        return f"Erlang(k={self.k}, mean={self._mean})"


@dataclass
class Gamma(RandomVariable):
    """
    Gamma distribution with shape (alpha) and scale (beta) parameters.
    
    Mean = alpha * beta
    Variance = alpha * beta^2
    """
    
    alpha: float  # Shape parameter
    beta: float   # Scale parameter
    _rng: Optional[random.Random] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("Alpha and beta must be positive")
    
    @classmethod
    def from_mean_variance(cls, mean: float, variance: float) -> Gamma:
        """Create a Gamma distribution with specified mean and variance."""
        if mean <= 0 or variance <= 0:
            raise ValueError("Mean and variance must be positive")
        beta = variance / mean
        alpha = mean / beta
        return cls(alpha=alpha, beta=beta)
    
    def sample(self) -> float:
        return self.rng.gammavariate(self.alpha, self.beta)
    
    def mean(self) -> float:
        return self.alpha * self.beta
    
    def variance(self) -> float:
        return self.alpha * self.beta ** 2
    
    def __str__(self) -> str:
        return f"Gamma(α={self.alpha}, β={self.beta})"


@dataclass  
class Weibull(RandomVariable):
    """
    Weibull distribution with shape (k) and scale (λ) parameters.
    
    Commonly used in reliability engineering and failure analysis.
    """
    
    shape: float  # k parameter
    scale: float  # λ parameter
    _rng: Optional[random.Random] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        if self.shape <= 0 or self.scale <= 0:
            raise ValueError("Shape and scale must be positive")
    
    def sample(self) -> float:
        return self.scale * self.rng.weibullvariate(1.0, self.shape)
    
    def mean(self) -> float:
        return self.scale * math.gamma(1 + 1/self.shape)
    
    def variance(self) -> float:
        return self.scale**2 * (
            math.gamma(1 + 2/self.shape) - math.gamma(1 + 1/self.shape)**2
        )
    
    def __str__(self) -> str:
        return f"Weibull(shape={self.shape}, scale={self.scale})"


@dataclass
class Empirical(RandomVariable):
    """
    Empirical distribution based on observed data.
    
    Samples are drawn uniformly from the provided data points.
    """
    
    data: Sequence[float]
    _rng: Optional[random.Random] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        if len(self.data) == 0:
            raise ValueError("Data cannot be empty")
        self._data_list = list(self.data)
        self._cached_mean: Optional[float] = None
        self._cached_variance: Optional[float] = None
    
    def sample(self) -> float:
        return self.rng.choice(self._data_list)
    
    def mean(self) -> float:
        if self._cached_mean is None:
            self._cached_mean = sum(self._data_list) / len(self._data_list)
        return self._cached_mean
    
    def variance(self) -> float:
        if self._cached_variance is None:
            mu = self.mean()
            self._cached_variance = sum((x - mu)**2 for x in self._data_list) / len(self._data_list)
        return self._cached_variance
    
    def __str__(self) -> str:
        return f"Empirical(n={len(self._data_list)})"


@dataclass
class DiscreteUniform(RandomVariable):
    """
    Discrete uniform distribution over integers [low, high].
    """
    
    low: int
    high: int
    _rng: Optional[random.Random] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("low must be <= high")
    
    def sample(self) -> float:
        return float(self.rng.randint(self.low, self.high))
    
    def mean(self) -> float:
        return (self.low + self.high) / 2.0
    
    def variance(self) -> float:
        n = self.high - self.low + 1
        return (n ** 2 - 1) / 12.0
    
    def __str__(self) -> str:
        return f"DiscreteUniform({self.low}, {self.high})"


# ============================================================================
# DISCRETE DISTRIBUTION WITH MUTABLE WEIGHTS
# ============================================================================

from typing import TypeVar, Generic, Dict, Iterator, Hashable
import bisect

K = TypeVar('K', bound=Hashable)


class DiscreteDistribution(RandomVariable, Generic[K]):
    """
    A discrete distribution over arbitrary keys with mutable weights.
    
    Optimized for distributions that change frequently, such as in the
    Barabási-Albert preferential attachment model where P(node) ∝ degree(node).
    
    Features:
    - O(log n) sampling using cumulative weights + binary search
    - O(1) weight updates
    - Dict-like interface for weights
    - Automatic normalization to probabilities
    
    Example:
        >>> dist = DiscreteDistribution({'a': 1, 'b': 5, 'c': 1})
        >>> dist.sample()  # 'b' is 5x more likely than 'a' or 'c'
        'b'
        >>> dist['d'] = 3  # Add new element
        >>> dist['b'] += 2  # Increase weight
        >>> del dist['a']  # Remove element
    
    For Barabási-Albert model:
        >>> degrees = DiscreteDistribution()
        >>> for node in initial_nodes:
        ...     degrees[node] = 1  # Initial degree
        >>> # When adding edges:
        >>> target = degrees.sample()  # Preferential attachment
        >>> degrees[target] += 1  # Update degree
    """
    
    def __init__(
        self,
        weights: Optional[Dict[K, float]] = None,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize discrete distribution.
        
        Args:
            weights: Dict mapping keys to weights (non-negative numbers)
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self._weights: Dict[K, float] = {}
        self._total: float = 0.0
        self._keys: list[K] = []
        self._cumulative: list[float] = []
        self._dirty: bool = False  # Track if cumulative needs rebuild
        
        if weights:
            for key, weight in weights.items():
                self._add_item(key, weight)
    
    def _add_item(self, key: K, weight: float) -> None:
        """Add or update an item (internal, marks as dirty)."""
        if weight < 0:
            raise ValueError(f"Weight must be non-negative, got {weight}")
        
        if key in self._weights:
            self._total -= self._weights[key]
        
        self._weights[key] = weight
        self._total += weight
        self._dirty = True
    
    def _rebuild_cumulative(self) -> None:
        """Rebuild cumulative distribution for binary search."""
        self._keys = list(self._weights.keys())
        self._cumulative = []
        cumsum = 0.0
        for key in self._keys:
            cumsum += self._weights[key]
            self._cumulative.append(cumsum)
        self._dirty = False
    
    def sample(self) -> K:
        """
        Return a random element according to the weighted distribution.
        
        Time complexity: O(log n) using binary search.
        """
        if not self._weights:
            raise ValueError("Cannot sample from empty distribution")
        
        if self._dirty:
            self._rebuild_cumulative()
        
        # Generate random point in [0, total)
        r = self.rng.random() * self._total
        
        # Binary search for the key
        idx = bisect.bisect_left(self._cumulative, r)
        
        # Handle edge case where r == total
        if idx >= len(self._keys):
            idx = len(self._keys) - 1
        
        return self._keys[idx]
    
    def sample_n(self, n: int, replacement: bool = True) -> list[K]:
        """
        Sample n elements from the distribution.
        
        Args:
            n: Number of samples
            replacement: If True, sample with replacement (default)
                        If False, sample without replacement
        
        Returns:
            List of n sampled keys
        """
        if replacement:
            return [self.sample() for _ in range(n)]
        else:
            if n > len(self._weights):
                raise ValueError(f"Cannot sample {n} items without replacement from {len(self._weights)} items")
            
            # Weighted sampling without replacement
            result = []
            temp_weights = dict(self._weights)
            temp_dist = DiscreteDistribution(temp_weights)
            
            for _ in range(n):
                key = temp_dist.sample()
                result.append(key)
                del temp_dist[key]
            
            return result
    
    @property
    def total(self) -> float:
        """Total sum of all weights."""
        return self._total
    
    def pdf(self) -> Dict[K, float]:
        """Return probability distribution (normalized weights)."""
        if self._total == 0:
            return {}
        return {k: v / self._total for k, v in self._weights.items()}
    
    def pmf(self, key: K) -> float:
        """Probability mass function - probability of a specific key."""
        if key not in self._weights or self._total == 0:
            return 0.0
        return self._weights[key] / self._total
    
    def mean(self) -> float:
        """
        Mean of the distribution (if keys are numeric).
        
        Raises TypeError if keys are not numeric.
        """
        if not self._weights:
            return 0.0
        try:
            return sum(k * (w / self._total) for k, w in self._weights.items())
        except TypeError:
            raise TypeError("mean() requires numeric keys")
    
    def variance(self) -> float:
        """
        Variance of the distribution (if keys are numeric).
        
        Raises TypeError if keys are not numeric.
        """
        if not self._weights:
            return 0.0
        try:
            mu = self.mean()
            return sum((k - mu)**2 * (w / self._total) for k, w in self._weights.items())
        except TypeError:
            raise TypeError("variance() requires numeric keys")
    
    def mode(self) -> K:
        """Return the key with highest weight."""
        if not self._weights:
            raise ValueError("Cannot get mode of empty distribution")
        return max(self._weights, key=self._weights.get)
    
    def entropy(self) -> float:
        """Shannon entropy of the distribution in nats."""
        if self._total == 0:
            return 0.0
        h = 0.0
        for w in self._weights.values():
            if w > 0:
                p = w / self._total
                h -= p * math.log(p)
        return h
    
    # Dict-like interface
    def __getitem__(self, key: K) -> float:
        return self._weights[key]
    
    def __setitem__(self, key: K, weight: float) -> None:
        self._add_item(key, weight)
    
    def __delitem__(self, key: K) -> None:
        if key in self._weights:
            self._total -= self._weights[key]
            del self._weights[key]
            self._dirty = True
    
    def __contains__(self, key: K) -> bool:
        return key in self._weights
    
    def __len__(self) -> int:
        return len(self._weights)
    
    def __iter__(self) -> Iterator[K]:
        return iter(self._weights)
    
    def keys(self):
        """Return keys."""
        return self._weights.keys()
    
    def values(self):
        """Return weights."""
        return self._weights.values()
    
    def items(self):
        """Return (key, weight) pairs."""
        return self._weights.items()
    
    def get(self, key: K, default: float = 0.0) -> float:
        """Get weight for key, or default if not present."""
        return self._weights.get(key, default)
    
    def update(self, other: Dict[K, float]) -> None:
        """Update weights from another dict."""
        for key, weight in other.items():
            self._add_item(key, weight)
    
    def clear(self) -> None:
        """Remove all items."""
        self._weights.clear()
        self._total = 0.0
        self._keys.clear()
        self._cumulative.clear()
        self._dirty = False
    
    def copy(self) -> 'DiscreteDistribution[K]':
        """Return a copy of this distribution."""
        return DiscreteDistribution(dict(self._weights))
    
    def normalize(self) -> 'DiscreteDistribution[K]':
        """Return a new distribution with weights normalized to sum to 1."""
        if self._total == 0:
            return DiscreteDistribution()
        return DiscreteDistribution({k: v / self._total for k, v in self._weights.items()})
    
    def __call__(self) -> K:
        """Allow calling distribution directly to sample."""
        return self.sample()
    
    def __str__(self) -> str:
        if len(self._weights) <= 5:
            items = ", ".join(f"{k}:{v}" for k, v in self._weights.items())
            return f"DiscreteDistribution({{{items}}})"
        else:
            return f"DiscreteDistribution(n={len(self._weights)}, total={self._total:.2f})"
    
    def __repr__(self) -> str:
        return f"DiscreteDistribution({self._weights!r})"


class AliasMethod(RandomVariable, Generic[K]):
    """
    Vose's Alias Method for O(1) sampling from a discrete distribution.
    
    Use this when the distribution is static (doesn't change after creation).
    For dynamic distributions, use DiscreteDistribution instead.
    
    Time complexity:
    - Initialization: O(n)
    - Sampling: O(1)
    
    Example:
        >>> dist = AliasMethod({'a': 1, 'b': 5, 'c': 1})
        >>> samples = [dist.sample() for _ in range(1000)]
    """
    
    def __init__(
        self,
        weights: Dict[K, float],
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize with Vose's Alias Method.
        
        Args:
            weights: Dict mapping keys to weights
            seed: Random seed
        """
        super().__init__(seed)
        
        if not weights:
            raise ValueError("Weights cannot be empty")
        
        n = len(weights)
        self._keys = list(weights.keys())
        self._n = n
        
        # Normalize probabilities
        total = sum(weights.values())
        prob = [weights[k] / total * n for k in self._keys]
        
        # Initialize alias and probability tables
        self._prob = [0.0] * n
        self._alias = [0] * n
        
        # Separate into small and large
        small: list[int] = []
        large: list[int] = []
        
        for i, p in enumerate(prob):
            if p < 1.0:
                small.append(i)
            else:
                large.append(i)
        
        # Build tables
        while small and large:
            l = small.pop()
            g = large.pop()
            
            self._prob[l] = prob[l]
            self._alias[l] = g
            
            prob[g] = prob[g] + prob[l] - 1.0
            
            if prob[g] < 1.0:
                small.append(g)
            else:
                large.append(g)
        
        # Remaining items have probability 1
        while large:
            g = large.pop()
            self._prob[g] = 1.0
        
        while small:
            l = small.pop()
            self._prob[l] = 1.0
        
        # Store original weights for statistics
        self._weights = dict(weights)
        self._total = total
    
    def sample(self) -> K:
        """
        Sample from the distribution in O(1) time.
        """
        # Pick a random column
        i = self.rng.randrange(self._n)
        
        # Flip a biased coin
        if self.rng.random() < self._prob[i]:
            return self._keys[i]
        else:
            return self._keys[self._alias[i]]
    
    def mean(self) -> float:
        """Mean (requires numeric keys)."""
        try:
            return sum(k * (w / self._total) for k, w in self._weights.items())
        except TypeError:
            raise TypeError("mean() requires numeric keys")
    
    def variance(self) -> float:
        """Variance (requires numeric keys)."""
        try:
            mu = self.mean()
            return sum((k - mu)**2 * (w / self._total) for k, w in self._weights.items())
        except TypeError:
            raise TypeError("variance() requires numeric keys")
    
    def pdf(self) -> Dict[K, float]:
        """Return probability distribution."""
        return {k: v / self._total for k, v in self._weights.items()}
    
    def __str__(self) -> str:
        return f"AliasMethod(n={self._n})"


# ============================================================================
# CATEGORICAL DISTRIBUTION (ALIAS FOR DISCRETE)
# ============================================================================

class Categorical(DiscreteDistribution[int]):
    """
    Categorical distribution over integers 0, 1, ..., n-1.
    
    Convenience wrapper around DiscreteDistribution for the common case
    of sampling from a fixed set of categories.
    
    Example:
        >>> cat = Categorical([0.1, 0.5, 0.3, 0.1])  # 4 categories
        >>> cat.sample()  # Returns 0, 1, 2, or 3
        1
    """
    
    def __init__(
        self,
        probabilities: Sequence[float],
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize categorical distribution.
        
        Args:
            probabilities: Sequence of probabilities (will be normalized)
            seed: Random seed
        """
        weights = {i: p for i, p in enumerate(probabilities)}
        super().__init__(weights, seed)
        self._n_categories = len(probabilities)
    
    @property
    def n_categories(self) -> int:
        """Number of categories."""
        return self._n_categories
    
    def __str__(self) -> str:
        return f"Categorical(n={self._n_categories})"


# Backward compatible alias
Dist = DiscreteDistribution


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Convenience function for backward compatibility
def exponential_distribution(mean: float) -> float:
    """Generate a sample from an exponential distribution (legacy function)."""
    return random.expovariate(1.0 / mean)


# Alias for backward compatibility
exp_distr = exponential_distribution

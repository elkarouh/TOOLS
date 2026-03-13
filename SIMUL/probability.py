"""
Probability and Bayesian Inference Module

Provides tools for:
- Probability Mass Functions (PMF) with exact arithmetic
- Conditional probability calculations
- Bayesian inference and updates
- Naive Bayes classification
- Joint and marginal distributions

Integrates with the RandomVariable framework for sampling.

Usage:
    from probability import PMF, BayesianInference, NaiveBayesClassifier

    # Simple PMF
    dice = PMF({'1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1})
    print(dice.probability('6'))  # 1/6

    # Conditional probability
    even = lambda x: int(x) % 2 == 0
    print(dice.given(even).probability('2'))  # 1/3

    # Bayesian inference
    prior = PMF({'biased': 1, 'fair': 9})
    # Update with evidence...
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from decimal import Decimal, ROUND_HALF_UP
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from random_variable import RandomVariable, DiscreteDistribution

# Type variables
T = TypeVar('T', bound=Hashable)
Predicate = Callable[[Any], bool]


# ============================================================================
# FRACTION UTILITIES
# ============================================================================

# Make Fraction repr cleaner
Fraction.__repr__ = lambda self: (
    f"{self.numerator}/{self.denominator}"
    if self.denominator > 1
    else str(self.numerator)
)

F = Fraction  # Shorthand alias


def to_fraction(value: Union[int, float, Fraction, Decimal]) -> Fraction:
    """Convert a value to a Fraction."""
    if isinstance(value, Fraction):
        return value
    elif isinstance(value, int):
        return Fraction(value, 1)
    elif isinstance(value, Decimal):
        return Fraction(value)
    else:
        # Float - use limit_denominator to avoid huge fractions
        return Fraction(value).limit_denominator(1000000)


# ============================================================================
# PROBABILITY MASS FUNCTION
# ============================================================================

class PMF(Generic[T]):
    """
    A Probability Mass Function for discrete random variables.

    Supports exact arithmetic using Fractions, conditional probabilities,
    Bayesian updates, and joint distributions.

    Examples:
        >>> dice = PMF.uniform('1', '2', '3', '4', '5', '6')
        >>> dice.probability('6')
        Fraction(1, 6)

        >>> even = lambda x: int(x) % 2 == 0
        >>> dice.given(even)  # Conditional PMF
        PMF({'2': 1/3, '4': 1/3, '6': 1/3})

        >>> # Joint distribution
        >>> two_dice = dice * dice
        >>> two_dice.probability(('3', '4'))
        Fraction(1, 36)

    For conditional probability calculations:
        >>> # P(A | B) using the & and | operators
        >>> P = dice.prob  # Shorthand
        >>> P(even)  # P(even)
        Fraction(1, 2)
    """

    def __init__(
        self,
        outcomes: Optional[Union[Dict[T, Union[int, float, Fraction]], Iterable[T]]] = None,
        **kwargs: Union[int, float, Fraction]
    ) -> None:
        """
        Initialize a PMF.

        Args:
            outcomes: Dict of {outcome: weight} or iterable of outcomes (equal weight)
            **kwargs: Additional {outcome: weight} pairs
        """
        self._counts: Dict[T, Fraction] = {}
        self._total: Fraction = Fraction(0)

        if outcomes is not None:
            if isinstance(outcomes, dict):
                for key, value in outcomes.items():
                    self._add(key, to_fraction(value))
            else:
                # Iterable of outcomes - count occurrences
                for outcome in outcomes:
                    self._add(outcome, Fraction(1))

        for key, value in kwargs.items():
            self._add(key, to_fraction(value))

    def _add(self, outcome: T, weight: Fraction) -> None:
        """Add weight to an outcome."""
        if weight < 0:
            raise ValueError(f"Weight must be non-negative: {weight}")
        if outcome in self._counts:
            self._counts[outcome] += weight
        else:
            self._counts[outcome] = weight
        self._total += weight

    @classmethod
    def uniform(cls, *outcomes: T) -> PMF[T]:
        """Create a uniform distribution over the given outcomes."""
        return cls({outcome: 1 for outcome in outcomes})

    @classmethod
    def from_counter(cls, counter: Counter) -> PMF:
        """Create a PMF from a Counter."""
        return cls(dict(counter))

    @classmethod
    def from_samples(cls, samples: Iterable[T]) -> PMF[T]:
        """Create a PMF from samples (empirical distribution)."""
        return cls.from_counter(Counter(samples))

    # -------------------- Probability Access --------------------

    @property
    def total(self) -> Fraction:
        """Total weight (for unnormalized PMFs)."""
        return self._total

    def probability(self, outcome: T) -> Fraction:
        """Get probability of a specific outcome."""
        if self._total == 0:
            return Fraction(0)
        return self._counts.get(outcome, Fraction(0)) / self._total

    def prob(self, predicate: Union[T, Predicate]) -> Fraction:
        """
        Get probability of an outcome or event.

        Args:
            predicate: Either a specific outcome or a predicate function

        Returns:
            P(outcome) or P(predicate is true)
        """
        if callable(predicate):
            return sum(
                self.probability(o) for o in self._counts if predicate(o)
            )
        else:
            return self.probability(predicate)

    def __getitem__(self, outcome: T) -> Fraction:
        """Get probability of outcome."""
        return self.probability(outcome)

    def weight(self, outcome: T) -> Fraction:
        """Get raw weight (unnormalized) of an outcome."""
        return self._counts.get(outcome, Fraction(0))

    def weights(self) -> Dict[T, Fraction]:
        """Get all weights."""
        return dict(self._counts)

    def as_dict(self) -> Dict[T, Fraction]:
        """Get probabilities as a dictionary."""
        return {o: self.probability(o) for o in self._counts}

    def as_float_dict(self) -> Dict[T, float]:
        """Get probabilities as floats."""
        return {o: float(self.probability(o)) for o in self._counts}

    # -------------------- Conditional Probability --------------------

    def given(self, predicate: Predicate) -> PMF[T]:
        """
        Return conditional distribution P(· | predicate).

        Example:
            >>> dice = PMF.uniform(1, 2, 3, 4, 5, 6)
            >>> dice.given(lambda x: x > 3)  # P(· | X > 3)
            PMF({4: 1/3, 5: 1/3, 6: 1/3})
        """
        filtered = {o: self._counts[o] for o in self._counts if predicate(o)}
        return PMF(filtered)

    def __and__(self, predicate: Predicate) -> PMF[T]:
        """
        Shorthand for conditional: pmf & predicate = pmf.given(predicate)

        Example:
            >>> even = lambda x: x % 2 == 0
            >>> (even & dice)  # P(· | even)
        """
        return self.given(predicate)

    def __rand__(self, predicate: Predicate) -> PMF[T]:
        """Allow predicate & pmf syntax."""
        return self.given(predicate)

    def __or__(self, predicate: Predicate) -> Fraction:
        """
        Compute P(predicate | self).

        Example:
            >>> P(even | dice) == dice.prob(even)
        """
        return self.prob(predicate)

    def __ror__(self, predicate: Predicate) -> Fraction:
        """Allow predicate | pmf syntax for P(predicate)."""
        return self.prob(predicate)

    # -------------------- Joint Distribution --------------------

    def __mul__(self, other: PMF) -> PMF[Tuple]:
        """
        Joint distribution of two independent PMFs.

        Example:
            >>> dice = PMF.uniform(1, 2, 3, 4, 5, 6)
            >>> two_dice = dice * dice
            >>> two_dice.probability((1, 1))
            Fraction(1, 36)
        """
        joint = {}
        for o1 in self._counts:
            for o2 in other._counts:
                key = (o1, o2)
                joint[key] = self._counts[o1] * other._counts[o2]
        return PMF(joint)

    def joint_with_labels(self, other: PMF, sep: str = "-") -> PMF[str]:
        """
        Joint distribution with string labels.

        Example:
            >>> PMF(cancer=1) * PMF(positive=9, negative=1)
            # Returns PMF with keys like 'cancer-positive'
        """
        joint = {}
        for o1 in self._counts:
            for o2 in other._counts:
                key = f"{o1}{sep}{o2}"
                joint[key] = self._counts[o1] * other._counts[o2]
        return PMF(joint)

    def __add__(self, other: PMF[T]) -> PMF[T]:
        """
        Combine two PMFs (mixture).

        This is NOT addition of random variables - it's combining
        the frequency counts (useful for aggregating evidence).
        """
        combined = dict(self._counts)
        for o, w in other._counts.items():
            if o in combined:
                combined[o] += w
            else:
                combined[o] = w
        return PMF(combined)

    def marginal(self, index: int) -> PMF:
        """
        Get marginal distribution from a joint distribution.

        Args:
            index: Which element of the tuple to keep (0 or 1)

        Example:
            >>> joint = PMF({('a', 1): 2, ('a', 2): 1, ('b', 1): 1})
            >>> joint.marginal(0)  # Distribution over first element
            PMF({'a': 3/4, 'b': 1/4})
        """
        marginal_counts: Dict[Any, Fraction] = {}
        for outcome, weight in self._counts.items():
            if not isinstance(outcome, tuple):
                raise ValueError("marginal() requires tuple outcomes (joint distribution)")
            key = outcome[index]
            if key in marginal_counts:
                marginal_counts[key] += weight
            else:
                marginal_counts[key] = weight
        return PMF(marginal_counts)

    # -------------------- Statistics --------------------

    def expected_value(self) -> float:
        """
        Expected value E[X] (requires numeric outcomes).
        """
        return sum(float(o) * float(self.probability(o)) for o in self._counts)

    def variance(self) -> float:
        """
        Variance Var[X] (requires numeric outcomes).
        """
        ev = self.expected_value()
        return sum(
            (float(o) - ev) ** 2 * float(self.probability(o))
            for o in self._counts
        )

    def std(self) -> float:
        """Standard deviation."""
        return math.sqrt(self.variance())

    def mode(self) -> T:
        """Most likely outcome."""
        return max(self._counts, key=self._counts.get)

    def median(self) -> T:
        """Median outcome (requires orderable outcomes)."""
        sorted_outcomes = sorted(self._counts.keys())
        cumulative = Fraction(0)
        for outcome in sorted_outcomes:
            cumulative += self.probability(outcome)
            if cumulative >= Fraction(1, 2):
                return outcome
        return sorted_outcomes[-1]

    def entropy(self) -> float:
        """Shannon entropy in bits."""
        h = 0.0
        for o in self._counts:
            p = float(self.probability(o))
            if p > 0:
                h -= p * math.log2(p)
        return h

    # -------------------- Sampling --------------------

    def sample(self) -> T:
        """Draw a random sample from the distribution."""
        r = random.random() * float(self._total)
        cumulative = 0.0
        for outcome, weight in self._counts.items():
            cumulative += float(weight)
            if cumulative >= r:
                return outcome
        return list(self._counts.keys())[-1]

    def samples(self, n: int) -> List[T]:
        """Draw n samples."""
        return [self.sample() for _ in range(n)]

    def to_discrete_distribution(self) -> DiscreteDistribution:
        """Convert to DiscreteDistribution for efficient repeated sampling."""
        return DiscreteDistribution({o: float(w) for o, w in self._counts.items()})

    # -------------------- Bayesian Update --------------------

    def bayesian_update(
        self,
        likelihood: Callable[[Any, T], float],
        data: Any
    ) -> PMF[T]:
        """
        Perform Bayesian update: P(H|D) ∝ P(D|H) × P(H)

        Args:
            likelihood: Function(data, hypothesis) -> P(data | hypothesis)
            data: Observed data

        Returns:
            Posterior distribution

        Example:
            >>> prior = PMF({'fair': 9, 'biased': 1})
            >>> def likelihood(data, hypo):
            ...     p_heads = 0.5 if hypo == 'fair' else 0.9
            ...     return p_heads if data == 'H' else 1 - p_heads
            >>> posterior = prior.bayesian_update(likelihood, 'H')
        """
        posterior_weights = {}
        for hypothesis in self._counts:
            prior = self._counts[hypothesis]
            lik = to_fraction(likelihood(data, hypothesis))
            posterior_weights[hypothesis] = prior * lik
        return PMF(posterior_weights)

    def bayesian_update_batch(
        self,
        likelihood: Callable[[Any, T], float],
        data_points: Iterable[Any]
    ) -> PMF[T]:
        """
        Perform sequential Bayesian updates with multiple data points.
        """
        current = self
        for data in data_points:
            current = current.bayesian_update(likelihood, data)
        return current

    # -------------------- Utility Methods --------------------

    def normalize(self) -> PMF[T]:
        """Return a normalized copy (probabilities sum to 1)."""
        return PMF(self._counts)  # Constructor normalizes automatically

    def outcomes(self) -> List[T]:
        """List of all outcomes."""
        return list(self._counts.keys())

    def __len__(self) -> int:
        return len(self._counts)

    def __iter__(self) -> Iterator[T]:
        return iter(self._counts)

    def __contains__(self, outcome: T) -> bool:
        return outcome in self._counts

    def items(self):
        """Iterate over (outcome, probability) pairs."""
        for o in self._counts:
            yield o, self.probability(o)

    def __repr__(self) -> str:
        if len(self._counts) <= 6:
            items = ", ".join(f"{o!r}: {self.probability(o)}" for o in self._counts)
            return f"PMF({{{items}}})"
        else:
            return f"PMF(n={len(self._counts)}, total={self._total})"

    def __str__(self) -> str:
        return self.__repr__()

    def display(self, max_width: int = 50) -> None:
        """Print a visual representation of the distribution."""
        if not self._counts:
            print("Empty PMF")
            return

        max_prob = max(float(self.probability(o)) for o in self._counts)

        for outcome in sorted(self._counts.keys(), key=str):
            p = float(self.probability(outcome))
            bar_len = int(p / max_prob * max_width) if max_prob > 0 else 0
            bar = "█" * bar_len
            print(f"{str(outcome):>15}: {p:6.3f} {bar}")


# ============================================================================
# NAIVE BAYES CLASSIFIER
# ============================================================================

@dataclass
class NaiveBayesClassifier:
    """
    A Naive Bayes classifier for discrete features.

    Assumes features are conditionally independent given the class:
    P(class | features) ∝ P(class) × ∏ P(feature_i | class)

    Example (Gender classification from Wikipedia):
        >>> clf = NaiveBayesClassifier()
        >>>
        >>> # Training data
        >>> clf.add_sample('male', height=6.0, weight=180, foot_size=12)
        >>> clf.add_sample('male', height=5.92, weight=190, foot_size=11)
        >>> clf.add_sample('female', height=5.0, weight=100, foot_size=6)
        >>> clf.add_sample('female', height=5.5, weight=150, foot_size=8)
        >>>
        >>> # Classify new sample
        >>> clf.classify(height=6.0, weight=130, foot_size=8)
        'female'
    """

    classes: Dict[str, int] = field(default_factory=dict)
    features: Dict[str, Dict[str, PMF]] = field(default_factory=dict)
    feature_names: Set[str] = field(default_factory=set)
    n_samples: int = 0

    # For continuous features
    bins: int = 10
    continuous_features: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=dict)

    def add_sample(self, class_label: str, **features: Any) -> None:
        """
        Add a training sample.

        Args:
            class_label: The class of this sample
            **features: Feature name-value pairs
        """
        # Update class count
        self.classes[class_label] = self.classes.get(class_label, 0) + 1
        self.n_samples += 1

        # Update feature distributions
        for feature_name, value in features.items():
            self.feature_names.add(feature_name)

            if feature_name not in self.features:
                self.features[feature_name] = {}

            if class_label not in self.features[feature_name]:
                self.features[feature_name][class_label] = PMF()

            # Discretize continuous features
            if isinstance(value, (int, float)):
                # Store range for later discretization
                if feature_name not in self.continuous_features:
                    self.continuous_features[feature_name] = {}
                if class_label not in self.continuous_features[feature_name]:
                    self.continuous_features[feature_name][class_label] = (value, value)
                else:
                    lo, hi = self.continuous_features[feature_name][class_label]
                    self.continuous_features[feature_name][class_label] = (
                        min(lo, value), max(hi, value)
                    )

            # Add to PMF
            self.features[feature_name][class_label]._add(value, Fraction(1))

    def add_samples(self, samples: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Add multiple training samples."""
        for class_label, features in samples:
            self.add_sample(class_label, **features)

    def prior(self, class_label: str) -> Fraction:
        """Get prior probability of a class."""
        if self.n_samples == 0:
            return Fraction(0)
        return Fraction(self.classes.get(class_label, 0), self.n_samples)

    def likelihood(
        self,
        class_label: str,
        feature_name: str,
        value: Any,
        smoothing: float = 1.0
    ) -> float:
        """
        Get P(feature=value | class) with Laplace smoothing.
        """
        if feature_name not in self.features:
            return 1.0 / self.bins  # Unknown feature

        if class_label not in self.features[feature_name]:
            return 1.0 / self.bins  # Class has no data for this feature

        pmf = self.features[feature_name][class_label]

        # Laplace smoothing
        n_values = len(pmf)
        count = float(pmf.weight(value))
        total = float(pmf.total)

        if n_values == 0:
            return 1.0 / self.bins

        return (count + smoothing) / (total + smoothing * n_values)

    def classify(
        self,
        smoothing: float = 1.0,
        return_probs: bool = False,
        **features: Any
    ) -> Union[str, Tuple[str, Dict[str, float]]]:
        """
        Classify a sample based on features.

        Args:
            smoothing: Laplace smoothing parameter
            return_probs: If True, also return class probabilities
            **features: Feature name-value pairs

        Returns:
            Predicted class (and probabilities if return_probs=True)
        """
        if not self.classes:
            raise ValueError("Classifier has no training data")

        # Compute log probabilities to avoid underflow
        log_probs = {}

        for class_label in self.classes:
            # Log prior
            log_prob = math.log(float(self.prior(class_label)) + 1e-10)

            # Log likelihoods
            for feature_name, value in features.items():
                lik = self.likelihood(class_label, feature_name, value, smoothing)
                log_prob += math.log(lik + 1e-10)

            log_probs[class_label] = log_prob

        # Find best class
        best_class = max(log_probs, key=log_probs.get)

        if return_probs:
            # Convert log probs to normalized probabilities
            max_log = max(log_probs.values())
            probs = {c: math.exp(lp - max_log) for c, lp in log_probs.items()}
            total = sum(probs.values())
            probs = {c: p / total for c, p in probs.items()}
            return best_class, probs

        return best_class

    def __repr__(self) -> str:
        return (
            f"NaiveBayesClassifier(classes={list(self.classes.keys())}, "
            f"features={list(self.feature_names)}, n_samples={self.n_samples})"
        )


# ============================================================================
# BAYESIAN INFERENCE HELPER
# ============================================================================

class BayesianInference:
    """
    Helper class for common Bayesian inference problems.

    Example - Medical testing:
        >>> bi = BayesianInference()
        >>> # Prior: 1% of population has disease
        >>> prior = PMF({'disease': 1, 'healthy': 99})
        >>>
        >>> # Likelihood: test accuracy
        >>> def likelihood(test_result, health_status):
        ...     if health_status == 'disease':
        ...         return 0.95 if test_result == 'positive' else 0.05
        ...     else:
        ...         return 0.10 if test_result == 'positive' else 0.90
        >>>
        >>> # Update with positive test
        >>> posterior = prior.bayesian_update(likelihood, 'positive')
        >>> print(posterior.probability('disease'))  # ~8.8%
    """

    @staticmethod
    def medical_test(
        prevalence: float,
        sensitivity: float,  # True positive rate
        specificity: float,  # True negative rate
        test_result: str = 'positive'
    ) -> PMF:
        """
        Compute posterior probability of disease given test result.

        Args:
            prevalence: P(disease) in population
            sensitivity: P(positive | disease) - true positive rate
            specificity: P(negative | healthy) - true negative rate
            test_result: 'positive' or 'negative'

        Returns:
            Posterior PMF over {'disease', 'healthy'}
        """
        prior = PMF({
            'disease': Fraction(prevalence).limit_denominator(10000),
            'healthy': Fraction(1 - prevalence).limit_denominator(10000)
        })

        def likelihood(result, status):
            if status == 'disease':
                return sensitivity if result == 'positive' else (1 - sensitivity)
            else:
                return (1 - specificity) if result == 'positive' else specificity

        return prior.bayesian_update(likelihood, test_result)

    @staticmethod
    def coin_bias(
        n_heads: int,
        n_tails: int,
        prior_beliefs: Optional[Dict[float, float]] = None,
        n_hypotheses: int = 11
    ) -> PMF:
        """
        Infer coin bias from observed flips.

        Args:
            n_heads: Number of heads observed
            n_tails: Number of tails observed
            prior_beliefs: Optional prior {bias: weight}
            n_hypotheses: Number of discrete bias values to consider

        Returns:
            Posterior over bias values
        """
        if prior_beliefs is None:
            # Uniform prior over discrete bias values
            biases = [i / (n_hypotheses - 1) for i in range(n_hypotheses)]
            prior = PMF.uniform(*biases)
        else:
            prior = PMF(prior_beliefs)

        def likelihood(flip, bias):
            p_head = bias
            if flip == 'H':
                return p_head
            else:
                return 1 - p_head

        # Create data sequence
        data = ['H'] * n_heads + ['T'] * n_tails

        return prior.bayesian_update_batch(likelihood, data)

    @staticmethod
    def urn_problem(
        draws: Sequence[str],
        urns: Dict[str, Dict[str, int]]
    ) -> PMF:
        """
        Solve an urn problem: which urn did the draws come from?

        Args:
            draws: Sequence of ball colors drawn (with replacement)
            urns: {urn_name: {color: count}} describing each urn

        Returns:
            Posterior over urn identities

        Example:
            >>> urns = {
            ...     'urn_A': {'red': 3, 'blue': 7},
            ...     'urn_B': {'red': 7, 'blue': 3}
            ... }
            >>> BayesianInference.urn_problem(['red', 'red', 'blue'], urns)
        """
        # Uniform prior over urns
        prior = PMF.uniform(*urns.keys())

        def likelihood(ball_color, urn_name):
            urn = urns[urn_name]
            total = sum(urn.values())
            return urn.get(ball_color, 0) / total

        return prior.bayesian_update_batch(likelihood, draws)


# ============================================================================
# EXPECTATION-MAXIMIZATION
# ============================================================================

class EMAlgorithm:
    """
    Expectation-Maximization algorithm for parameter estimation.

    Example - Two coins with unknown biases:
        >>> em = EMAlgorithm()
        >>> observations = ['HTTTHHTHTH', 'HHHHTHHHHH', 'HTHHHHHTHH']
        >>> theta_a, theta_b = em.two_coins(observations)
    """

    @staticmethod
    def two_coins(
        observations: List[str],
        initial_theta_a: float = 0.6,
        initial_theta_b: float = 0.5,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Tuple[float, float]:
        """
        Estimate bias of two coins from mixed observations.

        Each observation is a string of 'H' and 'T' from either coin A or B,
        but we don't know which coin produced each observation.

        Args:
            observations: List of flip sequences (e.g., ['HHTHT', 'TTHH'])
            initial_theta_a: Initial guess for coin A's P(heads)
            initial_theta_b: Initial guess for coin B's P(heads)
            max_iterations: Maximum EM iterations
            tolerance: Convergence threshold

        Returns:
            Estimated (theta_a, theta_b)
        """
        theta_a = initial_theta_a
        theta_b = initial_theta_b

        for iteration in range(max_iterations):
            # E-step: compute expected counts
            expected_a_heads = 0.0
            expected_a_tails = 0.0
            expected_b_heads = 0.0
            expected_b_tails = 0.0

            for obs in observations:
                n_heads = obs.count('H')
                n_tails = obs.count('T')
                n = len(obs)

                # Likelihood of observation under each coin
                lik_a = (theta_a ** n_heads) * ((1 - theta_a) ** n_tails)
                lik_b = (theta_b ** n_heads) * ((1 - theta_b) ** n_tails)

                # Responsibility (posterior probability of each coin)
                total_lik = lik_a + lik_b + 1e-10
                resp_a = lik_a / total_lik
                resp_b = lik_b / total_lik

                # Accumulate expected counts
                expected_a_heads += resp_a * n_heads
                expected_a_tails += resp_a * n_tails
                expected_b_heads += resp_b * n_heads
                expected_b_tails += resp_b * n_tails

            # M-step: update parameters
            new_theta_a = expected_a_heads / (expected_a_heads + expected_a_tails + 1e-10)
            new_theta_b = expected_b_heads / (expected_b_heads + expected_b_tails + 1e-10)

            # Check convergence
            if (abs(new_theta_a - theta_a) < tolerance and
                abs(new_theta_b - theta_b) < tolerance):
                break

            theta_a = new_theta_a
            theta_b = new_theta_b

        return theta_a, theta_b


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_pmf_basics():
    """Demonstrate basic PMF operations."""
    print("=" * 70)
    print("PMF BASICS")
    print("=" * 70)

    # Fair die
    print("\n1. FAIR DIE")
    print("-" * 50)
    die = PMF.uniform(1, 2, 3, 4, 5, 6)
    print(f"Die: {die}")
    print(f"P(6) = {die.probability(6)} = {float(die.probability(6)):.4f}")
    print(f"E[X] = {die.expected_value():.4f}")
    print(f"Var[X] = {die.variance():.4f}")

    # Conditional probability
    print("\n2. CONDITIONAL PROBABILITY")
    print("-" * 50)
    even = lambda x: x % 2 == 0
    greater_than_3 = lambda x: x > 3

    print(f"P(even) = {die.prob(even)}")
    print(f"P(X > 3) = {die.prob(greater_than_3)}")
    print(f"P(X > 3 | even) = {die.given(even).prob(greater_than_3)}")
    print(f"Die | even: {die.given(even)}")

    # Alternative syntax
    print(f"\nUsing operator syntax:")
    print(f"  even | die = {even | die}")
    print(f"  die & even = {die & even}")


def demo_medical_test():
    """Demonstrate Bayesian medical testing."""
    print("\n" + "=" * 70)
    print("BAYESIAN MEDICAL TESTING")
    print("=" * 70)

    print("""
    Scenario:
    - Disease prevalence: 1% of population
    - Test sensitivity: 90% (P(positive | disease))
    - Test specificity: 90% (P(negative | healthy))

    Question: If you test positive, what's P(disease)?
    """)

    posterior = BayesianInference.medical_test(
        prevalence=0.01,
        sensitivity=0.90,
        specificity=0.90,
        test_result='positive'
    )

    print(f"Prior P(disease) = 1%")
    print(f"Posterior P(disease | positive) = {float(posterior.probability('disease'))*100:.1f}%")
    print(f"\nThis is the 'base rate fallacy' - even with a positive test,")
    print(f"you're more likely healthy because the disease is rare!")

    # Second test
    print(f"\nAfter a SECOND positive test:")
    posterior2 = posterior.bayesian_update(
        lambda result, status: 0.90 if (result == 'positive') == (status == 'disease') else 0.10,
        'positive'
    )
    print(f"P(disease | two positives) = {float(posterior2.probability('disease'))*100:.1f}%")


def demo_joint_distribution():
    """Demonstrate joint distributions."""
    print("\n" + "=" * 70)
    print("JOINT DISTRIBUTIONS")
    print("=" * 70)

    # Cancer screening example
    print("\n1. CANCER SCREENING")
    print("-" * 50)

    # Prior: 1% have cancer
    cancer = PMF({'cancer': 1, 'healthy': 99})

    # Test results given health status
    test_if_cancer = PMF({'positive': F(9, 10), 'negative': F(1, 10)})
    test_if_healthy = PMF({'positive': F(1, 10), 'negative': F(9, 10)})

    # Joint distribution
    joint_cancer = cancer.given(lambda x: x == 'cancer').joint_with_labels(test_if_cancer)
    joint_healthy = cancer.given(lambda x: x == 'healthy').joint_with_labels(test_if_healthy)
    combined = joint_cancer + joint_healthy

    print("Joint distribution (health-test_result):")
    for outcome in combined:
        print(f"  {outcome}: {combined.probability(outcome)}")

    # Conditional
    is_positive = lambda o: 'positive' in o
    has_cancer = lambda o: 'cancer' in o

    positive_cases = combined.given(is_positive)
    p_cancer_if_positive = positive_cases.prob(has_cancer)
    print(f"\nP(cancer | positive) = {p_cancer_if_positive} ≈ {float(p_cancer_if_positive):.3f}")


def demo_naive_bayes():
    """Demonstrate Naive Bayes classifier."""
    print("\n" + "=" * 70)
    print("NAIVE BAYES CLASSIFIER")
    print("=" * 70)

    print("""
    Training data (from Wikipedia example):

    Gender  Height  Weight  Foot Size
    ------  ------  ------  ---------
    Male    6.00    180     12
    Male    5.92    190     11
    Male    5.58    170     12
    Male    5.92    165     10
    Female  5.00    100     6
    Female  5.50    150     8
    Female  5.42    130     7
    Female  5.75    150     9
    """)

    clf = NaiveBayesClassifier()

    # Training data
    clf.add_sample('male', height=6.00, weight=180, foot_size=12)
    clf.add_sample('male', height=5.92, weight=190, foot_size=11)
    clf.add_sample('male', height=5.58, weight=170, foot_size=12)
    clf.add_sample('male', height=5.92, weight=165, foot_size=10)
    clf.add_sample('female', height=5.00, weight=100, foot_size=6)
    clf.add_sample('female', height=5.50, weight=150, foot_size=8)
    clf.add_sample('female', height=5.42, weight=130, foot_size=7)
    clf.add_sample('female', height=5.75, weight=150, foot_size=9)

    print(f"Classifier: {clf}")

    # Classify new sample
    test_sample = {'height': 6.0, 'weight': 130, 'foot_size': 8}
    prediction, probs = clf.classify(return_probs=True, **test_sample)

    print(f"\nTest sample: {test_sample}")
    print(f"Prediction: {prediction}")
    print(f"Probabilities: {', '.join(f'{c}={p:.3f}' for c, p in probs.items())}")


def demo_em_algorithm():
    """Demonstrate EM algorithm."""
    print("\n" + "=" * 70)
    print("EXPECTATION-MAXIMIZATION")
    print("=" * 70)

    print("""
    Problem: Two coins A and B with unknown biases.
    We observe flip sequences but don't know which coin produced each.

    Observations:
    - HTTTHHTHTH (5H, 5T)
    - HHHHTHHHHH (9H, 1T)
    - HTHHHHHTHH (8H, 2T)
    - HTHTTTHHTT (4H, 6T)
    - THHHTHHHTH (7H, 3T)
    """)

    observations = [
        "HTTTHHTHTH",
        "HHHHTHHHHH",
        "HTHHHHHTHH",
        "HTHTTTHHTT",
        "THHHTHHHTH"
    ]

    theta_a, theta_b = EMAlgorithm.two_coins(
        observations,
        initial_theta_a=0.6,
        initial_theta_b=0.5
    )

    print(f"Initial guesses: θ_A=0.6, θ_B=0.5")
    print(f"EM estimates: θ_A={theta_a:.4f}, θ_B={theta_b:.4f}")
    print(f"\n(True values from original problem: θ_A≈0.80, θ_B≈0.52)")


def demo_coin_inference():
    """Demonstrate coin bias inference."""
    print("\n" + "=" * 70)
    print("COIN BIAS INFERENCE")
    print("=" * 70)

    print("\nObserved: 7 heads, 3 tails. What's the coin's bias?")

    posterior = BayesianInference.coin_bias(n_heads=7, n_tails=3)

    print(f"\nPosterior distribution:")
    posterior.display()

    print(f"\nMost likely bias: {posterior.mode()}")
    print(f"Expected bias: {posterior.expected_value():.3f}")


def main():
    """Run all demonstrations."""
    RandomVariable.set_global_seed(42)

    demo_pmf_basics()
    demo_medical_test()
    demo_joint_distribution()
    demo_naive_bayes()
    demo_em_algorithm()
    demo_coin_inference()

    print("\n" + "=" * 70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

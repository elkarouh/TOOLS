"""
Monte Carlo Simulation Examples

Demonstrates various Monte Carlo techniques using the RandomVariable framework:
1. Classic Monte Carlo - Estimating π
2. Financial - Option pricing and portfolio risk (VaR)
3. Project scheduling - PERT analysis
4. Queueing - Bank capacity planning with confidence intervals
5. Reliability - System failure analysis
6. Integration - Numerical integration via Monte Carlo

Usage:
    python montecarlo.py
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional
from concurrent.futures import ProcessPoolExecutor
import time

from random_variable import (
    RandomVariable,
    Exponential,
    Normal,
    LogNormal,
    Uniform,
    Triangular,
    Constant,
    Weibull,
    Erlang,
)


# ============================================================================
# MONTE CARLO STATISTICS HELPERS
# ============================================================================

@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo simulation."""

    mean: float
    std_dev: float
    std_error: float
    n_samples: int
    confidence_level: float = 0.95
    samples: list[float] = field(default_factory=list, repr=False)

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Return the confidence interval for the mean."""
        # Using normal approximation (valid for large n)
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(self.confidence_level, 1.96)
        margin = z * self.std_error
        return (self.mean - margin, self.mean + margin)

    @property
    def relative_error(self) -> float:
        """Coefficient of variation of the estimate."""
        return self.std_error / abs(self.mean) if self.mean != 0 else float('inf')

    def percentile(self, p: float) -> float:
        """Return the p-th percentile (0-100)."""
        if not self.samples:
            raise ValueError("No samples stored")
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * p / 100)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def __str__(self) -> str:
        ci = self.confidence_interval
        return (
            f"Monte Carlo Result (n={self.n_samples:,}):\n"
            f"  Mean: {self.mean:.6f}\n"
            f"  Std Dev: {self.std_dev:.6f}\n"
            f"  Std Error: {self.std_error:.6f}\n"
            f"  {self.confidence_level*100:.0f}% CI: [{ci[0]:.6f}, {ci[1]:.6f}]\n"
            f"  Relative Error: {self.relative_error*100:.2f}%"
        )


def monte_carlo(
    sample_func: Callable[[], float],
    n_samples: int = 10000,
    confidence: float = 0.95,
    store_samples: bool = False
) -> MonteCarloResult:
    """
    Run a Monte Carlo simulation.

    Args:
        sample_func: Function that generates one sample
        n_samples: Number of samples to generate
        confidence: Confidence level for interval estimation
        store_samples: Whether to store all samples (for percentiles)

    Returns:
        MonteCarloResult with statistics
    """
    samples = [sample_func() for _ in range(n_samples)]

    mean = statistics.mean(samples)
    std_dev = statistics.stdev(samples) if n_samples > 1 else 0.0
    std_error = std_dev / math.sqrt(n_samples)

    return MonteCarloResult(
        mean=mean,
        std_dev=std_dev,
        std_error=std_error,
        n_samples=n_samples,
        confidence_level=confidence,
        samples=samples if store_samples else []
    )


# ============================================================================
# EXAMPLE 1: CLASSIC MONTE CARLO - ESTIMATING π
# ============================================================================

def estimate_pi(n_samples: int = 100000) -> MonteCarloResult:
    """
    Estimate π using the classic Monte Carlo method.

    Throw random darts at a unit square. The ratio of darts inside
    the inscribed quarter circle to total darts ≈ π/4.
    """
    x = Uniform(0, 1)
    y = Uniform(0, 1)

    def sample() -> float:
        # Returns 4 if point is inside quarter circle, 0 otherwise
        px, py = x(), y()
        return 4.0 if (px*px + py*py) <= 1.0 else 0.0

    return monte_carlo(sample, n_samples)


def demo_estimate_pi():
    """Demonstrate π estimation with increasing sample sizes."""
    print("=" * 60)
    print("EXAMPLE 1: Estimating π via Monte Carlo")
    print("=" * 60)
    print(f"\nTrue value of π: {math.pi:.10f}\n")

    for n in [100, 1000, 10000, 100000, 1000000]:
        result = estimate_pi(n)
        error = abs(result.mean - math.pi)
        ci = result.confidence_interval
        print(f"n={n:>10,}: π ≈ {result.mean:.6f} ± {result.std_error:.6f} "
              f"(error={error:.6f}, CI=[{ci[0]:.4f}, {ci[1]:.4f}])")


# ============================================================================
# EXAMPLE 2: FINANCIAL MONTE CARLO - OPTION PRICING
# ============================================================================

@dataclass
class EuropeanOption:
    """European option parameters."""
    spot_price: float      # Current stock price (S)
    strike_price: float    # Strike price (K)
    time_to_expiry: float  # Time to expiration in years (T)
    risk_free_rate: float  # Risk-free interest rate (r)
    volatility: float      # Volatility (σ)
    is_call: bool = True   # True for call, False for put


def black_scholes_price(opt: EuropeanOption) -> float:
    """Analytical Black-Scholes price for comparison."""
    S, K, T, r, sigma = (opt.spot_price, opt.strike_price, opt.time_to_expiry,
                         opt.risk_free_rate, opt.volatility)

    d1 = (math.log(S/K) + (r + sigma**2/2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Standard normal CDF approximation
    def norm_cdf(x):
        return (1 + math.erf(x / math.sqrt(2))) / 2

    if opt.is_call:
        return S * norm_cdf(d1) - K * math.exp(-r*T) * norm_cdf(d2)
    else:
        return K * math.exp(-r*T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def monte_carlo_option_price(
    opt: EuropeanOption,
    n_paths: int = 100000
) -> MonteCarloResult:
    """
    Price a European option using Monte Carlo simulation.

    Uses geometric Brownian motion for the stock price:
    S(T) = S(0) * exp((r - σ²/2)*T + σ*√T*Z)
    where Z ~ N(0,1)
    """
    S, K, T, r, sigma = (opt.spot_price, opt.strike_price, opt.time_to_expiry,
                         opt.risk_free_rate, opt.volatility)

    z = Normal(_mean=0, _std=1)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * math.sqrt(T)
    discount = math.exp(-r * T)

    def sample() -> float:
        # Simulate terminal stock price
        S_T = S * math.exp(drift + diffusion * z())

        # Calculate payoff
        if opt.is_call:
            payoff = max(S_T - K, 0)
        else:
            payoff = max(K - S_T, 0)

        # Return discounted payoff
        return discount * payoff

    return monte_carlo(sample, n_paths, store_samples=True)


def demo_option_pricing():
    """Demonstrate Monte Carlo option pricing vs Black-Scholes."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: European Option Pricing")
    print("=" * 60)

    option = EuropeanOption(
        spot_price=100,
        strike_price=105,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        is_call=True
    )

    print(f"\nOption Parameters:")
    print(f"  Spot Price (S): ${option.spot_price}")
    print(f"  Strike Price (K): ${option.strike_price}")
    print(f"  Time to Expiry: {option.time_to_expiry} year")
    print(f"  Risk-free Rate: {option.risk_free_rate*100}%")
    print(f"  Volatility: {option.volatility*100}%")
    print(f"  Type: {'Call' if option.is_call else 'Put'}")

    bs_price = black_scholes_price(option)
    print(f"\nBlack-Scholes Analytical Price: ${bs_price:.4f}")

    print("\nMonte Carlo Estimates:")
    for n in [1000, 10000, 100000]:
        result = monte_carlo_option_price(option, n)
        error = abs(result.mean - bs_price)
        ci = result.confidence_interval
        print(f"  n={n:>6,}: ${result.mean:.4f} ± ${result.std_error:.4f} "
              f"(error=${error:.4f})")


# ============================================================================
# EXAMPLE 3: PORTFOLIO VALUE AT RISK (VaR)
# ============================================================================

@dataclass
class Asset:
    """A financial asset with expected return and volatility."""
    name: str
    value: float
    annual_return: float  # Expected annual return (e.g., 0.08 for 8%)
    volatility: float     # Annual volatility (std dev of returns)


def portfolio_var(
    assets: list[Asset],
    correlations: Optional[list[list[float]]] = None,
    time_horizon: float = 1/252,  # 1 day by default
    confidence: float = 0.95,
    n_simulations: int = 100000
) -> MonteCarloResult:
    """
    Calculate Value at Risk for a portfolio using Monte Carlo.

    VaR answers: "What is the maximum loss at the X% confidence level?"

    Note: This simplified version assumes uncorrelated assets.
    """
    n_assets = len(assets)
    sqrt_t = math.sqrt(time_horizon)

    # Create normal RVs for each asset's return
    z = [Normal(_mean=0, _std=1) for _ in range(n_assets)]

    def sample() -> float:
        portfolio_return = 0.0
        for i, asset in enumerate(assets):
            # Daily return = drift + volatility * random shock
            drift = asset.annual_return * time_horizon
            shock = asset.volatility * sqrt_t * z[i]()
            asset_return = asset.value * (drift + shock)
            portfolio_return += asset_return
        return portfolio_return

    result = monte_carlo(sample, n_simulations, store_samples=True)
    return result


def demo_portfolio_var():
    """Demonstrate Value at Risk calculation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Portfolio Value at Risk (VaR)")
    print("=" * 60)

    portfolio = [
        Asset("Stocks", 60000, 0.10, 0.20),   # $60k, 10% return, 20% vol
        Asset("Bonds", 30000, 0.04, 0.05),    # $30k, 4% return, 5% vol
        Asset("Commodities", 10000, 0.06, 0.25),  # $10k, 6% return, 25% vol
    ]

    total_value = sum(a.value for a in portfolio)

    print(f"\nPortfolio Composition (Total: ${total_value:,.0f}):")
    for asset in portfolio:
        pct = asset.value / total_value * 100
        print(f"  {asset.name}: ${asset.value:,.0f} ({pct:.0f}%) - "
              f"μ={asset.annual_return*100:.0f}%, σ={asset.volatility*100:.0f}%")

    result = portfolio_var(portfolio, n_simulations=100000)

    var_95 = result.percentile(5)  # 5th percentile = 95% VaR
    var_99 = result.percentile(1)  # 1st percentile = 99% VaR

    print(f"\n1-Day Value at Risk:")
    print(f"  Expected Daily P&L: ${result.mean:,.2f}")
    print(f"  P&L Std Dev: ${result.std_dev:,.2f}")
    print(f"  95% VaR: ${-var_95:,.2f} (5% chance of losing more)")
    print(f"  99% VaR: ${-var_99:,.2f} (1% chance of losing more)")

    # Annual VaR (simplified - multiply by sqrt(252))
    print(f"\n  Annualized 95% VaR: ${-var_95 * math.sqrt(252):,.2f}")


# ============================================================================
# EXAMPLE 4: PROJECT SCHEDULING (PERT-LIKE)
# ============================================================================

@dataclass
class Task:
    """A project task with uncertain duration."""
    name: str
    duration: RandomVariable
    predecessors: list[str] = field(default_factory=list)


def project_completion_time(
    tasks: list[Task],
    n_simulations: int = 10000
) -> MonteCarloResult:
    """
    Simulate project completion time using Monte Carlo.

    Uses a simple critical path calculation for each simulation.
    """
    # Build task lookup
    task_map = {t.name: t for t in tasks}

    def sample() -> float:
        # Sample durations for all tasks
        durations = {t.name: t.duration() for t in tasks}

        # Calculate completion times (simple forward pass)
        completion = {}

        def get_completion(name: str) -> float:
            if name in completion:
                return completion[name]

            task = task_map[name]
            if not task.predecessors:
                start = 0
            else:
                start = max(get_completion(p) for p in task.predecessors)

            completion[name] = start + durations[name]
            return completion[name]

        # Get completion time of all tasks
        return max(get_completion(t.name) for t in tasks)

    return monte_carlo(sample, n_simulations, store_samples=True)


def demo_project_scheduling():
    """Demonstrate project scheduling Monte Carlo."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Project Completion Time (PERT-like)")
    print("=" * 60)

    # Define a software project with uncertain task durations
    # Using Triangular distributions (low, high, mode)
    tasks = [
        Task("Requirements", Triangular(low=5, high=12, mode=7)),
        Task("Design", Triangular(low=8, high=15, mode=10), ["Requirements"]),
        Task("Database", Triangular(low=4, high=10, mode=6), ["Design"]),
        Task("Backend", Triangular(low=15, high=30, mode=20), ["Design"]),
        Task("Frontend", Triangular(low=12, high=22, mode=15), ["Design"]),
        Task("Integration", Triangular(low=5, high=14, mode=8), ["Database", "Backend", "Frontend"]),
        Task("Testing", Triangular(low=8, high=20, mode=12), ["Integration"]),
        Task("Deployment", Triangular(low=2, high=5, mode=3), ["Testing"]),
    ]

    print("\nProject Tasks (duration in days: min/likely/max):")
    for t in tasks:
        if isinstance(t.duration, Triangular):
            print(f"  {t.name}: {t.duration.low:.0f}/{t.duration.mode:.0f}/{t.duration.high:.0f} days", end="")
        else:
            print(f"  {t.name}: ~{t.duration.mean():.0f} days", end="")
        if t.predecessors:
            print(f" (after: {', '.join(t.predecessors)})")
        else:
            print()

    result = project_completion_time(tasks, n_simulations=50000)

    print(f"\nProject Completion Time Analysis ({result.n_samples:,} simulations):")
    print(f"  Expected Duration: {result.mean:.1f} days")
    print(f"  Std Deviation: {result.std_dev:.1f} days")
    print(f"  95% CI: [{result.confidence_interval[0]:.1f}, {result.confidence_interval[1]:.1f}] days")
    print(f"\n  Percentiles:")
    print(f"    10th (optimistic): {result.percentile(10):.1f} days")
    print(f"    50th (median): {result.percentile(50):.1f} days")
    print(f"    90th (pessimistic): {result.percentile(90):.1f} days")
    print(f"    95th (buffer target): {result.percentile(95):.1f} days")


# ============================================================================
# EXAMPLE 5: BANK CAPACITY PLANNING (Using our simulation framework)
# ============================================================================

def bank_capacity_analysis(
    interarrival_mean: float = 1.0,
    service_mean: float = 5.0,
    teller_counts: list[int] = None,
    simulation_hours: float = 8.0,
    n_replications: int = 30
) -> dict[int, MonteCarloResult]:
    """
    Analyze bank capacity using multiple simulation replications.

    Returns statistics on average customer wait time for different teller counts.
    """
    # Import here to avoid circular dependency
    from banksimulation import BankSimulation, SimulationConfig
    from random_variable import RandomVariable

    if teller_counts is None:
        teller_counts = [3, 4, 5, 6, 7]

    results = {}

    for n_tellers in teller_counts:
        def run_one_replication() -> float:
            # Each replication uses a different random seed
            seed = int(time.time() * 1000000) % (2**31)

            config = SimulationConfig(
                interarrival_time=Exponential(_mean=interarrival_mean),
                service_time=Exponential(_mean=service_mean),
                number_tellers=n_tellers,
                deterministic=True,
                random_seed=seed
            )

            # Quiet simulation
            class QuietSim(BankSimulation):
                def update(self): pass
                def terminate(self): pass

            sim = QuietSim(config)
            sim.run(0, simulation_hours * 60)
            return sim.bank.stats.average_time

        # Run multiple replications
        wait_times = []
        for i in range(n_replications):
            RandomVariable.set_global_seed(42 + i * 1000)  # Different seed each time
            wait_times.append(run_one_replication())

        mean = statistics.mean(wait_times)
        std_dev = statistics.stdev(wait_times) if len(wait_times) > 1 else 0
        std_error = std_dev / math.sqrt(n_replications)

        results[n_tellers] = MonteCarloResult(
            mean=mean,
            std_dev=std_dev,
            std_error=std_error,
            n_samples=n_replications,
            samples=wait_times
        )

    return results


def demo_bank_capacity():
    """Demonstrate bank capacity planning analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Bank Capacity Planning")
    print("=" * 60)

    print("\nAnalyzing average customer time in system...")
    print("(Running multiple simulation replications for each configuration)")

    # Theoretical traffic intensity: λ/μ = 1/5 = 0.2 per teller
    # For M/M/c queue: need c > λ/μ = 1/(1/5) = 5 tellers for stability

    results = bank_capacity_analysis(
        interarrival_mean=1.0,  # 1 customer per minute
        service_mean=5.0,       # 5 minutes per customer
        teller_counts=[4, 5, 6, 7, 8],
        simulation_hours=8.0,
        n_replications=20
    )

    print(f"\nResults (λ=1/min, μ=1/5 min, ρ=5.0):")
    print(f"{'Tellers':>8} {'Avg Wait':>12} {'Std Err':>10} {'95% CI':>24}")
    print("-" * 56)

    for n_tellers, result in sorted(results.items()):
        ci = result.confidence_interval
        print(f"{n_tellers:>8} {result.mean:>10.2f}m {result.std_error:>10.2f}m "
              f"[{ci[0]:>8.2f}, {ci[1]:>8.2f}]m")

    print("\nNote: With traffic intensity ρ=5.0, we need > 5 tellers for stability.")


# ============================================================================
# EXAMPLE 6: SYSTEM RELIABILITY
# ============================================================================

@dataclass
class Component:
    """A system component with a failure time distribution."""
    name: str
    time_to_failure: RandomVariable


def series_system_reliability(
    components: list[Component],
    mission_time: float,
    n_simulations: int = 100000
) -> MonteCarloResult:
    """
    Calculate probability of system survival for a series system.

    In a series system, ALL components must survive for the system to survive.
    """
    def sample() -> float:
        # System survives if all components survive past mission_time
        for comp in components:
            if comp.time_to_failure() < mission_time:
                return 0.0  # System failed
        return 1.0  # System survived

    return monte_carlo(sample, n_simulations)


def parallel_system_reliability(
    components: list[Component],
    mission_time: float,
    n_simulations: int = 100000
) -> MonteCarloResult:
    """
    Calculate probability of system survival for a parallel (redundant) system.

    In a parallel system, only ONE component needs to survive.
    """
    def sample() -> float:
        # System survives if at least one component survives
        for comp in components:
            if comp.time_to_failure() >= mission_time:
                return 1.0  # System survived
        return 0.0  # All components failed

    return monte_carlo(sample, n_simulations)


def demo_reliability():
    """Demonstrate system reliability analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: System Reliability Analysis")
    print("=" * 60)

    # Components with Weibull failure distributions (common in reliability)
    # shape > 1 means wear-out failures (failure rate increases with age)
    components = [
        Component("Pump A", Weibull(shape=2.0, scale=1000)),
        Component("Pump B", Weibull(shape=2.0, scale=1000)),
        Component("Motor", Weibull(shape=1.5, scale=1500)),
    ]

    mission_time = 500  # hours

    print(f"\nComponents (Weibull distributions):")
    for c in components:
        if isinstance(c.time_to_failure, Weibull):
            w = c.time_to_failure
            print(f"  {c.name}: shape={w.shape}, scale={w.scale}, "
                  f"MTTF={w.mean():.0f} hours")

    print(f"\nMission Time: {mission_time} hours")

    # Series system (all must work)
    series_result = series_system_reliability(components, mission_time)

    # Parallel system (redundant - any one works)
    parallel_result = parallel_system_reliability(components, mission_time)

    print(f"\nSeries System (all components must survive):")
    print(f"  Reliability: {series_result.mean*100:.2f}% ± {series_result.std_error*100:.2f}%")

    print(f"\nParallel System (any one component surviving is sufficient):")
    print(f"  Reliability: {parallel_result.mean*100:.2f}% ± {parallel_result.std_error*100:.2f}%")


# ============================================================================
# EXAMPLE 7: NUMERICAL INTEGRATION
# ============================================================================

def monte_carlo_integrate(
    func: Callable[[float], float],
    a: float,
    b: float,
    n_samples: int = 100000
) -> MonteCarloResult:
    """
    Estimate ∫[a,b] f(x) dx using Monte Carlo integration.

    Uses the formula: ∫f(x)dx ≈ (b-a) * mean(f(X)) where X ~ Uniform(a,b)
    """
    x = Uniform(a, b)
    width = b - a

    def sample() -> float:
        return width * func(x())

    return monte_carlo(sample, n_samples)


def demo_integration():
    """Demonstrate Monte Carlo integration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Monte Carlo Integration")
    print("=" * 60)

    # Example 1: ∫[0,1] x² dx = 1/3
    print("\n∫[0,1] x² dx:")
    result = monte_carlo_integrate(lambda x: x**2, 0, 1)
    print(f"  Monte Carlo: {result.mean:.6f} ± {result.std_error:.6f}")
    print(f"  Exact: {1/3:.6f}")

    # Example 2: ∫[0,π] sin(x) dx = 2
    print("\n∫[0,π] sin(x) dx:")
    result = monte_carlo_integrate(math.sin, 0, math.pi)
    print(f"  Monte Carlo: {result.mean:.6f} ± {result.std_error:.6f}")
    print(f"  Exact: 2.000000")

    # Example 3: ∫[0,1] e^x dx = e - 1
    print("\n∫[0,1] eˣ dx:")
    result = monte_carlo_integrate(math.exp, 0, 1)
    print(f"  Monte Carlo: {result.mean:.6f} ± {result.std_error:.6f}")
    print(f"  Exact: {math.e - 1:.6f}")

    # Example 4: ∫[0,1] 4/(1+x²) dx = π (another way to estimate π!)
    print("\n∫[0,1] 4/(1+x²) dx = π:")
    result = monte_carlo_integrate(lambda x: 4/(1+x**2), 0, 1, n_samples=1000000)
    print(f"  Monte Carlo: {result.mean:.6f} ± {result.std_error:.6f}")
    print(f"  Exact (π): {math.pi:.6f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all Monte Carlo demonstrations."""
    print("\n" + "=" * 70)
    print("   MONTE CARLO SIMULATION EXAMPLES")
    print("   Using the RandomVariable Framework")
    print("=" * 70)

    # Set global seed for reproducibility
    RandomVariable.set_global_seed(42)

    demo_estimate_pi()
    demo_option_pricing()
    demo_portfolio_var()
    demo_project_scheduling()
    # demo_bank_capacity()  # Slower - uncomment to run
    demo_reliability()
    demo_integration()

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

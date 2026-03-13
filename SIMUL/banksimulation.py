"""
Bank Queue Simulation

A discrete-event simulation of a bank with multiple tellers serving customers.
Demonstrates the use of the simulator framework for queueing system analysis.

Features:
- Configurable number of tellers
- Pluggable random distributions for arrivals and service times
- Running statistics with mean and standard deviation
- Visual queue display

Original: 04/03/02
Updated: Python 3.12+ with type hints, dataclasses, and RandomVariable abstraction
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from simulator import SchedulableObject, Simulation
from random_variable import (
    RandomVariable,
    Exponential,
    Constant,
    Uniform,
    Erlang,
    LogNormal,
)


@dataclass
class SimulationConfig:
    """
    Configuration parameters for the bank simulation.

    Uses RandomVariable instances for flexible distribution specification.
    """

    # Random variables for stochastic processes
    interarrival_time: RandomVariable = field(default_factory=lambda: Exponential(1.0))
    service_time: RandomVariable = field(default_factory=lambda: Exponential(5.0))

    # Bank configuration
    number_tellers: int = 5

    # Reproducibility
    deterministic: bool = True
    random_seed: int = 32

    def __post_init__(self) -> None:
        if self.deterministic:
            RandomVariable.set_global_seed(self.random_seed)

    def __str__(self) -> str:
        return (
            f"SimulationConfig(\n"
            f"  interarrival_time={self.interarrival_time},\n"
            f"  service_time={self.service_time},\n"
            f"  number_tellers={self.number_tellers},\n"
            f"  deterministic={self.deterministic},\n"
            f"  random_seed={self.random_seed}\n"
            f")"
        )


# Global configuration instance
CONFIG: SimulationConfig = SimulationConfig()


class Teller:
    """A bank teller who serves customers."""

    def __init__(self) -> None:
        self.idle: bool = True
        self.customer: Optional[Customer] = None

    def add_customer(self, customer: Customer) -> None:
        """Assign a customer to this teller."""
        self.customer = customer
        self.idle = False

    def remove_customer(self) -> None:
        """Remove the current customer (service complete)."""
        self.idle = True
        self.customer = None

    def __str__(self) -> str:
        return "." if self.idle else "C"


@dataclass
class BankStatistics:
    """
    Statistics collector for bank simulation.

    Tracks customer count, total time, and computes running mean and
    standard deviation using Welford's online algorithm.
    """

    ncust: int = 0
    total_time: float = 0.0
    mean: float = 0.0  # Running mean
    ssq: float = 0.0  # Sum of squared deviations (divide by n-1 for variance)

    def add(self, time: float) -> None:
        """Record a customer's time in the system."""
        self.ncust += 1
        self.total_time += time

        # Welford's online algorithm for running mean and variance
        deviation = time - self.mean
        self.mean += deviation / self.ncust
        self.ssq += deviation * (time - self.mean)

    @property
    def average_time(self) -> float:
        """Calculate the average time spent per customer."""
        return self.total_time / self.ncust if self.ncust > 0 else 0.0

    @property
    def standard_deviation(self) -> float:
        """Calculate the sample standard deviation."""
        if self.ncust < 2:
            return 0.0
        return math.sqrt(self.ssq / (self.ncust - 1))

    @property
    def variance(self) -> float:
        """Calculate the sample variance."""
        if self.ncust < 2:
            return 0.0
        return self.ssq / (self.ncust - 1)

    def print_summary(self) -> None:
        """Print detailed statistics summary."""
        print(f"Running average: {self.mean:.4f} minutes")
        print(f"Standard deviation: {self.standard_deviation:.4f} minutes")
        print(f"Variance: {self.variance:.4f}")

    def __str__(self) -> str:
        self.print_summary()
        return f"{self.ncust} customers. Average time: {self.average_time:.2f} minutes"


class Bank(SchedulableObject):
    """
    A bank with multiple tellers and a waiting queue.

    Customers are served FIFO (First In, First Out).
    """

    def __init__(self, number_tellers: int) -> None:
        super().__init__()
        self.waiting_queue: list[Customer] = []
        self.tellers = [Teller() for _ in range(number_tellers)]
        self.stats = BankStatistics()

    def add_customer(self, customer: Customer) -> None:
        """
        Add a customer to the bank.

        If a teller is available, service begins immediately.
        Otherwise, the customer joins the waiting queue.
        """
        # Find the first idle teller
        for teller in self.tellers:
            if teller.idle:
                teller.add_customer(customer)
                service_time = CONFIG.service_time.sample()
                customer.schedule(service_time, Customer.departure, teller)
                return

        # No teller is idle - customer must wait
        self.waiting_queue.append(customer)

    def remove_customer(self, customer: Customer, teller: Teller) -> None:
        """
        Remove a customer after service completion.

        Records statistics and assigns the next waiting customer (if any).
        """
        # Record time in system (arrival to departure)
        time_in_system = self.sim_engine.now - customer.arrival_time
        self.stats.add(time_in_system)

        teller.remove_customer()

        # Serve next customer in queue if any
        if self.waiting_queue:
            next_customer = self.waiting_queue.pop(0)  # FIFO
            teller.idle = False
            teller.customer = next_customer
            service_time = CONFIG.service_time.sample()
            next_customer.schedule(service_time, Customer.departure, teller)

    @property
    def queue_length(self) -> int:
        """Return the current number of customers waiting."""
        return len(self.waiting_queue)

    @property
    def busy_tellers(self) -> int:
        """Return the number of currently busy tellers."""
        return sum(1 for t in self.tellers if not t.idle)

    @property
    def utilization(self) -> float:
        """Return the current teller utilization (0.0 to 1.0)."""
        return self.busy_tellers / len(self.tellers) if self.tellers else 0.0

    def __str__(self) -> str:
        """Visual representation: tellers | waiting queue."""
        teller_status = "".join(str(t) for t in self.tellers)
        queue_display = "C" * len(self.waiting_queue)
        return f"{teller_status}#{queue_display}"


class Customer(SchedulableObject):
    """A customer arriving at the bank."""

    def __init__(self, arrival_time: float) -> None:
        super().__init__()
        self.arrival_time = arrival_time

    def arrival(self) -> None:
        """Handle customer arrival event."""
        # Add this customer to the bank
        self.sim_engine.bank.add_customer(self)

        # Schedule the arrival of the next customer
        interarrival = CONFIG.interarrival_time.sample()
        next_arrival_time = self.sim_engine.now + interarrival
        Customer(next_arrival_time).schedule(interarrival, Customer.arrival)

    def departure(self, teller: Teller) -> None:
        """Handle customer departure event."""
        self.sim_engine.bank.remove_customer(self, teller)


class BankSimulation(Simulation):
    """
    The main simulation engine for the bank simulation.

    Manages initialization, updates, and termination.
    """

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        super().__init__()
        global CONFIG
        if config is not None:
            CONFIG = config
        self.bank = Bank(number_tellers=CONFIG.number_tellers)

    def initialize(self) -> None:
        """Initialize the simulation with the first customer arrival."""
        Customer(self.now).arrival()

    def update(self) -> None:
        """Called after each event - displays the current bank state."""
        print(self.bank)

    def terminate(self) -> None:
        """Called at simulation end - prints final statistics."""
        print("\n" + "=" * 50)
        print("SIMULATION COMPLETE")
        print("=" * 50)
        print(self.bank.stats)


def run_simulation(
    duration_hours: float = 9.0,
    start_time: float = 0.0,
    config: Optional[SimulationConfig] = None
) -> BankStatistics:
    """
    Run a bank simulation and return the statistics.

    Args:
        duration_hours: How long to run the simulation (in hours)
        start_time: Start time in minutes (default 0)
        config: Optional simulation configuration

    Returns:
        BankStatistics object with the simulation results
    """
    MINUTES_PER_HOUR = 60
    sim = BankSimulation(config)
    end_time = start_time + (duration_hours * MINUTES_PER_HOUR)
    sim.run(start_time, end_time)
    return sim.bank.stats


def main() -> None:
    """Main entry point for the bank simulation."""
    # Configuration with RandomVariable instances
    config = SimulationConfig(
        interarrival_time=Exponential(_mean=1.0),  # 1 minute average between arrivals
        service_time=Exponential(_mean=5.0),       # 5 minutes average service time
        number_tellers=5,
        deterministic=True,
        random_seed=32
    )

    # Time configuration
    MINUTES_PER_HOUR = 60
    simulation_hours = 9  # Run for 9 hours

    print("=" * 60)
    print("BANK QUEUE SIMULATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Tellers: {config.number_tellers}")
    print(f"  Interarrival time: {config.interarrival_time}")
    print(f"    (theoretical mean: {config.interarrival_time.mean():.2f} min)")
    print(f"  Service time: {config.service_time}")
    print(f"    (theoretical mean: {config.service_time.mean():.2f} min)")
    print(f"  Simulation duration: {simulation_hours} hours")
    print(f"  Deterministic mode: {config.deterministic}")
    if config.deterministic:
        print(f"  Random seed: {config.random_seed}")
    print("=" * 60)
    print("\nBank state: [Tellers]#[Waiting Queue]")
    print("  '.' = idle teller, 'C' = customer\n")

    # Run simulation
    stats = run_simulation(
        duration_hours=simulation_hours,
        start_time=0,
        config=config
    )

    # Expected result with seed=32:
    # 156 customers with average busy time of ~8.69 minutes


def demo_different_distributions() -> None:
    """Demonstrate using different distributions for service times."""
    print("\n" + "=" * 60)
    print("DISTRIBUTION COMPARISON DEMO")
    print("=" * 60)

    distributions = [
        ("Exponential(5)", Exponential(_mean=5.0)),
        ("Constant(5)", Constant(value=5.0)),
        ("Uniform(2, 8)", Uniform(low=2.0, high=8.0)),
        ("Erlang(k=3, mean=5)", Erlang(k=3, _mean=5.0)),
    ]

    for name, dist in distributions:
        config = SimulationConfig(
            interarrival_time=Exponential(_mean=1.0),
            service_time=dist,
            number_tellers=5,
            deterministic=True,
            random_seed=42
        )

        # Suppress per-event output for comparison
        class QuietBankSimulation(BankSimulation):
            def update(self) -> None:
                pass  # Silent
            def terminate(self) -> None:
                pass  # Silent

        sim = QuietBankSimulation(config)
        sim.run(0, 3 * 60)  # 3 hours

        stats = sim.bank.stats
        print(f"\n{name}:")
        print(f"  Theoretical mean: {dist.mean():.2f} min")
        print(f"  Theoretical std:  {dist.std():.2f} min")
        print(f"  Customers served: {stats.ncust}")
        print(f"  Avg time in system: {stats.average_time:.2f} min")
        print(f"  Std deviation: {stats.standard_deviation:.2f} min")


if __name__ == "__main__":
    main()
    # Uncomment to see distribution comparison:
    # demo_different_distributions()

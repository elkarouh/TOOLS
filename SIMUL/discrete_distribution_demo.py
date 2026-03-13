"""
Discrete Distribution and Network Generation Demo

Demonstrates:
1. DiscreteDistribution - mutable weighted sampling
2. AliasMethod - O(1) sampling for static distributions
3. Categorical - convenience wrapper for integer categories
4. Barabási-Albert preferential attachment model
5. Other network models (Erdős-Rényi, Watts-Strogatz)
6. Performance comparison

Usage:
    python discrete_distribution_demo.py
"""

from __future__ import annotations

import time
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Set, Tuple

from random_variable import (
    DiscreteDistribution,
    AliasMethod,
    Categorical,
    RandomVariable,
    Uniform,
)


# ============================================================================
# BASIC DEMONSTRATIONS
# ============================================================================

def demo_discrete_distribution():
    """Demonstrate basic DiscreteDistribution functionality."""
    print("=" * 70)
    print("DISCRETE DISTRIBUTION DEMO")
    print("=" * 70)
    
    # Create distribution
    print("\n1. BASIC USAGE")
    print("-" * 50)
    
    dist = DiscreteDistribution({'a': 6, 'b': 1, 'c': 3})
    print(f"Created: {dist}")
    print(f"Total weight: {dist.total}")
    print(f"PDF: {dist.pdf()}")
    print(f"Mode: {dist.mode()}")
    print(f"Entropy: {dist.entropy():.4f} nats")
    
    # Sampling
    print("\n2. SAMPLING")
    print("-" * 50)
    print(f"Single sample: {dist.sample()}")
    print(f"5 samples: {[dist() for _ in range(5)]}")  # Using __call__
    
    # Verify distribution
    n_samples = 100000
    samples = [dist.sample() for _ in range(n_samples)]
    counts = {k: samples.count(k) for k in dist}
    
    print(f"\nSampling verification ({n_samples:,} samples):")
    print(f"  Expected: a=60%, b=10%, c=30%")
    print(f"  Observed: " + ", ".join(f"{k}={v/n_samples*100:.1f}%" for k, v in sorted(counts.items())))
    
    # Dict-like interface
    print("\n3. DICT-LIKE INTERFACE (Mutable Weights)")
    print("-" * 50)
    
    dist2 = DiscreteDistribution({'x': 1, 'y': 2})
    print(f"Initial: {dict(dist2.items())}")
    
    dist2['z'] = 3
    print(f"After dist['z'] = 3: {dict(dist2.items())}")
    
    dist2['x'] += 4
    print(f"After dist['x'] += 4: {dict(dist2.items())}")
    
    del dist2['y']
    print(f"After del dist['y']: {dict(dist2.items())}")
    
    print(f"'z' in dist: {'z' in dist2}")
    print(f"len(dist): {len(dist2)}")
    
    # Multi-sampling
    print("\n4. MULTI-SAMPLING")
    print("-" * 50)
    
    dist3 = DiscreteDistribution({'A': 1, 'B': 2, 'C': 3, 'D': 4})
    print(f"Distribution: {dict(dist3.items())}")
    print(f"5 samples with replacement: {dist3.sample_n(5, replacement=True)}")
    print(f"3 samples without replacement: {dist3.sample_n(3, replacement=False)}")


def demo_alias_method():
    """Demonstrate AliasMethod for O(1) sampling."""
    print("\n" + "=" * 70)
    print("ALIAS METHOD DEMO (O(1) Sampling)")
    print("=" * 70)
    
    weights = {'rare': 1, 'uncommon': 5, 'common': 20, 'very_common': 74}
    alias = AliasMethod(weights)
    
    print(f"\nDistribution: {weights}")
    print(f"PDF: {alias.pdf()}")
    
    # Verify distribution
    n_samples = 100000
    samples = [alias.sample() for _ in range(n_samples)]
    counts = Counter(samples)
    
    print(f"\nSampling verification ({n_samples:,} samples):")
    for key in weights:
        expected = weights[key]
        observed = counts[key] / n_samples * 100
        print(f"  {key}: expected={expected}%, observed={observed:.1f}%")


def demo_categorical():
    """Demonstrate Categorical distribution."""
    print("\n" + "=" * 70)
    print("CATEGORICAL DISTRIBUTION DEMO")
    print("=" * 70)
    
    # Probabilities for categories 0, 1, 2, 3
    probs = [0.1, 0.5, 0.3, 0.1]
    cat = Categorical(probs)
    
    print(f"\nProbabilities: {probs}")
    print(f"Number of categories: {cat.n_categories}")
    
    # Sample and verify
    n_samples = 10000
    samples = [cat.sample() for _ in range(n_samples)]
    counts = [samples.count(i) for i in range(len(probs))]
    
    print(f"\nSampling verification ({n_samples:,} samples):")
    for i, (expected, observed) in enumerate(zip(probs, counts)):
        print(f"  Category {i}: expected={expected*100:.0f}%, observed={observed/n_samples*100:.1f}%")


def demo_performance_comparison():
    """Compare performance of different sampling methods."""
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    
    # Create distributions of different sizes
    sizes = [10, 100, 1000, 10000]
    n_samples = 100000
    
    print(f"\nSampling {n_samples:,} times from distributions of various sizes:")
    print(f"\n{'Size':>8} {'DiscreteDistribution':>22} {'AliasMethod':>15} {'Speedup':>10}")
    print("-" * 60)
    
    for size in sizes:
        weights = {f'item_{i}': i + 1 for i in range(size)}
        
        # DiscreteDistribution (O(log n) per sample)
        dist = DiscreteDistribution(weights)
        start = time.perf_counter()
        for _ in range(n_samples):
            dist.sample()
        dd_time = time.perf_counter() - start
        
        # AliasMethod (O(1) per sample)
        alias = AliasMethod(weights)
        start = time.perf_counter()
        for _ in range(n_samples):
            alias.sample()
        alias_time = time.perf_counter() - start
        
        speedup = dd_time / alias_time
        print(f"{size:>8} {dd_time:>20.4f}s {alias_time:>14.4f}s {speedup:>9.1f}x")
    
    print("\nNote: Use AliasMethod when distribution is static.")
    print("      Use DiscreteDistribution when weights change frequently.")


# ============================================================================
# NETWORK GENERATION MODELS
# ============================================================================

@dataclass
class NetworkStats:
    """Statistics for a network."""
    n_nodes: int
    n_edges: int
    avg_degree: float
    max_degree: int
    min_degree: int
    degree_distribution: Dict[int, int]
    
    def __str__(self) -> str:
        return (
            f"Nodes: {self.n_nodes}, Edges: {self.n_edges}\n"
            f"Degree: avg={self.avg_degree:.2f}, min={self.min_degree}, max={self.max_degree}"
        )


def compute_network_stats(graph: Dict[int, Set[int]]) -> NetworkStats:
    """Compute statistics for a network."""
    degrees = [len(neighbors) for neighbors in graph.values()]
    return NetworkStats(
        n_nodes=len(graph),
        n_edges=sum(degrees) // 2,
        avg_degree=sum(degrees) / len(degrees) if degrees else 0,
        max_degree=max(degrees) if degrees else 0,
        min_degree=min(degrees) if degrees else 0,
        degree_distribution=Counter(degrees),
    )


def barabasi_albert(n: int, m: int, seed: int = None) -> Dict[int, Set[int]]:
    """
    Generate a Barabási-Albert preferential attachment network.
    
    This creates a scale-free network where the degree distribution
    follows a power law: P(k) ~ k^(-γ), typically with γ ≈ 3.
    
    The key insight: new nodes prefer to attach to well-connected nodes,
    creating a "rich get richer" dynamic.
    
    Args:
        n: Total number of nodes
        m: Number of edges each new node creates
        seed: Random seed
    
    Returns:
        Adjacency list as dict of sets
    """
    if seed is not None:
        RandomVariable.set_global_seed(seed)
    
    if m < 1 or m >= n:
        raise ValueError("m must be >= 1 and < n")
    
    # Start with complete graph of (m+1) nodes
    graph: Dict[int, Set[int]] = defaultdict(set)
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            graph[i].add(j)
            graph[j].add(i)
    
    # Degree distribution for preferential attachment
    # P(connect to node i) ∝ degree(i)
    degrees = DiscreteDistribution()
    for node in range(m + 1):
        degrees[node] = len(graph[node])
    
    # Add remaining nodes one by one
    for new_node in range(m + 1, n):
        # Sample m existing nodes with probability proportional to degree
        # (without replacement to avoid multi-edges)
        targets = degrees.sample_n(m, replacement=False)
        
        for target in targets:
            # Add edge
            graph[new_node].add(target)
            graph[target].add(new_node)
            # Update degree in distribution - THIS IS WHY WE NEED MUTABLE WEIGHTS!
            degrees[target] += 1
        
        # Add new node to degree distribution
        degrees[new_node] = m
    
    return dict(graph)


def erdos_renyi(n: int, p: float, seed: int = None) -> Dict[int, Set[int]]:
    """
    Generate an Erdős-Rényi random graph G(n, p).
    
    Each possible edge exists independently with probability p.
    Results in a Poisson degree distribution (not scale-free).
    
    Args:
        n: Number of nodes
        p: Probability of each edge
        seed: Random seed
    
    Returns:
        Adjacency list as dict of sets
    """
    if seed is not None:
        RandomVariable.set_global_seed(seed)
    
    graph: Dict[int, Set[int]] = {i: set() for i in range(n)}
    rng = Uniform(0, 1)
    
    for i in range(n):
        for j in range(i + 1, n):
            if rng() < p:
                graph[i].add(j)
                graph[j].add(i)
    
    return graph


def watts_strogatz(n: int, k: int, beta: float, seed: int = None) -> Dict[int, Set[int]]:
    """
    Generate a Watts-Strogatz small-world network.
    
    Starts with a ring lattice and randomly rewires edges.
    Creates networks with high clustering and short path lengths.
    
    Args:
        n: Number of nodes
        k: Each node connects to k nearest neighbors (must be even)
        beta: Rewiring probability (0 = regular lattice, 1 = random)
        seed: Random seed
    
    Returns:
        Adjacency list as dict of sets
    """
    if seed is not None:
        RandomVariable.set_global_seed(seed)
    
    if k % 2 != 0:
        raise ValueError("k must be even")
    
    # Start with ring lattice
    graph: Dict[int, Set[int]] = {i: set() for i in range(n)}
    
    for i in range(n):
        for j in range(1, k // 2 + 1):
            neighbor = (i + j) % n
            graph[i].add(neighbor)
            graph[neighbor].add(i)
    
    # Rewire edges
    rng = Uniform(0, 1)
    
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if rng() < beta:
                neighbor = (i + j) % n
                if neighbor in graph[i]:
                    # Choose new target (not self, not already connected)
                    candidates = [
                        x for x in range(n)
                        if x != i and x not in graph[i]
                    ]
                    if candidates:
                        # Use DiscreteDistribution with uniform weights
                        target_dist = DiscreteDistribution({c: 1 for c in candidates})
                        new_neighbor = target_dist.sample()
                        
                        # Rewire
                        graph[i].remove(neighbor)
                        graph[neighbor].remove(i)
                        graph[i].add(new_neighbor)
                        graph[new_neighbor].add(i)
    
    return graph


def demo_network_models():
    """Demonstrate network generation models."""
    print("\n" + "=" * 70)
    print("NETWORK GENERATION MODELS")
    print("=" * 70)
    
    n_nodes = 1000
    
    # Barabási-Albert
    print("\n1. BARABÁSI-ALBERT (Scale-Free Network)")
    print("-" * 50)
    print("   P(connect to node) ∝ degree(node)")
    print("   'Rich get richer' - creates hub nodes")
    
    ba_graph = barabasi_albert(n=n_nodes, m=3, seed=42)
    ba_stats = compute_network_stats(ba_graph)
    print(f"\n   {ba_stats}")
    
    # Show degree distribution (power law)
    print("\n   Degree distribution (should follow power law):")
    for deg in sorted(ba_stats.degree_distribution.keys())[:10]:
        count = ba_stats.degree_distribution[deg]
        bar = "█" * min(count // 10, 40)
        print(f"   degree {deg:3d}: {count:4d} {bar}")
    print("   ...")
    
    # Top hubs
    hub_nodes = sorted(ba_graph.keys(), key=lambda n: len(ba_graph[n]), reverse=True)[:5]
    print(f"\n   Top 5 hub nodes: {[(n, len(ba_graph[n])) for n in hub_nodes]}")
    
    # Erdős-Rényi
    print("\n2. ERDŐS-RÉNYI (Random Graph)")
    print("-" * 50)
    print("   Each edge exists with probability p")
    print("   Degree distribution is Poisson (not scale-free)")
    
    # Match expected edges to BA model
    p = 2 * ba_stats.n_edges / (n_nodes * (n_nodes - 1))
    er_graph = erdos_renyi(n=n_nodes, p=p, seed=42)
    er_stats = compute_network_stats(er_graph)
    print(f"\n   {er_stats}")
    
    # Show degree distribution (Poisson)
    print("\n   Degree distribution (should be Poisson-like):")
    for deg in sorted(er_stats.degree_distribution.keys())[:10]:
        count = er_stats.degree_distribution[deg]
        bar = "█" * min(count // 10, 40)
        print(f"   degree {deg:3d}: {count:4d} {bar}")
    
    # Watts-Strogatz
    print("\n3. WATTS-STROGATZ (Small-World Network)")
    print("-" * 50)
    print("   High clustering + short path lengths")
    print("   Interpolates between regular lattice and random graph")
    
    ws_graph = watts_strogatz(n=n_nodes, k=6, beta=0.3, seed=42)
    ws_stats = compute_network_stats(ws_graph)
    print(f"\n   {ws_stats}")
    
    # Compare key properties
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Nodes':>8} {'Edges':>8} {'Avg Deg':>10} {'Max Deg':>10}")
    print("-" * 60)
    print(f"{'Barabási-Albert':<20} {ba_stats.n_nodes:>8} {ba_stats.n_edges:>8} {ba_stats.avg_degree:>10.2f} {ba_stats.max_degree:>10}")
    print(f"{'Erdős-Rényi':<20} {er_stats.n_nodes:>8} {er_stats.n_edges:>8} {er_stats.avg_degree:>10.2f} {er_stats.max_degree:>10}")
    print(f"{'Watts-Strogatz':<20} {ws_stats.n_nodes:>8} {ws_stats.n_edges:>8} {ws_stats.avg_degree:>10.2f} {ws_stats.max_degree:>10}")
    
    print("\nKey insight: Barabási-Albert has much higher max degree (hubs)")
    print("             due to preferential attachment creating scale-free structure.")


def demo_dynamic_weights_use_case():
    """Show why mutable weights are essential."""
    print("\n" + "=" * 70)
    print("WHY MUTABLE WEIGHTS MATTER")
    print("=" * 70)
    
    print("""
    In the Barabási-Albert model, we need to:
    
    1. Sample nodes with P(node) ∝ degree(node)
    2. After adding an edge, UPDATE the degree of the target node
    3. Add the new node to the distribution
    
    This requires a distribution that can be modified efficiently!
    
    Code example:
    
        degrees = DiscreteDistribution()
        
        # Initialize with existing nodes
        for node in initial_nodes:
            degrees[node] = initial_degree
        
        # Add new nodes with preferential attachment
        for new_node in range(start, n):
            # Sample targets proportional to their degree
            targets = degrees.sample_n(m, replacement=False)
            
            for target in targets:
                add_edge(new_node, target)
                degrees[target] += 1  # <-- DYNAMIC UPDATE!
            
            degrees[new_node] = m  # <-- ADD NEW ELEMENT!
    
    Without mutable weights, we'd need to rebuild the entire
    distribution after each edge addition - O(n) per operation
    instead of O(1)!
    """)
    
    # Demonstrate the updates
    print("Live demonstration:")
    print("-" * 50)
    
    degrees = DiscreteDistribution({'A': 2, 'B': 3, 'C': 1})
    print(f"Initial: {dict(degrees.items())}, total={degrees.total}")
    
    # Simulate preferential attachment
    target = degrees.sample()
    print(f"Sampled (preferential): {target}")
    
    degrees[target] += 1
    print(f"After incrementing {target}: {dict(degrees.items())}, total={degrees.total}")
    
    degrees['D'] = 2
    print(f"After adding D: {dict(degrees.items())}, total={degrees.total}")
    
    print(f"\nAll operations are O(1) for weight updates!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all demonstrations."""
    RandomVariable.set_global_seed(42)
    
    demo_discrete_distribution()
    demo_alias_method()
    demo_categorical()
    demo_performance_comparison()
    demo_network_models()
    demo_dynamic_weights_use_case()
    
    print("\n" + "=" * 70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)
    print("""
    Summary of new classes in random_variable.py:
    
    DiscreteDistribution - Mutable weighted sampling, O(log n) per sample
                          Use for: Barabási-Albert, RL, dynamic scenarios
    
    AliasMethod         - Immutable, O(1) per sample after O(n) setup
                          Use for: Static distributions, high-volume sampling
    
    Categorical         - Convenience wrapper for integer categories
                          Use for: Simple classification, multinomial sampling
    """)


if __name__ == "__main__":
    main()

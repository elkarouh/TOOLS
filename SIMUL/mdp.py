"""
Markov Decision Process (MDP) Framework

Provides tools for defining and solving MDPs:
- Value Iteration
- Policy Iteration
- Q-Learning (model-free)

Includes classic examples:
- Jack's Car Rental (Sutton & Barto Example 4.2)
- Gridworld
- Gambler's Problem

Usage:
    from mdp import MDP, ValueIteration, PolicyIteration

    # Define MDP
    mdp = MDP(states, actions, transition_func, reward_func, gamma=0.9)

    # Solve
    V, policy = ValueIteration(mdp).solve()
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from collections import defaultdict
import itertools

from random_variable import RandomVariable, DiscreteDistribution


# Type variables
State = TypeVar('State', bound=Hashable)
Action = TypeVar('Action', bound=Hashable)


# ============================================================================
# CORE MDP CLASSES
# ============================================================================

@dataclass
class TransitionResult:
    """Result of a state transition."""
    next_state: Any
    probability: float
    reward: float


class MDP(ABC, Generic[State, Action]):
    """
    Abstract base class for Markov Decision Processes.

    An MDP is defined by:
    - S: Set of states
    - A: Set of actions (may depend on state)
    - T(s, a, s'): Transition probability P(s' | s, a)
    - R(s, a, s'): Reward function
    - γ: Discount factor

    Subclasses should implement:
    - states: Property returning all states
    - actions(s): Available actions in state s
    - transitions(s, a): List of (probability, next_state, reward) tuples
    """

    def __init__(self, gamma: float = 0.9):
        """
        Initialize MDP.

        Args:
            gamma: Discount factor (0 < γ ≤ 1)
        """
        if not 0 < gamma <= 1:
            raise ValueError("Discount factor must be in (0, 1]")
        self.gamma = gamma

    @property
    @abstractmethod
    def states(self) -> List[State]:
        """Return list of all states."""
        pass

    @abstractmethod
    def actions(self, state: State) -> List[Action]:
        """Return available actions in given state."""
        pass

    @abstractmethod
    def transitions(self, state: State, action: Action) -> List[TransitionResult]:
        """
        Return possible transitions from (state, action).

        Returns:
            List of TransitionResult(next_state, probability, reward)
        """
        pass

    def is_terminal(self, state: State) -> bool:
        """Check if state is terminal (no actions available)."""
        return len(self.actions(state)) == 0

    @property
    def n_states(self) -> int:
        return len(self.states)

    def expected_reward(self, state: State, action: Action) -> float:
        """Calculate expected immediate reward for (state, action)."""
        return sum(t.probability * t.reward for t in self.transitions(state, action))

    def sample_transition(self, state: State, action: Action) -> Tuple[State, float]:
        """
        Sample a single transition (for simulation/Q-learning).

        Returns:
            (next_state, reward)
        """
        transitions = self.transitions(state, action)

        # Build distribution and sample
        dist = DiscreteDistribution({
            i: t.probability for i, t in enumerate(transitions)
        })
        idx = dist.sample()
        t = transitions[idx]
        return t.next_state, t.reward


# ============================================================================
# TABLE-BASED MDP (for simple problems)
# ============================================================================

class TableMDP(MDP[State, Action]):
    """
    MDP defined by explicit transition and reward tables.

    Useful for small, discrete MDPs where you can enumerate everything.
    """

    def __init__(
        self,
        states: List[State],
        actions: Union[List[Action], Callable[[State], List[Action]]],
        transitions: Dict[Tuple[State, Action], List[Tuple[float, State]]],
        rewards: Union[Dict[Tuple[State, Action, State], float],
                      Callable[[State, Action, State], float]],
        gamma: float = 0.9,
    ):
        """
        Initialize table-based MDP.

        Args:
            states: List of all states
            actions: List of actions OR function(state) -> actions
            transitions: Dict mapping (s, a) -> [(prob, next_state), ...]
            rewards: Dict mapping (s, a, s') -> reward OR function
            gamma: Discount factor
        """
        super().__init__(gamma)
        self._states = states
        self._actions = actions
        self._transitions = transitions
        self._rewards = rewards

    @property
    def states(self) -> List[State]:
        return self._states

    def actions(self, state: State) -> List[Action]:
        if callable(self._actions):
            return self._actions(state)
        return self._actions

    def transitions(self, state: State, action: Action) -> List[TransitionResult]:
        key = (state, action)
        if key not in self._transitions:
            return []

        results = []
        for prob, next_state in self._transitions[key]:
            if callable(self._rewards):
                reward = self._rewards(state, action, next_state)
            else:
                reward = self._rewards.get((state, action, next_state), 0.0)
            results.append(TransitionResult(next_state, prob, reward))

        return results


# ============================================================================
# SOLUTION METHODS
# ============================================================================

@dataclass
class MDPSolution:
    """Solution to an MDP."""

    values: Dict[Any, float]  # V(s) for each state
    policy: Dict[Any, Any]    # π(s) -> action for each state
    q_values: Optional[Dict[Tuple[Any, Any], float]] = None  # Q(s, a)
    iterations: int = 0
    converged: bool = True

    def __str__(self) -> str:
        return (
            f"MDPSolution:\n"
            f"  States: {len(self.values)}\n"
            f"  Iterations: {self.iterations}\n"
            f"  Converged: {self.converged}"
        )


class ValueIteration:
    """
    Value Iteration algorithm for solving MDPs.

    Repeatedly applies Bellman optimality update:
    V(s) = max_a Σ P(s'|s,a) [R(s,a,s') + γV(s')]

    Until convergence, then extracts optimal policy.
    """

    def __init__(self, mdp: MDP, epsilon: float = 1e-6):
        """
        Initialize Value Iteration.

        Args:
            mdp: The MDP to solve
            epsilon: Convergence threshold
        """
        self.mdp = mdp
        self.epsilon = epsilon

    def solve(
        self,
        max_iterations: int = 1000,
        verbose: bool = False,
    ) -> MDPSolution:
        """
        Run Value Iteration.

        Returns:
            MDPSolution with optimal values and policy
        """
        # Initialize values to zero
        V: Dict[State, float] = {s: 0.0 for s in self.mdp.states}

        # Convergence threshold adjusted for discount
        theta = self.epsilon * (1 - self.mdp.gamma) / self.mdp.gamma if self.mdp.gamma < 1 else self.epsilon

        converged = False
        iteration = 0

        for iteration in range(1, max_iterations + 1):
            delta = 0.0

            for s in self.mdp.states:
                if self.mdp.is_terminal(s):
                    continue

                v_old = V[s]

                # Bellman optimality update
                action_values = []
                for a in self.mdp.actions(s):
                    q_value = sum(
                        t.probability * (t.reward + self.mdp.gamma * V[t.next_state])
                        for t in self.mdp.transitions(s, a)
                    )
                    action_values.append(q_value)

                V[s] = max(action_values) if action_values else 0.0
                delta = max(delta, abs(v_old - V[s]))

            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: max delta = {delta:.6f}")

            if delta < theta:
                converged = True
                break

        # Extract optimal policy
        policy = self._extract_policy(V)

        # Compute Q-values
        Q = self._compute_q_values(V)

        if verbose:
            print(f"Converged after {iteration} iterations")

        return MDPSolution(
            values=V,
            policy=policy,
            q_values=Q,
            iterations=iteration,
            converged=converged,
        )

    def _extract_policy(self, V: Dict[State, float]) -> Dict[State, Action]:
        """Extract greedy policy from value function."""
        policy = {}

        for s in self.mdp.states:
            if self.mdp.is_terminal(s):
                policy[s] = None
                continue

            best_action = None
            best_value = float('-inf')

            for a in self.mdp.actions(s):
                q_value = sum(
                    t.probability * (t.reward + self.mdp.gamma * V[t.next_state])
                    for t in self.mdp.transitions(s, a)
                )
                if q_value > best_value:
                    best_value = q_value
                    best_action = a

            policy[s] = best_action

        return policy

    def _compute_q_values(self, V: Dict[State, float]) -> Dict[Tuple[State, Action], float]:
        """Compute Q-values from value function."""
        Q = {}

        for s in self.mdp.states:
            for a in self.mdp.actions(s):
                Q[(s, a)] = sum(
                    t.probability * (t.reward + self.mdp.gamma * V[t.next_state])
                    for t in self.mdp.transitions(s, a)
                )

        return Q


class PolicyIteration:
    """
    Policy Iteration algorithm for solving MDPs.

    Alternates between:
    1. Policy Evaluation: Compute V^π for current policy
    2. Policy Improvement: Update policy greedily w.r.t. V^π

    Often converges in fewer iterations than Value Iteration.
    """

    def __init__(self, mdp: MDP, epsilon: float = 1e-6):
        """
        Initialize Policy Iteration.

        Args:
            mdp: The MDP to solve
            epsilon: Convergence threshold for policy evaluation
        """
        self.mdp = mdp
        self.epsilon = epsilon

    def solve(
        self,
        max_iterations: int = 100,
        max_eval_iterations: int = 1000,
        verbose: bool = False,
    ) -> MDPSolution:
        """
        Run Policy Iteration.

        Returns:
            MDPSolution with optimal values and policy
        """
        # Initialize with random policy
        policy: Dict[State, Action] = {}
        for s in self.mdp.states:
            actions = self.mdp.actions(s)
            if actions:
                policy[s] = random.choice(actions)
            else:
                policy[s] = None

        V: Dict[State, float] = {s: 0.0 for s in self.mdp.states}

        for iteration in range(1, max_iterations + 1):
            # Policy Evaluation
            V = self._policy_evaluation(policy, V, max_eval_iterations, verbose)

            # Policy Improvement
            policy_stable = True
            new_policy = {}

            for s in self.mdp.states:
                if self.mdp.is_terminal(s):
                    new_policy[s] = None
                    continue

                old_action = policy[s]

                # Find best action
                best_action = None
                best_value = float('-inf')

                for a in self.mdp.actions(s):
                    q_value = sum(
                        t.probability * (t.reward + self.mdp.gamma * V[t.next_state])
                        for t in self.mdp.transitions(s, a)
                    )
                    if q_value > best_value:
                        best_value = q_value
                        best_action = a

                new_policy[s] = best_action

                if old_action != best_action:
                    policy_stable = False

            policy = new_policy

            if verbose:
                print(f"Policy Iteration {iteration}: stable = {policy_stable}")

            if policy_stable:
                break

        # Compute Q-values
        Q = {}
        for s in self.mdp.states:
            for a in self.mdp.actions(s):
                Q[(s, a)] = sum(
                    t.probability * (t.reward + self.mdp.gamma * V[t.next_state])
                    for t in self.mdp.transitions(s, a)
                )

        return MDPSolution(
            values=V,
            policy=policy,
            q_values=Q,
            iterations=iteration,
            converged=policy_stable,
        )

    def _policy_evaluation(
        self,
        policy: Dict[State, Action],
        V: Dict[State, float],
        max_iterations: int,
        verbose: bool,
    ) -> Dict[State, float]:
        """Evaluate a policy by solving V^π."""
        theta = self.epsilon * (1 - self.mdp.gamma) / self.mdp.gamma if self.mdp.gamma < 1 else self.epsilon

        for iteration in range(1, max_iterations + 1):
            delta = 0.0

            for s in self.mdp.states:
                if self.mdp.is_terminal(s):
                    continue

                v_old = V[s]
                a = policy[s]

                if a is None:
                    continue

                V[s] = sum(
                    t.probability * (t.reward + self.mdp.gamma * V[t.next_state])
                    for t in self.mdp.transitions(s, a)
                )

                delta = max(delta, abs(v_old - V[s]))

            if delta < theta:
                break

        return V


class QLearning:
    """
    Q-Learning: Model-free reinforcement learning algorithm.

    Learns Q(s, a) directly through experience without knowing
    the transition probabilities.

    Update rule:
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    """

    def __init__(
        self,
        mdp: MDP,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.999,
    ):
        """
        Initialize Q-Learning.

        Args:
            mdp: The MDP to solve
            alpha: Learning rate
            epsilon: Exploration rate (for ε-greedy)
            epsilon_decay: Decay rate for epsilon
        """
        self.mdp = mdp
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Initialize Q-values
        self.Q: Dict[Tuple[State, Action], float] = defaultdict(float)

    def choose_action(self, state: State) -> Action:
        """Choose action using ε-greedy policy."""
        actions = self.mdp.actions(state)

        if not actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(actions)
        else:
            return max(actions, key=lambda a: self.Q[(state, a)])

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
    ) -> float:
        """
        Perform Q-learning update.

        Returns:
            TD error
        """
        # Get max Q-value for next state
        next_actions = self.mdp.actions(next_state)
        if next_actions:
            max_next_q = max(self.Q[(next_state, a)] for a in next_actions)
        else:
            max_next_q = 0.0

        # TD error
        td_target = reward + self.mdp.gamma * max_next_q
        td_error = td_target - self.Q[(state, action)]

        # Update
        self.Q[(state, action)] += self.alpha * td_error

        return td_error

    def train_episode(self, start_state: Optional[State] = None, max_steps: int = 1000) -> float:
        """
        Train for one episode.

        Returns:
            Total reward for the episode
        """
        if start_state is None:
            start_state = random.choice(self.mdp.states)

        state = start_state
        total_reward = 0.0

        for step in range(max_steps):
            if self.mdp.is_terminal(state):
                break

            action = self.choose_action(state)
            next_state, reward = self.mdp.sample_transition(state, action)

            self.update(state, action, reward, next_state)

            total_reward += reward
            state = next_state

        # Decay epsilon
        self.epsilon *= self.epsilon_decay

        return total_reward

    def train(
        self,
        n_episodes: int = 1000,
        verbose: bool = False,
    ) -> MDPSolution:
        """
        Train Q-Learning for multiple episodes.

        Returns:
            MDPSolution with learned Q-values and policy
        """
        rewards = []

        for episode in range(1, n_episodes + 1):
            reward = self.train_episode()
            rewards.append(reward)

            if verbose and episode % 100 == 0:
                avg_reward = sum(rewards[-100:]) / 100
                print(f"Episode {episode}: avg reward = {avg_reward:.2f}, ε = {self.epsilon:.3f}")

        # Extract policy and values from Q
        policy = {}
        values = {}

        for s in self.mdp.states:
            actions = self.mdp.actions(s)
            if actions:
                policy[s] = max(actions, key=lambda a: self.Q[(s, a)])
                values[s] = self.Q[(s, policy[s])]
            else:
                policy[s] = None
                values[s] = 0.0

        return MDPSolution(
            values=values,
            policy=policy,
            q_values=dict(self.Q),
            iterations=n_episodes,
            converged=True,
        )


# ============================================================================
# JACK'S CAR RENTAL PROBLEM
# ============================================================================

class JacksCarRental(MDP[Tuple[int, int], int]):
    """
    Jack's Car Rental Problem (Sutton & Barto Example 4.2)

    Jack manages two car rental locations. Each day:
    - Customers arrive to rent cars (Poisson distributed)
    - Customers return cars (Poisson distributed)
    - Jack earns $10 per car rented
    - Jack can move cars overnight for $2 per car
    - Max 20 cars per location, max 5 cars moved

    State: (cars_at_loc1, cars_at_loc2)
    Action: cars moved from loc1 to loc2 (negative = other direction)
    """

    def __init__(
        self,
        max_cars: int = 20,
        max_move: int = 5,
        rental_reward: float = 10.0,
        move_cost: float = 2.0,
        request_lambda: Tuple[float, float] = (3, 4),
        return_lambda: Tuple[float, float] = (3, 2),
        gamma: float = 0.9,
        poisson_cutoff: int = 12,  # Truncate Poisson at this value
    ):
        """
        Initialize Jack's Car Rental problem.

        Args:
            max_cars: Maximum cars at each location
            max_move: Maximum cars to move overnight
            rental_reward: Reward per car rented
            move_cost: Cost per car moved
            request_lambda: Poisson λ for rental requests (loc1, loc2)
            return_lambda: Poisson λ for returns (loc1, loc2)
            gamma: Discount factor
            poisson_cutoff: Truncate Poisson distribution
        """
        super().__init__(gamma)

        self.max_cars = max_cars
        self.max_move = max_move
        self.rental_reward = rental_reward
        self.move_cost = move_cost
        self.request_lambda = request_lambda
        self.return_lambda = return_lambda
        self.poisson_cutoff = poisson_cutoff

        # Precompute Poisson probabilities
        self._poisson_cache: Dict[Tuple[int, float], float] = {}

        # Precompute transitions for speed
        self._transition_cache: Dict[Tuple[Tuple[int, int], int], List[TransitionResult]] = {}

    def _poisson_prob(self, n: int, lam: float) -> float:
        """Get Poisson probability P(X=n) with caching."""
        key = (n, lam)
        if key not in self._poisson_cache:
            if n < 0:
                self._poisson_cache[key] = 0.0
            else:
                self._poisson_cache[key] = math.exp(-lam) * (lam ** n) / math.factorial(n)
        return self._poisson_cache[key]

    def _poisson_tail(self, n: int, lam: float) -> float:
        """P(X >= n) for Poisson."""
        return 1.0 - sum(self._poisson_prob(i, lam) for i in range(n))

    @property
    def states(self) -> List[Tuple[int, int]]:
        """All possible states (cars_loc1, cars_loc2)."""
        return [
            (n1, n2)
            for n1 in range(self.max_cars + 1)
            for n2 in range(self.max_cars + 1)
        ]

    def actions(self, state: Tuple[int, int]) -> List[int]:
        """
        Available actions: move cars from loc1 to loc2.
        Positive = loc1 -> loc2, Negative = loc2 -> loc1
        """
        n1, n2 = state
        actions = []

        for move in range(-self.max_move, self.max_move + 1):
            # Check if move is valid
            new_n1 = n1 - move
            new_n2 = n2 + move

            if 0 <= new_n1 <= self.max_cars and 0 <= new_n2 <= self.max_cars:
                actions.append(move)

        return actions

    def transitions(self, state: Tuple[int, int], action: int) -> List[TransitionResult]:
        """
        Compute transition probabilities.

        This is the complex part - we need to consider:
        1. Cars after moving
        2. Rental requests (Poisson)
        3. Returns (Poisson)
        """
        cache_key = (state, action)
        if cache_key in self._transition_cache:
            return self._transition_cache[cache_key]

        n1, n2 = state

        # After moving cars
        n1_after_move = n1 - action
        n2_after_move = n2 + action

        # Moving cost
        move_cost = self.move_cost * abs(action)

        # Compute distribution of (final_n1, final_n2, rental_reward)
        results: Dict[Tuple[int, int], Tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))

        cutoff = self.poisson_cutoff

        # Iterate over possible requests and returns
        for req1 in range(cutoff + 1):
            p_req1 = self._poisson_prob(req1, self.request_lambda[0])
            if req1 == cutoff:
                p_req1 = self._poisson_tail(cutoff, self.request_lambda[0])

            for req2 in range(cutoff + 1):
                p_req2 = self._poisson_prob(req2, self.request_lambda[1])
                if req2 == cutoff:
                    p_req2 = self._poisson_tail(cutoff, self.request_lambda[1])

                # Actual rentals (limited by available cars)
                actual_rent1 = min(n1_after_move, req1)
                actual_rent2 = min(n2_after_move, req2)

                # Reward from rentals
                rental_reward = self.rental_reward * (actual_rent1 + actual_rent2)

                # Cars after rentals
                n1_after_rent = n1_after_move - actual_rent1
                n2_after_rent = n2_after_move - actual_rent2

                for ret1 in range(cutoff + 1):
                    p_ret1 = self._poisson_prob(ret1, self.return_lambda[0])
                    if ret1 == cutoff:
                        p_ret1 = self._poisson_tail(cutoff, self.return_lambda[0])

                    for ret2 in range(cutoff + 1):
                        p_ret2 = self._poisson_prob(ret2, self.return_lambda[1])
                        if ret2 == cutoff:
                            p_ret2 = self._poisson_tail(cutoff, self.return_lambda[1])

                        # Final car counts (capped at max)
                        final_n1 = min(n1_after_rent + ret1, self.max_cars)
                        final_n2 = min(n2_after_rent + ret2, self.max_cars)

                        # Probability
                        prob = p_req1 * p_req2 * p_ret1 * p_ret2

                        # Accumulate
                        next_state = (final_n1, final_n2)
                        old_prob, old_reward = results[next_state]

                        # Weighted average of rewards
                        total_reward = rental_reward - move_cost
                        new_prob = old_prob + prob
                        new_reward = (old_prob * old_reward + prob * total_reward) / new_prob if new_prob > 0 else 0

                        results[next_state] = (new_prob, new_reward)

        # Convert to TransitionResult list
        transition_list = [
            TransitionResult(next_state=ns, probability=prob, reward=reward)
            for ns, (prob, reward) in results.items()
            if prob > 1e-10
        ]

        self._transition_cache[cache_key] = transition_list
        return transition_list


# ============================================================================
# GRIDWORLD
# ============================================================================

class GridWorld(MDP[Tuple[int, int], str]):
    """
    Classic GridWorld MDP.

    Agent navigates a grid with:
    - Walls (impassable)
    - Goals (terminal states with reward)
    - Traps (negative reward)
    - Regular cells (small negative reward to encourage efficiency)

    Actions: 'up', 'down', 'left', 'right'
    """

    def __init__(
        self,
        width: int = 5,
        height: int = 5,
        walls: Optional[Set[Tuple[int, int]]] = None,
        goals: Optional[Dict[Tuple[int, int], float]] = None,
        traps: Optional[Dict[Tuple[int, int], float]] = None,
        step_cost: float = -0.04,
        gamma: float = 0.9,
        slip_prob: float = 0.0,  # Probability of moving perpendicular
    ):
        """
        Initialize GridWorld.

        Args:
            width, height: Grid dimensions
            walls: Set of wall positions
            goals: Dict mapping goal positions to rewards
            traps: Dict mapping trap positions to rewards
            step_cost: Reward for each step (usually negative)
            gamma: Discount factor
            slip_prob: Probability of slipping (stochastic movement)
        """
        super().__init__(gamma)

        self.width = width
        self.height = height
        self.walls = walls or set()
        self.goals = goals or {(width-1, height-1): 1.0}
        self.traps = traps or {}
        self.step_cost = step_cost
        self.slip_prob = slip_prob

        self._actions = ['up', 'down', 'left', 'right']
        self._deltas = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0),
        }
        self._perpendicular = {
            'up': ['left', 'right'],
            'down': ['left', 'right'],
            'left': ['up', 'down'],
            'right': ['up', 'down'],
        }

    @property
    def states(self) -> List[Tuple[int, int]]:
        """All non-wall grid positions."""
        return [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in self.walls
        ]

    def actions(self, state: Tuple[int, int]) -> List[str]:
        """Available actions (none in terminal states)."""
        if state in self.goals or state in self.traps:
            return []
        return self._actions

    def _move(self, state: Tuple[int, int], action: str) -> Tuple[int, int]:
        """Compute next state from action (with wall collision)."""
        dx, dy = self._deltas[action]
        new_x = state[0] + dx
        new_y = state[1] + dy

        # Check bounds and walls
        if (0 <= new_x < self.width and
            0 <= new_y < self.height and
            (new_x, new_y) not in self.walls):
            return (new_x, new_y)
        else:
            return state  # Stay in place

    def transitions(self, state: Tuple[int, int], action: str) -> List[TransitionResult]:
        """Compute transitions with possible slipping."""
        results = []

        # Intended direction
        intended_prob = 1.0 - self.slip_prob
        next_state = self._move(state, action)
        reward = self.step_cost

        if next_state in self.goals:
            reward = self.goals[next_state]
        elif next_state in self.traps:
            reward = self.traps[next_state]

        results.append(TransitionResult(next_state, intended_prob, reward))

        # Perpendicular slipping
        if self.slip_prob > 0:
            slip_per_dir = self.slip_prob / 2
            for perp_action in self._perpendicular[action]:
                perp_state = self._move(state, perp_action)
                perp_reward = self.step_cost

                if perp_state in self.goals:
                    perp_reward = self.goals[perp_state]
                elif perp_state in self.traps:
                    perp_reward = self.traps[perp_state]

                results.append(TransitionResult(perp_state, slip_per_dir, perp_reward))

        return results

    def display(self, values: Optional[Dict] = None, policy: Optional[Dict] = None) -> str:
        """Create ASCII visualization of the gridworld."""
        policy_arrows = {
            'up': '↑', 'down': '↓', 'left': '←', 'right': '→', None: '·'
        }

        lines = []
        for y in range(self.height - 1, -1, -1):
            row = []
            for x in range(self.width):
                pos = (x, y)
                if pos in self.walls:
                    cell = '███'
                elif pos in self.goals:
                    cell = ' G '
                elif pos in self.traps:
                    cell = ' X '
                elif policy and pos in policy:
                    cell = f' {policy_arrows.get(policy[pos], "?")} '
                elif values and pos in values:
                    cell = f'{values[pos]:+.1f}'[:4].center(4)
                else:
                    cell = ' · '
                row.append(cell)
            lines.append('│'.join(row))

        separator = '┼'.join(['───'] * self.width)
        return ('\n' + separator + '\n').join(lines)


# ============================================================================
# GAMBLER'S PROBLEM
# ============================================================================

class GamblersProblem(MDP[int, int]):
    """
    Gambler's Problem (Sutton & Barto Example 4.3)

    A gambler bets on coin flips:
    - Win: money doubles
    - Lose: money lost
    - Goal: reach $100

    State: current capital (1-99)
    Action: stake amount (1 to min(capital, 100-capital))
    """

    def __init__(
        self,
        goal: int = 100,
        p_heads: float = 0.4,
        gamma: float = 1.0,
    ):
        """
        Initialize Gambler's Problem.

        Args:
            goal: Target capital
            p_heads: Probability of winning a bet
            gamma: Discount factor (usually 1.0 for episodic)
        """
        super().__init__(gamma)

        self.goal = goal
        self.p_heads = p_heads

    @property
    def states(self) -> List[int]:
        """States 0 to goal (0 and goal are terminal)."""
        return list(range(self.goal + 1))

    def actions(self, state: int) -> List[int]:
        """Possible stakes from current capital."""
        if state == 0 or state == self.goal:
            return []
        return list(range(1, min(state, self.goal - state) + 1))

    def transitions(self, state: int, action: int) -> List[TransitionResult]:
        """Win or lose the bet."""
        win_state = min(state + action, self.goal)
        lose_state = state - action

        win_reward = 1.0 if win_state == self.goal else 0.0

        return [
            TransitionResult(win_state, self.p_heads, win_reward),
            TransitionResult(lose_state, 1 - self.p_heads, 0.0),
        ]


# ============================================================================
# DEMONSTRATIONS
# ============================================================================

def demo_gridworld():
    """Demonstrate GridWorld MDP."""
    print("=" * 70)
    print("GRIDWORLD MDP")
    print("=" * 70)

    # Create a simple gridworld
    gw = GridWorld(
        width=4,
        height=3,
        walls={(1, 1)},
        goals={(3, 2): 1.0},
        traps={(3, 1): -1.0},
        step_cost=-0.04,
        gamma=0.9,
        slip_prob=0.2,  # 20% chance of slipping
    )

    print("\nGridWorld (4x3 with walls, goals, traps):")
    print("G = Goal (+1), X = Trap (-1), █ = Wall")
    print()
    print(gw.display())

    # Solve with Value Iteration
    print("\n\nSolving with Value Iteration...")
    vi = ValueIteration(gw)
    solution = vi.solve(verbose=False)

    print(f"\nOptimal Policy:")
    print(gw.display(policy=solution.policy))

    print(f"\nValue Function:")
    print(gw.display(values=solution.values))

    print(f"\nIterations: {solution.iterations}")


def demo_gamblers_problem():
    """Demonstrate Gambler's Problem."""
    print("\n" + "=" * 70)
    print("GAMBLER'S PROBLEM")
    print("=" * 70)

    print("""
    A gambler bets on coin flips (P(heads) = 0.4):
    - Win: double the stake
    - Lose: lose the stake
    - Goal: reach $100 from current capital

    What's the optimal betting strategy?
    """)

    gp = GamblersProblem(goal=100, p_heads=0.4, gamma=1.0)

    vi = ValueIteration(gp)
    solution = vi.solve(verbose=False)

    print("Optimal Policy (stake for each capital level):")
    print("-" * 50)

    # Show policy at selected capitals
    capitals = [1, 10, 25, 50, 75, 99]
    for c in capitals:
        stake = solution.policy[c]
        value = solution.values[c]
        print(f"  Capital ${c:2d}: bet ${stake:2d} (P(win) = {value:.4f})")

    print(f"\nNote: Optimal strategy often bets conservatively, but")
    print(f"      sometimes 'goes all in' at specific capital levels.")


def demo_jacks_car_rental():
    """Demonstrate Jack's Car Rental problem."""
    print("\n" + "=" * 70)
    print("JACK'S CAR RENTAL PROBLEM")
    print("=" * 70)

    print("""
    Jack manages two car rental locations:
    - Earns $10 per car rented
    - Pays $2 per car moved overnight
    - Max 20 cars per location, max 5 cars moved
    - Requests: Poisson(λ=3) and Poisson(λ=4)
    - Returns: Poisson(λ=3) and Poisson(λ=2)
    """)

    # Use smaller problem for speed
    print("Solving with Value Iteration (10x10 grid for demo)...")

    jack = JacksCarRental(
        max_cars=10,  # Reduced for demo speed
        max_move=3,
        gamma=0.9,
        poisson_cutoff=8,
    )

    vi = ValueIteration(jack, epsilon=0.1)
    solution = vi.solve(max_iterations=100, verbose=True)

    print(f"\n{solution}")

    # Display policy as grid
    print("\nOptimal Policy (cars to move from loc1 to loc2):")
    print("Positive = loc1→loc2, Negative = loc2→loc1")
    print()

    max_cars = jack.max_cars
    print("     " + "".join(f"{j:4d}" for j in range(max_cars + 1)))
    print("    +" + "-" * (4 * (max_cars + 1)))

    for i in range(max_cars, -1, -1):
        row = f"{i:3d} |"
        for j in range(max_cars + 1):
            action = solution.policy.get((i, j), 0)
            if action is None:
                action = 0
            row += f"{action:4d}"
        print(row)

    print("\n     (columns = cars at loc1, rows = cars at loc2)")

    # Show some specific states
    print("\nExample optimal actions:")
    examples = [(10, 10), (5, 5), (10, 0), (0, 10)]
    for n1, n2 in examples:
        if (n1, n2) in solution.policy:
            action = solution.policy[(n1, n2)]
            value = solution.values[(n1, n2)]
            print(f"  State ({n1}, {n2}): move {action:+d} cars, V={value:.1f}")


def demo_q_learning():
    """Demonstrate Q-Learning."""
    print("\n" + "=" * 70)
    print("Q-LEARNING (Model-Free)")
    print("=" * 70)

    # Simple gridworld for Q-learning
    gw = GridWorld(
        width=4,
        height=3,
        walls={(1, 1)},
        goals={(3, 2): 1.0},
        traps={(3, 1): -1.0},
        step_cost=-0.04,
        gamma=0.9,
        slip_prob=0.0,  # Deterministic for easier learning
    )

    print("\nLearning optimal policy through exploration...")

    ql = QLearning(gw, alpha=0.1, epsilon=0.3, epsilon_decay=0.995)
    solution = ql.train(n_episodes=1000, verbose=True)

    print(f"\nLearned Policy:")
    print(gw.display(policy=solution.policy))

    # Compare with Value Iteration
    print("\nComparing with Value Iteration solution:")
    vi = ValueIteration(gw)
    vi_solution = vi.solve()
    print(gw.display(policy=vi_solution.policy))

    # Count matches
    matches = sum(
        1 for s in gw.states
        if solution.policy.get(s) == vi_solution.policy.get(s)
    )
    print(f"\nPolicy match: {matches}/{len(gw.states)} states")


def main():
    """Run all demonstrations."""
    random.seed(42)
    RandomVariable.set_global_seed(42)

    demo_gridworld()
    demo_gamblers_problem()
    demo_jacks_car_rental()
    demo_q_learning()

    print("\n" + "=" * 70)
    print("ALL MDP DEMONSTRATIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

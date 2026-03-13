"""
Discrete-Event Simulation Framework

A base module for performing discrete-time simulations with support for:
- Event scheduling (one-time and periodic)
- Cooperative multithreading via generators
- Resource management with blocking
- Condition variables for synchronization
- Statistics collection and histogram generation

TODO: add a "delete_timer" method
TODO: add priorities to microthreading
TODO: add signal handling
"""

from __future__ import annotations

import heapq
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import GeneratorType
from typing import Any, Callable, Generic, TypeVar, Optional

# Time unit constants
MSEC = 1
SEC = 1000 * MSEC
MIN = 60 * SEC
HOUR = 60 * MIN


# ======================== UTILITIES ========================

class PriorityQueue:
    """A priority queue implemented using a heap."""
    
    def __init__(self) -> None:
        self._heap: list[tuple[float, int, Any]] = []
        self._counter = 0
    
    def push(self, item: Any, priority: float) -> None:
        """Add an item with given priority (lower = higher priority)."""
        heapq.heappush(self._heap, (priority, self._counter, item))
        self._counter += 1
    
    def pop(self) -> Any:
        """Remove and return the highest priority item."""
        return heapq.heappop(self._heap)[-1]
    
    def __len__(self) -> int:
        return len(self._heap)
    
    def __bool__(self) -> bool:
        return bool(self._heap)


# ======================== OBSERVER PATTERN ========================

class Observer(ABC):
    """Base class for views in the Observer Pattern."""
    
    @abstractmethod
    def update(self, subject: Subject, *args: Any, **kwargs: Any) -> None:
        """Called when the observed subject changes. Implement in subclass."""
        pass


class Subject:
    """Base class for a model in the Observer Pattern."""
    
    def __init__(self) -> None:
        """Initialize the list of observers. Don't forget to call in derived classes."""
        self._observers: list[Observer] = []
    
    def attach(self, observer: Observer) -> None:
        """Attach an observer to this model."""
        self._observers.append(observer)
    
    def detach(self, observer: Observer) -> None:
        """Detach an observer from this model."""
        self._observers.remove(observer)
    
    def notify(self, *args: Any, **kwargs: Any) -> None:
        """Notify all observers about changes in the model."""
        for observer in self._observers:
            observer.update(self, *args, **kwargs)


# ======================== YIELD OBJECTS ========================

@dataclass
class YieldObject:
    """Yield object for resource blocking."""
    success: bool = True
    number: int = 0
    resource: Optional[Resource] = None


@dataclass
class YieldConditionObject:
    """Yield object for condition variable blocking."""
    success: bool = True
    condition: Optional[Callable[[], bool]] = None
    condition_variable: Optional[ConditionVariable] = None


@dataclass
class Hold:
    """Sleep for the specified amount of time."""
    time: float = 0
    success: bool = field(default=False, init=False)


@dataclass
class Sleep:
    """
    Sleep until awakened.
    
    Usage: in a method of a SchedulableObject called my_object:
        yield Sleep(self)
    
    It can get awakened by:
        my_object.awake()
    """
    obj: SchedulableObject
    success: bool = field(default=False, init=False)


# ======================== EVENTS ========================

class Event:
    """A scheduled event in the simulation."""
    
    def __init__(
        self,
        time: float,
        func: Optional[Callable] = None,
        arg: Optional[list] = None
    ) -> None:
        self.time = time
        self.func = func
        self.arg = arg or []
        self.periodic = False
        self.threaded = False
    
    def __call__(self) -> Any:
        if self.func is not None:
            return self.func(*self.arg)
        return None


class PeriodicEvent(Event):
    """An event that repeats at regular intervals."""
    
    def __init__(
        self,
        time: float,
        period: float,
        func: Optional[Callable] = None,
        arg: Optional[list] = None
    ) -> None:
        super().__init__(time, func, arg)
        self.period = period
        self.periodic = True


class IteratorEvent(Event):
    """An event that resumes a generator/thread."""
    
    def __init__(self, time: float, iterator: GeneratorType) -> None:
        super().__init__(time, iterator)
        self.threaded = True
    
    def __call__(self) -> Any:
        return next(self.func)


# ======================== SIMULATION ENGINE ========================

class Simulation(Subject):
    """
    The main simulation engine.
    
    Manages event scheduling, thread execution, and simulation lifecycle.
    """
    
    object_queue: list[SchedulableObject] = []
    
    def __init__(self) -> None:
        super().__init__()
        self.event_queue = PriorityQueue()
        self.thread_queue: list[GeneratorType] = []
        self.release_queue: list[GeneratorType] = []
        self.waiting_queue: list[GeneratorType] = []
        self.now: float = 0
        self.to: float = 0
        self.stop_requested = False
        # Notify all schedulable objects of this engine
        SchedulableObject.sim_engine = self
    
    def run(self, from_: float = 0, to_: float = 999999999) -> None:
        """Run the simulation from time from_ to time to_."""
        self.now = from_
        self.to = to_
        self.stop_requested = False
        self.initialize()
        
        while True:
            # Process one time event
            try:
                event = self.event_queue.pop()
                self.now = event.time
                
                if event.threaded:
                    # This is a threaded time event
                    self.schedule_thread(event.func)
                else:
                    # This is a regular time event
                    event()
                    if event.periodic:
                        event.time += event.period
                        self.schedule(event)
            except IndexError:
                print("Event queue is empty... terminating!")
                self.terminate()
                return
            
            # Process one time-slice for each current thread
            for thread in self.thread_queue[:]:
                try:
                    result = next(thread)
                    if not result.success:
                        # We're blocking
                        if isinstance(result, YieldObject):
                            # Blocking on a resource
                            result.resource.resource_queue.append((result.number, thread))
                        elif isinstance(result, YieldConditionObject):
                            # Blocking on a condition variable
                            result.condition_variable.waiting_queue.append(
                                (result.condition, thread)
                            )
                        elif isinstance(result, Hold):
                            # This is a hold object
                            self.schedule_hold(result.time, thread)
                        else:
                            # This is a sleep object
                            result.obj.sleep(thread)
                        self.thread_queue.remove(thread)
                except StopIteration:
                    self.thread_queue.remove(thread)
            
            self.update()
            
            if self.stop_requested or self.now > self.to:
                self.terminate()
                return
    
    def schedule(self, event: Event) -> None:
        """Schedule a time event."""
        self.event_queue.push(event, event.time)
    
    def schedule_thread(self, iterator: GeneratorType) -> None:
        """Add a thread to the execution queue."""
        self.thread_queue.append(iterator)
    
    def schedule_release(self, iterator: GeneratorType) -> None:
        """Insert a thread at the beginning of the queue (high priority)."""
        self.thread_queue.insert(0, iterator)
    
    def schedule_hold(self, time: float, iterator: GeneratorType) -> None:
        """Schedule a thread to resume after a delay."""
        ev = IteratorEvent(self.now + time, iterator)
        self.schedule(ev)
    
    def stop_periodic(self, event: PeriodicEvent) -> None:
        """Stop a periodic event from recurring."""
        event.periodic = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the simulation. Must schedule at least one event."""
        raise NotImplementedError("Simulation needs at least one event to start!")
    
    def stop(self) -> None:
        """Request the simulation to stop."""
        self.stop_requested = True
    
    def update(self) -> None:
        """Called after each simulation step. Override to add visualization."""
        for obj in self.object_queue:
            obj.notify()
    
    def terminate(self) -> None:
        """Called when simulation ends. Override to add cleanup/reporting."""
        pass


# ======================== SCHEDULABLE OBJECT ========================

class SchedulableObject(Subject):
    """
    Base class for objects that can schedule events in the simulation.
    
    Provides methods for scheduling events, periodic events, and threads.
    """
    
    sim_engine: Optional[Simulation] = None
    
    def __init__(self) -> None:
        super().__init__()
        self.iterator: Optional[GeneratorType] = None
    
    def schedule(self, time: float, callback: Callable, *args: Any) -> None:
        """Schedule an event to occur after 'time' units."""
        if isinstance(callback, GeneratorType):
            ev = IteratorEvent(self.get_now() + time, callback)
        else:
            ev = Event(self.get_now() + time, callback, [self] + list(args))
        self.sim_engine.schedule(ev)
    
    def schedule_periodic(
        self,
        time: float,
        period: float,
        callback: Callable,
        *args: Any
    ) -> PeriodicEvent:
        """Schedule a periodic event. Returns the event for later cancellation."""
        now = self.sim_engine.now
        ev = PeriodicEvent(now + time, period, callback, [self] + list(args))
        self.sim_engine.schedule(ev)
        return ev
    
    def stop_periodic(self, event: PeriodicEvent) -> None:
        """Stop a periodic event."""
        self.sim_engine.stop_periodic(event)
    
    def schedule_thread(self, iterator: GeneratorType) -> None:
        """Schedule a generator as a cooperative thread."""
        self.sim_engine.schedule_thread(iterator)
    
    def schedule_release(self, iterator: GeneratorType) -> None:
        """Schedule a thread with high priority."""
        self.sim_engine.schedule_release(iterator)
    
    @staticmethod
    def s_schedule(time: float, callback: Callable, *args: Any) -> None:
        """Static method to schedule an event."""
        now = SchedulableObject.sim_engine.now
        ev = Event(now + time, callback, list(args))
        SchedulableObject.sim_engine.schedule(ev)
    
    @staticmethod
    def s_schedule_periodic(
        time: float,
        period: float,
        callback: Callable,
        *args: Any
    ) -> PeriodicEvent:
        """Static method to schedule a periodic event."""
        now = SchedulableObject.sim_engine.now
        ev = PeriodicEvent(now + time, period, callback, list(args))
        SchedulableObject.sim_engine.schedule(ev)
        return ev
    
    def get_now(self) -> float:
        """Get the current simulation time."""
        return self.sim_engine.now
    
    # Alias for backward compatibility
    def now(self) -> float:
        """Get the current simulation time (deprecated, use get_now())."""
        return self.sim_engine.now
    
    def sleep(self, iterator: GeneratorType) -> None:
        """Put a thread to sleep until awakened."""
        self.iterator = iterator
        self.sim_engine.waiting_queue.append(self.iterator)
    
    def awake(self) -> None:
        """Wake up a sleeping thread."""
        if self.iterator is None:
            return
        
        self.sim_engine.waiting_queue.remove(self.iterator)
        iterator = self.iterator
        
        try:
            result = next(iterator)
            if not result.success:
                # We're blocking again
                if isinstance(result, YieldObject):
                    result.resource.resource_queue.append((result.number, iterator))
                elif isinstance(result, YieldConditionObject):
                    result.condition_variable.waiting_queue.append(
                        (result.condition, iterator)
                    )
                elif isinstance(result, Hold):
                    self.sim_engine.schedule_hold(result.time, iterator)
                else:
                    result.obj.sleep(iterator)
        except StopIteration:
            pass


# ======================== RESOURCE MANAGEMENT ========================

class ResourceBusy(Exception):
    """Exception raised when a resource is busy."""
    
    def __init__(
        self,
        success: bool,
        number: int = 0,
        resource: Optional[Resource] = None
    ) -> None:
        super().__init__("Resource is busy")
        self.success = success
        self.number = number
        self.resource = resource
    
    def __call__(self) -> YieldObject:
        return YieldObject(self.success, self.number, self.resource)


class Resource(SchedulableObject):
    """
    A resource that can be acquired and released.
    
    Supports blocking when the resource is unavailable.
    """
    
    def __init__(self, initial_size: int) -> None:
        super().__init__()
        self.size = initial_size
        self.max_size = initial_size
        self.resource_queue: list[tuple[int, GeneratorType]] = []
    
    def acquire(self, number: int = 1) -> None:
        """
        Acquire 'number' units of the resource.
        
        Raises ResourceBusy if not enough units available.
        """
        if self.size >= number:
            self.size -= number
        else:
            raise ResourceBusy(False, number, self)
    
    def release(self, number: int = 1) -> None:
        """Release 'number' units of the resource."""
        self.size = min(self.size + number, self.max_size)
        
        # Check if any waiting threads can now proceed
        for needed, waiter in self.resource_queue[:]:
            if self.size >= needed:
                self.size -= needed
                self.resource_queue.remove((needed, waiter))
                
                try:
                    result = next(waiter)
                    if not result.success:
                        # Re-blocking
                        if isinstance(result, YieldObject):
                            result.resource.resource_queue.append(
                                (result.number, waiter)
                            )
                        elif isinstance(result, YieldConditionObject):
                            result.condition_variable.waiting_queue.append(
                                (result.condition, waiter)
                            )
                        elif isinstance(result, Hold):
                            self.sim_engine.schedule_hold(result.time, waiter)
                        else:
                            result.obj.sleep(waiter)
                except StopIteration:
                    pass
                else:
                    self.schedule_release(waiter)


# ======================== CURRY FUNCTION ========================

def curry(*args: Any, **create_time_kwds: Any) -> Callable:
    """
    Create a curried function with pre-bound arguments.
    
    Example:
        add = lambda x, y: x + y
        add5 = curry(add, 5)
        add5(3)  # Returns 8
    """
    func = args[0]
    create_time_args = args[1:]
    
    def curried_function(*call_time_args: Any, **call_time_kwds: Any) -> Any:
        all_args = create_time_args + call_time_args
        kwds = create_time_kwds.copy()
        kwds.update(call_time_kwds)
        return func(*all_args, **kwds)
    
    return curried_function


# ======================== CONDITION VARIABLES ========================

class ConditionNotFulfilled(Exception):
    """Exception raised when a condition is not fulfilled."""
    
    def __init__(
        self,
        condition: Callable[[], bool],
        condition_variable: ConditionVariable
    ) -> None:
        super().__init__("Condition not fulfilled")
        self.condition = condition
        self.condition_variable = condition_variable
    
    def __call__(self) -> YieldConditionObject:
        return YieldConditionObject(False, self.condition, self.condition_variable)


class ConditionVariable(SchedulableObject):
    """
    A condition variable for thread synchronization.
    
    Example usage:
    
        class MyConditionVariable(ConditionVariable):
            def condition1(self):
                return len(self.buff) > 0
            
            def condition2(self, number_items):
                return len(self.buff) > number_items
        
        my_buffer_is_empty = MyConditionVariable()
        
        def put_element(self, number_items):
            while True:
                try:
                    my_buffer_is_empty.wait_until(my_buffer_is_empty.condition1)
                except ConditionNotFulfilled as exc:
                    yield exc()
                else:
                    break
            # continue here
        
        # A consumer might execute:
        # consume some buffer elements and signal a change to all producers
        my_buffer_is_empty.signal()
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.waiting_queue: list[tuple[Callable[[], bool], GeneratorType]] = []
    
    def wait_until(self, *cond: Any) -> None:
        """
        Wait until the condition is fulfilled.
        
        Raises ConditionNotFulfilled if the condition is not met.
        """
        condition = curry(*cond)
        if not condition():
            raise ConditionNotFulfilled(condition, self)
    
    def signal(self) -> None:
        """Signal all waiting threads to check their conditions."""
        for cond, waiter in self.waiting_queue[:]:
            if cond():
                try:
                    result = next(waiter)
                    if not result.success:
                        # Re-blocking
                        if isinstance(result, YieldObject):
                            result.resource.resource_queue.append(
                                (result.number, waiter)
                            )
                        elif isinstance(result, YieldConditionObject):
                            self.waiting_queue.append((cond, waiter))
                        elif isinstance(result, Hold):
                            self.sim_engine.schedule_hold(result.time, waiter)
                        else:
                            result.obj.sleep(waiter)
                        raise StopIteration
                except StopIteration:
                    pass
                else:
                    self.schedule_thread(waiter)
                
                self.waiting_queue.remove((cond, waiter))


class Entity(SchedulableObject):
    """A basic entity in the simulation."""
    
    def __init__(self, initial_size: int = 0) -> None:
        super().__init__()
    
    def wait(self) -> None:
        """Override in subclass."""
        pass


# ======================== STATISTICS ========================

class Bin:
    """A histogram bin for statistics collection."""
    
    max_weight: int = 0
    min_weight: int = 0
    
    def __init__(self, low_mark: float, high_mark: float) -> None:
        self.low = low_mark
        self.high = high_mark
        self.weight = 0
    
    def add_value(self, value: float) -> bool:
        """Add a value to this bin if it falls within range."""
        if self.low <= value < self.high:
            self.weight += 1
            if self.weight > Bin.max_weight:
                Bin.max_weight = self.weight
            if self.weight < Bin.min_weight:
                Bin.min_weight = self.weight
            return True
        return False
    
    def contains_value(self, value: float) -> bool:
        """Check if a value falls within this bin's range."""
        return self.low <= value < self.high
    
    def __lt__(self, other: Bin) -> bool:
        return self.low < other.low
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bin):
            return NotImplemented
        return self.low == other.low
    
    def __str__(self) -> str:
        return f"[{int(self.low)}, {int(self.high)}]"


class MaxBin(Bin):
    """A bin for the maximum range (to infinity)."""
    
    def __init__(self, low_mark: float) -> None:
        super().__init__(low_mark, float('inf'))
    
    def __str__(self) -> str:
        return f"[{int(self.low)}, +∞]"


class MinBin(Bin):
    """A bin for the minimum range (from negative infinity)."""
    
    def __init__(self, high_mark: float) -> None:
        super().__init__(float('-inf'), high_mark)
    
    def __str__(self) -> str:
        return f"[-∞, {int(self.high)}]"


class Statistics:
    """
    Statistics collector with running mean, variance, and histogram support.
    
    Uses Welford's algorithm for numerically stable running statistics.
    """
    
    def __init__(self, name: str, name_item: str = "item") -> None:
        self.nitems = 0
        self.total_time = 0.0
        self.name = name
        self.name_item = name_item
        # Running average and variance (Welford's algorithm)
        self.mean = 0.0
        self.ssq = 0.0  # Sum of squared deviations
        self.bins: list[Bin] = []
        self.values: list[float] = []
    
    def add(self, value: float) -> None:
        """Add a value to the statistics."""
        self.nitems += 1
        self.total_time += value
        
        # Welford's online algorithm for running mean and variance
        dev = value - self.mean
        self.mean += dev / self.nitems
        self.ssq += dev * (value - self.mean)
        
        self.values.append(value)
    
    def average_delay(self) -> float:
        """Return the running mean."""
        return self.mean
    
    def standard_deviation(self) -> float:
        """Return the sample standard deviation."""
        if self.nitems < 2:
            return 0.0
        return math.sqrt(self.ssq / (self.nitems - 1))
    
    def variance(self) -> float:
        """Return the sample variance."""
        if self.nitems < 2:
            return 0.0
        return self.ssq / (self.nitems - 1)
    
    def average_item_delay(self) -> float:
        """Return average delay in milliseconds."""
        if not self.values:
            return 0.0
        min_value = min(self.values)
        positivised_values = [v - min_value for v in self.values]
        return sum(positivised_values) / self.nitems / MSEC
    
    def recalculate_bins(self, confidence: float = 0.05) -> None:
        """Recalculate histogram bins based on data distribution."""
        if self.nitems < 10:
            print("Not enough data for histogram")
            return
        
        self.values.sort()
        self.bins.clear()
        
        value_range = max(self.values) - min(self.values)
        if value_range == 0:
            return
        
        typical_bin_pop = int(1 / ((confidence ** 2) + (1 / self.nitems)))
        number_of_bins = min(50, self.nitems // max(1, typical_bin_pop))
        if number_of_bins == 0:
            number_of_bins = 1
        
        bin_width = value_range / number_of_bins + 2
        bin_widths = (bin_width / 2, bin_width * 2)
        number_values_per_bin = (typical_bin_pop // 2, typical_bin_pop * 2)
        
        min_index = min(number_values_per_bin)
        max_index = self.nitems - min(number_values_per_bin)
        
        if min_index >= len(self.values) or max_index <= min_index:
            # Not enough data, create simple bins
            self.bins.append(MinBin(self.values[0]))
            self.bins.append(Bin(self.values[0], self.values[-1]))
            self.bins.append(MaxBin(self.values[-1]))
        else:
            self.bins.append(MinBin(self.values[min_index]))
            low_mark_index = min_index
            
            while True:
                high_mark_index = min(low_mark_index + typical_bin_pop, max_index)
                actual_bin_width = self.values[high_mark_index] - self.values[low_mark_index]
                
                if actual_bin_width > max(bin_widths):
                    high_mark_index = low_mark_index + min(number_values_per_bin)
                elif actual_bin_width < min(bin_widths):
                    high_mark_index = low_mark_index + max(number_values_per_bin)
                
                if high_mark_index >= max_index:
                    self.bins.append(Bin(self.values[low_mark_index], self.values[max_index]))
                    self.bins.append(MaxBin(self.values[max_index]))
                    break
                
                self.bins.append(Bin(self.values[low_mark_index], self.values[high_mark_index]))
                low_mark_index = high_mark_index
        
        print(f"Typical bin population size: {typical_bin_pop}")
        
        # Add values to bins
        for value in self.values:
            for bin_ in self.bins:
                if bin_.add_value(value):
                    break
    
    def freq(self, value: float) -> tuple[float, float, float, float]:
        """
        Estimate probability density for a value.
        
        Returns: (frequency, confidence, bin_low, bin_high)
        """
        target_bin = None
        for bin_ in self.bins:
            if bin_.contains_value(value):
                target_bin = bin_
                break
        
        if target_bin is None:
            raise ValueError("The bins do not cover all possible values")
        
        freq = target_bin.weight / self.nitems
        confidence = math.sqrt((1 - freq) / (freq * (self.nitems + 3)))
        return freq, confidence, target_bin.low, target_bin.high
    
    def print_histogram(self) -> None:
        """Print a text-based histogram."""
        self.recalculate_bins(0.10)
        self.bins.sort()
        
        if Bin.max_weight == Bin.min_weight:
            resolution = 1
        else:
            resolution = max(1, (Bin.max_weight - Bin.min_weight) // 10)
        
        print(f"Resolution: {resolution}")
        for bin_ in self.bins:
            bar = "*" * (bin_.weight // resolution)
            print(f"{bin_} {bar}")
        print()
        for bin_ in self.bins:
            print(f"{bin_} {bin_.weight}")
    
    def __str__(self) -> str:
        avg = self.average_item_delay()
        max_val = max(self.values) if self.values else 0
        return (
            f"{self.nitems} {self.name_item}. "
            f"Mean value for {self.name}={avg:.2f} msec. "
            f"Max jitter of {max_val} msec"
        )


# Backward compatibility aliases
PrioQueue = PriorityQueue
IterEvent = IteratorEvent

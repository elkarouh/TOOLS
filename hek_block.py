#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hek_block - Use class syntax as a configuration block for function calls.

This module provides a decorator that transforms a class definition into
a function call, passing all class attributes and methods as keyword
arguments. It enables a Ruby-like "block" syntax in Python.

Example usage:

    @call(dict)
    class my_config:
        name = "aircraft"
        altitude = 35000

        def validate(self, x):
            # 'self' is injected by the wrapper (can be ignored)
            return x > 0

    # my_config is now: {'name': 'aircraft', 'altitude': 35000, 'validate': <function>}

This is particularly useful for:

    - Configuration objects with many parameters
    - Builder patterns where you want readable, structured definitions
    - DSLs (Domain Specific Languages) embedded in Python
    - Cases where you have multiple callback functions to pass

Instead of:

    create_widget(
        name="button",
        width=100,
        height=50,
        on_click=lambda: print("clicked"),
        on_hover=lambda: print("hovering"),
    )

You can write:

    @call(create_widget)
    class _:
        name = "button"
        width = 100
        height = 50

        def on_click(self):
            print("clicked")

        def on_hover(self):
            print("hovering")

The class name is irrelevant (commonly '_' is used) since the class
itself is replaced by the return value of the decorated function.

Note: Methods must include 'self' as the first parameter because the
wrapper injects a temporary class instance. This 'self' can be ignored
in the method body if not needed.
"""


def call(func):
    """
    Decorator factory that converts a class definition into a function call.

    Args:
        func: The function to call with the class attributes as keyword arguments.

    Returns:
        A decorator that, when applied to a class:
        1. Creates a temporary instance of the class
        2. Extracts all non-dunder attributes and methods
        3. Wraps methods so they can be called without 'self'
        4. Calls func(**attributes) and returns the result

    Example:
        @call(some_builder_function)
        class _:
            param1 = "value1"
            param2 = 42

            def callback(self):
                return "result"

        # Equivalent to:
        # some_builder_function(param1="value1", param2=42, callback=<wrapped function>)
    """

    def _deco(cls):
        # Create a temporary instance for method binding
        bogus_instance = cls()

        # Collect all non-dunder class attributes
        fields = {}
        for k, v in cls.__dict__.items():
            # Skip Python's internal/magic attributes
            if k.startswith('__'):
                continue

            if callable(v):
                # Wrap methods so they receive the bogus instance as 'self'
                # This allows methods defined without 'self' to still work,
                # and methods with 'self' to have it pre-bound
                def mymeth(name=k):
                    def _mymeth(*args):
                        args = (bogus_instance,) + args
                        return getattr(cls, name)(*args)
                    return _mymeth
                fields[k] = mymeth(k)
            else:
                # Regular attributes are passed through as-is
                fields[k] = v

        # Call the target function with all collected fields as kwargs
        return func(**fields)

    return _deco


if __name__ == "__main__":
    # Demonstration

    print("=== Example 1: Creating a dict ===")

    @call(dict)
    class config:
        database = "postgresql"
        host = "localhost"
        port = 5432

    print(f"config = {config}")

    print("\n=== Example 2: Custom builder function ===")

    def create_validator(name, min_val=0, max_val=100, check=None):
        """Example builder that creates a validation dict."""
        return {
            "name": name,
            "range": (min_val, max_val),
            "check": check,
        }

    @call(create_validator)
    class altitude_validator:
        name = "altitude"
        min_val = 0
        max_val = 45000

        def check(self, value):
            # Note: 'self' is required here because the wrapper injects
            # the bogus instance as the first argument
            return isinstance(value, (int, float))

    print(f"validator = {altitude_validator}")
    print(f"check(35000) = {altitude_validator['check'](35000)}")

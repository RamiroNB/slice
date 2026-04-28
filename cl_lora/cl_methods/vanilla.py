"""Vanilla CL method: no extra mechanics.

Equivalent to the framework's pre-existing per-stage train+merge behavior.
"""
from __future__ import annotations

from .base import CLMethod


class VanillaCLMethod(CLMethod):
    name = "vanilla"

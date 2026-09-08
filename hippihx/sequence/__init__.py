"""Sequence helpers. Scans stay under attn/; this is short-window conv."""

from . import causal_conv

__all__ = ["causal_conv"]

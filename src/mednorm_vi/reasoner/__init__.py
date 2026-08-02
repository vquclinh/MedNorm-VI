"""Constrained ~8B unified clinical reasoner (sprint 0075).

One instruction model proposes entities, types, assertions and candidate selections; a
deterministic validator then refuses anything it cannot prove against the source text and the
governed KB. The model never emits a code, an offset, or a type of its own authority.
"""

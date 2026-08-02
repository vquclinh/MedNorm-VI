"""GraphCENT (0080): multi-encoder biomedical retrieval + ontology context + constrained LLM.

Frozen pretrained retrievers shape a small candidate context; a local Qwen3-8B chooses among
governed ids or abstains; deterministic evidence tiers decide what is actually emitted. No
component may invent an entity, a span, a type or a code.
"""

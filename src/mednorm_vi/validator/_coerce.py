"""Internal helper: coerce entities to the organizer JSON dict shape.

Validators accept either :class:`EntityPrediction` objects or plain mappings in
the organizer output shape (``text``, ``type``, ``assertions``, ``candidates``,
``position``). This keeps the validators usable both on in-memory predictions
and on already-serialized ``output/N.json`` content.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..schemas import EntityPrediction

EntityLike = EntityPrediction | Mapping[str, Any]


def as_entity_dict(entity: EntityLike) -> dict[str, Any]:
    """Return a plain dict in the organizer output shape."""
    if isinstance(entity, EntityPrediction):
        return entity.to_output_dict()
    if isinstance(entity, Mapping):
        return dict(entity)
    raise TypeError(f"unsupported entity type: {type(entity)!r}")

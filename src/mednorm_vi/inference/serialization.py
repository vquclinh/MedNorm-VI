"""Organizer JSON serialization for decoded entities."""

from __future__ import annotations

import json

from ..metric_decoder import DecodedEntity
from ..schemas.constants import TYPE_BY_ORGANIZER_LABEL
from ..schemas.prediction import EntityPrediction
from ..schemas.spans import Span, SpanCoordinates


def to_entity_predictions(decoded: tuple[DecodedEntity, ...]) -> tuple[EntityPrediction, ...]:
    predictions: list[EntityPrediction] = []
    for entity in decoded:
        internal_type = TYPE_BY_ORGANIZER_LABEL.get(
            entity.hypothesis.entity_type, entity.hypothesis.entity_type
        )
        predictions.append(
            EntityPrediction(
                text=entity.hypothesis.text,
                type=internal_type,
                coords=SpanCoordinates(Span(entity.hypothesis.start, entity.hypothesis.end)),
                assertions=entity.assertions,
                candidates=entity.candidates,
            )
        )
    return tuple(predictions)


def to_submission_json(predictions: tuple[EntityPrediction, ...]) -> str:
    payload = [p.to_submission_dict() for p in predictions]
    return json.dumps(payload, ensure_ascii=False, sort_keys=False, indent=2) + "\n"


__all__ = ["to_entity_predictions", "to_submission_json"]

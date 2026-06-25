"""
serialization.py - JSON and JSONL serialization for evaluation records.

Provides deterministic serialization and strict deserialization for benchmark
metadata, trajectory events, content identities, and measured context
deliveries.

Metadata is stored as one formatted JSON document. Trajectory events and
context deliveries are stored as JSON Lines so records can be appended while
a benchmark is running.

Full-file writes are atomic. Individual JSONL append operations assume a
single writer.
"""

import json
import os
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

from indexer.errors import EvaluationSerializationError, UnsupportedTrajectoryEventError

from .models import (
    BenchmarkRunMetadata,
    ContentIdentity,
    ContextDelivery,
    JsonValue,
    ModelUsage,
    ToolCall,
    ToolResult,
    ToolStatus,
    TrajectoryEvent,
)

type JsonObject = dict[str, JsonValue]


# ---------------------------------------------------------------------------
# Metadata conversion
# ---------------------------------------------------------------------------


def metadata_to_dict(metadata: BenchmarkRunMetadata) -> JsonObject:
    """
    Convert benchmark metadata to a JSON-compatible dictionary.

    Args:
        metadata: The benchmark metadata to serialize.

    Returns:
        A JSON-compatible metadata dictionary.
    """
    return {
        "run_id": metadata.run_id,
        "policy": metadata.policy,
        "repository": metadata.repository,
        "revision": metadata.revision,
        "started_at": metadata.started_at,
        "token_counter": metadata.token_counter,
        "task_id": metadata.task_id,
        "provider": metadata.provider,
        "model": metadata.model,
        "schema_version": metadata.schema_version,
    }


def metadata_from_dict(
    payload: Mapping[str, object],
) -> BenchmarkRunMetadata:
    """
    Parse benchmark metadata from a dictionary.

    Unknown fields are ignored so records can gain optional fields without
    breaking older readers.

    Args:
        payload: The metadata dictionary to parse.

    Returns:
        Parsed benchmark metadata.

    Raises:
        EvaluationSerializationError: If a required field is missing or has
            the wrong type.
    """
    context = "benchmark metadata"
    schema_version = _optional_non_negative_int(
        payload,
        "schema_version",
        context,
        default=1,
    )
    if schema_version < 1:  # type: ignore
        raise EvaluationSerializationError(
            message="benchmark metadata.schema_version must be at least 1"
        )

    return BenchmarkRunMetadata(
        run_id=_require_str(payload, "run_id", context),
        policy=_require_str(payload, "policy", context),
        repository=_require_str(payload, "repository", context),
        revision=_require_str(payload, "revision", context),
        started_at=_require_str(payload, "started_at", context),
        token_counter=_require_str(payload, "token_counter", context),
        task_id=_optional_str(payload, "task_id", context),
        provider=_optional_str(payload, "provider", context),
        model=_optional_str(payload, "model", context),
        schema_version=schema_version,  # type: ignore
    )


# ---------------------------------------------------------------------------
# Trajectory-event conversion
# ---------------------------------------------------------------------------


def event_to_dict(event: TrajectoryEvent) -> JsonObject:
    """
    Convert one trajectory event to a JSON-compatible dictionary.

    Args:
        event: The trajectory event to serialize.

    Returns:
        A JSON-compatible event dictionary.

    Raises:
        TypeError: If the event type is unsupported.
    """
    if isinstance(event, ToolCall):
        return {
            "event_type": event.event_type,
            "call_id": event.call_id,
            "turn": event.turn,
            "sequence": event.sequence,
            "name": event.name,
            "arguments": event.arguments,
            "purpose": event.purpose,
        }

    if isinstance(event, ToolResult):
        return {
            "event_type": event.event_type,
            "call_id": event.call_id,
            "sequence": event.sequence,
            "status": event.status,
            "output": event.output,
            "error": event.error,
            "duration_ms": event.duration_ms,
        }

    if isinstance(event, ModelUsage):
        return {
            "event_type": event.event_type,
            "request_id": event.request_id,
            "turn": event.turn,
            "sequence": event.sequence,
            "provider": event.provider,
            "model": event.model,
            "input_tokens": event.input_tokens,
            "output_tokens": event.output_tokens,
            "cached_input_tokens": event.cached_input_tokens,
            "reasoning_tokens": event.reasoning_tokens,
        }

    raise UnsupportedTrajectoryEventError(type(event).__name__)


def event_from_dict(
    payload: Mapping[str, object],
) -> TrajectoryEvent:
    """
    Parse one trajectory event from a dictionary.

    Args:
        payload: The event dictionary to parse.

    Returns:
        Parsed ToolCall, ToolResult, or ModelUsage.

    Raises:
        EvaluationSerializationError: If the event type is unknown or a field
            is missing or invalid.
    """
    event_type = _require_str(
        payload,
        "event_type",
        "trajectory event",
    )

    if event_type == "tool_call":
        return _tool_call_from_dict(payload)

    if event_type == "tool_result":
        return _tool_result_from_dict(payload)

    if event_type == "model_usage":
        return _model_usage_from_dict(payload)

    raise UnsupportedTrajectoryEventError(event_type)


def _tool_call_from_dict(
    payload: Mapping[str, object],
) -> ToolCall:
    """
    Parse a tool-call event.

    Args:
        payload: The tool-call dictionary.

    Returns:
        Parsed ToolCall.
    """
    context = "tool_call"
    return ToolCall(
        call_id=_require_str(payload, "call_id", context),
        turn=_require_non_negative_int(payload, "turn", context),
        sequence=_require_non_negative_int(
            payload,
            "sequence",
            context,
        ),
        name=_require_str(payload, "name", context),
        arguments=_optional_json_object(
            payload,
            "arguments",
            context,
        ),
        purpose=_optional_str(payload, "purpose", context),
    )


def _tool_result_from_dict(
    payload: Mapping[str, object],
) -> ToolResult:
    """
    Parse a tool-result event.

    Args:
        payload: The tool-result dictionary.

    Returns:
        Parsed ToolResult.

    Raises:
        EvaluationSerializationError: If the status is unsupported.
    """
    context = "tool_result"
    status = _require_str(payload, "status", context)
    if status not in {"success", "error", "cancelled"}:
        raise EvaluationSerializationError(
            message="tool_result.status must be success, error, or cancelled"
        )

    return ToolResult(
        call_id=_require_str(payload, "call_id", context),
        sequence=_require_non_negative_int(
            payload,
            "sequence",
            context,
        ),
        status=cast(ToolStatus, status),
        output=_optional_str(payload, "output", context),
        error=_optional_str(payload, "error", context),
        duration_ms=_optional_non_negative_float(
            payload,
            "duration_ms",
            context,
        ),
    )


def _model_usage_from_dict(
    payload: Mapping[str, object],
) -> ModelUsage:
    """
    Parse a model-usage event.

    Args:
        payload: The model-usage dictionary.

    Returns:
        Parsed ModelUsage.
    """
    context = "model_usage"
    return ModelUsage(
        request_id=_require_str(payload, "request_id", context),
        turn=_require_non_negative_int(payload, "turn", context),
        sequence=_require_non_negative_int(
            payload,
            "sequence",
            context,
        ),
        provider=_require_str(payload, "provider", context),
        model=_require_str(payload, "model", context),
        input_tokens=_require_non_negative_int(
            payload,
            "input_tokens",
            context,
        ),
        output_tokens=_require_non_negative_int(
            payload,
            "output_tokens",
            context,
        ),
        cached_input_tokens=_optional_non_negative_int(
            payload,
            "cached_input_tokens",
            context,
        ),
        reasoning_tokens=_optional_non_negative_int(
            payload,
            "reasoning_tokens",
            context,
        ),
    )


# ---------------------------------------------------------------------------
# Content-identity conversion
# ---------------------------------------------------------------------------


def content_identity_to_dict(
    identity: ContentIdentity,
) -> JsonObject:
    """
    Convert a content identity to a JSON-compatible dictionary.

    Args:
        identity: The content identity to serialize.

    Returns:
        A JSON-compatible content identity dictionary.
    """
    return {
        "content_id": identity.content_id,
        "kind": identity.kind,
        "digest": identity.digest,
        "byte_length": identity.byte_length,
    }


def content_identity_from_dict(
    payload: Mapping[str, object],
) -> ContentIdentity:
    """
    Parse a content identity from a dictionary.

    Args:
        payload: The content identity dictionary to parse.

    Returns:
        Parsed ContentIdentity.

    Raises:
        EvaluationSerializationError: If a field is missing or invalid.
    """
    context = "content identity"
    return ContentIdentity(
        content_id=_require_str(payload, "content_id", context),
        kind=_require_str(payload, "kind", context),
        digest=_require_str(payload, "digest", context),
        byte_length=_require_non_negative_int(
            payload,
            "byte_length",
            context,
        ),
    )


# ---------------------------------------------------------------------------
# Context-delivery conversion
# ---------------------------------------------------------------------------


def delivery_to_dict(delivery: ContextDelivery) -> JsonObject:
    """
    Convert a context delivery to a JSON-compatible dictionary.

    Args:
        delivery: The context delivery to serialize.

    Returns:
        A JSON-compatible delivery dictionary.
    """
    return {
        "call_id": delivery.call_id,
        "sequence": delivery.sequence,
        "content_id": delivery.content_id,
        "content_kind": delivery.content_kind,
        "byte_length": delivery.byte_length,
        "counter_name": delivery.counter_name,
        "tokens": delivery.tokens,
        "novel_tokens": delivery.novel_tokens,
        "repeated_tokens": delivery.repeated_tokens,
    }


def delivery_from_dict(
    payload: Mapping[str, object],
) -> ContextDelivery:
    """
    Parse a context delivery from a dictionary.

    Args:
        payload: The context-delivery dictionary to parse.

    Returns:
        Parsed ContextDelivery.

    Raises:
        EvaluationSerializationError: If a field is missing, invalid, or has
            inconsistent token totals.
    """
    context = "context delivery"
    delivery = ContextDelivery(
        call_id=_require_str(payload, "call_id", context),
        sequence=_require_non_negative_int(
            payload,
            "sequence",
            context,
        ),
        content_id=_require_str(payload, "content_id", context),
        content_kind=_require_str(
            payload,
            "content_kind",
            context,
        ),
        byte_length=_require_non_negative_int(
            payload,
            "byte_length",
            context,
        ),
        counter_name=_require_str(
            payload,
            "counter_name",
            context,
        ),
        tokens=_require_non_negative_int(
            payload,
            "tokens",
            context,
        ),
        novel_tokens=_require_non_negative_int(
            payload,
            "novel_tokens",
            context,
        ),
        repeated_tokens=_require_non_negative_int(
            payload,
            "repeated_tokens",
            context,
        ),
    )

    measured_tokens = delivery.novel_tokens + delivery.repeated_tokens
    if delivery.tokens != measured_tokens:
        raise EvaluationSerializationError(
            message=(
                "context delivery tokens must equal "
                "novel_tokens plus repeated_tokens"
            )
        )

    return delivery


# ---------------------------------------------------------------------------
# Metadata JSON
# ---------------------------------------------------------------------------


def write_metadata(
    path: str | Path,
    metadata: BenchmarkRunMetadata,
) -> None:
    """
    Write benchmark metadata as formatted JSON.

    The destination is replaced atomically after the complete document has
    been written.

    Args:
        path: The destination JSON path.
        metadata: The metadata to write.
    """
    _write_json_atomic(
        Path(path),
        metadata_to_dict(metadata),
    )


def read_metadata(
    path: str | Path,
) -> BenchmarkRunMetadata:
    """
    Read benchmark metadata from a JSON file.

    Args:
        path: The metadata JSON path.

    Returns:
        Parsed benchmark metadata.

    Raises:
        EvaluationSerializationError: If the file does not contain valid
            benchmark metadata.
    """
    payload = _read_json_object(
        Path(path),
        "benchmark metadata",
    )
    return metadata_from_dict(payload)


# ---------------------------------------------------------------------------
# Trajectory event JSONL
# ---------------------------------------------------------------------------


def write_events(
    path: str | Path,
    events: Iterable[TrajectoryEvent],
) -> None:
    """
    Replace a JSONL file with trajectory events.

    The destination is replaced atomically after every event has been
    serialized.

    Args:
        path: The destination JSONL path.
        events: The trajectory events to write.
    """
    _write_jsonl_atomic(
        Path(path),
        (event_to_dict(event) for event in events),
    )


def append_event(
    path: str | Path,
    event: TrajectoryEvent,
) -> None:
    """
    Append one trajectory event to a JSONL file.

    Args:
        path: The destination JSONL path.
        event: The trajectory event to append.
    """
    _append_jsonl(
        Path(path),
        event_to_dict(event),
    )


def iter_events(
    path: str | Path,
) -> Iterator[TrajectoryEvent]:
    """
    Iterate over trajectory events in a JSONL file.

    Blank lines are ignored.

    Args:
        path: The trajectory JSONL path.

    Yields:
        Parsed events in file order.

    Raises:
        EvaluationSerializationError: If a nonblank line is invalid.
    """
    for payload in _iter_jsonl_objects(
        Path(path),
        "trajectory event",
    ):
        yield event_from_dict(payload)


def read_events(
    path: str | Path,
) -> list[TrajectoryEvent]:
    """
    Read all trajectory events from a JSONL file.

    Args:
        path: The trajectory JSONL path.

    Returns:
        Parsed events in file order.
    """
    return list(iter_events(path))


# ---------------------------------------------------------------------------
# Context delivery JSONL
# ---------------------------------------------------------------------------


def write_deliveries(
    path: str | Path,
    deliveries: Iterable[ContextDelivery],
) -> None:
    """
    Replace a JSONL file with context-delivery measurements.

    The destination is replaced atomically after every delivery has been
    serialized.

    Args:
        path: The destination JSONL path.
        deliveries: The context deliveries to write.
    """
    _write_jsonl_atomic(
        Path(path),
        (delivery_to_dict(delivery) for delivery in deliveries),
    )


def append_delivery(
    path: str | Path,
    delivery: ContextDelivery,
) -> None:
    """
    Append one context delivery to a JSONL file.

    Args:
        path: The destination JSONL path.
        delivery: The context delivery to append.
    """
    _append_jsonl(
        Path(path),
        delivery_to_dict(delivery),
    )


def iter_deliveries(
    path: str | Path,
) -> Iterator[ContextDelivery]:
    """
    Iterate over context deliveries in a JSONL file.

    Blank lines are ignored.

    Args:
        path: The context-delivery JSONL path.

    Yields:
        Parsed context deliveries in file order.

    Raises:
        EvaluationSerializationError: If a nonblank line is invalid.
    """
    for payload in _iter_jsonl_objects(
        Path(path),
        "context delivery",
    ):
        yield delivery_from_dict(payload)


def read_deliveries(
    path: str | Path,
) -> list[ContextDelivery]:
    """
    Read all context deliveries from a JSONL file.

    Args:
        path: The context-delivery JSONL path.

    Returns:
        Parsed context deliveries in file order.
    """
    return list(iter_deliveries(path))


# ---------------------------------------------------------------------------
# Internal file helpers
# ---------------------------------------------------------------------------


def _write_json_atomic(
    path: Path,
    payload: JsonObject,
) -> None:
    """
    Atomically write one formatted JSON object.

    Args:
        path: The destination file path.
        payload: The JSON-compatible object to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _read_json_object(
    path: Path,
    context: str,
) -> Mapping[str, object]:
    """
    Read and validate one JSON object.

    Args:
        path: The JSON file path.
        context: The record description used in errors.

    Returns:
        Parsed JSON object.

    Raises:
        EvaluationSerializationError: If the JSON is invalid or its root is
            not an object.
    """
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationSerializationError(
            message=f"Invalid {context} JSON in {path}: {exc.msg}"
        ) from exc

    return _require_mapping(raw, context)


def _write_jsonl_atomic(
    path: Path,
    payloads: Iterable[JsonObject],
) -> None:
    """
    Atomically replace a JSONL file.

    Args:
        path: The destination JSONL path.
        payloads: The JSON-compatible objects to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)

            for payload in payloads:
                handle.write(_encode_json_line(payload))
                handle.write("\n")

            handle.flush()
            os.fsync(handle.fileno())

        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _append_jsonl(
    path: Path,
    payload: JsonObject,
) -> None:
    """
    Append one object to a JSONL file.

    Args:
        path: The destination JSONL path.
        payload: The JSON-compatible object to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(_encode_json_line(payload))
        handle.write("\n")


def _iter_jsonl_objects(
    path: Path,
    context: str,
) -> Iterator[Mapping[str, object]]:
    """
    Iterate over validated objects in a JSONL file.

    Args:
        path: The JSONL file path.
        context: The record description used in errors.

    Yields:
        Parsed JSON objects.

    Raises:
        EvaluationSerializationError: If a nonblank line is invalid.
    """
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                raw: object = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise EvaluationSerializationError(
                    message=f"Invalid {context} JSON in {path} "
                    f"at line {line_number}: {exc.msg}"
                ) from exc

            try:
                yield _require_mapping(raw, context)
            except EvaluationSerializationError as exc:
                raise EvaluationSerializationError(
                    message=f"Invalid {context} in {path} "
                    f"at line {line_number}: {exc}"
                ) from exc


def _encode_json_line(payload: JsonObject) -> str:
    """
    Encode one object as deterministic compact JSON.

    Args:
        payload: The JSON-compatible object to encode.

    Returns:
        Compact JSON text without a trailing newline.
    """
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------


def _require_mapping(
    value: object,
    context: str,
) -> Mapping[str, object]:
    """
    Require a JSON object with string keys.

    Args:
        value: The value to validate.
        context: The record description used in errors.

    Returns:
        The validated mapping.

    Raises:
        EvaluationSerializationError: If the value is not a string-keyed
            mapping.
    """
    if not isinstance(value, dict):
        raise EvaluationSerializationError(message=f"{context} must be a JSON object")

    if not all(isinstance(key, str) for key in value):
        raise EvaluationSerializationError(message=f"{context} keys must be strings")

    return cast(Mapping[str, object], value)


def _require_str(
    payload: Mapping[str, object],
    key: str,
    context: str,
) -> str:
    """
    Read a required string field.

    Args:
        payload: The record being parsed.
        key: The field name.
        context: The record description used in errors.

    Returns:
        The string value.

    Raises:
        EvaluationSerializationError: If the field is missing or invalid.
    """
    value = payload.get(key)
    if not isinstance(value, str):
        raise EvaluationSerializationError(message=f"{context}.{key} must be a string")

    return value


def _optional_str(
    payload: Mapping[str, object],
    key: str,
    context: str,
) -> str | None:
    """
    Read an optional string field.

    Args:
        payload: The record being parsed.
        key: The field name.
        context: The record description used in errors.

    Returns:
        The string value or None.

    Raises:
        EvaluationSerializationError: If the field has the wrong type.
    """
    value = payload.get(key)
    if value is None:
        return None

    if not isinstance(value, str):
        raise EvaluationSerializationError(
            message=f"{context}.{key} must be a string or null"
        )

    return value


def _require_non_negative_int(
    payload: Mapping[str, object],
    key: str,
    context: str,
) -> int:
    """
    Read a required non-negative integer field.

    Boolean values are rejected even though bool subclasses int.

    Args:
        payload: The record being parsed.
        key: The field name.
        context: The record description used in errors.

    Returns:
        The non-negative integer value.

    Raises:
        EvaluationSerializationError: If the field is missing or invalid.
    """
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationSerializationError(
            message=f"{context}.{key} must be an integer"
        )

    if value < 0:
        raise EvaluationSerializationError(
            message=f"{context}.{key} must be non-negative"
        )

    return value


def _optional_non_negative_int(
    payload: Mapping[str, object],
    key: str,
    context: str,
    *,
    default: int | None = None,
) -> int | None:
    """
    Read an optional non-negative integer field.

    Args:
        payload: The record being parsed.
        key: The field name.
        context: The record description used in errors.
        default: The value returned when the field is absent or null.

    Returns:
        The integer value or the supplied default.

    Raises:
        EvaluationSerializationError: If the field is invalid.
    """
    value = payload.get(key)
    if value is None:
        return default

    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationSerializationError(
            message=f"{context}.{key} must be an integer or null"
        )

    if value < 0:
        raise EvaluationSerializationError(
            message=f"{context}.{key} must be non-negative"
        )

    return value


def _optional_non_negative_float(
    payload: Mapping[str, object],
    key: str,
    context: str,
) -> float | None:
    """
    Read an optional non-negative numeric field.

    Args:
        payload: The record being parsed.
        key: The field name.
        context: The record description used in errors.

    Returns:
        The numeric value as a float, or None.

    Raises:
        EvaluationSerializationError: If the field is invalid.
    """
    value = payload.get(key)
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EvaluationSerializationError(
            message=f"{context}.{key} must be a number or null"
        )

    numeric_value = float(value)
    if numeric_value < 0:
        raise EvaluationSerializationError(
            message=f"{context}.{key} must be non-negative"
        )

    return numeric_value


def _optional_json_object(
    payload: Mapping[str, object],
    key: str,
    context: str,
) -> dict[str, JsonValue]:
    """
    Read an optional JSON-compatible object field.

    Args:
        payload: The record being parsed.
        key: The field name.
        context: The record description used in errors.

    Returns:
        The validated object, or an empty dictionary when absent.

    Raises:
        EvaluationSerializationError: If the field is not JSON-compatible.
    """
    value = payload.get(key)
    if value is None:
        return {}

    mapping = _require_mapping(
        value,
        f"{context}.{key}",
    )

    for item_key, item_value in mapping.items():
        _validate_json_value(
            item_value,
            f"{context}.{key}.{item_key}",
        )

    return cast(
        dict[str, JsonValue],
        dict(mapping),
    )


def _validate_json_value(
    value: object,
    context: str,
) -> None:
    """
    Validate one recursive JSON value.

    Args:
        value: The value to validate.
        context: The field path used in errors.

    Raises:
        EvaluationSerializationError: If the value is not JSON-compatible.
    """
    if value is None or isinstance(
        value,
        str | int | float | bool,
    ):
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(
                item,
                f"{context}[{index}]",
            )
        return

    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvaluationSerializationError(
                    message=f"{context} contains a non-string object key"
                )

            _validate_json_value(
                item,
                f"{context}.{key}",
            )
        return

    raise EvaluationSerializationError(message=f"{context} is not JSON-compatible")

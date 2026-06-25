"""
tests/evaluation/test_serialization.py - Tests for evaluation persistence.

Coverage:
  - Dictionary conversion and round trips
  - Metadata JSON reads and atomic replacement
  - Event JSONL writes, appends, and iteration
  - Delivery JSONL writes, appends, and iteration
  - Invalid JSON and invalid field types
  - Token-total invariants
  - Unicode and recursive JSON tool arguments
"""

import json
from pathlib import Path
from typing import cast

import pytest

from indexer.errors import (
    EvaluationSerializationError,
    UnsupportedTrajectoryEventError,
)
from indexer.evaluation.models import (
    BenchmarkRunMetadata,
    ContentIdentity,
    ContextDelivery,
    ModelUsage,
    ToolCall,
    ToolResult,
    TrajectoryEvent,
)
from indexer.evaluation.serialization import (
    append_delivery,
    append_event,
    content_identity_from_dict,
    content_identity_to_dict,
    delivery_from_dict,
    delivery_to_dict,
    event_from_dict,
    event_to_dict,
    iter_events,
    metadata_from_dict,
    metadata_to_dict,
    read_deliveries,
    read_events,
    read_metadata,
    write_deliveries,
    write_events,
    write_metadata,
)


@pytest.fixture
def metadata() -> BenchmarkRunMetadata:
    return BenchmarkRunMetadata(
        run_id="run-1",
        policy="progressive",
        repository="owner/repo",
        revision="abc123",
        started_at="2026-06-25T12:00:00Z",
        token_counter="heuristic-chars-4",
        task_id="task-1",
        provider="example",
        model="example-model",
    )


@pytest.fixture
def events() -> list[TrajectoryEvent]:
    return [
        ToolCall(
            call_id="call-1",
            turn=0,
            sequence=1,
            name="search",
            arguments={
                "query": "incremental indexing",
                "options": {
                    "limit": 10,
                    "include_tests": True,
                },
                "paths": ["src", "tests"],
            },
            purpose="Find relevant symbols.",
        ),
        ToolResult(
            call_id="call-1",
            sequence=2,
            status="success",
            output="résultat\n",
            duration_ms=12.5,
        ),
        ModelUsage(
            request_id="request-1",
            turn=0,
            sequence=3,
            provider="example",
            model="example-model",
            input_tokens=100,
            output_tokens=20,
            cached_input_tokens=40,
            reasoning_tokens=5,
        ),
    ]


@pytest.fixture
def deliveries() -> list[ContextDelivery]:
    return [
        ContextDelivery(
            call_id="call-1",
            sequence=2,
            content_id="tool_output:sha256:abc",
            content_kind="tool_output",
            byte_length=12,
            counter_name="heuristic-chars-4",
            tokens=3,
            novel_tokens=3,
            repeated_tokens=0,
        ),
        ContextDelivery(
            call_id="call-2",
            sequence=4,
            content_id="tool_output:sha256:abc",
            content_kind="tool_output",
            byte_length=12,
            counter_name="heuristic-chars-4",
            tokens=3,
            novel_tokens=0,
            repeated_tokens=3,
        ),
    ]


class TestMetadataSerialization:
    def test_dictionary_round_trip(
        self,
        metadata: BenchmarkRunMetadata,
    ) -> None:
        payload = metadata_to_dict(metadata)

        assert metadata_from_dict(payload) == metadata

    def test_missing_schema_version_defaults_to_one(
        self,
        metadata: BenchmarkRunMetadata,
    ) -> None:
        payload = metadata_to_dict(metadata)
        del payload["schema_version"]

        restored = metadata_from_dict(payload)

        assert restored.schema_version == 1

    def test_unknown_fields_are_ignored(
        self,
        metadata: BenchmarkRunMetadata,
    ) -> None:
        payload = metadata_to_dict(metadata)
        payload["future_field"] = "value"

        assert metadata_from_dict(payload) == metadata

    def test_write_and_read(
        self,
        tmp_path: Path,
        metadata: BenchmarkRunMetadata,
    ) -> None:
        path = tmp_path / "nested" / "metadata.json"

        write_metadata(path, metadata)

        assert read_metadata(path) == metadata
        assert path.read_text(encoding="utf-8").endswith("\n")

    def test_write_replaces_existing_file(
        self,
        tmp_path: Path,
        metadata: BenchmarkRunMetadata,
    ) -> None:
        path = tmp_path / "metadata.json"
        path.write_text("old", encoding="utf-8")

        write_metadata(path, metadata)

        assert read_metadata(path) == metadata
        assert "old" not in path.read_text(encoding="utf-8")

    def test_invalid_schema_version_is_rejected(
        self,
        metadata: BenchmarkRunMetadata,
    ) -> None:
        payload = metadata_to_dict(metadata)
        payload["schema_version"] = 0

        with pytest.raises(
            EvaluationSerializationError,
            match="schema_version",
        ):
            metadata_from_dict(payload)

    def test_missing_required_field_is_rejected(
        self,
        metadata: BenchmarkRunMetadata,
    ) -> None:
        payload = metadata_to_dict(metadata)
        del payload["run_id"]

        with pytest.raises(
            EvaluationSerializationError,
            match="run_id",
        ):
            metadata_from_dict(payload)


class TestEventSerialization:
    @pytest.mark.parametrize(
        "event",
        [
            ToolCall(
                call_id="call-1",
                turn=0,
                sequence=1,
                name="search",
                arguments={"query": "foo"},
            ),
            ToolResult(
                call_id="call-1",
                sequence=2,
                status="success",
                output="result",
                duration_ms=1.5,
            ),
            ModelUsage(
                request_id="request-1",
                turn=0,
                sequence=3,
                provider="example",
                model="model",
                input_tokens=10,
                output_tokens=2,
            ),
        ],
    )
    def test_dictionary_round_trip(
        self,
        event: TrajectoryEvent,
    ) -> None:
        assert event_from_dict(event_to_dict(event)) == event

    def test_unsupported_instance_is_rejected(self) -> None:
        with pytest.raises(UnsupportedTrajectoryEventError):
            event_to_dict(cast(TrajectoryEvent, object()))

    def test_unknown_event_type_is_rejected(self) -> None:
        with pytest.raises(UnsupportedTrajectoryEventError):
            event_from_dict({"event_type": "unknown"})

    def test_invalid_tool_status_is_rejected(self) -> None:
        with pytest.raises(
            EvaluationSerializationError,
            match="status",
        ):
            event_from_dict(
                {
                    "event_type": "tool_result",
                    "call_id": "call-1",
                    "sequence": 1,
                    "status": "pending",
                }
            )

    def test_negative_sequence_is_rejected(self) -> None:
        with pytest.raises(
            EvaluationSerializationError,
            match="sequence",
        ):
            event_from_dict(
                {
                    "event_type": "tool_call",
                    "call_id": "call-1",
                    "turn": 0,
                    "sequence": -1,
                    "name": "search",
                }
            )

    def test_boolean_is_not_accepted_as_integer(self) -> None:
        with pytest.raises(
            EvaluationSerializationError,
            match="turn",
        ):
            event_from_dict(
                {
                    "event_type": "tool_call",
                    "call_id": "call-1",
                    "turn": True,
                    "sequence": 1,
                    "name": "search",
                }
            )

    def test_recursive_json_arguments_round_trip(self) -> None:
        event = ToolCall(
            call_id="call-1",
            turn=0,
            sequence=1,
            name="search",
            arguments={
                "query": "foo",
                "flags": [True, False],
                "nested": {"limit": 5, "value": None},
            },
        )

        assert event_from_dict(event_to_dict(event)) == event


class TestContentIdentitySerialization:
    def test_round_trip(self) -> None:
        identity = ContentIdentity(
            content_id="tool_output:sha256:abc",
            kind="tool_output",
            digest="abc",
            byte_length=20,
        )

        assert (
            content_identity_from_dict(content_identity_to_dict(identity)) == identity
        )

    def test_negative_byte_length_is_rejected(self) -> None:
        with pytest.raises(
            EvaluationSerializationError,
            match="byte_length",
        ):
            content_identity_from_dict(
                {
                    "content_id": "id",
                    "kind": "tool_output",
                    "digest": "abc",
                    "byte_length": -1,
                }
            )


class TestDeliverySerialization:
    def test_round_trip(
        self,
        deliveries: list[ContextDelivery],
    ) -> None:
        delivery = deliveries[0]

        assert delivery_from_dict(delivery_to_dict(delivery)) == delivery

    def test_inconsistent_token_total_is_rejected(self) -> None:
        with pytest.raises(
            EvaluationSerializationError,
            match="novel_tokens plus repeated_tokens",
        ):
            delivery_from_dict(
                {
                    "call_id": "call-1",
                    "sequence": 1,
                    "content_id": "id",
                    "content_kind": "tool_output",
                    "byte_length": 10,
                    "counter_name": "counter",
                    "tokens": 10,
                    "novel_tokens": 4,
                    "repeated_tokens": 5,
                }
            )


class TestJsonlFiles:
    def test_events_write_and_read(
        self,
        tmp_path: Path,
        events: list[TrajectoryEvent],
    ) -> None:
        path = tmp_path / "run" / "events.jsonl"

        write_events(path, events)

        assert read_events(path) == events

    def test_append_event(
        self,
        tmp_path: Path,
        events: list[TrajectoryEvent],
    ) -> None:
        path = tmp_path / "events.jsonl"

        append_event(path, events[0])
        append_event(path, events[1])

        assert read_events(path) == events[:2]

    def test_events_preserve_file_order(
        self,
        tmp_path: Path,
        events: list[TrajectoryEvent],
    ) -> None:
        path = tmp_path / "events.jsonl"

        write_events(path, reversed(events))

        assert read_events(path) == list(reversed(events))

    def test_event_iterator_ignores_blank_lines(
        self,
        tmp_path: Path,
        events: list[TrajectoryEvent],
    ) -> None:
        path = tmp_path / "events.jsonl"
        first = json.dumps(event_to_dict(events[0]))
        second = json.dumps(event_to_dict(events[1]))
        path.write_text(
            f"{first}\n\n   \n{second}\n",
            encoding="utf-8",
        )

        assert list(iter_events(path)) == events[:2]

    def test_deliveries_write_and_read(
        self,
        tmp_path: Path,
        deliveries: list[ContextDelivery],
    ) -> None:
        path = tmp_path / "run" / "deliveries.jsonl"

        write_deliveries(path, deliveries)

        assert read_deliveries(path) == deliveries

    def test_append_delivery(
        self,
        tmp_path: Path,
        deliveries: list[ContextDelivery],
    ) -> None:
        path = tmp_path / "deliveries.jsonl"

        append_delivery(path, deliveries[0])
        append_delivery(path, deliveries[1])

        assert read_deliveries(path) == deliveries

    def test_invalid_json_reports_line_number(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "events.jsonl"
        path.write_text(
            (
                '{"event_type":"tool_call","call_id":"1",'
                '"turn":0,"sequence":1,"name":"search"}\n'
                "not-json\n"
            ),
            encoding="utf-8",
        )

        with pytest.raises(
            EvaluationSerializationError,
            match="line 2",
        ):
            read_events(path)

    def test_non_object_jsonl_record_is_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "events.jsonl"
        path.write_text('["not", "an", "object"]\n', encoding="utf-8")

        with pytest.raises(
            EvaluationSerializationError,
            match="line 1",
        ):
            read_events(path)

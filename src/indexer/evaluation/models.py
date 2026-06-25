"""
models.py - Dataclasses for token evaluation and trajectory records.

Defines provider-neutral records for benchmark metadata, agent tool activity,
model token usage, content identities, and measured context deliveries.

These models contain raw or directly measured data. Token counting, content
hashing, ledger updates, serialization, and aggregate reporting are implemented
in their respective evaluation modules.
"""

from dataclasses import dataclass, field
from typing import Literal

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

type ToolStatus = Literal[
    "success",
    "error",
    "cancelled",
]


@dataclass(frozen=True)
class BenchmarkRunMetadata:
    """
    Describes one benchmark or evaluation run.

    Attributes:
        run_id: The unique identifier for the run.
        policy: The context policy or baseline being evaluated.
        repository: The repository identifier or canonical repository name.
        revision: The repository revision used for the run.
        started_at: The run start time as an ISO 8601 string.
        token_counter: The stable name of the token counter used.
        task_id: The optional task definition identifier.
        provider: The optional model provider name.
        model: The optional model name.
        schema_version: The metadata schema version.
    """

    run_id: str
    policy: str
    repository: str
    revision: str
    started_at: str
    token_counter: str
    task_id: str | None = None
    provider: str | None = None
    model: str | None = None
    schema_version: int = 1


@dataclass(frozen=True)
class ToolCall:
    """
    Represents one tool invocation requested by an agent.

    ToolCall records the request only. Execution status, returned content,
    timing, and token measurements are recorded separately.

    Attributes:
        call_id: The unique identifier for this invocation.
        turn: The zero-based agent turn containing the request.
        sequence: The global event sequence number within the run.
        name: The provider-neutral tool name.
        arguments: The normalized JSON-compatible tool arguments.
        purpose: The optional explanation for why the tool was requested.
        event_type: The trajectory event discriminator.
    """

    call_id: str
    turn: int
    sequence: int
    name: str
    arguments: dict[str, JsonValue] = field(default_factory=dict)
    purpose: str | None = None
    event_type: Literal["tool_call"] = field(
        default="tool_call",
        init=False,
    )


@dataclass(frozen=True)
class ToolResult:
    """
    Represents the execution result of one ToolCall.

    ToolResult stores raw execution data. Derived token measurements and
    duplicate-context accounting are stored in ContextDelivery.

    Attributes:
        call_id: The identifier of the corresponding ToolCall.
        sequence: The global event sequence number within the run.
        status: The execution outcome.
        output: The textual output supplied to the agent.
        error: The error message for an unsuccessful execution.
        duration_ms: The elapsed execution time in milliseconds.
        event_type: The trajectory event discriminator.
    """

    call_id: str
    sequence: int
    status: ToolStatus
    output: str | None = None
    error: str | None = None
    duration_ms: float | None = None
    event_type: Literal["tool_result"] = field(
        default="tool_result",
        init=False,
    )


@dataclass(frozen=True)
class ModelUsage:
    """
    Represents provider-reported token usage for one model request.

    This record stores provider usage independently from locally estimated
    context tokens. Optional fields remain None when the provider does not
    expose the corresponding token category.

    Attributes:
        request_id: The unique identifier for the model request.
        turn: The zero-based agent turn associated with the request.
        sequence: The global event sequence number within the run.
        provider: The model provider name.
        model: The model name.
        input_tokens: The total provider-reported input tokens.
        output_tokens: The total provider-reported output tokens.
        cached_input_tokens: The optional cached portion of input tokens.
        reasoning_tokens: The optional provider-reported reasoning tokens.
        event_type: The trajectory event discriminator.
    """

    request_id: str
    turn: int
    sequence: int
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    event_type: Literal["model_usage"] = field(
        default="model_usage",
        init=False,
    )


@dataclass(frozen=True)
class ContentIdentity:
    """
    Represents the stable identity of one delivered context unit.

    The identity is produced from normalized content by content_ids.py.
    This model does not define the normalization or hashing algorithm.

    Attributes:
        content_id: The stable type-prefixed content identifier.
        kind: The semantic content type.
        digest: The digest of the normalized content.
        byte_length: The UTF-8 byte length of the normalized content.
    """

    content_id: str
    kind: str
    digest: str
    byte_length: int


@dataclass(frozen=True)
class ContextDelivery:
    """
    Represents token measurements for one context delivery.

    The first occurrence of a content ID is considered novel. Subsequent
    identical occurrences are considered repeated. The ledger is responsible
    for enforcing that tokens equals novel_tokens plus repeated_tokens.

    Attributes:
        call_id: The originating ToolCall identifier.
        sequence: The sequence number of the corresponding ToolResult.
        content_id: The stable identity of the delivered content.
        content_kind: The semantic type of the delivered content.
        byte_length: The normalized UTF-8 byte length.
        counter_name: The token-counter implementation used.
        tokens: The total tokens in this delivery.
        novel_tokens: The tokens not previously delivered in this run.
        repeated_tokens: The tokens previously delivered in identical content.
    """

    call_id: str
    sequence: int
    content_id: str
    content_kind: str
    byte_length: int
    counter_name: str
    tokens: int
    novel_tokens: int
    repeated_tokens: int


type TrajectoryEvent = ToolCall | ToolResult | ModelUsage

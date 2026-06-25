from dataclasses import dataclass, field

from .content_ids import identify_content
from .models import ContextDelivery, ToolResult
from .token_counter import TokenCounter


@dataclass
class TokenLedger:
    """
    Track delivered context and duplicate token expenditure.

    Attributes:
        counter: token counter used for all measurements.
        deliveries: measurements recorded in trajectory order.
    """

    counter: TokenCounter
    deliveries: list[ContextDelivery] = field(default_factory=list)
    _seen_content_ids: set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
    )

    def record(
        self,
        result: ToolResult,
        *,
        kind: str = "tool_output",
    ) -> ContextDelivery | None:
        """
        Measure and record a successful tool result.

        Empty and unsuccessful results are not context deliveries.

        Args:
            result: completed tool result.
            kind: semantic content type used in the content ID.

        Returns:
            Recorded delivery, or None when no output was delivered.
        """
        if result.status != "success" or not result.output:
            return None

        identity = identify_content(kind, result.output)
        tokens = self.counter.count(result.output)
        repeated = identity.content_id in self._seen_content_ids

        delivery = ContextDelivery(
            call_id=result.call_id,
            sequence=result.sequence,
            content_id=identity.content_id,
            content_kind=kind,
            byte_length=identity.byte_length,
            counter_name=self.counter.name,
            tokens=tokens,
            novel_tokens=0 if repeated else tokens,
            repeated_tokens=tokens if repeated else 0,
        )
        self.deliveries.append(delivery)
        self._seen_content_ids.add(identity.content_id)
        return delivery

    @property
    def total_tokens(self) -> int:
        """
        Return all delivered context tokens.
        """
        return sum(item.tokens for item in self.deliveries)

    @property
    def novel_tokens(self) -> int:
        """
        Return tokens from first-time content deliveries.
        """
        return sum(item.novel_tokens for item in self.deliveries)

    @property
    def repeated_tokens(self) -> int:
        """
        Return tokens from repeated identical content.
        """
        return sum(item.repeated_tokens for item in self.deliveries)

    @property
    def duplication_rate(self) -> float:
        """
        Return the fraction of delivered tokens that were repeated.
        """
        if self.total_tokens == 0:
            return 0.0
        return self.repeated_tokens / self.total_tokens

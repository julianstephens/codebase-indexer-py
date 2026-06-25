"""
tests/evaluation/test_ledger.py - Tests for context token accounting.

Coverage:
  - First and repeated deliveries
  - Different content and content kinds
  - Empty and unsuccessful results
  - Aggregate token metrics
  - Duplication rate
  - Counter metadata
"""

import pytest

from indexer.evaluation.ledger import TokenLedger
from indexer.evaluation.models import ToolResult


class FixedTokenCounter:
    """
    Counts one token per character for deterministic tests.
    """

    @property
    def name(self) -> str:
        return "fixed-character-counter"

    def count(self, text: str) -> int:
        return len(text)


def _result(
    output: str | None,
    *,
    call_id: str = "call-1",
    sequence: int = 1,
    status: str = "success",
) -> ToolResult:
    return ToolResult(
        call_id=call_id,
        sequence=sequence,
        status=status,  # type: ignore[arg-type]
        output=output,
    )


class TestTokenLedgerRecord:
    def test_first_delivery_is_entirely_novel(self) -> None:
        ledger = TokenLedger(counter=FixedTokenCounter())

        delivery = ledger.record(_result("abcd"))

        assert delivery is not None
        assert delivery.tokens == 4
        assert delivery.novel_tokens == 4
        assert delivery.repeated_tokens == 0

    def test_identical_second_delivery_is_entirely_repeated(self) -> None:
        ledger = TokenLedger(counter=FixedTokenCounter())

        first = ledger.record(_result("abcd", call_id="call-1", sequence=1))
        second = ledger.record(_result("abcd", call_id="call-2", sequence=2))

        assert first is not None
        assert second is not None
        assert first.content_id == second.content_id
        assert second.tokens == 4
        assert second.novel_tokens == 0
        assert second.repeated_tokens == 4

    def test_different_content_is_novel(self) -> None:
        ledger = TokenLedger(counter=FixedTokenCounter())

        first = ledger.record(_result("abcd", call_id="call-1", sequence=1))
        second = ledger.record(_result("efgh", call_id="call-2", sequence=2))

        assert first is not None
        assert second is not None
        assert first.content_id != second.content_id
        assert second.novel_tokens == 4
        assert second.repeated_tokens == 0

    def test_same_text_with_different_kind_is_novel(self) -> None:
        ledger = TokenLedger(counter=FixedTokenCounter())

        source = ledger.record(
            _result("abcd", call_id="call-1", sequence=1),
            kind="symbol_source",
        )
        search = ledger.record(
            _result("abcd", call_id="call-2", sequence=2),
            kind="search_result",
        )

        assert source is not None
        assert search is not None
        assert source.content_id != search.content_id
        assert search.novel_tokens == 4

    @pytest.mark.parametrize("status", ["error", "cancelled"])
    def test_unsuccessful_result_is_not_recorded(
        self,
        status: str,
    ) -> None:
        ledger = TokenLedger(counter=FixedTokenCounter())

        delivery = ledger.record(_result("output", status=status))

        assert delivery is None
        assert ledger.deliveries == []

    @pytest.mark.parametrize("output", [None, ""])
    def test_empty_output_is_not_recorded(
        self,
        output: str | None,
    ) -> None:
        ledger = TokenLedger(counter=FixedTokenCounter())

        delivery = ledger.record(_result(output))

        assert delivery is None
        assert ledger.deliveries == []

    def test_whitespace_output_is_recorded(self) -> None:
        ledger = TokenLedger(counter=FixedTokenCounter())

        delivery = ledger.record(_result(" "))

        assert delivery is not None
        assert delivery.tokens == 1

    def test_delivery_keeps_source_metadata(self) -> None:
        ledger = TokenLedger(counter=FixedTokenCounter())

        delivery = ledger.record(
            _result("abcd", call_id="call-9", sequence=12),
            kind="symbol_source",
        )

        assert delivery is not None
        assert delivery.call_id == "call-9"
        assert delivery.sequence == 12
        assert delivery.content_kind == "symbol_source"
        assert delivery.counter_name == "fixed-character-counter"
        assert delivery.byte_length == 4


class TestTokenLedgerAggregates:
    def test_empty_ledger_totals_are_zero(self) -> None:
        ledger = TokenLedger(counter=FixedTokenCounter())

        assert ledger.total_tokens == 0
        assert ledger.novel_tokens == 0
        assert ledger.repeated_tokens == 0
        assert ledger.duplication_rate == 0.0

    def test_aggregate_totals(self) -> None:
        ledger = TokenLedger(counter=FixedTokenCounter())
        ledger.record(_result("abcd", call_id="call-1", sequence=1))
        ledger.record(_result("abcd", call_id="call-2", sequence=2))
        ledger.record(_result("xy", call_id="call-3", sequence=3))

        assert ledger.total_tokens == 10
        assert ledger.novel_tokens == 6
        assert ledger.repeated_tokens == 4
        assert ledger.total_tokens == ledger.novel_tokens + ledger.repeated_tokens

    def test_duplication_rate(self) -> None:
        ledger = TokenLedger(counter=FixedTokenCounter())
        ledger.record(_result("abcd", call_id="call-1", sequence=1))
        ledger.record(_result("abcd", call_id="call-2", sequence=2))

        assert ledger.duplication_rate == pytest.approx(0.5)

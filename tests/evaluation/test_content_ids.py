"""
tests/evaluation/test_content_ids.py - Tests for stable content identities.

Coverage:
  - Deterministic hashing
  - Line-ending normalization
  - Kind-specific identifiers
  - UTF-8 byte lengths
  - Empty content
"""

from indexer.evaluation.content_ids import identify_content


class TestIdentifyContent:
    def test_same_content_has_same_identity(self) -> None:
        first = identify_content("symbol_source", "def run():\n    pass\n")
        second = identify_content("symbol_source", "def run():\n    pass\n")

        assert first == second

    def test_line_endings_are_normalized(self) -> None:
        unix = identify_content("symbol_source", "a\nb\n")
        windows = identify_content("symbol_source", "a\r\nb\r\n")
        classic_mac = identify_content("symbol_source", "a\rb\r")

        assert unix.digest == windows.digest
        assert unix.digest == classic_mac.digest
        assert unix.content_id == windows.content_id
        assert unix.content_id == classic_mac.content_id

    def test_kind_changes_content_id(self) -> None:
        source = identify_content("symbol_source", "same content")
        search = identify_content("search_result", "same content")

        assert source.digest == search.digest
        assert source.content_id != search.content_id
        assert source.kind == "symbol_source"
        assert search.kind == "search_result"

    def test_content_id_contains_kind_and_algorithm(self) -> None:
        identity = identify_content("tool_output", "hello")

        assert identity.content_id.startswith("tool_output:sha256:")
        assert identity.content_id.endswith(identity.digest)

    def test_byte_length_uses_normalized_utf8(self) -> None:
        identity = identify_content("tool_output", "é\r\n")

        assert identity.byte_length == len("é\n".encode("utf-8"))

    def test_empty_content_has_valid_identity(self) -> None:
        identity = identify_content("tool_output", "")

        assert identity.byte_length == 0
        assert len(identity.digest) == 64

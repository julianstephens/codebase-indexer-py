"""
token_counter.py - Token counting interfaces and implementations.

Defines the token-counter protocol used by evaluation code and the default
character-based approximation retained for compatibility with existing
benchmarks.
"""

from dataclasses import dataclass
from typing import Protocol

from indexer.errors import InvalidHeuristicError


class TokenCounter(Protocol):
    """
    Counts tokens for benchmark input.
    """

    @property
    def name(self) -> str:
        """
        Return a stable counter identifier.
        """
        ...

    def count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: The text to measure.

        Returns:
            A non-negative token count.
        """
        ...


@dataclass(frozen=True)
class HeuristicTokenCounter:
    """
    Approximates tokens using a fixed characters-per-token ratio.

    Attributes:
        characters_per_token: The approximate number of characters per token.
    """

    characters_per_token: int = 4

    def __post_init__(self) -> None:
        """
        Validate the configured ratio.

        Raises:
            InvalidHeuristicError: If characters_per_token is not positive.
        """
        if self.characters_per_token <= 0:
            raise InvalidHeuristicError(message="characters_per_token must be positive")

    @property
    def name(self) -> str:
        """
        Return a stable counter identifier.
        """
        return f"heuristic-chars-{self.characters_per_token}"

    def count(self, text: str) -> int:
        """
        Estimate tokens in text.

        Args:
            text: The text to measure.

        Returns:
            The approximate token count.
        """
        if not text:
            return 0
        return max(1, len(text) // self.characters_per_token)

from typing import Protocol


class TokenCounter(Protocol):
    """
    Count tokens for benchmark input.

    Implementations may use a heuristic, a model tokenizer, or usage values
    reported directly by a provider.
    """

    character_per_token: int = 4

    @property
    def name(self) -> str:
        """
        Return a stable counter identifier.
        """
        return f"heuristic-chars-{self.character_per_token}"

    def count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: text to measure.

        Returns:
            Non-negative token count.
        """
        if not text:
            return 0
        return max(1, len(text) // self.character_per_token)

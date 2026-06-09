"""Prompt generator for creating exact-length prompts."""

import tiktoken
from .prompts import get_template, FILLER_TEXT


class PromptGenerator:
    """Generate prompts with exact token counts."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize the prompt generator.

        Args:
            encoding_name: Name of the tiktoken encoding to use
                          (cl100k_base is used by GPT-4 and similar models)
        """
        self.encoder = tiktoken.get_encoding(encoding_name)

    def generate_prompt(self, target_tokens: int, template: str = "qa") -> str:
        """Generate a prompt with exactly target_tokens tokens.

        Args:
            target_tokens: Target number of tokens
            template: Template type ('qa', 'code', or 'creative')

        Returns:
            A prompt string with exactly target_tokens tokens

        Raises:
            ValueError: If template is not recognized or target_tokens is invalid
        """
        if target_tokens < 10:
            raise ValueError("target_tokens must be at least 10")

        # Work at the token level so this stays O(target_tokens): the filler is
        # encoded once and tiled, rather than re-encoding a growing string each
        # loop (which was O(n^2) and unusable at 100k+ tokens).
        base_tokens = self.encoder.encode(get_template(template))
        if len(base_tokens) >= target_tokens:
            return self._normalize_to_exact_tokens(
                self.encoder.decode(base_tokens[:target_tokens]), target_tokens
            )

        filler_tokens = self.encoder.encode("\n\n" + FILLER_TEXT)
        if not filler_tokens:
            return self._normalize_to_exact_tokens(
                self.encoder.decode(base_tokens[:target_tokens]), target_tokens
            )

        tokens = list(base_tokens)
        while len(tokens) < target_tokens:
            need = target_tokens - len(tokens)
            tokens.extend(filler_tokens if need >= len(filler_tokens) else filler_tokens[:need])

        return self._normalize_to_exact_tokens(
            self.encoder.decode(tokens[:target_tokens]), target_tokens
        )

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self.encoder.encode(text))

    def _trim_to_exact_tokens(self, text: str, target: int) -> str:
        """Trim text to exact token count.

        Args:
            text: Text to trim
            target: Target token count

        Returns:
            Text trimmed to exactly target tokens
        """
        tokens = self.encoder.encode(text)
        if len(tokens) <= target:
            return text

        return self._normalize_to_exact_tokens(self.encoder.decode(tokens[:target]), target)

    def _normalize_to_exact_tokens(self, text: str, target: int) -> str:
        """Normalize decoded token slices back to an exact re-encoded count."""
        for _ in range(10):
            tokens = self.encoder.encode(text)
            diff = target - len(tokens)
            if diff == 0:
                return text
            if diff > 0:
                text += " data" * diff
                continue
            text = self.encoder.decode(tokens[:target])
        raise RuntimeError(f"Could not normalize prompt to {target} tokens")

    def validate_token_count(self, text: str, expected: int, tolerance: int = 0) -> bool:
        """Validate that text has expected token count.

        Args:
            text: Text to validate
            expected: Expected token count
            tolerance: Acceptable deviation from expected count

        Returns:
            True if token count is within tolerance, False otherwise
        """
        actual = self._count_tokens(text)
        return abs(actual - expected) <= tolerance

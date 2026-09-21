from .State import State
from .ParameterEndState import ParameterEndState
from ..context import DecoderContext


class StringState(State):
    """Constrains generation to a valid JSON string.

    Unlike NumberState, a string has a natural terminator: an unescaped
    closing quote. That means "stop" can simply be offered as one of the
    normally-allowed tokens at every step (once not mid-escape), rather
    than requiring the two-phase "detect the boundary, then force the
    rest of a fixed string" trick NumberState needs.
    """

    def __init__(
        self,
        context: DecoderContext,
        start_position: int,
    ) -> None:
        self.context = context
        self.start_position = start_position

    def _get_value_tokens(
        self,
        generated_tokens: list[int],
    ) -> list[int]:
        return generated_tokens[self.start_position:]

    def _get_value(
        self,
        generated_tokens: list[int],
    ) -> str:
        tokens = self._get_value_tokens(generated_tokens)
        return self.context.decode_tokens(tokens)

    @staticmethod
    def _has_open_escape(value: str) -> bool:
        backslashes = 0

        for char in reversed(value):
            if char != "\\":
                break

            backslashes += 1

        return backslashes % 2 == 1

    @staticmethod
    def _is_valid_escape(value: str) -> bool:
        """Return True if every backslash-escape started in value is legal.

        Scans from the start rather than only inspecting the very end:
        checking the trailing backslash count of the *post-append*
        candidate is not enough, because appending almost any single
        character "resolves" a pending escape (the string no longer ends
        in an odd run of backslashes) regardless of whether that character
        was actually a legal escape code. `\\x` would slip through a
        trailing-only check even though `x` is not a valid JSON escape.
        """
        index = 0
        length = len(value)

        while index < length:
            if value[index] == "\\":
                if index + 1 >= length:
                    # Trailing backslash: escape not resolved yet, but not
                    # invalid either -- more text may still complete it.
                    return True

                if value[index + 1] not in '"\\/bfnrt':
                    return False

                index += 2
            else:
                index += 1

        return True

    @staticmethod
    def _is_closing_quote(value: str) -> bool:
        if not value.endswith('"'):
            return False

        return not StringState._has_open_escape(value[:-1])

    def get_allowed_tokens(
        self,
        generated_tokens: list[int],
    ) -> list[int]:
        current = self._get_value(generated_tokens)
        allowed: list[int] = []

        for token_id in self.context.token_text_cache:
            token_text = self.context.get_token_text(token_id)
            candidate = current + token_text

            # An unescaped quote closes the string.
            if self._is_closing_quote(candidate):
                continue

            # Don't allow invalid escape sequences.
            if not self._is_valid_escape(candidate):
                continue

            # JSON strings cannot contain raw control characters.
            if any(
                ord(char) < 0x20
                for char in token_text
                if char not in "\n\t\r"
            ):
                continue

            # Newlines, tabs and carriage returns must also be escaped.
            if "\n" in token_text or "\r" in token_text or "\t" in token_text:
                continue

            allowed.append(token_id)

        # If we're not currently in the middle of an escape,
        # the closing quote is always a possible next token.
        if not self._has_open_escape(current):
            allowed.extend(
                self.context.get_token_ids('"')
            )

        return allowed

    def update(
        self,
        generated_tokens: list[int],
    ) -> State | None:
        value = self._get_value(generated_tokens)

        if self._is_closing_quote(value):
            return ParameterEndState(
                self.context,
                len(generated_tokens),
            )

        return None

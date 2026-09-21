from .State import State
from ..context import DecoderContext


class NumberState(State):
    """Constrains generation to a valid JSON number.

    Unlike FunctionNameState/ParameterNameState, the set of valid numbers
    isn't a small closed vocabulary, so this can't use a TokenTrie. Instead
    it checks, for every token in the vocabulary, whether appending that
    token's text to what's been generated so far still forms a valid
    *prefix* of a JSON number.

    A JSON number has no terminating character (unlike a string's closing
    quote), so "the value is already a complete number" is not, by itself,
    a reason to stop: "4" is a complete number, but so is "42" and "425".
    Whenever the current value is already complete, this state additionally
    offers the first token of whatever fixed text follows a parameter value
    (a comma to loop back for another parameter, or the closing braces) as
    one more legal choice alongside every digit-continuation token, and lets
    the model's own logits decide whether to keep extending the number or
    stop. Once that boundary token is chosen, the state forces the rest of
    that fixed text itself (the same way SuffixState/ParameterPrefixState
    force their own fixed text) before handing off.
    """

    def __init__(
        self,
        context: DecoderContext,
        start_position: int,
    ) -> None:
        self.context = context
        self.start_position = start_position
        self._closing_tokens: list[int] = []
        self._closing_start: int | None = None
        self._done = False

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
    def _strip_leading_space(value: str) -> str:
        """Strip exactly one leading space, if present.

        Since ParameterValuePrefixState no longer forces the space before
        a number (see its docstring for why), the model's first token for
        the value is free to include a single leading space of its own --
        JSON permits arbitrary whitespace between `:` and the value, so
        `"a": -2` and `"a":-2` are equally valid. Only one leading space is
        tolerated, not arbitrary whitespace: this is meant to accommodate
        one real, commonly-merged tokenization boundary, not to loosen the
        number grammar generally.
        """
        if value.startswith(" "):
            return value[1:]

        return value

    @staticmethod
    def _is_valid_prefix(value: str) -> bool:
        """Return True if value could be extended into a valid JSON number."""
        value = NumberState._strip_leading_space(value)

        if value == "":
            return True

        if value == "-":
            return True

        if value.startswith("-"):
            value = value[1:]

        if value == "":
            return True

        if value.startswith("."):
            return False

        if value.startswith("0"):
            if value == "0":
                return True

            return (
                value.startswith("0.")
                or value.startswith("0e")
                or value.startswith("0E")
            )

        if not value[0].isdigit():
            return False

        index = 0

        while index < len(value) and value[index].isdigit():
            index += 1

        remainder = value[index:]

        if remainder == "":
            return True

        if remainder.startswith("."):
            fraction = remainder[1:]

            if fraction == "":
                return True

            return fraction.isdigit()

        if remainder.startswith(("e", "E")):
            exponent = remainder[1:]

            if exponent == "":
                return True

            if exponent[0] in "+-":
                exponent = exponent[1:]

            if exponent == "":
                return True

            return exponent.isdigit()

        return False

    @staticmethod
    def _is_complete(value: str) -> bool:
        """Return True if value is already a fully valid JSON number."""
        value = NumberState._strip_leading_space(value)

        if value == "":
            return False

        if value == "-":
            return False

        if value.endswith("."):
            return False

        if value.endswith(("e", "E")):
            return False

        if value.endswith(("e+", "e-", "E+", "E-")):
            return False

        try:
            float(value)
        except ValueError:
            return False

        return True

    def get_allowed_tokens(
        self,
        generated_tokens: list[int],
    ) -> list[int]:
        if self._done:
            return []

        if self._closing_start is not None:
            position = len(generated_tokens) - self._closing_start

            if position >= len(self._closing_tokens):
                return []

            return [self._closing_tokens[position]]

        current = self._get_value(generated_tokens)
        allowed: list[int] = []

        for token_id in self.context.token_text_cache:
            token_text = self.context.get_token_text(token_id)
            candidate = current + token_text

            if self._is_valid_prefix(candidate):
                allowed.append(token_id)

        if self._is_complete(current):
            closing_tokens = self._get_closing_tokens()

            if closing_tokens and closing_tokens[0] not in allowed:
                allowed.append(closing_tokens[0])

        return allowed

    def update(
        self,
        generated_tokens: list[int],
    ) -> State | None:
        if self._done:
            return None

        if self._closing_start is not None:
            position = len(generated_tokens) - self._closing_start

            if position < len(self._closing_tokens):
                return None

            return self._finish_parameter(generated_tokens)

        value_tokens = self._get_value_tokens(generated_tokens)

        if not value_tokens:
            return None

        # Check the value *before* the just-generated token, to tell apart
        # a genuine extra digit from the boundary token offered above.
        prior_value = self.context.decode_tokens(value_tokens[:-1])
        last_token = value_tokens[-1]

        if self._is_complete(prior_value):
            closing_tokens = self._get_closing_tokens()

            if closing_tokens and last_token == closing_tokens[0]:
                self._closing_tokens = closing_tokens
                self._closing_start = len(generated_tokens) - 1
                return None

        return None

    def _get_closing_tokens(self) -> list[int]:
        closing_text = self.context.get_parameter_end_text()
        tokens: list[int] = self.context.model.encode(closing_text)[0].tolist()
        return tokens

    def _finish_parameter(self, generated_tokens: list[int]) -> State | None:
        if self.context.current_parameter is not None:
            self.context.generated_parameters.add(self.context.current_parameter)
            self.context.current_parameter = None

        function = self.context.selected_function
        remaining = (
            set(function.parameters) - self.context.generated_parameters
            if function is not None
            else set()
        )

        if not remaining:
            self._done = True
            return None

        # Imported here rather than at module level to break the same
        # import cycle ParameterEndState avoids: ParameterNameState ->
        # ParameterValuePrefixState -> NumberState.
        from .ParameterNameState import ParameterNameState

        return ParameterNameState(self.context, len(generated_tokens))
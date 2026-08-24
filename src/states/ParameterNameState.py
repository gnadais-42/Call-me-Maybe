from .State import State
from .ParameterValuePrefixState import ParameterValuePrefixState
from ..context import DecoderContext
from ..trie import TokenTrie


class ParameterNameState(State):
    def __init__(self, context: DecoderContext, start_position: int) -> None:
        self.context = context
        self.start_position = start_position
        self.parameter_trie: TokenTrie = (context.get_parameter_trie())

    def get_allowed_tokens(self, generated_tokens: list[int]) -> list[int]:
        parameter_tokens = generated_tokens[self.start_position:]

        return self.parameter_trie.get_next_tokens(
            parameter_tokens
        )

    def update(self, generated_tokens: list[int]) -> State | None:
        parameter_tokens = generated_tokens[self.start_position:]

        if self.parameter_trie.is_terminal(parameter_tokens):
            parameter_name = self.context.model.decode(
                parameter_tokens
            )

            self.context.current_parameter = parameter_name

            return ParameterValuePrefixState(
                self.context,
                len(generated_tokens),
            )

        return None
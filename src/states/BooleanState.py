from .State import State
from .ParameterEndState import ParameterEndState
from ..context import DecoderContext
from ..trie import TokenTrie


class BooleanState(State):
    """Constrains generation to exactly the JSON literal `true` or `false`.

    A closed two-entry set is exactly what TokenTrie is for, so this
    reuses the same pattern as FunctionNameState/ParameterNameState rather
    than a bespoke character-level check like NumberState/StringState use.
    """

    def __init__(self, context: DecoderContext, start_position: int) -> None:
        self.context = context
        self.start_position = start_position
        self.trie = TokenTrie()
        self.trie.insert(context.model.encode("true")[0].tolist())
        self.trie.insert(context.model.encode("false")[0].tolist())

    def get_allowed_tokens(self, generated_tokens: list[int]) -> list[int]:
        value_tokens = generated_tokens[self.start_position:]

        return self.trie.get_next_tokens(value_tokens)

    def update(self, generated_tokens: list[int]) -> State | None:
        value_tokens = generated_tokens[self.start_position:]

        if self.trie.is_terminal(value_tokens):
            return ParameterEndState(
                self.context,
                len(generated_tokens),
            )

        return None

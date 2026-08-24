from .State import State
from .SuffixState import SuffixState
from ..trie import TokenTrie
from ..context import DecoderContext


class FunctionNameState(State):
    def __init__(self, context: DecoderContext, start_position: int) -> None:
        self.context = context
        self.function_trie: TokenTrie = context.get_function_trie()
        self.start_position = start_position

    def get_allowed_tokens(self, generated_tokens: list[int]) -> list[int]:
        function_tokens = generated_tokens[self.start_position:]

        return self.function_trie.get_next_tokens(
            function_tokens
        )

    def update(self, generated_tokens: list[int]) -> State | None:
        function_tokens = generated_tokens[self.start_position:]

        if self.function_trie.is_terminal(function_tokens):
            self.context.selected_function = self.context.get_function(function_tokens)
            return SuffixState(self.context, len(generated_tokens))

        return None
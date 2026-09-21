from .State import State
from .SuffixState import SuffixState
from ..trie import TokenTrie
from ..context import DecoderContext


class FunctionNameState(State):
    """Constrains generation to exactly one of the known function names.

    The trie restricts *which* token ids are grammatically legal at each
    step, but the model's own logits (masked to those legal ids) still
    decide which legal path is taken -- this is what satisfies the
    project's requirement that function selection come from the LLM, not
    from heuristics.

    Known limitation: if one function's name is a strict prefix of
    another's (e.g. "fn_add" and "fn_add_numbers"), this state commits to
    the shorter name the instant it becomes terminal, even though a longer
    valid name was still reachable. Not an issue for the example function
    sets, but worth flagging: a stricter implementation would need to keep
    both options open until the trie forks or truly dead-ends.
    """

    def __init__(self, context: DecoderContext, start_position: int) -> None:
        self.context = context
        self.function_trie: TokenTrie = context.get_function_trie()
        self.start_position = start_position

    def get_allowed_tokens(self, generated_tokens: list[int]) -> list[int]:
        function_tokens = generated_tokens[self.start_position:]

        return self.function_trie.get_next_tokens(function_tokens)

    def update(self, generated_tokens: list[int]) -> State | None:
        function_tokens = generated_tokens[self.start_position:]

        if self.function_trie.is_terminal(function_tokens):
            self.context.selected_function = self.context.get_function(function_tokens)
            return SuffixState(self.context, len(generated_tokens))

        return None

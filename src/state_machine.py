from enum import Enum, auto
from .models import FunctionDefinition
from .trie import TokenTrie
from llm_sdk import Small_LLM_Model



class DecoderState(Enum):
    START = auto()
    FUNCTION_NAME = auto()
    FUNCTION_SUFFIX = auto()
    PARAMETERS = auto()


class DecoderContext:
    def __init__(self) -> None:
        self.state: DecoderState = DecoderState.START
        self.selected_function: FunctionDefinition | None = None
        self.current_parameter: str | None = None

class StateMachine:
    def __init__(self, model: Small_LLM_Model, function_trie: TokenTrie, functions: list[FunctionDefinition]) -> None:
        self.model = model
        self.context = DecoderContext()
        self.function_trie = function_trie
        self.prefix = self._encode('{\n   "name": "')
        self.function_suffix = self._encode('",\n  "parameters": {\n')
        self.function_len = 0
        self.functions = functions

    def _encode(self, text: str) -> list[int]:
        return self.model.encode(text)[0].tolist()

    def update_state(self, generated_tokens: list[int]) -> None:
        if (
            self.context.state == DecoderState.START
            and len(generated_tokens) == len(self.prefix)
        ):
            self.context.state = DecoderState.FUNCTION_NAME
        elif self.context.state == DecoderState.FUNCTION_NAME:
            function_tokens = self._get_function_tokens(
                generated_tokens
            )

            if self.function_trie.is_terminal(function_tokens):
                function_name = self.model.decode(function_tokens)

                for function in self.functions:
                    if function.name == function_name:
                        self.context.selected_function = function
                        break

                self.context.state = DecoderState.FUNCTION_SUFFIX
                self.function_len = len(function_tokens)

    def get_allowed_tokens(self, generated_tokens: list[int]) -> list[int]:
        if self.context.state == DecoderState.START:
            return self._get_fixed_tokens(
                generated_tokens,
                self.prefix
            )

        if self.context.state == DecoderState.FUNCTION_NAME:
            function_tokens = self._get_function_tokens(
                generated_tokens
            )
            return self.function_trie.get_next_tokens(
                function_tokens
            )

        if self.context.state == DecoderState.FUNCTION_SUFFIX:
            return self._get_suffix_tokens(
                            generated_tokens,
                            self.function_suffix
                        )

        return []

    @staticmethod
    def _get_fixed_tokens(generated_tokens: list[int], expected_tokens: list[int]) -> list[int]:
        position = len(generated_tokens)

        if position >= len(expected_tokens):
            return []

        return [expected_tokens[position]]

    def _get_function_tokens(
        self,
        generated_tokens: list[int],
    ) -> list[int]:
        prefix_length = len(self.prefix)

        return generated_tokens[prefix_length:]

    def _get_suffix_tokens(
        self,
        generated_tokens: list[int],
        expected_tokens: list[int]
    ) -> list[int]:
        prefix_length = len(self.prefix)
        position = len(generated_tokens) - prefix_length - self.function_len

        if position >= len(expected_tokens):
            return []

        return [expected_tokens[position]]
from .models import FunctionDefinition
from .trie import TokenTrie
from llm_sdk import Small_LLM_Model


class DecoderContext:
    def __init__(self, model: Small_LLM_Model, functions: list[FunctionDefinition]) -> None:
        self.model = model
        self.functions = functions
        self.selected_function: FunctionDefinition | None = None
        self.current_parameter: str | None = None
        self.generated_parameters: set[str] = set()

    def get_function_trie(self) -> TokenTrie:
        trie = TokenTrie()

        for function in self.functions:
            encoded: list[int] = self.model.encode(function.name)[0].tolist()
            trie.insert(encoded)

        return trie

    def get_function(self, function_tokens: list[int]) -> FunctionDefinition | None:
        name = self.model.decode(function_tokens)

        for function in self.functions:
            if function.name == name:
                return function

        return None

    def get_parameter_trie(self) -> TokenTrie:
        trie = TokenTrie()

        if self.selected_function is None:
            return trie

        for name in self.selected_function.parameters:
            tokens = self.model.encode(name)[0].tolist()
            trie.insert(tokens)

        return trie
import json

from .models import FunctionDefinition
from .trie import TokenTrie
from llm_sdk import Small_LLM_Model

TokenCache = tuple[dict[int, str], dict[str, list[int]]]


def build_token_cache(model: Small_LLM_Model) -> TokenCache:
    """Precompute the decoded text of every token in the vocabulary.

    This calls decode() once per vocabulary entry (tens of thousands of
    calls for a typical BPE vocabulary). The vocabulary never changes
    across prompts within a run, so this is meant to be called *once per
    process* and its result passed into every DecoderContext, rather than
    rebuilt from scratch for every prompt -- turning an
    O(num_prompts * vocab_size) cost into O(vocab_size).

    Returns:
        A (token_text_cache, token_ids_by_text) pair: token id -> decoded
        text, and the reverse mapping, decoded text -> every token id that
        decodes to exactly that text.
    """
    path_to_file = model.get_path_to_vocab_file()

    with open(path_to_file) as file:
        vocabulary = json.load(file)

    token_text_cache: dict[int, str] = {}
    token_ids_by_text: dict[str, list[int]] = {}

    for token_id in vocabulary.values():
        text = model.decode([token_id])
        token_text_cache[token_id] = text
        token_ids_by_text.setdefault(text, []).append(token_id)

    return token_text_cache, token_ids_by_text


class DecoderContext:
    """Shared state and helpers used by every state in the state machine.

    One DecoderContext (and therefore one StateMachine) is created per
    natural language prompt. It owns:

    - the model handle,
    - a cache mapping every vocabulary token id to its decoded text, and
      the reverse mapping (text -> token ids). Callers processing more
      than one prompt should build this once via build_token_cache() and
      pass it in via `token_cache`, rather than let each DecoderContext
      rebuild it redundantly (see build_token_cache's docstring),
    - the function selected so far (if any), and
    - which parameter is currently being generated and which parameters
      are already complete, so the grammar can decide whether to loop
      back for another parameter or close the JSON object.
    """

    def __init__(
        self,
        model: Small_LLM_Model,
        functions: list[FunctionDefinition],
        token_cache: TokenCache | None = None,
    ) -> None:
        self.model = model
        self.functions = functions
        self.selected_function: FunctionDefinition | None = None
        self.current_parameter: str | None = None
        self.generated_parameters: set[str] = set()

        if token_cache is not None:
            self.token_text_cache, self.token_ids_by_text = token_cache
        else:
            self.token_text_cache, self.token_ids_by_text = build_token_cache(model)

    def get_function_trie(self) -> TokenTrie:
        """Build a trie over every candidate function's name (in tokens)."""
        trie = TokenTrie()

        for function in self.functions:
            encoded: list[int] = self.model.encode(function.name)[0].tolist()
            trie.insert(encoded)

        return trie

    def get_function(self, function_tokens: list[int]) -> FunctionDefinition | None:
        """Resolve generated name tokens back to a FunctionDefinition."""
        name = self.model.decode(function_tokens)

        for function in self.functions:
            if function.name == name:
                return function

        return None

    def get_parameter_trie(self) -> TokenTrie:
        """Build a trie over the parameters not yet generated.

        Excluding already-generated parameter names here is what prevents
        the model from producing the same parameter twice: once a name has
        been consumed (tracked in generated_parameters), it simply stops
        being a valid path through this trie.
        """
        trie = TokenTrie()

        if self.selected_function is None:
            return trie

        remaining = set(self.selected_function.parameters) - self.generated_parameters

        for name in remaining:
            tokens = self.model.encode(name)[0].tolist()
            trie.insert(tokens)

        return trie

    def get_token_text(self, token_id: int) -> str:
        """Return the cached decoded text for a single token id."""
        return self.token_text_cache[token_id]

    def get_token_ids(self, text: str) -> list[int]:
        """Return every token id whose decoded text matches exactly.

        Returns an empty list if no token decodes to exactly this text.
        Used by states that need to force a specific literal (e.g. the
        closing quote of a string) without hardcoding a token id.
        """
        return self.token_ids_by_text.get(text, [])

    def get_parameter_end_text(self) -> str:
        """Return the fixed text that follows a just-completed parameter value.

        This mirrors the decision ParameterEndState makes, but as a pure
        query with no side effects, so states without a natural terminator
        character (currently only NumberState) can preview it and offer it
        as an explicit "stop here" option alongside "keep extending the
        value" -- letting the model's own logits choose between them.
        """
        function = self.selected_function

        if function is None:
            return "\n  }\n}"

        done = set(self.generated_parameters)
        if self.current_parameter is not None:
            done.add(self.current_parameter)

        remaining = set(function.parameters) - done

        return ',\n    "' if remaining else "\n  }\n}"

    def decode_tokens(self, tokens: list[int]) -> str:
        """Decode a sequence of generated tokens into text.

        Used to reconstruct the JSON produced so far, both mid-generation
        (to check e.g. "is this a valid number prefix?") and at the end
        (to get the final JSON string to parse).
        """
        result = ""

        for token in tokens:
            result += self.model.decode([token])

        return result

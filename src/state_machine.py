from .models import FunctionDefinition
from .context import DecoderContext, TokenCache
from .states.State import State
from .states.StartState import StartState
from llm_sdk import Small_LLM_Model


class StateMachine:
    """Drives the constrained-decoding grammar for a single prompt.

    Owns a DecoderContext (the shared, mutable state used across every
    grammar state) and the currently active State. The generation loop
    (see generation.py) only ever talks to this class: ask which tokens
    are allowed right now, append the model's chosen token, then tell the
    machine to update -- it internally swaps states as each grammar
    segment completes.
    """

    def __init__(
        self,
        model: Small_LLM_Model,
        functions: list[FunctionDefinition],
        token_cache: TokenCache | None = None,
    ) -> None:
        self.context = DecoderContext(model, functions, token_cache)
        self.state: State = StartState(self.context)

    def get_allowed_tokens(self, generated_tokens: list[int]) -> list[int]:
        """Return the token ids allowed as the next generated token."""
        return self.state.get_allowed_tokens(generated_tokens)

    def update(self, generated_tokens: list[int]) -> None:
        """Advance the machine's state after a token has been generated."""
        new_state = self.state.update(generated_tokens)

        if new_state is not None:
            self.state = new_state

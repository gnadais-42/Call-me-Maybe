from .models import FunctionDefinition
from .context import DecoderContext
from .states.State import State
from .states.StartState import StartState
from .states.FunctionNameState import FunctionNameState
from .states.SuffixState import SuffixState
from llm_sdk import Small_LLM_Model


class StateMachine:

    def __init__(self, model: Small_LLM_Model,
        functions: list[FunctionDefinition],
    ) -> None:
        self.context = DecoderContext(model, functions)

        self.state: State = StartState(
            self.context,
        )

    def get_allowed_tokens(self, generated_tokens: list[int]) -> list[int]:
        return self.state.get_allowed_tokens(
            generated_tokens
        )

    def update(self, generated_tokens: list[int]) -> None:
        new_state = self.state.update(
            generated_tokens
        )

        if new_state is not None:
            self.state = new_state
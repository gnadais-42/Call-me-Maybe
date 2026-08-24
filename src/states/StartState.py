from .State import State
from .FunctionNameState import FunctionNameState
from ..context import DecoderContext

class StartState(State):
    def __init__(self, context: DecoderContext):
        self.context = context
        self.prefix: list[int] = context.model.encode('{\n  "name": "')[0].tolist()

    def get_allowed_tokens(self, generated_tokens: list[int]) -> list[int]:
        position = len(generated_tokens)

        if position >= len(self.prefix):
            return []

        return [self.prefix[position]]

    def update(self, generated_tokens: list[int]) -> State | None:
        if len(generated_tokens) >= len(self.prefix):
            return FunctionNameState(self.context, len(self.prefix))

        return None
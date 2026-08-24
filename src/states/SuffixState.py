from .State import State
from ..context import DecoderContext
from .ParameterPrefixState import ParameterPrefixState


class SuffixState(State):
    def __init__(self, context: DecoderContext, start_position: int):
        self.context = context
        self.start_position = start_position
        self.suffix = context.model.encode('",\n  "parameters": {\n')[0].tolist()

    def get_allowed_tokens(self, generated_tokens: list[int]) -> list[int]:
        position = len(generated_tokens) - self.start_position

        if position >= len(self.suffix):
            return []

        return [self.suffix[position]]

    def update(self, generated_tokens: list[int]) -> State | None:
        generated_len = len(generated_tokens) - self.start_position

        if generated_len >= len(self.suffix):
            return ParameterPrefixState(self.context, len(generated_tokens))

        return None
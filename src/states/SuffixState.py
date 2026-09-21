from .State import State
from ..context import DecoderContext
from .ParameterPrefixState import ParameterPrefixState
from .ParameterEndState import ParameterEndState


class SuffixState(State):
    """Forces the fixed text `",\\n  "parameters": {\\n` after the name.

    Once that text is done, generation either continues into the first
    parameter's name (ParameterPrefixState), or -- if the selected function
    takes no parameters at all -- jumps straight to ParameterEndState,
    which will see there is nothing left to generate and close the object.
    """

    def __init__(self, context: DecoderContext, start_position: int) -> None:
        self.context = context
        self.start_position = start_position
        self.suffix = context.model.encode('",\n  "parameters": {\n')[0].tolist()

    def get_allowed_tokens(self, generated_tokens: list[int]) -> list[int]:
        position = len(generated_tokens) - self.start_position

        if position >= len(self.suffix):
            return []

        return [self.suffix[position]]

    def update(self, generated_tokens: list[int]) -> State | None:
        position = len(generated_tokens) - self.start_position

        if position < len(self.suffix):
            return None

        function = self.context.selected_function

        if function is not None and function.parameters:
            return ParameterPrefixState(self.context, len(generated_tokens))

        return ParameterEndState(self.context, len(generated_tokens))

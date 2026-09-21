from .State import State
from .ParameterNameState import ParameterNameState
from ..context import DecoderContext


class ParameterPrefixState(State):
    """Forces the fixed text `    "` before a parameter's name.

    Only reached for the *first* parameter of an object; subsequent
    parameters are introduced by ParameterEndState forcing `,\\n    "`
    directly, to avoid duplicating this same fixed text in two places.
    """

    def __init__(self, context: DecoderContext, start_position: int) -> None:
        self.context = context
        self.start_position = start_position
        self.prefix = context.model.encode('    "')[0].tolist()

    def get_allowed_tokens(self, generated_tokens: list[int]) -> list[int]:
        position = len(generated_tokens) - self.start_position

        if position >= len(self.prefix):
            return []

        return [self.prefix[position]]

    def update(self, generated_tokens: list[int]) -> State | None:
        position = len(generated_tokens) - self.start_position

        if position >= len(self.prefix):
            return ParameterNameState(
                self.context,
                len(generated_tokens),
            )

        return None

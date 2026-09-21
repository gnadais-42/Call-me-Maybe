from .State import State
from ..context import DecoderContext


class ParameterEndState(State):
    """Reached right after one parameter's value has been fully generated
    (or immediately after "parameters": { if the function takes none).

    This state needs no input from the model to decide what happens next:
    given the function's full parameter schema and which parameter names
    have already been generated, there are exactly two possibilities, and
    both are fixed strings:

    - parameters remain -> force `,\\n    "` and hand off to a fresh
      ParameterNameState for the next one.
    - none remain -> force `\\n  }\\n}` to close "parameters" and the
      outer object, and finish. get_allowed_tokens() then returns []
      once that closing text is fully emitted, which is the generation
      loop's signal that there is nothing left the grammar permits.
    """

    def __init__(self, context: DecoderContext, start_position: int) -> None:
        self.context = context
        self.start_position = start_position

        if context.current_parameter is not None:
            context.generated_parameters.add(context.current_parameter)
            context.current_parameter = None

        forced_text = context.get_parameter_end_text()
        self._has_remaining_parameters = forced_text != "\n  }\n}"
        self.forced: list[int] = context.model.encode(forced_text)[0].tolist()

    def get_allowed_tokens(self, generated_tokens: list[int]) -> list[int]:
        position = len(generated_tokens) - self.start_position

        if position >= len(self.forced):
            return []

        return [self.forced[position]]

    def update(self, generated_tokens: list[int]) -> State | None:
        position = len(generated_tokens) - self.start_position

        if position < len(self.forced):
            return None

        if not self._has_remaining_parameters:
            return None

        # Imported here rather than at module level to break the import
        # cycle: ParameterNameState -> ParameterValuePrefixState ->
        # StringState/NumberState/BooleanState -> ParameterEndState.
        from .ParameterNameState import ParameterNameState

        return ParameterNameState(self.context, len(generated_tokens))

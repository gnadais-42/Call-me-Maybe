from .State import State
from ..context import DecoderContext
from .StringState import StringState
from .NumberState import NumberState
from .BooleanState import BooleanState


class ParameterValuePrefixState(State):
    """Forces the fixed text after a parameter name, then dispatches to the
    value state matching that parameter's declared type.

    For string-typed parameters, the forced text includes the value's
    opening quote (`": "` rather than `": `). Nothing else in the grammar
    emits it: StringState only ever emits the closing quote (via its own
    terminator logic), so without this, string values would come out with
    a single unbalanced quote instead of a proper opening/closing pair --
    invalid JSON despite "constrained" decoding.

    For number-typed parameters, the trailing space is deliberately *not*
    forced (`":` rather than `": `). JSON doesn't require it -- `"a":-2` is
    exactly as valid as `"a": -2` -- and forcing it as part of an isolated,
    context-free encode() call turned out to actively hurt: it consumes
    the space as its own token before the model gets a turn, meaning a
    tokenizer that would naturally merge the space with a following minus
    sign into one token (very common: many tokenizers represent " -2"
    after a colon as a single token, not "-" then "2") can never produce
    that merged token here, since it no longer exists as an option by the
    time NumberState starts. NumberState now accepts an optional leading
    space as part of its own first token instead, so the model can still
    choose to write one -- just as part of a token it actually has real
    confidence in, rather than one we pre-committed it to.
    """

    def __init__(self, context: DecoderContext, start_position: int) -> None:
        self.context = context
        self.start_position = start_position

        parameter_type = self._parameter_type()

        if parameter_type == "string":
            prefix_text = '": "'
        elif parameter_type == "number":
            prefix_text = '":'
        else:
            prefix_text = '": '

        self.prefix = context.model.encode(prefix_text)[0].tolist()

    def _parameter_type(self) -> str | None:
        function = self.context.selected_function
        name = self.context.current_parameter

        if function is None or name is None:
            return None

        return function.parameters[name].type

    def get_allowed_tokens(self, generated_tokens: list[int]) -> list[int]:
        position = len(generated_tokens) - self.start_position

        if position >= len(self.prefix):
            return []

        return [self.prefix[position]]

    def update(self, generated_tokens: list[int]) -> State | None:
        position = len(generated_tokens) - self.start_position

        if position < len(self.prefix):
            return None

        if self.context.selected_function is None:
            raise RuntimeError("No function selected")

        if self.context.current_parameter is None:
            raise RuntimeError("No parameter selected")

        parameter_type = self._parameter_type()

        if parameter_type == "string":
            return StringState(
                self.context,
                len(generated_tokens),
            )

        if parameter_type == "number":
            return NumberState(
                self.context,
                len(generated_tokens),
            )

        if parameter_type == "boolean":
            return BooleanState(
                self.context,
                len(generated_tokens),
            )

        raise ValueError(
            f"Unsupported parameter type: {parameter_type}"
        )
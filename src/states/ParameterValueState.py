from .State import State
from .ParameterEndState import ParameterEndState
from ..context import DecoderContext


class ParameterValueState(State):
    def __init__(
        self,
        context: DecoderContext,
        start_position: int,
    ) -> None:
        self.context = context
        self.start_position = start_position

        if context.selected_function is None:
            raise RuntimeError("No function selected")

        if context.current_parameter is None:
            raise RuntimeError("No parameter selected")

        parameter = context.selected_function.parameters[
            context.current_parameter
        ]

        self.parameter_type = parameter.type

    def get_allowed_tokens(
        self,
        generated_tokens: list[int],
    ) -> list[int]:
        if self.parameter_type == "string":
            return self._get_string_tokens()

        if self.parameter_type == "number":
            return self._get_number_tokens()

        raise ValueError(
            f"Unsupported parameter type: {self.parameter_type}"
        )

    def _get_string_tokens(self) -> list[int]:
        return self._get_tokens_for_string()

    def _get_number_tokens(self) -> list[int]:
        return self._get_tokens_for_number()

    def update(
        self,
        generated_tokens: list[int],
    ) -> State | None:
        return None

    def _get_tokens_for_string(self) -> list[int]:
        return []

    def _get_tokens_for_number(self) -> list[int]:
        return []
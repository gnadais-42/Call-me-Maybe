from enum import Enum, auto
from .models import FunctionDefinition


class DecoderState(Enum):
    START = auto()
    FUNCTION_NAME = auto()
    PARAMETERS = auto()


class DecoderContext:
    def __init__(self) -> None:
        self.state: DecoderState = DecoderState.START
        self.selected_function: FunctionDefinition | None = None
        self.current_parameter: str | None = None

class StateMachine:
    def __init__(self) -> None:
        self.context = DecoderContext()

    def transition(self) -> None:
        if self.context.state == DecoderState.START:
            self.context.state = DecoderState.FUNCTION_NAME

        elif self.context.state == DecoderState.FUNCTION_NAME:
            self.context.state = DecoderState.PARAMETERS
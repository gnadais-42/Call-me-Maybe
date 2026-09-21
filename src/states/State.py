from abc import ABC, abstractmethod


class State(ABC):
    """Base class for one node of the constrained JSON generation grammar.

    Every concrete state answers two questions about the tokens generated
    so far:

    - get_allowed_tokens: which token ids are grammatically valid as the
      next generated token, given this state?
    - update: has this state's segment of the grammar finished, and if so,
      which state should handle generation from here? Returns None while
      the state is still in progress.

    A state whose get_allowed_tokens returns an empty list with nothing
    left to force (see ParameterEndState) signals that generation is
    complete: there is nothing left the grammar permits, so the caller's
    generation loop should stop.
    """

    @abstractmethod
    def get_allowed_tokens(self, generated_tokens: list[int]) -> list[int]:
        """Return the token ids allowed as the next generated token."""

    @abstractmethod
    def update(self, generated_tokens: list[int]) -> "State | None":
        """Return the next state once this state's segment is complete."""

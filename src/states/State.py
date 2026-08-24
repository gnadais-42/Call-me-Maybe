from abc import ABC, abstractmethod

class State(ABC):

    @abstractmethod
    def get_allowed_tokens(self, generated_tokens: list[int]) -> list[int]:
        pass

    @abstractmethod
    def update(self, generated_tokens: list[int]) -> "State | None":
        pass
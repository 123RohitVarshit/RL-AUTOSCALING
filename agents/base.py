from abc import ABC, abstractmethod


class AutoscalerAgent(ABC):
    @abstractmethod
    def decide(self, state: dict) -> int:
        pass

    def reset(self):
        pass

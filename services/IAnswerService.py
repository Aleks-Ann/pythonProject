from abc import ABC, abstractmethod


class IAnswerService(ABC):

    @abstractmethod
    def answer(self, text: str):
        pass

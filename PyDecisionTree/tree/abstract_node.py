from abc import ABC, abstractmethod


class Node(ABC):
    """ Abstract class of a Node. """

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

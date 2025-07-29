from abc import ABC, abstractmethod
from typing import Any


class Processor(ABC):
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def batch_process(self, *args, **kwargs) -> Any:
        pass
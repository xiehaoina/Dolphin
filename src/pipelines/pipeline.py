from abc import ABC, abstractmethod
from typing import Any, List, Union
from PIL import Image

class Pipeline(ABC):
    """
    所有流水线的基类。
    """

    def __init__(self, config: Any):
        """
        使用配置对象初始化流水线。

        Args:
            config: 配置对象。
        """
        self.config = config

    @abstractmethod
    def run(self, debug: bool = False, **kwargs) -> Any:
        """
        在给定的输入上运行流水线。

        Args:
            **kwargs: 流水线的输入数据。

        Returns:
            流水线的输出。
        """
        raise NotImplementedError
    
from abc import ABC, abstractmethod
from typing import Any, Optional


class ModelLoader(ABC):
    """모델 로더 인터페이스"""

    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        pass

    def load_processor(self, processor_path: str) -> Any:
        pass

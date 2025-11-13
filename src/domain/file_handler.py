from abc import ABC, abstractmethod
from typing import Any, Optional


class FileHandler(ABC):
    """파일 핸들러 인터페이스"""

    @abstractmethod
    def get_file_obj(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def download_file_obj(self, key: str) -> Any:
        pass

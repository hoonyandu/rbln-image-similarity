from abc import ABC, abstractmethod
from typing import Any

class SearchClient(ABC):
    """검색 클라이언트 인터페이스"""

    @abstractmethod
    def __init__(self, env: str) -> None:
        pass

    @abstractmethod
    def get_cluster_info(self) -> Any:
        pass

    @abstractmethod
    def __repr__(self) -> Any:
        pass
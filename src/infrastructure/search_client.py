from opensearchpy import OpenSearch, RequestsHttpConnection
from typing import Dict, Any

from src.domain.search_client import SearchClient
from src.utils.logger import get_logger


class OpenSearchClient(SearchClient):
    """
    OpenSearch 클라이언트
    Args:
        env: 환경

    Returns:
        OpenSearchClient
    """

    def __init__(self, host, port, timeout, response_timeout, use_ssl, verify_certs, username=None, password=None) -> None:
        """
        OpenSearch 클라이언트 초기화
        Args:
            host: 호스트
            port: 포트
            timeout: 타임아웃

        Returns:
            OpenSearchClient
        """
        self.logger = get_logger(__name__)
        
        # 인증 정보 설정
        auth = None
        if username and password:
            auth = (username, password)

        self.client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=auth,
            timeout=timeout,
            response_timeout=response_timeout,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_show_warn=False,  # self-signed 인증서 경고 억제
            connection_class=RequestsHttpConnection,
        )   
        self.logger.info(f"OpenSearchClient initialized with host: {host}, port: {port}, use_ssl: {use_ssl}, auth: {'Yes' if auth else 'No'}")


    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get cluster info
        Args:
            None

        Returns:
            cluster_info: cluster info
        """
        try:
            # () 를 붙여서 실제 API를 호출해야 합니다
            return self.client.cluster.health()
        except Exception as e:
            self.logger.error(f"Error getting cluster info: {e}")
            raise e

    def __repr__(self) -> str:
        """return string representation of the object"""
        return f"OpenSearchClient(host={self.host}, port={self.port}"